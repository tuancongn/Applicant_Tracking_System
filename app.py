"""
PAVN ATS - Flask API Server
Hệ thống đối sánh CV thông minh
"""

import os
import json
import glob
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from cv_parser import parse_cv
from analyzer import analyze_cv_jd, compare_modes, RateLimitError, AI_PROVIDER, AI_MODEL

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Config
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
MAU_1_FOLDER = os.path.join(os.path.dirname(__file__), 'Mau_1')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def load_sample_jds():
    """Load các JD mẫu từ folder Mau_1."""
    jds = []
    jd_files = glob.glob(os.path.join(MAU_1_FOLDER, '*.txt'))
    for filepath in sorted(jd_files):
        filename = os.path.basename(filepath)
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Trích xuất tên công việc và công ty từ filename
        name = filename.replace('.txt', '')

        jds.append({
            'id': filename,
            'name': name,
            'content': content,
            'filepath': filepath,
        })
    return jds


def load_all_cvs():
    """Load tất cả CV mẫu từ folder Mau_1."""
    cvs = []
    pdf_files = sorted(glob.glob(os.path.join(MAU_1_FOLDER, '*.pdf')))
    for cv_path in pdf_files:
        try:
            cv_data = parse_cv(cv_path)
            cvs.append({
                'filename': os.path.basename(cv_path),
                'filepath': cv_path,
                'data': cv_data,
            })
        except Exception as e:
            cvs.append({
                'filename': os.path.basename(cv_path),
                'filepath': cv_path,
                'error': str(e),
            })
    return cvs


# Cache sample data
SAMPLE_JDS = load_sample_jds()
SAMPLE_CVS = load_all_cvs()
SAMPLE_CV = SAMPLE_CVS[0] if SAMPLE_CVS else None  # backward compat


@app.route('/')
def index():
    """Trang chủ."""
    return render_template('index.html')


@app.route('/api/sample-data', methods=['GET'])
def get_sample_data():
    """Lấy dữ liệu mẫu (CVs + JDs)."""
    cv_list = []
    for cv in SAMPLE_CVS:
        if 'data' in cv:
            cv_list.append({
                'filename': cv['filename'],
                'metadata': cv['data']['metadata'],
                'word_count': cv['data']['word_count'],
                'sections': list(cv['data']['sections'].keys()),
            })
        else:
            cv_list.append({'filename': cv['filename'], 'error': cv.get('error')})

    return jsonify({
        'cv': cv_list[0] if cv_list else None,  # backward compat
        'cv_list': cv_list,
        'jd_list': [{'id': jd['id'], 'name': jd['name']} for jd in SAMPLE_JDS],
        'jds': [{
            'id': jd['id'],
            'name': jd['name'],
            'preview': jd['content'][:200] + '...',
        } for jd in SAMPLE_JDS],
    })


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Phân tích CV vs JD. Supports mode: local, ai, hybrid."""
    data = request.json

    cv_text = data.get('cv_text', '')
    jd_text = data.get('jd_text', '')
    jd_id = data.get('jd_id', '')
    cv_id = data.get('cv_id', '')
    mode = data.get('mode', 'hybrid')  # 'local', 'ai', 'hybrid'

    if jd_id and not jd_text:
        for jd in SAMPLE_JDS:
            if jd['id'] == jd_id:
                jd_text = jd['content']
                break

    # Chọn CV theo cv_id hoặc dùng mặc định
    if not cv_text:
        target_cv = SAMPLE_CV  # default
        if cv_id:
            for cv in SAMPLE_CVS:
                if cv['filename'] == cv_id:
                    target_cv = cv
                    break
        if target_cv and 'data' in target_cv:
            cv_text = target_cv['data']['raw_text']

    if not cv_text or not jd_text:
        return jsonify({'error': 'Cần cung cấp CV text và JD text'}), 400

    print(f"\n[API] Phân tích CV: {cv_id or 'Upload CV'} × JD: {jd_id or 'Custom JD'} (Mode: {mode})")

    try:
        result = analyze_cv_jd(cv_text, jd_text, mode=mode)

        if result.get('rate_limited'):
            return jsonify({
                'error': result['error'],
                'rate_limited': True,
                'local_analysis': result['local_analysis'],
                'final_scores': result['final_scores'],
            }), 429

        return jsonify(result)

    except RateLimitError as e:
        return jsonify({'error': str(e), 'rate_limited': True}), 429
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/compare-modes', methods=['POST'])
def compare_modes_endpoint():
    """Chạy 3 mode (local, ai, hybrid) và trả về so sánh."""
    data = request.json
    cv_text = data.get('cv_text', '')
    jd_text = data.get('jd_text', '')
    jd_id = data.get('jd_id', '')

    if jd_id and not jd_text:
        for jd in SAMPLE_JDS:
            if jd['id'] == jd_id:
                jd_text = jd['content']
                break

    if not cv_text and SAMPLE_CV and 'data' in SAMPLE_CV:
        cv_text = SAMPLE_CV['data']['raw_text']

    if not cv_text or not jd_text:
        return jsonify({'error': 'Cần cung cấp CV text và JD text'}), 400

    try:
        results = compare_modes(cv_text, jd_text)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/batch-analyze', methods=['POST'])
def batch_analyze():
    """So sánh CV với nhiều JD cùng lúc."""
    data = request.json
    cv_text = data.get('cv_text', '')
    jd_ids = data.get('jd_ids', [])
    mode = data.get('mode', 'hybrid')

    if not cv_text and SAMPLE_CV and 'data' in SAMPLE_CV:
        cv_text = SAMPLE_CV['data']['raw_text']

    if not cv_text:
        return jsonify({'error': 'Cần cung cấp CV text'}), 400

    results = []
    for jd_id in jd_ids:
        jd_text = ''
        jd_name = ''
        for jd in SAMPLE_JDS:
            if jd['id'] == jd_id:
                jd_text = jd['content']
                jd_name = jd['name']
                break

        if not jd_text:
            continue

        try:
            result = analyze_cv_jd(cv_text, jd_text, mode=mode)
            results.append({
                'jd_id': jd_id,
                'jd_name': jd_name,
                'final_scores': result['final_scores'],
                'local_analysis': result['local_analysis'],
                'ai_analysis': result.get('ai_analysis'),
                'ai_available': result.get('ai_available', False),
                'error': result.get('error'),
                'rate_limited': result.get('rate_limited', False),
            })

            # Nếu bị rate limit, dừng batch
            if result.get('rate_limited'):
                break

        except RateLimitError as e:
            results.append({
                'jd_id': jd_id,
                'jd_name': jd_name,
                'error': str(e),
                'rate_limited': True,
            })
            break
        except Exception as e:
            results.append({
                'jd_id': jd_id,
                'jd_name': jd_name,
                'error': str(e),
            })

    return jsonify({'results': results})


@app.route('/api/upload-cv', methods=['POST'])
def upload_cv():
    """Upload CV PDF."""
    if 'file' not in request.files:
        return jsonify({'error': 'Không tìm thấy file'}), 400

    file = request.files['file']
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Chỉ chấp nhận file PDF'}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        cv_data = parse_cv(filepath)
        return jsonify({
            'filename': file.filename,
            'data': {
                'raw_text': cv_data['raw_text'],
                'metadata': cv_data['metadata'],
                'word_count': cv_data['word_count'],
                'sections': list(cv_data['sections'].keys()),
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("  PAVN ATS - Hệ Thống Đối Sánh CV Thông Minh")
    print("  http://localhost:5000")
    print("=" * 60)
    print(f"  📄 CV mẫu: {SAMPLE_CV['filename'] if SAMPLE_CV else 'Không có'}")
    print(f"  📋 JD mẫu: {len(SAMPLE_JDS)} file(s)")
    for jd in SAMPLE_JDS:
        print(f"     - {jd['name']}")
    print("=" * 60)
    app.run(debug=True, port=5000)
