"""
Test script: So sánh 3 chế độ (Local, AI, Hybrid)
Chạy với JD "AI Engineer at REECO" làm benchmark.
"""
import requests
import json
import time

BASE_URL = "http://localhost:5000"
# JD mặc định: AI Engineer
JD_ID = "AI Engineer at Công ty TNHH Khoa học và Kỹ thuật REECO.txt"

def test_mode(mode: str):
    """Test a single mode."""
    print(f"\n{'='*60}")
    print(f"  MODE: {mode.upper()}")
    print(f"{'='*60}")
    
    start = time.time()
    resp = requests.post(
        f"{BASE_URL}/api/analyze",
        json={'jd_id': JD_ID, 'mode': mode},
        timeout=300,
    )
    elapsed = time.time() - start
    
    if resp.status_code != 200:
        print(f"  ❌ Error {resp.status_code}: {resp.text[:200]}")
        return None
    
    data = resp.json()
    scores = data.get('final_scores', {})
    ai_info = data.get('ai_analysis', {})
    
    print(f"  ⏱  Time: {elapsed:.1f}s")
    print(f"  🤖 AI Provider: {data.get('ai_provider', 'N/A')} | Model: {data.get('ai_model', 'N/A')}")
    print(f"  🔧 Mode: {data.get('mode', 'N/A')}")
    print(f"  ✅ AI Available: {data.get('ai_available', False)}")
    if data.get('error'):
        print(f"  ⚠️  Error: {data['error'][:200]}")
    
    print(f"\n  📊 OVERALL SCORE: {scores.get('overall', 'N/A')}% ({scores.get('match_level', '')})")
    print(f"  ┌──────────────────────────────────────────")
    print(f"  │ Hard Skills:  {scores.get('hard_skills', 'N/A')}%")
    print(f"  │ Experience:   {scores.get('experience', 'N/A')}%")
    print(f"  │ Education:    {scores.get('education', 'N/A')}%")
    print(f"  │ Language:     {scores.get('language', 'N/A')}%")
    print(f"  │ Soft Skills:  {scores.get('soft_skills', 'N/A')}%")
    print(f"  │ Culture Fit:  {scores.get('culture_fit', 'N/A')}%")
    print(f"  └──────────────────────────────────────────")
    
    # AI-specific info
    if ai_info and mode != 'local':
        print(f"\n  📝 AI Summary: {ai_info.get('summary', 'N/A')[:200]}")
        strengths = ai_info.get('strengths', [])
        if strengths:
            print(f"  💪 Strengths: {', '.join(strengths[:3])}")
        weaknesses = ai_info.get('weaknesses', [])
        if weaknesses:
            print(f"  ⚡ Weaknesses: {', '.join(weaknesses[:3])}")
    
    return {'mode': mode, 'time': elapsed, 'scores': scores, 'ai': bool(ai_info), 'error': data.get('error')}


def run_comparison():
    print("\n" + "🔥" * 30)
    print("  PAVN ATS — SO SÁNH 3 CHẾ ĐỘ PHÂN TÍCH")
    print("  JD: AI Engineer at REECO")
    print("  CV: CV_NGUYEN_CONG_LAP_NHAN_AI_ENGINEER_2.pdf")
    print("🔥" * 30)
    
    results = {}
    for mode in ['local', 'ai', 'hybrid']:
        r = test_mode(mode)
        if r:
            results[mode] = r
    
    # Comparison table
    print("\n\n" + "=" * 70)
    print("  📊 BẢNG SO SÁNH 3 CHẾ ĐỘ")
    print("=" * 70)
    print(f"{'Metric':<18} {'LOCAL':>10} {'AI':>10} {'HYBRID':>10}")
    print("-" * 50)
    
    for metric in ['overall', 'hard_skills', 'experience', 'education', 'language', 'soft_skills', 'culture_fit']:
        vals = []
        for mode in ['local', 'ai', 'hybrid']:
            if mode in results:
                vals.append(f"{results[mode]['scores'].get(metric, 'N/A')}%")
            else:
                vals.append("ERR")
        print(f"{metric:<18} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10}")
    
    print("-" * 50)
    
    # Time comparison
    for mode in ['local', 'ai', 'hybrid']:
        if mode in results:
            print(f"⏱  {mode:<14} → {results[mode]['time']:.1f}s")
    
    print("\n" + "=" * 70)
    print("  📋 PHÂN TÍCH")
    print("=" * 70)
    
    if 'local' in results and 'ai' in results:
        local_s = results['local']['scores'].get('overall', 0)
        ai_s = results['ai']['scores'].get('overall', 0)
        diff = abs(local_s - ai_s) if isinstance(local_s, (int, float)) and isinstance(ai_s, (int, float)) else 'N/A'
        print(f"  → Chênh lệch Local vs AI: {diff} điểm")
        
        if isinstance(diff, (int, float)):
            if diff <= 5:
                print(f"  → Đánh giá: ✅ Local ML rất chính xác, gần như bằng AI")
            elif diff <= 15:
                print(f"  → Đánh giá: ⚠️ Có chênh lệch nhưng chấp nhận được")
            else:
                print(f"  → Đánh giá: ❌ Chênh lệch lớn — cần cải thiện Local ML")
    
    if 'local' in results:
        local_time = results['local']['time']
        print(f"\n  → Local ML chạy trong {local_time:.1f}s (deterministic, miễn phí)")
    if 'ai' in results:
        ai_time = results['ai']['time']
        print(f"  → AI chạy trong {ai_time:.1f}s (phụ thuộc API, có thể mất phí)")


if __name__ == '__main__':
    run_comparison()
