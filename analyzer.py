"""PAVN ATS - Analyzer Module v4
Domain-Agnostic Hybrid Semantic + AI Engine

Layer 1: Semantic Embedding + Keyword Extraction
  - True semantic similarity via multilingual-e5-large
  - Supports 100+ languages (Vietnamese, English, Korean, Japanese, Chinese...)
  - Section-level Late Fusion (skills↔skills, exp↔exp)
Layer 2: AI API — auto-detects provider from API key:
  - ghp_* → GitHub Models (DeepSeek-R1-0528)
  - sk-or-v1-* → OpenRouter
  - AIzaSy* → Google Gemini 2.5 Flash

Works with ANY industry — not just IT.
"""

import re
import os
import json
import hashlib
import time
import random
from collections import Counter
import requests
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# AI PROVIDER CONFIGURATION
# Reads ghp_ keys from keyfull.txt for GitHub Models (DeepSeek)
# Supports: ghp_ (GitHub), sk-or-v1- (OpenRouter), AIzaSy (Gemini)
# ============================================================
ENABLE_AI = os.getenv('ENABLE_AI', 'YES').strip().upper() == 'YES'

# Path to keyfull.txt containing API keys
KEYFULL_PATH = os.getenv(
    'KEYFULL_PATH',
    r'C:\Users\Admin\Desktop\Gpm_Sript\Testnet_Trading_BTC\Bybit_testnet\keyfull.txt'
)

def _load_ghp_keys() -> list:
    """Load all ghp_ keys from keyfull.txt."""
    keys = []
    try:
        with open(KEYFULL_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                for part in parts:
                    part = part.strip()
                    if part.startswith('ghp_'):
                        keys.append(part)
    except FileNotFoundError:
        print(f"[AI Config] keyfull.txt not found at {KEYFULL_PATH}")
    return keys

# Load all available ghp_ keys
GHP_KEYS = _load_ghp_keys() if ENABLE_AI else []

# Fallback: also check .env API_KEY
API_KEY = os.getenv('API_KEY', '').strip()

def _detect_provider():
    """Auto-detect AI provider. Priority: ghp_ keys from file > .env API_KEY."""
    if not ENABLE_AI:
        return None, None
    if GHP_KEYS:
        return 'github', 'DeepSeek-R1-0528'
    if API_KEY.startswith('sk-or-v1-'):
        return 'openrouter', 'stepfun/step-3.5-flash:free'
    elif API_KEY.startswith('ghp_'):
        return 'github', 'DeepSeek-R1-0528'
    elif API_KEY.startswith('AIzaSy'):
        return 'gemini', 'gemini-2.5-flash'
    elif API_KEY:
        return 'unknown', 'unknown'
    return None, None

def _get_api_key() -> str:
    """Get an API key. For GitHub, pick a random ghp_ key from pool."""
    if GHP_KEYS:
        return random.choice(GHP_KEYS)
    return API_KEY

AI_PROVIDER, AI_MODEL = _detect_provider()
if ENABLE_AI:
    key_count = len(GHP_KEYS) if GHP_KEYS else (1 if API_KEY else 0)
    print(f"[AI Config] Provider: {AI_PROVIDER or 'None'} | Model: {AI_MODEL or 'No key'} | Keys: {key_count}")
else:
    print("[AI Config] AI is DISABLED (local ML only).")

# ============================================================
# SEMANTIC EMBEDDING MODEL — multilingual-e5-large
# Lazy-loaded: only loads model on first use
# ============================================================
_embedding_model = None
_embedding_ready = False

def _get_embedding_model():
    """Lazy-load multilingual-e5-large (loads once only)."""
    global _embedding_model, _embedding_ready
    if _embedding_ready:
        return _embedding_model
    try:
        from sentence_transformers import SentenceTransformer
        print("[Embedding] Loading multilingual-e5-large...")
        start = time.time()
        try:
            # Prefer loading from local cache (fast, no Internet required)
            _embedding_model = SentenceTransformer(
                'intfloat/multilingual-e5-large',
                local_files_only=True
            )
        except Exception:
            # First run — no cache yet, downloading from HuggingFace Hub (~1.1GB)
            print("[Embedding] No local cache, downloading from HuggingFace Hub (~1.1GB)...")
            _embedding_model = SentenceTransformer('intfloat/multilingual-e5-large')
        elapsed = time.time() - start
        print(f"[Embedding] Model loaded in {elapsed:.1f}s ✔")
        _embedding_ready = True
    except Exception as e:
        print(f"[Embedding] Failed to load: {e}")
        print("[Embedding] Falling back to TF-IDF")
        _embedding_model = None
        _embedding_ready = True  # Don't retry
    return _embedding_model


def _compute_similarity(text_a: str, text_b: str) -> float:
    """Compute semantic similarity between two texts using embedding model.
    Returns cosine similarity [0, 1]."""
    model = _get_embedding_model()
    if model is None:
        return -1.0  # Signal to use fallback

    # multilingual-e5 requires "query: " or "passage: " prefix
    emb = model.encode(
        [f"query: {text_a}", f"passage: {text_b}"],
        normalize_embeddings=True,
    )
    # Cosine similarity (vectors already normalized)
    similarity = float(np.dot(emb[0], emb[1]))
    return similarity


# ============================================================
# RESULT CACHE — same input → same output (no redundant API calls)
# ============================================================
_result_cache = {}


class RateLimitError(Exception):
    """Error when rate limited by AI API."""
    pass


# ============================================================
# BILINGUAL MAP — for cross-language matching
# English CV + Vietnamese JD (or vice versa) → still matches
# ============================================================
BILINGUAL_MAP = {
    # Common job terms
    'kinh nghiệm': 'experience', 'kỹ năng': 'skill', 'yêu cầu': 'requirement',
    'trách nhiệm': 'responsibility', 'mô tả công việc': 'job description',
    'ứng viên': 'candidate', 'tuyển dụng': 'recruitment',
    'phát triển': 'development', 'thiết kế': 'design',
    'phân tích': 'analysis', 'báo cáo': 'report',
    'khách hàng': 'customer', 'dự án': 'project',
    'đào tạo': 'training', 'nghiên cứu': 'research',
    'hỗ trợ': 'support', 'bán hàng': 'sales',
    'quản lý': 'management', 'giám sát': 'supervision',
    'triển khai': 'deployment', 'vận hành': 'operation',
    'tài chính': 'finance', 'kế toán': 'accounting',
    'nhân sự': 'human resources',
    # Soft skills
    'giao tiếp': 'communication', 'sáng tạo': 'creative',
    'chủ động': 'proactive', 'làm việc nhóm': 'teamwork',
    'lãnh đạo': 'leadership', 'giải quyết vấn đề': 'problem solving',
    'tư duy': 'thinking', 'thích ứng': 'adaptable',
    'tỉ mỉ': 'detail oriented', 'năng động': 'dynamic',
    'chịu áp lực': 'work under pressure',
    # Education
    'trình độ': 'qualification', 'bằng cấp': 'degree',
    'đại học': 'university', 'cao đẳng': 'college',
    'cử nhân': 'bachelor', 'thạc sĩ': 'master', 'tiến sĩ': 'phd',
    'tốt nghiệp': 'graduated',
    # Language
    'tiếng anh': 'english', 'tiếng việt': 'vietnamese',
    'tiếng nhật': 'japanese', 'tiếng trung': 'chinese',
    # IT-specific (for when JD is in Vietnamese but CV is in English)
    'phần mềm': 'software', 'lập trình': 'programming',
    'cơ sở dữ liệu': 'database', 'máy chủ': 'server',
    'trí tuệ nhân tạo': 'artificial intelligence',
    'học máy': 'machine learning', 'mạng': 'network',
    # Non-IT domains
    'luật': 'law', 'pháp luật': 'legal', 'hợp đồng': 'contract',
    'bất động sản': 'real estate', 'nhà hàng': 'restaurant',
    'khách sạn': 'hotel', 'du lịch': 'tourism',
    'chế biến': 'cooking', 'thực phẩm': 'food',
    'lái xe': 'driver', 'giao thông': 'traffic',
    'xây dựng': 'construction', 'vận tải': 'transportation',
}

# Vietnamese stop words
VI_STOP_WORDS = {
    'và', 'của', 'có', 'là', 'được', 'cho', 'các', 'một', 'trong', 'với',
    'này', 'đã', 'để', 'từ', 'về', 'theo', 'đến', 'khi', 'không', 'cần',
    'phải', 'nếu', 'hoặc', 'hay', 'thì', 'vì', 'do', 'bởi', 'tại',
    'những', 'nhưng', 'mà', 'bằng', 'qua', 'trên', 'dưới', 'ngoài',
    'giữa', 'sau', 'trước', 'lại', 'cũng', 'đang', 'sẽ', 'rất', 'chỉ',
    'vào', 'lên', 'xuống', 'ra', 'đi', 'đây', 'đó', 'ở', 'nào', 'ai',
    'gì', 'thể', 'còn', 'như', 'tuy', 'mỗi', 'nhiều', 'ít', 'đều',
    'việc', 'người', 'năm', 'tháng', 'ngày', 'tốt', 'tốn', 'làm',
    'trở', 'nên', 'hơn', 'nhất', 'khác', 'mới', 'hiện',
}

# English stop words (minimal set)
EN_STOP_WORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
    'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'between', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
    'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
    'than', 'too', 'very', 'just', 'or', 'and', 'but', 'if', 'while',
    'about', 'up', 'out', 'off', 'over', 'this', 'that', 'it', 'its',
    'we', 'they', 'you', 'he', 'she', 'our', 'your', 'their', 'my',
}

# Generic job terms that appear in ALL job descriptions regardless of domain.
# These should NOT count as meaningful keyword matches.
JOB_GENERIC_STOP_WORDS = {
    # EN generic
    'experience', 'skill', 'skills', 'requirement', 'requirements',
    'responsibility', 'responsibilities', 'qualification', 'qualifications',
    'ability', 'knowledge', 'proficiency', 'strong', 'good',
    'work', 'working', 'team', 'management', 'manage', 'manager',
    'company', 'organization', 'salary', 'bonus', 'benefit', 'benefits',
    'insurance', 'position', 'candidate', 'apply', 'job', 'career',
    'development', 'project', 'projects', 'environment',
    'communication', 'problem', 'solving', 'well', 'time',
    'report', 'reporting', 'support', 'provide', 'ensure',
    'include', 'including', 'related', 'based', 'minimum', 'least',
    'preferred', 'required', 'plus', 'years', 'year', 'months',
    'full', 'part', 'level', 'senior', 'junior', 'lead',
    'training', 'degree', 'bachelor', 'master', 'university',
    'english', 'vietnamese', 'professional', 'relevant',
    # VI generic
    'công', 'việc', 'tuyển', 'dụng', 'lương', 'thưởng', 'bảo', 'hiểm',
    'quyền', 'lợi', 'yêu', 'cầu', 'trách', 'nhiệm', 'ứng', 'viên',
    'vị', 'trí', 'kinh', 'nghiệm', 'trình', 'độ', 'tốt', 'nghiệp',
    'khả', 'năng', 'tốt', 'giỏi', 'thành', 'thạo', 'liên', 'quan',
    'môi', 'trường', 'chuyên', 'nghiệp', 'phát', 'triển',
    'quản', 'lý', 'nhóm', 'báo', 'cáo', 'hỗ', 'trợ',
}

ALL_STOP_WORDS = VI_STOP_WORDS | EN_STOP_WORDS | JOB_GENERIC_STOP_WORDS

# IT job indicators (for auto-detection)
IT_INDICATORS = [
    'software', 'developer', 'engineer', 'programming', 'lập trình',
    'phần mềm', 'backend', 'frontend', 'fullstack', 'devops',
    'database', 'api', 'framework', 'coding', 'code',
    'python', 'java', 'javascript', 'typescript', 'react', 'node',
    'machine learning', 'deep learning', 'artificial intelligence',
    'data scientist', 'data engineer',
    'tester', 'qa', 'qc', 'automation test', 'selenium',
    'cloud', 'aws', 'azure', 'docker', 'kubernetes',
    'web developer', 'mobile developer', 'ios', 'android',
    'sql', 'nosql', 'mongodb', 'postgresql',
    'git', 'ci/cd', 'agile', 'scrum',
]

# IT-specific skills (for bonus matching when JD is IT)
IT_SKILLS = [
    'python', 'java', 'javascript', 'typescript', 'c\\+\\+', 'c#',
    'go', 'rust', 'ruby', 'php', 'swift', 'kotlin',
    'react', 'reactjs', 'vue', 'vuejs', 'angular',
    'next\\.?js', 'node\\.?js', 'express', 'django', 'flask', 'fastapi',
    'spring', 'spring boot', 'microservice', 'rest\\s*api', 'graphql',
    'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch',
    'docker', 'kubernetes', 'aws', 'azure', 'gcp', 'ci/cd',
    'jenkins', 'github actions', 'git', 'linux', 'nginx',
    'kafka', 'rabbitmq',
    'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy',
    'langchain', 'llamaindex', 'llm', 'rag', 'openai', 'gpt', 'gemini',
    'selenium', 'playwright', 'cypress', 'jest', 'pytest', 'junit',
    'agile', 'scrum', 'jira', 'confluence',
    'html', 'css', 'sass', 'tailwind',
    'maven', 'gradle', 'webpack', 'vite',
    'tdd', 'bdd', 'oop', 'solid', 'design pattern',
    'machine learning', 'deep learning', 'nlp', 'computer vision',
    'vector database', 'pinecone', 'chromadb', 'weaviate',
    'etl', 'data pipeline', 'airflow',
    'supabase', 'firebase', 'prisma',
]

SOFT_SKILLS = [
    # English
    'teamwork', 'leadership', 'communication', 'problem.solving',
    'critical thinking', 'time management', 'creative', 'adaptab',
    'collaborat', 'mentor', 'presentation', 'analytical',
    'detail.oriented', 'self.motivated', 'proactive', 'interpersonal',
    # Vietnamese
    'làm việc nhóm', 'lãnh đạo', 'giao tiếp', 'giải quyết vấn đề',
    'tư duy', 'quản lý thời gian', 'sáng tạo', 'thích ứng',
    'hợp tác', 'hướng dẫn', 'phân tích', 'tỉ mỉ',
    'chủ động', 'năng động', 'chịu áp lực',
]

EDUCATION_LEVELS = {
    'phd': 5, 'tiến sĩ': 5, 'doctorate': 5,
    'master': 4, 'thạc sĩ': 4, "master's": 4,
    'bachelor': 3, 'cử nhân': 3, 'đại học': 3, "bachelor's": 3,
    'college': 2, 'cao đẳng': 2, 'associate': 2, 'trung cấp': 2,
    'high school': 1, 'trung học': 1, 'phổ thông': 1,
}


# ============================================================
# LOCAL ANALYZER — Domain-Agnostic, Deterministic
# ============================================================
class LocalAnalyzer:
    """
    Domain-agnostic local ML analyzer.
    Works with ANY industry, not just IT.
    Supports cross-language (EN↔VI).
    """

    def __init__(self, cv_text: str, jd_text: str):
        self.cv_original = cv_text
        self.jd_original = jd_text
        self.cv_text = cv_text.lower()
        self.jd_text = jd_text.lower()

        # Normalize for cross-language matching
        self.cv_normalized = self._normalize_bilingual(self.cv_text)
        self.jd_normalized = self._normalize_bilingual(self.jd_text)

        # Auto-detect if IT job
        self.is_it = self._detect_it_job()

    def _normalize_bilingual(self, text: str) -> str:
        """Add English equivalents alongside Vietnamese terms
        so TF-IDF can find matches across languages."""
        normalized = text
        for vi, en in BILINGUAL_MAP.items():
            if vi in normalized:
                normalized = normalized.replace(vi, f'{vi} {en}')
        return normalized

    def _detect_it_job(self) -> bool:
        """Auto-detect if JD is for an IT position."""
        count = sum(1 for ind in IT_INDICATORS if ind in self.jd_text)
        return count >= 3

    def analyze(self) -> dict:
        """Run comprehensive domain-agnostic analysis with semantic embeddings."""
        # 1. Dynamic keyword extraction + matching
        jd_keywords = self._extract_keywords(self.jd_normalized)
        cv_keywords = self._extract_keywords(self.cv_normalized)
        keyword_match = self._match_keywords(jd_keywords, cv_keywords)

        # 2. Semantic Similarity (multilingual-e5-large)
        semantic_result = self._semantic_similarity()
        semantic_score = semantic_result['overall']

        # 3. N-gram overlap (keep as supplementary signal)
        ngram_score = self._ngram_overlap()

        # 4. IT-specific skill matching (only if IT job detected)
        it_skills = self._match_it_skills() if self.is_it else {
            'score': 0.0, 'matched': [], 'missing': [],
            'extra_in_cv': [], 'jd_required': [], 'cv_has': [],
        }

        # 5. Domain-agnostic dimensions
        experience = self._match_experience(semantic_score)
        education = self._match_education()
        language = self._match_language()
        soft_skills = self._match_soft_skills()

        # 6. Calculate composite local score
        # Semantic similarity now plays the MAIN role (replaces TF-IDF)
        if self.is_it:
            local_score = (
                semantic_score * 0.30 +          # Semantic (main signal)
                keyword_match['score'] * 0.15 +  # Keyword refinement
                it_skills['score'] * 0.20 +      # IT-specific
                experience['score'] * 0.15 +
                education['score'] * 0.08 +
                language['score'] * 0.05 +
                soft_skills['score'] * 0.04 +
                ngram_score * 0.03
            )
        else:
            local_score = (
                semantic_score * 0.40 +          # Semantic (main signal)
                keyword_match['score'] * 0.20 +  # Keyword refinement
                experience['score'] * 0.15 +
                education['score'] * 0.08 +
                language['score'] * 0.07 +
                soft_skills['score'] * 0.05 +
                ngram_score * 0.05
            )

        # Domain mismatch penalty (only for clearly unrelated domains)
        combined_domain_signal = (semantic_score + keyword_match['score']) / 2
        if combined_domain_signal < 10:
            local_score *= 0.30  # 70% penalty: completely different domain (e.g. Chef vs IT)
        elif combined_domain_signal < 20:
            local_score *= 0.55  # 45% penalty: very different
        elif combined_domain_signal < 30:
            local_score *= 0.80  # 20% penalty: somewhat different

        # Build backward-compatible result
        if self.is_it:
            hard_skills = it_skills
        else:
            hard_skills = {
                'score': keyword_match['score'],
                'matched': keyword_match['matched'],
                'missing': keyword_match['missing'][:15],
                'extra_in_cv': [k for k in cv_keywords[:20] if k not in set(jd_keywords)],
                'jd_required': jd_keywords[:20],
                'cv_has': cv_keywords[:20],
            }

        return {
            'local_score': round(local_score, 1),
            'is_it_job': self.is_it,
            'semantic_similarity': semantic_result,
            'semantic_score': semantic_score,
            'keyword_match': keyword_match,
            'ngram_overlap': ngram_score,
            'hard_skills': hard_skills,
            'it_skills': it_skills,
            'soft_skills': soft_skills,
            'experience': experience,
            'education': education,
            'language_match': language,
        }

    def _extract_keywords(self, text: str, top_n: int = 50) -> list:
        """Extract important keywords from text using word frequency analysis.
        Supports Vietnamese compound words via bigram extraction to prevent
        false positive matches from individual syllables (e.g., 'giao', 'toàn').

        Vietnamese is an isolating language where compound words are written
        with spaces between syllables. This method extracts bigrams as primary
        keywords for Vietnamese text, ensuring 'giao tiếp' (communication) and
        'giao thông' (traffic) are treated as distinct keywords."""
        # Extract all words (including Vietnamese with diacritics)
        word_pattern = r'\b[a-zA-Zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]{2,}\b'
        all_words = re.findall(word_pattern, text)

        # Filter stop words for single-word analysis
        filtered = [w for w in all_words if w not in ALL_STOP_WORDS and len(w) > 2]

        # Detect significant Vietnamese content (≥3 words with diacritics)
        vi_diacritics = set('àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ')
        vi_count = sum(1 for w in all_words if any(c in vi_diacritics for c in w))

        if vi_count >= 3:
            # Vietnamese: extract bigrams as compound words
            # Bigrams are extracted from raw word list (preserving true adjacency)
            bigrams = []
            for i in range(len(all_words) - 1):
                w1, w2 = all_words[i], all_words[i + 1]
                if (w1 not in ALL_STOP_WORDS and w2 not in ALL_STOP_WORDS
                        and len(w1) > 1 and len(w2) > 1):
                    bigrams.append(f"{w1} {w2}")

            bigram_freq = Counter(bigrams)
            word_freq = Counter(filtered)

            # Track short Vietnamese syllables (≤4 chars with diacritics)
            # that appear in frequent bigrams — these are compound word parts
            syllables_covered = set()
            for bg, _ in bigram_freq.most_common(30):
                for part in bg.split():
                    if any(c in vi_diacritics for c in part) and len(part) <= 4:
                        syllables_covered.add(part)

            # Build combined keyword list
            keywords = []
            # Priority 1: Vietnamese compound words (bigrams)
            for bg, _ in bigram_freq.most_common(25):
                keywords.append(bg)
            # Priority 2: single words NOT already covered as syllables
            for w, _ in word_freq.most_common(top_n):
                if w in syllables_covered:
                    continue  # Skip: VN syllable already captured by bigram
                keywords.append(w)

            # Deduplicate preserving order
            seen = set()
            return [k for k in keywords if not (k in seen or seen.add(k))][:top_n]
        else:
            # Non-Vietnamese text: standard single-word extraction
            word_freq = Counter(filtered)
            return [word for word, _ in word_freq.most_common(top_n)]

    def _match_keywords(self, jd_keywords: list, cv_keywords: list) -> dict:
        """Match dynamically extracted keywords — works for ANY domain."""
        if not jd_keywords:
            return {'score': 50.0, 'matched': [], 'missing': [], 'total_jd': 0}

        # Take top 30 JD keywords
        jd_top = set(jd_keywords[:30])
        cv_set = set(cv_keywords)

        matched = jd_top & cv_set
        missing = jd_top - cv_set

        # Score based on match ratio
        match_ratio = len(matched) / max(len(jd_top), 1)
        score = min(match_ratio * 110, 100)

        return {
            'score': round(score, 1),
            'matched': sorted(list(matched)),
            'missing': sorted(list(missing)),
            'total_jd': len(jd_top),
        }

    def _semantic_similarity(self) -> dict:
        """Compute semantic similarity using multilingual-e5-large.
        Includes overall + section-level Late Fusion."""
        # Overall similarity
        overall_sim = _compute_similarity(self.cv_original, self.jd_original)

        if overall_sim < 0:
            # Embedding model not available — return 0 with clear signal
            # (no TF-IDF fallback: it produces inaccurate results)
            print("[Embedding] Model unavailable. Install sentence-transformers and ensure multilingual-e5-large is cached.")
            return {
                'overall': 0.0,
                'method': 'model_unavailable',
                'sections': {},
            }

        # Section-level Late Fusion:
        # Extract sections and compare each pair separately
        sections = {}
        cv_sections = self._extract_sections(self.cv_original)
        jd_sections = self._extract_sections(self.jd_original)

        section_pairs = [
            ('skills', 'skills'),
            ('experience', 'requirements'),
            ('education', 'education'),
        ]

        for cv_key, jd_key in section_pairs:
            cv_sec = cv_sections.get(cv_key, '')
            jd_sec = jd_sections.get(jd_key, '')
            if cv_sec and jd_sec:
                sim = _compute_similarity(cv_sec, jd_sec)
                sections[cv_key] = round(sim * 100, 1)

        # Scale: e5-large cosine similarity interpretation
        # Cross-language pairs (EN↔JP, VN↔JP) typically get 0.74-0.82
        # Same-language, same-domain pairs get 0.82-0.92
        # Completely unrelated pairs get 0.65-0.72
        # Use wider range to avoid over-penalizing cross-language matches
        floor = 0.65
        ceiling = 0.90
        overall_scaled = max(0, (overall_sim - floor) / (ceiling - floor)) * 100
        overall_scaled = min(overall_scaled, 100)

        return {
            'overall': round(overall_scaled, 1),
            'raw_cosine': round(overall_sim, 4),
            'method': 'multilingual-e5-large',
            'sections': sections,
        }

    def _extract_sections(self, text: str) -> dict:
        """Extract rough sections from CV or JD for Late Fusion."""
        text_lower = text.lower()
        sections = {}

        # Skills section
        skills_patterns = [
            r'(?:skills?|kỹ năng|technical|công nghệ)[:\s]*(.{50,500})',
            r'(?:technologies?|tools?|công cụ)[:\s]*(.{30,300})',
        ]
        for pat in skills_patterns:
            m = re.search(pat, text_lower, re.IGNORECASE | re.DOTALL)
            if m:
                sections['skills'] = m.group(1)[:500]
                break

        # Experience/Requirements section
        exp_patterns = [
            r'(?:experience|kinh nghiệm|responsibilities?|trách nhiệm|yêu cầu)[:\s]*(.{50,800})',
            r'(?:requirements?|job description|mô tả)[:\s]*(.{50,800})',
        ]
        for pat in exp_patterns:
            m = re.search(pat, text_lower, re.IGNORECASE | re.DOTALL)
            if m:
                sections['requirements' if 'requirements' not in sections else 'experience'] = m.group(1)[:800]
                if 'requirements' in sections and 'experience' not in sections:
                    sections['experience'] = sections['requirements']
                break

        # Education section
        edu_patterns = [
            r'(?:education|học vấn|bằng cấp|trình độ)[:\s]*(.{30,300})',
        ]
        for pat in edu_patterns:
            m = re.search(pat, text_lower, re.IGNORECASE | re.DOTALL)
            if m:
                sections['education'] = m.group(1)[:300]
                break

        return sections

    def _ngram_overlap(self, n: int = 2) -> float:
        """Calculate bi-gram overlap ratio."""
        def get_ngrams(text, n):
            words = re.findall(r'\b\w{2,}\b', text.lower())
            words = [w for w in words if w not in ALL_STOP_WORDS]
            return set(tuple(words[i:i+n]) for i in range(len(words) - n + 1))

        cv_ngrams = get_ngrams(self.cv_normalized, n)
        jd_ngrams = get_ngrams(self.jd_normalized, n)

        if not jd_ngrams:
            return 50.0

        overlap = cv_ngrams & jd_ngrams
        ratio = len(overlap) / max(len(jd_ngrams), 1)

        # Scale: raw overlap ratio is typically 0.01-0.20
        scaled = min(ratio * 400, 100)
        return round(scaled, 1)

    def _match_it_skills(self) -> dict:
        """IT-specific skill matching (only called for IT jobs)."""
        jd_skills = []
        cv_skills = []

        for skill in IT_SKILLS:
            pattern = r'\b' + skill + r'\b'
            clean_name = re.sub(r'\\[+.?*s]', '', skill).replace('\\', '')
            if re.search(pattern, self.jd_text, re.IGNORECASE):
                jd_skills.append(clean_name)
            if re.search(pattern, self.cv_text, re.IGNORECASE):
                cv_skills.append(clean_name)

        jd_set = set(jd_skills)
        cv_set = set(cv_skills)
        matched = jd_set & cv_set
        missing = jd_set - cv_set
        extra = cv_set - jd_set

        score = (len(matched) / max(len(jd_set), 1)) * 100

        return {
            'score': round(score, 1),
            'jd_required': sorted(list(jd_set)),
            'cv_has': sorted(list(cv_set)),
            'matched': sorted(list(matched)),
            'missing': sorted(list(missing)),
            'extra_in_cv': sorted(list(extra)),
        }

    def _match_soft_skills(self) -> dict:
        """Match soft skills (bilingual)."""
        jd_soft, cv_soft = [], []

        for skill in SOFT_SKILLS:
            pattern = skill
            if re.search(pattern, self.jd_text, re.IGNORECASE):
                jd_soft.append(skill.replace('.', ' '))
            if re.search(pattern, self.cv_text, re.IGNORECASE):
                cv_soft.append(skill.replace('.', ' '))

        jd_set = set(jd_soft)
        cv_set = set(cv_soft)
        matched = jd_set & cv_set

        score = (len(matched) / max(len(jd_set), 1)) * 100

        return {
            'score': round(min(score, 100), 1),
            'matched': sorted(list(matched)),
            'missing': sorted(list(jd_set - cv_set)),
        }

    def _match_experience(self, semantic_score: float) -> dict:
        """Match experience years (domain-agnostic, bilingual)."""
        jd_years_patterns = [
            r'(\d+)\+?\s*(?:năm|years?)\s*(?:kinh\s*nghiệm|experience)',
            r'(?:tối\s*thiểu|at\s*least|minimum)\s*(\d+)\s*(?:năm|years?)',
            r'(\d+)\s*-\s*\d+\s*(?:năm|years?)',
        ]
        jd_years = 0
        for pat in jd_years_patterns:
            m = re.search(pat, self.jd_text, re.IGNORECASE)
            if m:
                jd_years = int(m.group(1))
                break

        cv_years_patterns = [
            r'(\d+)\+?\s*(?:năm|years?)\s*(?:of\s+)?(?:kinh\s*nghiệm|experience)',
            r'(?:over|hơn|trên)\s*(\d+)\s*(?:năm|years?)',
        ]
        cv_years = 0
        for pat in cv_years_patterns:
            m = re.search(pat, self.cv_text, re.IGNORECASE)
            if m:
                cv_years = max(cv_years, int(m.group(1)))

        if cv_years == 0:
            year_mentions = re.findall(r'20[12]\d', self.cv_text)
            if len(year_mentions) >= 2:
                years = [int(y) for y in year_mentions]
                cv_years = max(years) - min(years)

        if jd_years == 0:
            score = 40.0  # Unknown requirement → neutral-low (not assumed match)
        elif cv_years >= jd_years:
            score = 100.0
        elif cv_years > 0:
            score = (cv_years / jd_years) * 100
        else:
            score = 30.0

        # DOMAIN PENALTY for Experience:
        # If high experience but wrong domain (low semantic_score), it's worthless
        # Penalty thresholds calibrated for cross-language matching
        if semantic_score < 25.0:
            score *= 0.1  # 90% penalty: completely wrong field (Driver ↔ IT)
        elif semantic_score < 40.0:
            score *= 0.4  # 60% penalty: significantly different field
        elif semantic_score < 55.0:
            score *= 0.7  # 30% penalty: related but not exact domain match


        return {
            'score': round(min(score, 100), 1),
            'jd_requires_years': jd_years,
            'cv_has_years': cv_years,
        }

    def _match_education(self) -> dict:
        """Match education level (bilingual)."""
        jd_level, cv_level = 0, 0
        jd_edu_name, cv_edu_name = 'Unknown', 'Unknown'

        for kw, level in EDUCATION_LEVELS.items():
            if kw in self.jd_text:
                if level > jd_level:
                    jd_level = level
                    jd_edu_name = kw.title()
            if kw in self.cv_text:
                if level > cv_level:
                    cv_level = level
                    cv_edu_name = kw.title()

        if jd_level == 0:
            score = 40.0  # Unknown requirement → neutral-low
        elif cv_level >= jd_level:
            score = 100.0
        elif cv_level > 0:
            score = (cv_level / jd_level) * 80
        else:
            score = 30.0

        return {
            'score': round(min(score, 100), 1),
            'jd_requires': jd_edu_name,
            'cv_has': cv_edu_name,
        }

    def _match_language(self) -> dict:
        """Match language requirements (bilingual)."""
        lang_patterns = {
            'English': r'(?i)(english|tiếng\s*anh|ielts|toeic|toefl|intermediate|upper.intermediate|advanced)',
            'Japanese': r'(?i)(japanese|tiếng\s*nhật|jlpt|n[1-5])',
            'Chinese': r'(?i)(chinese|tiếng\s*trung|hsk)',
            'Korean': r'(?i)(korean|tiếng\s*hàn|topik)',
            'Vietnamese': r'(?i)(vietnamese|tiếng\s*việt)',
        }

        jd_langs, cv_langs = [], []
        for lang, pattern in lang_patterns.items():
            if re.search(pattern, self.jd_text):
                jd_langs.append(lang)
            if re.search(pattern, self.cv_text):
                cv_langs.append(lang)

        jd_set = set(jd_langs)
        cv_set = set(cv_langs)
        matched = jd_set & cv_set

        if not jd_set:
            score = 40.0  # Unknown requirement → neutral-low
        else:
            score = (len(matched) / len(jd_set)) * 100

        return {
            'score': round(score, 1),
            'jd_requires': sorted(list(jd_set)),
            'cv_has': sorted(list(cv_set)),
            'matched': sorted(list(matched)),
            'missing': sorted(list(jd_set - cv_set)),
        }


# ============================================================
# AI ANALYZER — Multi-provider (Gemini + DeepSeek)
# Auto-detect provider from API key prefix
# ============================================================
class AIAnalyzer:
    """
    Layer 2: AI-powered semantic analysis.
    Supports:
      - ghp_* → GitHub Models API (DeepSeek-R1-0528)
      - AIzaSy* → Google Gemini 2.5 Flash
    """

    def __init__(self):
        if not API_KEY and not GHP_KEYS:
            raise ValueError("No API key configured. Check .env or keyfull.txt")
        self.provider = AI_PROVIDER
        self.model = AI_MODEL

    def analyze(self, cv_text: str, jd_text: str, local_results: dict) -> dict:
        prompt = self._build_prompt(cv_text, jd_text, local_results)
        response = self._call_api(prompt)
        return self._parse_response(response)

    def _build_prompt(self, cv_text: str, jd_text: str, local_results: dict) -> str:
        # Always respond in English for international presentation
        lang = "English"
        is_it = local_results.get('is_it_job', True)
        local_score = local_results.get('local_score', 50)

        return f"""You are an expert ATS (Applicant Tracking System) analyst and career advisor.
Analyze the following CV against the Job Description and provide a detailed assessment.

IMPORTANT RULES:
1. Respond ONLY with a valid JSON object. No markdown, no code blocks, no explanation.
2. Respond in {lang}.
3. This job is {"an IT/Tech position" if is_it else "a NON-IT position"}.
4. The LOCAL ML analysis gave an overall score of {local_score}%. Your overall_score MUST be within ±15 points of this ({max(0, local_score-15)} to {min(100, local_score+15)}).
5. Be STRICT and ACCURATE — if the CV doesn't match the job field at all, give very low scores.

## CV Content:
{cv_text[:4000]}

## Job Description:
{jd_text[:3000]}

## Local ML Analysis Results:
- Overall Local Score: {local_score}%
- TF-IDF Similarity: {local_results.get('tfidf_similarity', 0)}%
- Keyword Match: {local_results.get('keyword_match', {}).get('score', 0)}%
- Is IT Job: {is_it}
- Hard Skills Match: {local_results.get('hard_skills', {}).get('score', 0)}%
- Missing Skills: {json.dumps(local_results.get('hard_skills', {}).get('missing', [])[:10])}

Provide your analysis as a JSON object with this EXACT structure:
{{
    "overall_score": <number 0-100, MUST be within ±15 of {local_score}>,
    "dimension_scores": {{
        "hard_skills": <number 0-100>,
        "soft_skills": <number 0-100>,
        "experience": <number 0-100>,
        "education": <number 0-100>,
        "language": <number 0-100>,
        "culture_fit": <number 0-100>
    }},
    "match_level": "<High/Medium/Low>",
    "summary": "<2-3 sentence overall assessment>",
    "strengths": ["<strength 1>", "<strength 2>", "<strength 3>"],
    "weaknesses": ["<weakness 1>", "<weakness 2>"],
    "missing_skills": [
        {{"skill": "<skill name>", "priority": "<required/preferred/nice-to-have>", "suggestion": "<how to acquire>"}}
    ],
    "cv_improvement_suggestions": [
        {{"section": "<CV section>", "current_issue": "<what's wrong>", "suggestion": "<improvement>", "example": "<rewritten text>"}}
    ],
    "ats_compatibility": {{
        "score": <number 0-100>,
        "issues": ["<issue 1>"],
        "tips": ["<tip 1>"]
    }},
    "recommended_courses": [
        {{"name": "<course name>", "platform": "<Coursera/Udemy/etc>", "reason": "<why>"}}
    ],
    "keyword_optimization": {{
        "missing_keywords": ["<keyword 1>"],
        "suggested_additions": ["<phrase to add>"]
    }}
}}"""

    def _call_api(self, prompt: str) -> str:
        """Route to correct provider with retry logic for GitHub keys."""
        if self.provider == 'github' and GHP_KEYS:
            return self._call_github_with_retry(prompt)
        elif self.provider == 'openrouter':
            return self._call_openrouter(prompt)
        elif self.provider == 'gemini':
            return self._call_gemini(prompt)
        else:
            raise Exception(f"Unknown AI provider: {self.provider}")

    def _call_github_with_retry(self, prompt: str, max_retries: int = 3) -> str:
        """Try random ghp_ keys up to max_retries times."""
        available_keys = list(GHP_KEYS)  # copy
        random.shuffle(available_keys)
        keys_to_try = available_keys[:max_retries]

        last_error = None
        for attempt, key in enumerate(keys_to_try, 1):
            try:
                key_preview = key[:8] + '...' + key[-4:]
                print(f"[GitHub] Attempt {attempt}/{max_retries} with key {key_preview}")
                return self._call_github_single(prompt, key)
            except RateLimitError as e:
                print(f"[GitHub] Key {key_preview} rate limited, trying next...")
                last_error = e
            except Exception as e:
                print(f"[GitHub] Key {key_preview} failed: {e}")
                last_error = e

        print(f"[GitHub] All {max_retries} keys failed. Falling back to local-only.")
        raise last_error or Exception("All GitHub API keys exhausted")

    def _call_github_single(self, prompt: str, api_key: str) -> str:
        """Call GitHub Models API with a specific key."""
        url = 'https://models.inference.ai.azure.com/chat/completions'
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        resp = requests.post(
            url,
            headers=headers,
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2
            },
            timeout=60
        )
        print(f"[{self.model}] Response status: {resp.status_code}")
        if resp.status_code == 429:
            raise RateLimitError("GitHub Models Rate Limit")
        if resp.status_code != 200:
            raise Exception(f"API error {resp.status_code}: {resp.text[:300]}")
        content = resp.json()['choices'][0]['message']['content']
        return re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

    def _call_openrouter(self, prompt: str) -> str:
        """Call OpenRouter API."""
        url = 'https://openrouter.ai/api/v1/chat/completions'
        headers = {
            "Authorization": f"Bearer {_get_api_key()}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:5000",
            "X-Title": "PAVN ATS"
        }
        resp = requests.post(
            url,
            headers=headers,
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2
            },
            timeout=60
        )
        if resp.status_code == 429:
            raise RateLimitError("OpenRouter Rate Limit")
        if resp.status_code != 200:
            raise Exception(f"API error {resp.status_code}: {resp.text[:300]}")
        content = resp.json()['choices'][0]['message']['content']
        return re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

    # ---- GitHub Models (DeepSeek-R1-0528) ----
    def _call_github(self, prompt: str) -> str:
        url = "https://models.inference.ai.azure.com/chat/completions"
        headers = {
            'Authorization': f'Bearer {API_KEY}',
            'Content-Type': 'application/json',
        }
        payload = {
            'model': self.model,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0.2,
        }

        try:
            print(f"[{self.model}] Calling GitHub Models API...")
            response = requests.post(url, headers=headers, json=payload, timeout=180)
            print(f"[{self.model}] Response status: {response.status_code}")

            if response.status_code == 429:
                raise RateLimitError("⚠️ GitHub Models Rate Limit!")

            if response.status_code != 200:
                detail = response.text[:500]
                print(f"[{self.model}] Error: {detail}")
                raise Exception(f"GitHub API error {response.status_code}: {detail}")

            data = response.json()
            choices = data.get('choices', [])
            if not choices:
                raise Exception("GitHub Models returned empty response")

            content = choices[0].get('message', {}).get('content', '')

            # ========== DeepSeek Response Cleaning ==========
            # 1. Remove <think>...</think> tags (reasoning traces)
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)

            # 2. Remove incomplete/partial <think> tags (no closing tag)
            content = re.sub(r'<think>.*$', '', content, flags=re.DOTALL)

            # 3. Strip leading/trailing whitespace
            content = content.strip()

            print(f"[{self.model}] Got response ({len(content)} chars)")
            return content

        except requests.exceptions.Timeout:
            raise Exception(f"{self.model} timeout (180s).")
        except RateLimitError:
            raise
        except requests.exceptions.ConnectionError:
            raise Exception("Cannot connect to GitHub Models API.")

    # ---- Google Gemini ----
    def _call_gemini(self, prompt: str) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={API_KEY}"
        headers = {'Content-Type': 'application/json'}
        payload = {
            'contents': [{'parts': [{'text': prompt}]}],
            'generationConfig': {
                'temperature': 0.0,
                'maxOutputTokens': 8192,
                'responseMimeType': 'application/json',
            },
        }

        try:
            print("[Gemini] Calling API (temperature=0)...")
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            print(f"[Gemini] Response status: {response.status_code}")

            if response.status_code == 429:
                raise RateLimitError("⚠️ Gemini API Rate Limit!")

            if response.status_code != 200:
                detail = response.text[:500]
                print(f"[Gemini] Error: {detail}")
                raise Exception(f"Gemini API error {response.status_code}: {detail}")

            data = response.json()
            candidates = data.get('candidates', [])
            if not candidates:
                raise Exception("Gemini trả về response rỗng")

            parts = candidates[0].get('content', {}).get('parts', [])
            text = ''
            for part in parts:
                if 'text' in part and 'thought' not in part:
                    text = part['text']
                    break
            if not text and parts:
                text = parts[-1].get('text', '')

            print(f"[Gemini] Got response ({len(text)} chars)")
            return text

        except requests.exceptions.Timeout:
            raise Exception("Gemini timeout (120s).")
        except RateLimitError:
            raise
        except requests.exceptions.ConnectionError:
            raise Exception("Không thể kết nối Gemini API.")

    def _parse_response(self, response_text: str) -> dict:
        """Parse AI response text into dict. Handles:
        - ```json ... ``` code blocks
        - Markdown formatting (#, *, **)
        - Trailing text after JSON
        - <think> remnants
        - Comments and annotations
        """
        try:
            cleaned = response_text.strip()

            # 1. Remove any remaining <think> tags
            cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL).strip()
            cleaned = re.sub(r'</?think>', '', cleaned).strip()

            # 2. Remove ```json ... ``` wrapper
            if '```' in cleaned:
                # Extract content between first ``` and last ```
                match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', cleaned, re.DOTALL)
                if match:
                    cleaned = match.group(1).strip()
                else:
                    # Fallback: remove all ``` markers
                    cleaned = re.sub(r'```(?:json)?', '', cleaned).strip()

            # 3. Remove markdown headers (# ## ###) that DeepSeek sometimes adds
            cleaned = re.sub(r'^#{1,6}\s+.*$', '', cleaned, flags=re.MULTILINE).strip()

            # 4. Remove bold/italic markers from OUTSIDE JSON (not inside values)
            # Only clean if the text doesn't start with { (to avoid breaking JSON values)
            if not cleaned.startswith('{'):
                cleaned = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', cleaned)

            # 5. Try direct JSON parse
            if cleaned.startswith('{'):
                # Find the matching closing brace
                brace_count = 0
                json_end = 0
                for i, ch in enumerate(cleaned):
                    if ch == '{':
                        brace_count += 1
                    elif ch == '}':
                        brace_count -= 1
                    if brace_count == 0 and i > 0:
                        json_end = i + 1
                        break
                if json_end > 0:
                    json_str = cleaned[:json_end]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        pass

            # 6. Fallback: extract any JSON object from the text
            json_match = re.search(r'\{[\s\S]*\}', cleaned)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    # Try to fix common issues:
                    raw = json_match.group()
                    # Remove trailing commas before }
                    raw = re.sub(r',\s*}', '}', raw)
                    raw = re.sub(r',\s*]', ']', raw)
                    try:
                        return json.loads(raw)
                    except json.JSONDecodeError:
                        pass

            # 7. Last resort: try the whole string
            return json.loads(cleaned)

        except (json.JSONDecodeError, Exception):
            return {
                'error': f'Không thể parse response từ {self.model}',
                'raw_response': response_text[:1000],
            }


# ============================================================
# HYBRID ANALYZER — Combines Local ML + AI with proper weighting
# ============================================================
class HybridAnalyzer:
    """
    Supports 3 modes:
      - 'local': Only Local ML (deterministic, no API)
      - 'ai': Only AI analysis (API-dependent)
      - 'hybrid': 75% Local + 25% AI (default)
    """

    WEIGHTS = {
        'hard_skills': 0.30,
        'experience': 0.25,
        'education': 0.15,
        'language': 0.10,
        'soft_skills': 0.10,
        'culture_fit': 0.10,
    }

    def __init__(self, cv_text: str, jd_text: str, mode: str = 'hybrid'):
        self.cv_text = cv_text
        self.jd_text = jd_text
        self.mode = mode  # 'local', 'ai', 'hybrid'

    def analyze(self) -> dict:
        """Run analysis in specified mode."""
        cache_key = hashlib.md5(
            (self.cv_text + '|||' + self.jd_text + '|||' + self.mode).encode()
        ).hexdigest()

        if cache_key in _result_cache:
            print(f"[Cache] Returning cached result (mode={self.mode})")
            return _result_cache[cache_key]

        # Layer 1: Local ML (always runs — needed even for AI-only as context)
        local = LocalAnalyzer(self.cv_text, self.jd_text)
        local_results = local.analyze()

        result = {
            'mode': self.mode,
            'ai_provider': AI_PROVIDER,
            'ai_model': AI_MODEL,
            'local_analysis': local_results,
            'ai_analysis': None,
            'final_scores': {},
            'ai_available': False,
            'error': None,
        }

        # Layer 2: AI analysis (skip if mode='local' or ENABLE_AI is False)
        if self.mode in ('ai', 'hybrid') and ENABLE_AI:
            try:
                ai = AIAnalyzer()
                ai_results = ai.analyze(self.cv_text, self.jd_text, local_results)

                if 'error' not in ai_results:
                    result['ai_analysis'] = ai_results
                    result['ai_available'] = True
                else:
                    result['error'] = ai_results.get('error')
            except RateLimitError as e:
                result['error'] = str(e)
                result['rate_limited'] = True
            except Exception as e:
                result['error'] = f"AI error ({AI_PROVIDER}): {str(e)}"

        # Calculate final scores based on mode
        result['final_scores'] = self._calculate_final_scores(
            local_results, result.get('ai_analysis'), self.mode
        )

        _result_cache[cache_key] = result
        return result

    def _calculate_final_scores(self, local: dict, ai: dict | None, mode: str) -> dict:
        scores = {}

        local_dims = {
            'hard_skills': local['hard_skills']['score'],
            'soft_skills': local['soft_skills']['score'],
            'experience': local['experience']['score'],
            'education': local['education']['score'],
            'language': local['language_match']['score'],
            'culture_fit': 30.0,
        }

        # Domain relevance factor based on semantic similarity
        # When semantic similarity is very low (completely unrelated domains like Driver<->IT),
        # penalize Experience, Soft Skills, and Culture Fit — because they're measuring
        # the WRONG domain's experience/skills.
        semantic_score = local.get('semantic_similarity', {}).get('overall', 50.0)
        # NOTE: multilingual-e5-large gives high baseline cosine for same-language texts
        # even when domains are completely unrelated (e.g., Vietnamese IT CV vs Vietnamese driver JD
        # gets ~0.79 cosine → ~56% scaled). Thresholds must account for this.
        if semantic_score < 40.0:
            # Completely unrelated domains (e.g., IT CV ↔ Chef JD in different languages)
            domain_factor = 0.10  # 90% penalty
        elif semantic_score < 55.0:
            domain_factor = 0.30  # 70% penalty
        elif semantic_score < 65.0:
            # Same language but different domain (IT CV ↔ Driver JD, both Vietnamese)
            domain_factor = 0.50  # 50% penalty
        else:
            domain_factor = 1.0  # No penalty — domains are genuinely related

        # Apply domain penalty to context-dependent dimensions
        # Hard skills and education/language are already correctly evaluated
        # But Experience and Soft Skills need domain context
        if domain_factor < 1.0:
            local_dims['experience'] = round(local_dims['experience'] * domain_factor, 1)
            local_dims['soft_skills'] = round(local_dims['soft_skills'] * domain_factor, 1)
            local_dims['culture_fit'] = round(local_dims['culture_fit'] * domain_factor, 1)

        if mode == 'local' or not ai or 'dimension_scores' not in ai:
            # Pure local scores
            scores = {dim: val for dim, val in local_dims.items()}
        elif mode == 'ai' and ai and 'dimension_scores' in ai:
            # Pure AI scores (no local weighting)
            ai_dim = ai['dimension_scores']
            for dim in self.WEIGHTS:
                scores[dim] = round(ai_dim.get(dim, local_dims[dim]), 1)
        else:
            # Hybrid: 75% local + 25% AI (constrained ±15)
            ai_dim = ai['dimension_scores']
            for dim in self.WEIGHTS:
                local_val = local_dims[dim]
                ai_val = ai_dim.get(dim, local_val)
                ai_constrained = max(local_val - 15, min(local_val + 15, ai_val))
                scores[dim] = round(local_val * 0.75 + ai_constrained * 0.25, 1)

        overall = sum(scores[dim] * w for dim, w in self.WEIGHTS.items())
        scores['overall'] = round(overall, 1)

        if scores['overall'] >= 75:
            scores['match_level'] = 'High'
            scores['color'] = '#22c55e'
        elif scores['overall'] >= 50:
            scores['match_level'] = 'Medium'
            scores['color'] = '#f59e0b'
        elif scores['overall'] >= 25:
            scores['match_level'] = 'Low'
            scores['color'] = '#ef4444'
        else:
            scores['match_level'] = 'Very Low'
            scores['color'] = '#dc2626'

        scores['is_it_job'] = local.get('is_it_job', False)
        return scores


def analyze_cv_jd(cv_text: str, jd_text: str, mode: str = 'hybrid') -> dict:
    """Convenience function with mode support."""
    analyzer = HybridAnalyzer(cv_text, jd_text, mode=mode)
    return analyzer.analyze()


def compare_modes(cv_text: str, jd_text: str) -> dict:
    """Run 3 modes and return comparison."""
    results = {}
    for mode in ['local', 'ai', 'hybrid']:
        print(f"\n{'='*40}")
        print(f"  Running mode: {mode.upper()}")
        print(f"{'='*40}")
        try:
            analyzer = HybridAnalyzer(cv_text, jd_text, mode=mode)
            results[mode] = analyzer.analyze()
        except Exception as e:
            results[mode] = {'error': str(e)}
    return results