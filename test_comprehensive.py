"""
PAVN ATS v4 — Comprehensive Test Suite
Tests: 2 CVs × 9 JDs (including JP/KR) × 3 modes
"""
import requests
import json
import time
import sys

BASE = "http://localhost:5000"

# CV files
CVS = [
    "CV_NGUYEN_CONG_LAP_NHAN_AI_ENGINEER_2.pdf",        # EN CV
    "CV_NGUYEN_CONG_LAP_NHAN_LAP_TRINH_VIEN_5.pdf",     # VN CV
]

# Key JDs to test
JDS_FOCUS = [
    "AI Engineer at Công ty TNHH Khoa học và Kỹ thuật REECO.txt",
    "MiddleSenior Backend Engineer (Fullstack) at PORTERS ASIA VIETNAM COMPANY LIMITED.txt",
    "Software Tester (Automation & Manual) at TEQNOLOGICAL ASIA Co., Ltd.txt",
    "Tai xe GrabCar TPHCM.txt",
    "Bep truong nha hang Hai San Hoang Gia.txt",
    "Software Engineer Backend tai TechInnovation Tokyo.txt",   # JP
    "Backend Developer tai Kakao Enterprise Seoul.txt",         # KR
]


def test_single(cv_id, jd_id, mode):
    """Test one combination."""
    try:
        r = requests.post(
            f"{BASE}/api/analyze",
            json={"cv_id": cv_id, "jd_id": jd_id, "mode": mode},
            timeout=300,
        )
        if r.status_code == 200:
            d = r.json()
            return {
                "overall": d.get("final_scores", {}).get("overall", "?"),
                "level": d.get("final_scores", {}).get("match_level", "?"),
                "semantic": d.get("local_analysis", {}).get("semantic_similarity", {}).get("overall", "?"),
                "raw_cosine": d.get("local_analysis", {}).get("semantic_similarity", {}).get("raw_cosine", "?"),
                "method": d.get("local_analysis", {}).get("semantic_similarity", {}).get("method", "?"),
                "ai_available": d.get("ai_available", False),
                "error": None,
            }
        elif r.status_code == 429:
            d = r.json()
            return {
                "overall": d.get("final_scores", {}).get("overall", "?"),
                "level": "RATE_LIMITED",
                "semantic": "?",
                "raw_cosine": "?",
                "method": "rate_limited",
                "ai_available": False,
                "error": "Rate Limited",
            }
        else:
            return {"overall": "ERR", "level": f"HTTP {r.status_code}", "error": r.text[:100]}
    except Exception as e:
        return {"overall": "ERR", "level": "EXCEPTION", "error": str(e)[:100]}


def short_name(name, max_len=40):
    name = name.replace(".txt", "").replace(".pdf", "")
    return name[:max_len] + ".." if len(name) > max_len else name


def run_full_test():
    # 1. Check server
    print("=" * 80)
    print("  PAVN ATS v4 — COMPREHENSIVE TEST SUITE")
    print("  Semantic: multilingual-e5-large | AI: DeepSeek-R1-0528")
    print("=" * 80)

    try:
        r = requests.get(f"{BASE}/api/sample-data", timeout=10)
        d = r.json()
        cv_list = d.get("cv_list", [])
        jd_list = d.get("jd_list", [])
        print(f"  📄 CVs loaded: {len(cv_list)}")
        for cv in cv_list:
            print(f"     - {cv['filename']}")
        print(f"  📋 JDs loaded: {len(jd_list)}")
        for jd in jd_list:
            print(f"     - {jd['name']}")
    except Exception as e:
        print(f"  ❌ Server error: {e}")
        return

    # 2. Test LOCAL mode — all CVs × focus JDs
    print(f"\n\n{'='*80}")
    print("  🔧 TEST 1: LOCAL MODE ONLY (Semantic Embedding)")
    print(f"{'='*80}")

    for cv_id in CVS:
        cv_short = short_name(cv_id, 50)
        print(f"\n  📄 CV: {cv_short}")
        print(f"  {'JD':<42} {'Score':>6} {'Semantic':>9} {'Cosine':>8} {'Level':>12}")
        print(f"  {'-'*78}")

        for jd_id in JDS_FOCUS:
            jd_short = short_name(jd_id)
            start = time.time()
            result = test_single(cv_id, jd_id, "local")
            elapsed = time.time() - start

            print(f"  {jd_short:<42} {str(result.get('overall','?')):>5}% {str(result.get('semantic','?')):>8}% {str(result.get('raw_cosine','?')):>8} {str(result.get('level','?')):>12} ({elapsed:.1f}s)")

    # 3. Test AI mode — only with AI Engineer JD to avoid rate limits
    print(f"\n\n{'='*80}")
    print("  🤖 TEST 2: AI MODE ONLY (DeepSeek-R1-0528)")
    print(f"{'='*80}")

    for cv_id in CVS:
        cv_short = short_name(cv_id, 50)
        ai_jd = JDS_FOCUS[0]  # AI Engineer JD
        jd_short = short_name(ai_jd)

        print(f"\n  📄 CV: {cv_short} × JD: {jd_short}")
        start = time.time()
        result = test_single(cv_id, ai_jd, "ai")
        elapsed = time.time() - start

        print(f"  Score: {result.get('overall','?')}% ({result.get('level','?')})")
        print(f"  AI Available: {result.get('ai_available', '?')}")
        if result.get("error"):
            print(f"  Error: {result['error']}")
        print(f"  Time: {elapsed:.1f}s")

    # 4. Test HYBRID mode
    print(f"\n\n{'='*80}")
    print("  📊 TEST 3: HYBRID MODE (Local + AI)")
    print(f"{'='*80}")

    for cv_id in CVS:
        cv_short = short_name(cv_id, 50)
        ai_jd = JDS_FOCUS[0]
        jd_short = short_name(ai_jd)

        print(f"\n  📄 CV: {cv_short} × JD: {jd_short}")
        start = time.time()
        result = test_single(cv_id, ai_jd, "hybrid")
        elapsed = time.time() - start

        print(f"  Score: {result.get('overall','?')}% ({result.get('level','?')})")
        if result.get("error"):
            print(f"  Error: {result['error']}")
        print(f"  Time: {elapsed:.1f}s")

    # 5. Cross-language test
    print(f"\n\n{'='*80}")
    print("  🌍 TEST 4: CROSS-LANGUAGE (VI CV × JP/KR JD)")
    print(f"{'='*80}")

    vi_cv = CVS[1]  # Vietnamese CV
    cross_jds = JDS_FOCUS[-2:]  # JP and KR JDs

    for jd_id in cross_jds:
        jd_short = short_name(jd_id)
        start = time.time()
        result = test_single(vi_cv, jd_id, "local")
        elapsed = time.time() - start

        print(f"\n  JD: {jd_short}")
        print(f"  Score: {result.get('overall','?')}% | Semantic: {result.get('semantic','?')}% | Cosine: {result.get('raw_cosine','?')}")
        print(f"  Level: {result.get('level','?')} | Time: {elapsed:.1f}s")

    # EN CV × JP/KR JDs
    en_cv = CVS[0]
    for jd_id in cross_jds:
        jd_short = short_name(jd_id)
        start = time.time()
        result = test_single(en_cv, jd_id, "local")
        elapsed = time.time() - start

        print(f"\n  EN CV × JD: {jd_short}")
        print(f"  Score: {result.get('overall','?')}% | Semantic: {result.get('semantic','?')}% | Cosine: {result.get('raw_cosine','?')}")
        print(f"  Level: {result.get('level','?')} | Time: {elapsed:.1f}s")

    print(f"\n\n{'='*80}")
    print("  ✅ ALL TESTS COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    run_full_test()
