"""Test v4: Semantic Embedding (multilingual-e5-large) vs cac JD."""
import requests, json, time

BASE = "http://localhost:5000"

JDS = [
    "AI Engineer at C\u00f4ng ty TNHH Khoa h\u1ecdc v\u00e0 K\u1ef9 thu\u1eadt REECO.txt",
    "Bep truong nha hang Hai San Hoang Gia.txt",
    "Corporate Lawyer at Baker McKenzie Vietnam.txt",
    "Tai xe GrabCar TPHCM.txt",
    "Nhan vien Kinh doanh Bat dong san Phu Long.txt",
    "Software Tester (Automation & Manual) at TEQNOLOGICAL ASIA Co., Ltd.txt",
    "MiddleSenior Backend Engineer (Fullstack) at PORTERS ASIA VIETNAM COMPANY LIMITED.txt",
]

print("=" * 70)
print("  PAVN ATS v4 \u2014 Semantic Embedding Test")
print("  Model: multilingual-e5-large")
print("=" * 70)

results = []
for jd_id in JDS:
    name = jd_id.replace(".txt", "")[:50]
    print(f"\n  Testing: {name}...")
    
    start = time.time()
    try:
        r = requests.post(
            f"{BASE}/api/analyze",
            json={"jd_id": jd_id, "mode": "local"},
            timeout=600,
        )
        elapsed = time.time() - start
        
        if r.status_code != 200:
            print(f"    \u274c Error {r.status_code}: {r.text[:200]}")
            continue
        
        d = r.json()
        la = d.get("local_analysis", {})
        s = d.get("final_scores", {})
        sem = la.get("semantic_similarity", {})
        
        overall = s.get("overall", "?")
        level = s.get("match_level", "?")
        sem_score = sem.get("overall", "?")
        sem_method = sem.get("method", "?")
        raw_cos = sem.get("raw_cosine", "?")
        sections = sem.get("sections", {})
        
        results.append({
            "name": name,
            "overall": overall,
            "level": level,
            "semantic": sem_score,
            "raw_cosine": raw_cos,
            "sections": sections,
            "time": elapsed,
        })
        
        print(f"    Overall: {overall}% ({level})")
        print(f"    Semantic: {sem_score}% (cosine={raw_cos})")
        print(f"    Method: {sem_method}")
        print(f"    Sections: {json.dumps(sections)}")
        print(f"    Time: {elapsed:.1f}s")
    except Exception as e:
        print(f"    \u274c Exception: {e}")

# Summary table
print("\n\n" + "=" * 80)
print(f"{'JD':<52} {'Overall':>8} {'Semantic':>9} {'Cosine':>8} {'Level':>10}")
print("-" * 80)
for r in sorted(results, key=lambda x: x.get("overall", 0), reverse=True):
    print(f"{r['name']:<52} {r['overall']:>7}% {r['semantic']:>8}% {r['raw_cosine']:>8} {r['level']:>10}")
print("-" * 80)
