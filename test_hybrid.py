"""Test Hybrid mode only."""
import requests, json, time

JD_ID = "AI Engineer at C\u00f4ng ty TNHH Khoa h\u1ecdc v\u00e0 K\u1ef9 thu\u1eadt REECO.txt"

print("Testing HYBRID mode...")
start = time.time()
r = requests.post(
    "http://localhost:5000/api/analyze",
    json={"jd_id": JD_ID, "mode": "hybrid"},
    timeout=300,
)
elapsed = time.time() - start

print(f"Status: {r.status_code}")
print(f"Time: {elapsed:.1f}s")

if r.status_code == 200:
    d = r.json()
    s = d.get("final_scores", {})
    ai = d.get("ai_analysis") or {}
    la = d.get("local_analysis", {})
    
    print(f"\nProvider: {d.get('ai_provider')} | Model: {d.get('ai_model')}")
    print(f"Mode: {d.get('mode')}")
    print(f"AI Available: {d.get('ai_available')}")
    if d.get("error"):
        print(f"Error: {d['error'][:300]}")
    
    print(f"\n{'='*50}")
    print(f"HYBRID OVERALL: {s.get('overall')}% ({s.get('match_level')})")
    print(f"{'='*50}")
    print(f"Hard Skills:  {s.get('hard_skills')}%")
    print(f"Experience:   {s.get('experience')}%")
    print(f"Education:    {s.get('education')}%")
    print(f"Language:     {s.get('language')}%")
    print(f"Soft Skills:  {s.get('soft_skills')}%")
    print(f"Culture Fit:  {s.get('culture_fit')}%")
    
    print(f"\n--- LOCAL SCORES ---")
    print(f"Local Overall: {la.get('local_score')}%")
    
    if ai:
        print(f"\n--- AI SCORES ---")
        print(f"AI Overall: {ai.get('overall_score')}")
        print(f"Summary: {str(ai.get('summary',''))[:300]}")
        print(f"Strengths: {ai.get('strengths', [])}")
        print(f"Weaknesses: {ai.get('weaknesses', [])}")
elif r.status_code == 429:
    d = r.json()
    print(f"Rate Limited! Error: {d.get('error','')[:200]}")
    s = d.get("final_scores", {})
    if s:
        print(f"Fallback score: {s.get('overall')}%")
else:
    print(f"Error: {r.text[:500]}")
