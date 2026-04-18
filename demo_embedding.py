# -*- coding: utf-8 -*-
"""
Demo 3: Why is the output ALWAYS 1024 dimensions?
==================================================
Proves that 2 words and 200 words both produce the same 1024-dim vector.
Explains the POOLING mechanism inside the transformer.
"""
import sys, io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
from sentence_transformers import SentenceTransformer

print("Loading model...")
model = SentenceTransformer('intfloat/multilingual-e5-large', local_files_only=True)
print("Model loaded!\n")

# ============================================================
# TEST: Different text lengths → same vector size
# ============================================================

texts = [
    ("2 words",    "Software Engineer"),
    ("5 words",    "Senior Backend Software Engineer Python"),
    ("1 sentence", "Experienced software engineer with 5 years in Python and Node.js"),
    ("1 paragraph",
     "Experienced software engineer with 5 years in Python, Node.js, "
     "and cloud infrastructure. Built microservices architecture serving 1M users. "
     "Proficient in Docker, Kubernetes, CI/CD pipelines, and agile methodologies. "
     "Strong background in database design with PostgreSQL and MongoDB."),
    ("3 paragraphs",
     "Experienced software engineer with 5 years in Python, Node.js, "
     "and cloud infrastructure. Built microservices architecture serving 1M users. "
     "Proficient in Docker, Kubernetes, CI/CD pipelines, and agile methodologies. "
     "Strong background in database design with PostgreSQL and MongoDB. "
     "Led a team of 5 developers to deliver a real-time data processing pipeline "
     "that reduced latency by 60%. Implemented automated testing frameworks "
     "achieving 95% code coverage. Experience with AWS Lambda, S3, DynamoDB, "
     "and CloudFormation for infrastructure as code. Published 2 technical papers "
     "on distributed systems optimization and contributed to open-source projects "
     "including FastAPI and SQLAlchemy."),
]

print("=" * 75)
print("PROOF: All text lengths produce the SAME 1024-dimensional vector")
print("=" * 75)
print(f"{'Label':<15} {'Word Count':>10} {'Vector Shape':>15} {'First 3 values'}")
print("-" * 75)

for label, text in texts:
    embedding = model.encode([f"query: {text}"], normalize_embeddings=True)
    word_count = len(text.split())
    first3 = embedding[0][:3].round(4)
    print(f"{label:<15} {word_count:>10} {str(embedding.shape):>15} {first3}")

# ============================================================
# EXPLANATION: What happens INSIDE the transformer
# ============================================================
print("\n" + "=" * 75)
print("HOW IT WORKS: Inside the Transformer")
print("=" * 75)

# Use the tokenizer to show internal steps
tokenizer = model.tokenizer

for label, text in texts[:3]:
    tokens = tokenizer.tokenize(text)
    print(f"\n--- '{text}' ---")
    print(f"  Step 1 - Tokenize:  {len(tokens)} tokens: {tokens[:8]}{'...' if len(tokens) > 8 else ''}")
    print(f"  Step 2 - Transform: {len(tokens)} vectors of 1024 dimensions each")
    print(f"                      = matrix [{len(tokens)} x 1024]")
    print(f"  Step 3 - POOLING:   Average all {len(tokens)} vectors into 1 vector")
    print(f"                      = final vector [1 x 1024]")

print("\n" + "=" * 75)
print("ANALOGY")
print("=" * 75)
print("""
Imagine a CLASSROOM AVERAGE:

  Class A (5 students):   scores = [80, 90, 85, 70, 95] → average = 84.0
  Class B (50 students):  scores = [80, 90, 85, ...x50]  → average = 82.3

  Both produce ONE number (the average), regardless of class size.
  But the 50-student average is MORE RELIABLE than the 5-student average.

  Similarly:
  "Software Engineer" (3 tokens)  → average of 3 vectors  → 1 vector [1024]
  Full CV paragraph (50+ tokens)  → average of 50 vectors → 1 vector [1024]

  Both produce ONE vector [1024], but the paragraph's vector is
  MORE MEANINGFUL because it averages more information.
  
  This is called MEAN POOLING.
""")
