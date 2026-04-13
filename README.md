# PAVN ATS — Intelligent CV Matching System

> AI-powered multi-dimensional CV-to-Job matching engine with cross-language support.

## Overview

PAVN ATS is a hybrid AI + ML system that evaluates candidate CVs against Job Descriptions across **6 dimensions**: Technical Skills, Experience, Education, Language, Soft Skills, and Culture Fit.

Built for the modern recruitment industry, it supports **cross-language matching** — a Vietnamese CV can be matched against an English or Japanese Job Description with high accuracy.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  PAVN ATS Engine                 │
├──────────────────┬──────────────────────────────┤
│   Layer 1: ML    │   Layer 2: AI (Optional)     │
│                  │                              │
│  • Semantic      │  • DeepSeek-R1-0528          │
│    Embeddings    │  • Google Gemini 2.5          │
│    (e5-large)    │  • OpenRouter                │
│                  │                              │
│  • TF-IDF        │  • CV improvement            │
│    Similarity    │    suggestions               │
│                  │                              │
│  • Keyword       │  • ATS compatibility         │
│    Extraction    │    analysis                   │
│                  │                              │
│  • Domain-Aware  │  • Recommended               │
│    Scoring       │    courses                    │
└──────────────────┴──────────────────────────────┘
```

## Key Features

- **Semantic Matching** — Uses `multilingual-e5-large` embeddings (supports 100+ languages)
- **Hybrid Scoring** — 75% local ML + 25% AI for balanced, reliable results
- **Cross-Language** — Vietnamese CV ↔ English/Japanese JD matching
- **Domain-Aware** — Penalizes irrelevant experience (Driver applying for IT role)
- **Offline-First** — Fully functional without internet (local ML mode)
- **Multi-Provider AI** — Auto-detects GitHub Models, OpenRouter, or Gemini from API key
- **Retry Logic** — Rotates API keys from pool with automatic failover

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Python 3.x, Flask |
| ML Engine | Sentence-Transformers, scikit-learn |
| Embedding Model | intfloat/multilingual-e5-large |
| AI Providers | DeepSeek-R1, Gemini 2.5, OpenRouter |
| Frontend | HTML5, CSS3, JavaScript, Chart.js |
| PDF Parsing | pdfplumber |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure (optional — works offline without API keys)
# Edit .env to set ENABLE_AI=YES/NO

# Run
python app.py

# Open http://localhost:5000
```

## Project Structure

```
PAVN_ATS/
├── app.py              # Flask API server
├── analyzer.py         # Core matching engine (ML + AI)
├── cv_parser.py        # PDF text extraction
├── requirements.txt    # Python dependencies
├── .env                # Configuration
├── Mau_1/              # Sample CVs and JDs
├── static/
│   ├── css/style.css   # UI styling
│   └── js/app.js       # Frontend logic
└── templates/
    └── index.html      # Web interface
```

## License

Private — Built by Nguyen Cong Lap for demonstration purposes.
