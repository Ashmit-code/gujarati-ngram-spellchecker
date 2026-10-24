# Gujarati Spell Checker (ગુજરાતી જોડણી તપાસક)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3%2B-green.svg)](https://flask.palletsprojects.com/)
[![Tests: Unittest](https://img.shields.io/badge/Tests-26%2F26%20Passing-brightgreen.svg)](tests/)
[![Architecture](https://img.shields.io/badge/Architecture-Statistical%20NLP%20%7C%20N--Gram-purple.svg)](ngram_model.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> An educational, transparent, and lightweight Gujarati Spell Checker developed for Natural Language Processing (NLP). Engineered completely on classical NLP principles: Unicode NFC Text Normalization, Tokenization, Statistical N-Gram Language Modeling with Laplace & Jelinek-Mercer Smoothing, Dynamic Programming Levenshtein Distance, and an interactive Flask Web UI.

---

## 01 — Overview

Natural Language Processing for Indian regional languages like Gujarati presents unique orthographic complexities due to non-Latin character sets, dependent vowel signs (*matras*), conjunct consonants (*jodakshar*), and anusvara homographs.

Unlike opaque deep learning models that function as black boxes, **Gujarati Spell Checker** provides a completely visible, mathematically auditable, and interpretable NLP pipeline. Every stage—from character normalization and Levenshtein distance matrix computation to n-gram conditional probabilities and smoothing distributions—is implemented cleanly from scratch using standard Python.

### Why This Project Matters
- **100% Explainable Classical NLP:** No heavy neural networks or third-party proprietary APIs. Every correction decision can be traced back to mathematical edit costs and corpus transition probabilities.
- **Context-Sensitive Trigram Ranking:** Evaluates sentence context so homophonic candidates are selected based on preceding word sequences.
- **Zero Heavy Dependencies:** Runs entirely on standard Python with `Flask` as the single lightweight dependency for the web dashboard.

---

## 02 — Demo & Web Interface

The application features a clean single-page dashboard styled with modern CSS cards and real-time response rendering:

```text
┌────────────────────────────────────────────────────────────────────────┐
│ [Logo] Gujarati Spell Checker (ગુજરાતી જોડણી તપાસક)                    │
│        Model Status: Active  |  Vocabulary Size: 550 Words             │
├────────────────────────────────────────────────────────────────────────┤
│ Enter Gujarati Sentence or Word:                                       │
│ ┌────────────────────────────────────────────────────────────────────┐ │
│ │ હું શાલામાં જાઉં છું.                                              │ │
│ └────────────────────────────────────────────────────────────────────┘ │
│ [ 🔍 Check Spelling ]                                                  │
├────────────────────────────────────────────────────────────────────────┤
│ Diagnostic Pipeline Analysis:                                          │
│ • Input Token: શાલામાં (Misspelled detected)                            │
│ • Edit Distance Candidates: ['શાળામાં', 'શાખામાં', 'શાકમાં']         │
│ • Trigram Context: P(w | હું) * P(જાઉં | w)                            │
│ • Suggested Correction: શાળામાં (Confidence: 89.4%)                    │
│                                                                        │
│ Candidate Suggestion Chips:                                            │
│ [ શાળામાં (Score: 0.894) ]  [ શાખામાં (Score: 0.062) ]  [ શાકમાં ]    │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 03 — Features

### Implemented & Verified
- **Full Sentence & Single-Word Inspection:** Processes isolated words or multi-word prose with punctuation preservation.
- **Orthographic Normalization:** Converts strings to Unicode NFC, normalizes zero-width joiners/non-joiners, and collapses irregular whitespaces.
- **Dynamic Programming Levenshtein Engine:** Computes optimal string alignment and minimum edit distances with custom matra weightings.
- **Statistical N-Gram Language Model:** Computes Unigram, Bigram, and Trigram transition probabilities over training corpora with Laplace (Add-1) and Jelinek-Mercer interpolation smoothing.
- **Contextual Candidate Scoring:** Scores candidates by combining inverse edit distance with language model sequence likelihood.
- **Interactive Flask Web Dashboard:** Renders candidate suggestion chips with transparent intermediate metric logs.
- **Interactive Preprocessing Demo:** Standalone script (`preprocessing_demo.py`) showing character-by-character transformations.

---

## 04 — Architecture & Pipeline Workflow

```mermaid
flowchart TD
    A[Raw Gujarati Input Text] --> B[preprocessing.py: Unicode NFC & Whitespace Clean]
    B --> C[Tokenization: Split Words & Retain Punctuation Boundaries]
    C --> D{Word in Trained Vocabulary?}
    D -- Yes --> E[Mark Valid: P(word) Retained]
    D -- No --> F[levenshtein.py: Dynamic Programming Candidate Generation]
    F --> G[Extract Candidates with Edit Distance <= 2]
    G --> H[ngram_model.py: Contextual Scoring]
    H --> I[Compute Unigram, Bigram, & Trigram Probabilities]
    I --> J[Apply Laplace & Jelinek-Mercer Smoothing]
    J --> K[Rank Suggestions: Combined Metric Score]
    K --> L[app.py: Return JSON Diagnostics & Suggestion Chips]
```

### Mathematical Foundations

#### 1. Levenshtein Dynamic Programming Distance
Given query token $s$ of length $m$ and dictionary word $t$ of length $n$:
$$D(i, j) = \min \begin{cases} 
D(i-1, j) + 1 & \text{(Deletion)} \\
D(i, j-1) + 1 & \text{(Insertion)} \\
D(i-1, j-1) + \text{cost}(s_i, t_j) & \text{(Substitution)}
\end{cases}$$

#### 2. Trigram Language Modeling with Smoothing
For candidate word $w_i$ given preceding context $(w_{i-2}, w_{i-1})$:
$$P(w_i \mid w_{i-2}, w_{i-1}) = \lambda_3 \frac{C(w_{i-2}, w_{i-1}, w_i) + 1}{C(w_{i-2}, w_{i-1}) + |V|} + \lambda_2 \frac{C(w_{i-1}, w_i) + 1}{C(w_{i-1}) + |V|} + \lambda_1 \frac{C(w_i) + 1}{N + |V|}$$

---

## 05 — Technology Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.9+ | Primary runtime environment |
| **Web Framework** | Flask 2.3+ | Lightweight HTTP server and API routing |
| **Storage & Models** | JSON & CSV | Human-readable serialized n-gram frequency matrices |
| **Frontend** | HTML5, CSS3, Vanilla JS | Interactive web UI with Google Keyboard style chip suggestions |
| **Testing** | `unittest` | Built-in automated test suite covering all NLP modules |

---

## 06 — Installation & Setup

### Prerequisites
- Python `3.9` or higher.
- A virtual environment tool (`venv`).

### Setup Commands

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/Gujarati_spell_checker.git
   cd Gujarati_spell_checker
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv

   # Windows
   .venv\Scripts\activate

   # Linux / macOS
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Train or regenerate the N-Gram Model:**
   The repository already includes the pre-trained model at `model/ngram_model.json`. To retrain:
   ```bash
   python train_ngram.py
   ```

5. **Start the web application:**
   ```bash
   python app.py
   ```
   Open `http://localhost:5000` in your web browser.

---

## 07 — Usage Examples

### 1. Web API Endpoint (`POST /check`)
You can query the spell checker directly via cURL or any HTTP client:

```bash
curl -X POST http://localhost:5000/check \
  -H "Content-Type: application/json" \
  -d '{"text": "હું શાલામાં જાઉં છું"}'
```

**JSON Response:**
```json
{
  "original_text": "હું શાલામાં જાઉં છું",
  "tokens": [
    {"word": "હું", "is_correct": true, "suggestions": []},
    {
      "word": "શાલામાં",
      "is_correct": false,
      "suggestions": [
        {"candidate": "શાળામાં", "score": 0.894, "edit_distance": 1},
        {"candidate": "શાખામાં", "score": 0.062, "edit_distance": 1}
      ]
    },
    {"word": "જાઉં", "is_correct": true, "suggestions": []},
    {"word": "છું", "is_correct": true, "suggestions": []}
  ],
  "corrected_sentence": "હું શાળામાં જાઉં છું"
}
```

### 2. Standalone Preprocessing Demo
To inspect the character tokenization and normalization pipeline in your terminal:
```bash
python preprocessing_demo.py
```

### 3. Quantitative Test Evaluation
To run full evaluation against the 109 held-out test cases in `data/gujarati_test.csv`:
```bash
python evaluate.py
```

---

## 08 — Testing & Verification

The project includes an automated test suite verifying preprocessing, normalization, Levenshtein distance calculations, and Flask routes.

### Test Results
Run the test suite with Python's built-in `unittest` runner:
```bash
python -m unittest discover -s tests
```

**Output:**
```text
..........................
----------------------------------------------------------------------
Ran 26 tests in 0.053s

OK
```
All **26 unit tests passed** with 100% success across all components.

---

## 09 — Project Structure

```text
Gujarati_spell_checker/
├── app.py                      # Flask web application & REST API routes
├── spell_checker.py            # High-level pipeline coordinator
├── ngram_model.py              # Statistical N-gram language model (Uni/Bi/Tri)
├── levenshtein.py              # Dynamic programming edit distance algorithm
├── preprocessing.py            # Unicode NFC normalization & tokenization
├── preprocessing_demo.py       # Standalone CLI demonstration script
├── train_ngram.py              # Corpus trainer producing model/ngram_model.json
├── evaluate.py                 # Evaluation benchmark script on test dataset
├── requirements.txt            # Python dependencies (Flask)
├── .env.example                # Example environment variables (PORT)
├── .gitignore                  # Ignores virtualenvs, pycache, temporary files
├── LICENSE                     # MIT License terms
├── PRD.md                      # Product Requirements & NLP Design Specification
├── PROGRESS.md                 # Implementation changelog & milestone log
│
├── data/
│   ├── gujarati_training.csv   # 829-sentence training corpus
│   └── gujarati_test.csv       # 109-instance held-out benchmark dataset
│
├── model/
│   └── ngram_model.json        # Precomputed n-gram vocabulary & frequency counts
│
├── static/
│   ├── favicon.png             # Application favicon
│   ├── script.js               # Frontend AJAX caller & suggestion chip renderer
│   └── style.css               # Clean responsive styling & layout rules
│
├── templates/
│   └── index.html              # Single-page application UI template
│
└── tests/
    ├── __init__.py
    └── test_spell_checker.py   # 26 automated unit & regression tests
```

---

## 10 — License

This project is licensed under the **MIT License**. See the [`LICENSE`](./LICENSE) file for complete details.

- **Author & Copyright:** (c) 2026 Ashmit Prasad.
