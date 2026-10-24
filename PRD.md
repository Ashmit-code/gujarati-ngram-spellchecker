# Product Requirements Document (PRD)

## 1. Project Title
**Gujarati Spell Checker (ગુજરાતી જોડણી તપાસક)**
*Semester 7 NLP Mini-Project*

---

## 2. Problem Statement
Gujarati is an Indo-Aryan language characterized by a complex orthography comprising independent vowels, dependent vowel signs (matras), anusvara, visarga, conjunct consonants (jodakshar), and halanta (virama). Spelling inaccuracies frequently arise from phonetic substitutions (e.g., confusing 'શ' [sha], 'ષ' [shha], and 'સ' [sa], or 'લ' [la] and 'ળ' [lda]), incorrect matra placement (e.g., 'િ' vs 'ી', 'ુ' vs 'ૂ'), omission/insertion of characters, or typographical slips.

Traditional deep-learning approaches and large language models function as opaque black boxes that are computationally heavy, require GPU acceleration, and obscure foundational NLP principles. For academic learning and university examination (viva), there is an essential need for an explainable, transparent, and lightweight automated Gujarati spell checker built strictly upon classical NLP algorithms.

---

## 3. Objective
The objective of this project is to develop an educational, explainable, and fully functional Gujarati Spell Checker that:
1. Accepts raw Gujarati words or full sentences.
2. Executes and visualizes every stage of standard NLP text preprocessing.
3. Automatically tokenizes text and preserves syntactic stopwords for contextual modeling.
4. Validates words against a corpus-derived vocabulary.
5. Generates candidate corrections using an explicit dynamic-programming Levenshtein distance algorithm.
6. Ranks candidates utilizing transition probabilities from a statistical N-gram language model (Unigram, Bigram, Trigram).
7. Suggests accurate corrections while keeping correct words intact.
8. Exposes the entire computational pipeline through an intuitive web interface and an independent terminal demonstration.
9. Performs objective quantitative evaluation on a held-out test dataset without data leakage.

---

## 4. Scope

### In-Scope:
- Processing single Gujarati words and multi-word sentences.
- Classical NLP text preprocessing: Unicode NFC normalization, whitespace normalization, punctuation handling.
- Deliberate retention of stopwords to preserve contextual integrity.
- Automatic vocabulary and N-gram extraction from `data/gujarati_training.csv`.
- Manual implementation of Levenshtein (edit) distance using Dynamic Programming.
- Manual statistical N-gram language model with Laplace (add-one) smoothing.
- Candidate generation filtering by edit distance ($\le 1$, then fallback to $\le 2$).
- Lightweight Flask backend with a responsive HTML/CSS/JavaScript interface.
- Complete transparency: interactive inspection of every pipeline stage.
- Independent standalone scripts: `preprocessing_demo.py`, `train_ngram.py`, and `evaluate.py`.

### Out-of-Scope (Strictly Excluded):
- Deep learning architectures (CNN, RNN, LSTM, GRU, Transformers, BERT).
- High-level external NLP/spell-checking libraries (TensorFlow, Keras, PyTorch, Scikit-learn, Hugging Face).
- Morphological analyzers (GujMORPH) and Part-of-Speech tagging dropdowns.
- Heavy frameworks such as Streamlit.
- Cloud APIs, external commercial spell-checking endpoints, or GPU training.
- Artificial dataset synthesis or data leakage between training and testing.

---

## 5. Target User
- **University Professors and Evaluators**: Seeking transparent demonstration of foundational NLP techniques during semester viva.
- **Computer Science Students**: Studying tokenization, language modeling, dynamic programming, and error detection.
- **General Users**: Checking spelling of everyday Gujarati words and sentences.

---

## 6. Functional Requirements
1. **Text Input**: Accepts Gujarati words or sentences via web UI or command-line interfaces.
2. **Visible Preprocessing**: Provides visible outputs for Unicode NFC normalization, whitespace normalization, and punctuation cleaning.
3. **Tokenization**: Splits normalized text into isolated word tokens.
4. **Stopword Preservation**: Retains common functional words (હું, છે, અને, માં, etc.) and clearly explains the rationale.
5. **Vocabulary Verification**: Cross-references tokens against the vocabulary built from `data/gujarati_training.csv`.
6. **Candidate Generation**: Identifies potential corrections from vocabulary within edit distance 1 or 2.
7. **Levenshtein Distance Calculation**: Computes minimum single-character insertions, deletions, and substitutions.
8. **N-Gram Contextual Scoring**: Evaluates candidate fit in sentence context using Trigram and Bigram transition probabilities, backed by Unigram priors.
9. **Candidate Ranking & Selection**: Suggests the optimal correction minimizing edit distance and maximizing contextual fit.
10. **Interactive Pipeline Inspection**: Displays all intermediate data structures in an expandable UI view.
11. **Standalone Preprocessing Demo**: Executable via `python preprocessing_demo.py`.
12. **Rigorous Evaluation**: Calculates un-fabricated performance metrics using `data/gujarati_test.csv` via `python evaluate.py`.

---

## 7. Non-Functional Requirements
- **Explainability**: Every algorithm, formula, and decision can be explained line-by-line during a university viva.
- **Simplicity**: Prefers clean, direct implementations over convoluted heuristics.
- **Local Execution**: Runs entirely offline on standard consumer hardware without GPU requirements.
- **Maintainability & Clean Architecture**: Modular Python codebase adhering to standard software engineering practices.
- **Performance**: Instantaneous inference (<50ms per sentence) using pre-calculated JSON statistical models.

---

## 8. NLP Techniques

### 8.1 Preprocessing & Normalization
- **Unicode Normalization (NFC)**: Resolves composite Gujarati characters (e.g. combining consonant + dependent matra).
- **Whitespace Normalization**: Normalizes tabs, newlines, and variable spacing.
- **Punctuation Stripping**: Isolates words by removing standard (`.`, `,`, `?`, `!`) and Indic punctuation (`।`, `॥`).
- **Tokenization**: Segmenting sentences into ordered token lists.
- **Stopword Policy**: Preserving functional words ensures sentence structure is intact for N-gram transition modeling.

### 8.2 Statistical N-Gram Language Model
- **Unigram Probability**: Prior likelihood of a word occurring independently:
  $$P(w) = \frac{\text{Count}(w)}{N}$$
- **Bigram Transition Probability**: Likelihood of word $w_i$ given preceding word $w_{i-1}$:
  $$P(w_i | w_{i-1}) = \frac{\text{Count}(w_{i-1}, w_i)}{\text{Count}(w_{i-1})}$$
- **Trigram Transition Probability**: Likelihood given previous two words:
  $$P(w_i | w_{i-2}, w_{i-1}) = \frac{\text{Count}(w_{i-2}, w_{i-1}, w_i)}{\text{Count}(w_{i-2}, w_{i-1})}$$
- **Laplace (Add-One) Smoothing**: Mitigates zero-probability issues for unseen transitions:
  $$P_{\text{Laplace}}(w_i | w_{i-1}) = \frac{\text{Count}(w_{i-1}, w_i) + 1}{\text{Count}(w_{i-1}) + V}$$

### 8.3 Levenshtein Distance (Edit Distance)
Dynamic programming algorithm tracking the minimum cost of transforming string $A$ to string $B$ using unit cost operations:
$$DP[i][j] = \min \begin{cases}
DP[i-1][j] + 1 & \text{(Deletion)} \\
DP[i][j-1] + 1 & \text{(Insertion)} \\
DP[i-1][j-1] + \text{cost} & \text{(Substitution, cost = 0 if match, else 1)}
\end{cases}$$

---

## 9. Datasets

The project uses **EXACTLY TWO** datasets located in `data/`:
1. `data/gujarati_training.csv`:
   - Columns: `id,sentence`
   - Content: 829 authentic Gujarati sentences.
   - Purpose: Builds vocabulary (550 unique words) and computes Unigram, Bigram, and Trigram frequencies.
2. `data/gujarati_test.csv`:
   - Columns: `id,input_word,correct_word,error_type,expected_correction`
   - Content: 109 test instances (89 deliberate spelling errors across 4 error types: `matra_substitution`, `substitution`, `insertion`, `deletion`, and 20 `no_error` instances).
   - Purpose: Held-out evaluation only. Never used during training.

---

## 10. System Architecture

```
                      +-----------------------------+
                      |         User Input          |
                      |   (Word or Full Sentence)   |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |    Visible Preprocessing    |
                      |  - Unicode NFC Normalization|
                      |  - Whitespace Normalization |
                      |  - Punctuation Handling     |
                      |  - Tokenization             |
                      |  - Stopwords Retained       |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |      Vocabulary Check       |
                      | (Trained Model Vocabulary)  |
                      +--------------+--------------+
                                     |
                     +---------------+---------------+
                     |                               |
              [Word In Vocab]                [Word Not In Vocab]
                     |                               |
                     v                               v
            +----------------+             +-------------------+
            | Retain as Valid|             | Candidate Gen.    |
            | Word (dist = 0)|             | (Dist <= 1, <= 2) |
            +--------+-------+             +---------+---------+
                     |                               |
                     |                               v
                     |                     +-------------------+
                     |                     | Levenshtein Dist. |
                     |                     | (Dynamic Program) |
                     |                     +---------+---------+
                     |                               |
                     |                               v
                     |                     +-------------------+
                     |                     | N-Gram Contextual |
                     |                     | Probability Rank  |
                     |                     +---------+---------+
                     |                               |
                     +---------------+---------------+
                                     |
                                     v
                      +-----------------------------+
                      |    Final Spell Decision     |
                      | & Reconstructed Text Output |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |    Flask Web Application    |
                      |     (templates / static)    |
                      |  - Result Verdict Card      |
                      |  - Visual Token Pills       |
                      |  - Full Pipeline Inspector  |
                      +-----------------------------+
```

---

## 11. Workflow
1. **Input Reception**: User supplies text via the web interface or terminal.
2. **Preprocessing Pipeline**: Input undergoes Unicode normalization, whitespace trimming, and punctuation cleaning.
3. **Token Extraction**: Preprocessed text is segmented into individual words.
4. **Vocabulary Lookup**: Each token is checked against the 550 vocabulary terms learned from the training corpus.
5. **Candidate Filtering**: For words not in vocabulary, vocabulary entries within Levenshtein distance $\le 1$ (or fallback $\le 2$) are selected.
6. **Context Scoring**: Candidates are ranked by edit distance first, then contextual probability via Trigram/Bigram transitions.
7. **Result Synthesis**: Best candidates replace misspelled words; correct tokens are preserved.
8. **UI Presentation**: Results and every intermediate stage are rendered clearly for interactive examination.

---

## 12. Evaluation
Quantitative testing is executed via `evaluate.py` using `data/gujarati_test.csv` (109 test instances).
- **Error Detection Rate**: Measures the system's sensitivity in identifying out-of-vocabulary misspellings without false acceptance.
- **Exact Correction Accuracy**: Measures exact string equality between `suggested_correction` and `expected_correction`.
- **False Positives**: Measures correct words outside the domain vocabulary flagged as unknown.
- **Zero Data Leakage Guarantee**: The test dataset is strictly isolated from model training.

---

## 13. Limitations
1. **Domain-Specific Vocabulary**: Vocabulary is limited to 550 words derived from the training corpus. Words outside this corpus cannot be generated as corrections unless the training corpus is expanded.
2. **Statistical Model Scope**: N-gram statistics reflect the sentence patterns present in the training corpus.
3. **Orthographic Ambiguity**: Phonetic variants with equal edit distance and zero corpus co-occurrence fall back to unigram frequency priors.

---

## 14. Future Scope
- Incorporating phonetic Gujarati Soundex / metaphone rules for phonetic substitutions.
- Training on larger public Gujarati corpora (e.g., Leipzig Corpora, IndicCorp) to expand vocabulary coverage.
- Context-sensitive real-word error detection (detecting valid Gujarati words used improperly in context).
- Exportable PDF or CSV viva reporting.

---

## 15. Technology Stack
- **Backend**: Python 3.11, Flask
- **Frontend**: HTML5, Vanilla CSS3, Vanilla JavaScript (ES6+)
- **Standard Library Modules**: `csv`, `json`, `re`, `unicodedata`, `collections`, `pathlib`, `sys`, `os`
- **Data Storage**: CSV (datasets), JSON (trained statistical model)
- **Dependencies**: Exclusively `Flask` (`requirements.txt`)
