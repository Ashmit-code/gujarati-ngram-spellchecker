"""
test_spell_checker.py - Automated Unit & Integration Tests

Validates:
1. Levenshtein Dynamic Programming algorithm
2. Preprocessing & Gujarati Unicode normalization
3. N-gram statistical calculations & Laplace smoothing
4. Spell-checker candidate generation & multi-factor ranking
5. Flask API endpoints and response structure
"""

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from levenshtein import levenshtein_distance
from preprocessing import (
    unicode_normalize,
    whitespace_normalize,
    handle_punctuation,
    tokenize,
    handle_stopwords,
    preprocess,
    preprocess_with_details,
)
from ngram_model import NGramLanguageModel
from spell_checker import GujaratiSpellChecker
from app import app


class TestLevenshteinDistance(unittest.TestCase):
    """Tests dynamic programming Levenshtein distance calculations."""

    def test_identical_words(self):
        self.assertEqual(levenshtein_distance("શાળા", "શાળા"), 0)

    def test_single_substitution(self):
        # Substituting 'લ' with 'ળ'
        self.assertEqual(levenshtein_distance("શાલામાં", "શાળામાં"), 1)

    def test_single_deletion(self):
        self.assertEqual(levenshtein_distance("પુસ્તક", "પુસ્ત"), 1)

    def test_single_insertion(self):
        # Adding 'ત' at the end of 'પુસ્ત' -> 'પુસ્તક' (distance 1)
        self.assertEqual(levenshtein_distance("પુસ્ત", "પુસ્તક"), 1)

    def test_empty_string(self):
        self.assertEqual(levenshtein_distance("", "ગુજરાતી"), 7)
        self.assertEqual(levenshtein_distance("ગુજરાતી", ""), 7)
        self.assertEqual(levenshtein_distance("", ""), 0)


class TestPreprocessing(unittest.TestCase):
    """Tests Unicode normalization, whitespace collapsing, punctuation, and tokenization."""

    def test_unicode_normalization(self):
        text = "ગુજરાતી"
        normalized = unicode_normalize(text)
        self.assertEqual(normalized, text)

    def test_whitespace_normalization(self):
        text = "  હું   શાળામાં\tજાઉં\n\nછું.  "
        cleaned = whitespace_normalize(text)
        self.assertEqual(cleaned, "હું શાળામાં જાઉં છું.")

    def test_punctuation_handling(self):
        text = "હું શાળામાં જાઉં છું! તમે આવશો? હા, ચોક્કસ।"
        punct_cleaned = handle_punctuation(text)
        self.assertNotIn("!", punct_cleaned)
        self.assertNotIn("?", punct_cleaned)
        self.assertNotIn(",", punct_cleaned)
        self.assertNotIn("।", punct_cleaned)
        self.assertIn("હું", punct_cleaned)
        self.assertIn("શાળામાં", punct_cleaned)

    def test_tokenization(self):
        tokens = tokenize("હું શાળામાં જાઉં છું")
        self.assertEqual(tokens, ["હું", "શાળામાં", "જાઉં", "છું"])

    def test_stopword_preservation(self):
        # Stopword removal must be DISABLED to preserve N-gram context
        tokens = ["હું", "શાળામાં", "જાઉં", "છું"]
        res = handle_stopwords(tokens)
        self.assertEqual(res["status"], "DISABLED")
        self.assertEqual(res["tokens"], tokens)

    def test_preprocess_pipeline(self):
        tokens = preprocess("હું   શાલામાં જાઉં   છું.")
        self.assertEqual(tokens, ["હું", "શાલામાં", "જાઉં", "છું"])

    def test_preprocess_with_details(self):
        details = preprocess_with_details("હું શાલામાં જાઉં છું.")
        self.assertIn("unicode_normalized", details)
        self.assertIn("whitespace_normalized", details)
        self.assertIn("punctuation_handled", details)
        self.assertIn("tokens", details)
        self.assertIn("stopword_handling", details)
        self.assertIn("final_tokens", details)
        self.assertEqual(details["stopword_handling"]["status"], "DISABLED")


class TestNGramLanguageModel(unittest.TestCase):
    """Tests N-gram statistical calculations and Laplace smoothing."""

    def setUp(self):
        self.model = NGramLanguageModel()
        self.corpus = [
            ["હું", "શાળામાં", "જાઉં", "છું"],
            ["મારો", "મિત્ર", "શાળામાં", "અભ્યાસ", "કરે", "છે"],
            ["વિદ્યાર્થી", "શાળામાં", "ભણે", "છે"]
        ]
        self.model.train(self.corpus)

    def test_vocab_extraction(self):
        self.assertIn("શાળામાં", self.model.vocab_set)
        self.assertIn("છે", self.model.vocab_set)
        self.assertTrue(self.model.vocab_size > 5)

    def test_unigram_probability(self):
        prob = self.model.get_unigram_prob("શાળામાં")
        self.assertGreater(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_bigram_probability_smoothed(self):
        prob = self.model.get_bigram_prob("શાળામાં", "જાઉં", smooth=True)
        self.assertGreater(prob, 0.0)

    def test_laplace_smoothing_prevents_zero(self):
        # Unseen bigram transition
        prob = self.model.get_bigram_prob("વિદ્યાર્થી", "જાઉં", smooth=True)
        self.assertGreater(prob, 0.0)

    def test_serialization(self):
        data_dict = self.model.to_dict()
        new_model = NGramLanguageModel()
        new_model.from_dict(data_dict)
        self.assertEqual(new_model.vocab_size, self.model.vocab_size)
        self.assertEqual(new_model.total_words, self.model.total_words)


class TestGujaratiSpellChecker(unittest.TestCase):
    """Tests spell-checker end-to-end functionality using the trained model."""

    @classmethod
    def setUpClass(cls):
        cls.checker = GujaratiSpellChecker()

    def test_model_loaded(self):
        self.assertTrue(self.checker.is_loaded)
        self.assertGreater(self.checker.model.vocab_size, 100)

    def test_correct_word(self):
        res = self.checker.check_word("આવકાર")
        self.assertTrue(res["is_correct"])
        self.assertTrue(res["in_vocabulary"])
        self.assertEqual(res["levenshtein_distance"], 0)

    def test_misspelled_word(self):
        # 'શાલામાં' -> 'શાળામાં'
        res = self.checker.check_word("શાલામાં")
        self.assertFalse(res["is_correct"])
        self.assertFalse(res["in_vocabulary"])
        self.assertEqual(res["suggested_correction"], "શાળામાં")
        self.assertEqual(res["levenshtein_distance"], 1)

    def test_multi_candidate_suggestions(self):
        res = self.checker.check_word("ગરવ")
        self.assertFalse(res["is_correct"])
        # Should return at least 6 candidate suggestions
        self.assertGreaterEqual(len(res["top_candidates"]), 6)

    def test_check_text_sentence(self):
        sentence = "હું શાલામાં જાઉં છું."
        res = self.checker.check_text(sentence)
        self.assertTrue(res["has_error"])
        self.assertEqual(res["tokens"], ["હું", "શાલામાં", "જાઉં", "છું"])
        self.assertIn("શાળામાં", res["corrected_text"])


class TestFlaskEndpoints(unittest.TestCase):
    """Tests Flask web server endpoints."""

    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()

    def test_index_route(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Gujarati spell checker", response.data)

    def test_api_status_route(self):
        response = self.client.get("/api/status")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "ready")
        self.assertGreater(data["vocab_size"], 100)

    def test_check_endpoint_valid_json(self):
        response = self.client.post(
            "/check",
            json={"text": "શાલામાં"},
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertTrue(data["has_error"])
        self.assertEqual(data["corrected_text"], "શાળામાં")

    def test_check_endpoint_empty_input(self):
        response = self.client.post(
            "/check",
            json={"text": "   "},
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data["success"])


if __name__ == "__main__":
    unittest.main()
