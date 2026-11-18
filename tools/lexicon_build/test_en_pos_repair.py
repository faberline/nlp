#!/usr/bin/env python3
"""Synthetic checks for English POS inference without corpus or model inputs."""
import importlib.util
import pathlib
import unittest

spec = importlib.util.spec_from_file_location("build_en", pathlib.Path(__file__).with_name("build_en.py"))
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class PosRepairTests(unittest.TestCase):
    def test_reviewed_evidence_is_exact_spelling_and_preserves_ambiguity(self):
        reviewed = {"samplewise": frozenset({"ADJ", "ADV"}), "samplewize": frozenset()}
        self.assertEqual(builder.apply_reviewed_pos_evidence("samplewise", set(), reviewed),
                         {"ADJ", "ADV"})
        self.assertEqual(builder.apply_reviewed_pos_evidence("samplewize", set(), reviewed),
                         set())
        self.assertEqual(builder.apply_reviewed_pos_evidence("samplewize", {"NOUN"}, reviewed),
                         {"NOUN"})
        self.assertIsNone(builder.apply_reviewed_pos_evidence("crosswise", set(), reviewed))
        self.assertEqual(builder.infer_derivational_pos_and_lemma("crosswise", {"cross": {"NOUN"}}),
                         ({"ADV"}, "-"))

    def test_noun_based_wise_and_unverified_partner(self):
        known = {"cross": {"NOUN"}}
        self.assertEqual(builder.infer_derivational_pos_and_lemma("crosswise", known), ({"ADV"}, "-"))
        self.assertEqual(builder.infer_derivational_pos_and_lemma("unwise", known), (set(), "-"))

    def test_sis_noun_does_not_capture_ising_verb(self):
        known = {"photosynthesis": {"NOUN"}}
        self.assertEqual(builder.infer_regular_lemma("photosynthesising", known), (None, set()))
        self.assertEqual(
            builder.infer_derivational_pos_and_lemma("photosynthesizing", known),
            ({"ADJ", "VERB"}, "photosynthesize"),
        )

    def test_wize_requires_actual_pair_and_noun_stem(self):
        known = {"cross": {"NOUN"}, "clock": {"VERB"}}
        self.assertTrue(builder.infer_paired_wize_adverb("crosswize", "crosswise", known))
        self.assertFalse(builder.infer_paired_wize_adverb("crosswize", "-", known))
        self.assertFalse(builder.infer_paired_wize_adverb("crosswize", "clockwise", known))
        self.assertFalse(builder.infer_paired_wize_adverb("clockwize", "clockwise", known))
        self.assertFalse(builder.infer_paired_wize_adverb("memorize", "memorise", known))
        self.assertEqual(builder.infer_derivational_pos_and_lemma("memorize", known), ({"VERB"}, "-"))

    def test_real_noun_ing_base_and_known_verb_are_preserved(self):
        self.assertEqual(builder.infer_regular_lemma("painting", {"paint": {"NOUN"}}), ("paint", {"NOUN"}))
        self.assertEqual(builder.infer_regular_lemma("washing", {"wash": {"VERB"}}), ("wash", {"VERB"}))


if __name__ == "__main__":
    unittest.main()
