#!/usr/bin/env python3
"""Synthetic checks for English POS inference without corpus or model inputs."""
import importlib.util
import pathlib
import unittest

spec = importlib.util.spec_from_file_location("build_en", pathlib.Path(__file__).with_name("build_en.py"))
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class PosRepairTests(unittest.TestCase):
    def test_reviewed_sample_evidence_preserves_supported_ambiguity(self):
        for word, expected in {
            "initialer": {"NOUN"},
            "scepters": {"NOUN", "VERB"},
            "recolouring": {"NOUN", "VERB"},
            "moulder": {"NOUN", "VERB"},
            "morselling": {"NOUN", "VERB"},
        }.items():
            self.assertEqual(
                builder.apply_reviewed_pos_evidence(word, set(), builder.REVIEWED_POS_EVIDENCE),
                expected,
            )
        self.assertEqual(builder.REVIEWED_LEMMA_EVIDENCE["reedifying"], "reedify")
        self.assertEqual(builder.REVIEWED_LEMMA_EVIDENCE["initialer"], "-")
        for counterpart in ("initialler", "sceptres", "recoloring", "molder", "morseling", "reaedifying"):
            self.assertNotIn(counterpart, builder.REVIEWED_POS_EVIDENCE)
            self.assertNotIn(counterpart, builder.REVIEWED_LEMMA_EVIDENCE)

    def test_reviewed_counterpart_removal_is_exact_and_keeps_variant(self):
        pairs = {
            "ceratin": ("en-US", "creatin"),
            "creatin": ("en-GB", "ceratin"),
            "ceratins": ("en-US", "creatins"),
            "creatins": ("en-GB", "ceratins"),
            "color": ("en-US", "colour"),
            "colour": ("en-GB", "color"),
            "unpaired": ("en-US", "-"),
        }
        repaired = builder.remove_reviewed_bad_counterparts(pairs)
        for word in ("ceratin", "creatin", "ceratins", "creatins"):
            self.assertEqual(repaired[word], (pairs[word][0], "-"))
        for word in ("color", "colour", "unpaired"):
            self.assertEqual(repaired[word], pairs[word])
        self.assertEqual(pairs["ceratin"], ("en-US", "creatin"))
        self.assertEqual(builder.remove_reviewed_bad_counterparts({}), {})
        self.assertEqual(
            builder.remove_reviewed_bad_counterparts({"ceratin": ("en-US", "other")}),
            {"ceratin": ("en-US", "other")},
        )

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
