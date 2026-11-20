import unittest

from apply_ja_glosses import apply_reviewed_345


class ReviewedCorrectionsTest(unittest.TestCase):
    def rows(self):
        return {
            "供応": [["供応", "ja", "763", "-", "NOUN|VERB", "-", "供應", "-"]],
            "敷衍": [["敷衍", "ja", "763", "-", "NOUN|VERB", "-", "敷衍", "-"]],
            "パンチ": [["パンチ", "ja", "501", "-", "NOUN|VERB", "-", "拳擊", "src=first_party_mit"]],
            "暗算": [["暗算", "ja", "759", "-", "NOUN|VERB", "-", "暗算", "-"]],
            "気絶": [["気絶", "ja", "763", "-", "NOUN|VERB", "-", "氣絕", "-"]],
        }

    def test_shipped_and_blank_rows_are_idempotent(self):
        for blank in (False, True):
            rows = self.rows()
            if blank:
                for entries in rows.values():
                    entries[0][6:] = ["-", "-"]
            self.assertEqual(apply_reviewed_345(rows), 5)
            self.assertEqual([rows[s][0][6] for s in rows], ["宴請", "詳細說明", "一拳", "心算", "昏厥"])
            self.assertEqual(apply_reviewed_345(rows), 0)

    def test_unknown_gloss_cost_or_source_is_rejected(self):
        for index, value in ((6, "未知"), (2, "764"), (7, "src=other")):
            rows = self.rows()
            rows["供応"][0][index] = value
            with self.assertRaises(ValueError):
                apply_reviewed_345(rows)


if __name__ == "__main__":
    unittest.main()
