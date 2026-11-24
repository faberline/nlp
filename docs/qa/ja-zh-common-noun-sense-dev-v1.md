# 常用日文多義名詞的公開開發對照

這 40 道題使用第一方原創日文與台灣繁中參考，依 MIT 授權。資料由未參與引擎實作的 Codex 子代理撰寫，N5 表示近似程度。 [資料單 #528](https://github.com/faberline/nlp/issues/528)

| 詞條 | 題號 | 核對語義 |
| --- | --- | --- |
| ソフト | N01–N06 | 軟體、遊戲軟體、霜淇淋、壘球 |
| アルバム | N07–N10、N38 | 相簿、音樂專輯 |
| ドラマ | N11–N16、N40 | 電視劇、戲劇作品、劇本、戲劇性事件 |
| ポスト | N17–N22 | 郵筒、職位、球門柱 |
| バッテリー | N23–N26、N39 | 電池、棒球投捕搭檔 |
| カウンター | N27–N32 | 櫃檯、計數器、反擊 |
| ケーブル | N33–N37 | 一般纜線、鋼索、搭乘的纜車 |

每組有明確語境、跨子句或跨句的對照，以及保留不確定性的題目。題目用 `semantic_checks` 記錄必須保留的語意，用 `semantic_unknowns` 記錄原文未明示的資訊。N22 和 N38 檢查引述，N32 保留行內程式碼。

## 設定和範圍

執行時使用共用 `fixture.translate_options`。詞表從題目的 `fixture.glossary` 讀取，缺少時沿用共用空詞表。N40 使用本題的詞表，`expected_glossary_hit` 用於核對輸出中的詞彙命中。來源範圍 `[0, 3)` 以 Unicode 字元從零起算，結尾位置開放。

題庫合併到 `main` 後生效，供 #527 獨立量測詞義判斷。現行 chrF 評估與評分底線沿用。 [開發規則](../ja-zh-development-rules.md#4-measurement-the-gate)

## 固定資料

- 核對基底 `c7b81377e4748f92b60064c9b3646bba16d317d6`。
- 題庫 SHA-256 `806f246cdf5d79811d292185a35d0cf44af17a8b5a2db6780543018d8cfcba1e`。
- 把日文統一全形半形並去除空白與標點，和 359 道已採納開發題的完整日文比較。40 題的完整日文彼此不同，也未與既有題重複。
- 題庫採 MIT 授權。字典用於查證詞義。 [題庫](../../crates/faberline-nlp-translate/tests/data/ja_zh/common_noun_sense_dev_v1.json)
