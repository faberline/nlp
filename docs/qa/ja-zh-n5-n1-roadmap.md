# Japanese → Taiwan Traditional Chinese: N5–N1 roadmap

**Status:** Issue #80 roadmap/setup snapshot. All five levels were unaccepted at setup. Read the live issue records for the currently selected issue and later status.

**Scope:** Japanese to Taiwan Traditional Chinese (`ja → zh-Hant`). This roadmap does not set a schedule or acceptance plan for other language directions.

## Goal and order

Work through N5, N4, N3, N2, and N1 in that order. The labels are approximate project levels. They are not official JLPT certification.

Keep the goal active until all five levels have passed acceptance, each has a verified new formal release, and no open issue blocks that acceptance.

At roadmap setup, N5 was the first level in the sequence. Start the next level only after the prior level has passed its full acceptance checks and its new three-platform release has been downloaded and verified. The five milestones are [N5 #6](https://github.com/faberline/nlp/milestone/6), [N4 #7](https://github.com/faberline/nlp/milestone/7), [N3 #8](https://github.com/faberline/nlp/milestone/8), [N2 #9](https://github.com/faberline/nlp/milestone/9), and [N1 #10](https://github.com/faberline/nlp/milestone/10). Read their live records for current status.

## Level scope and current visible bank counts

These counts report rows in the two adopted development banks. They do not show that a level is covered or accepted. The bank content is visible to developers and is not held out.

| Level | Scope for this project | Adopted visible rows | Sequence gate |
|---|---|---:|---|
| N5 | Basic positive and negative forms; present and past; topic and subject; places and destinations; common verbs; numbers and dates; names and neutral address | 10 (4 N-level + 6 さん bank) | First in sequence; unaccepted at setup. |
| N4 | Basic connected clauses; cause and sequence; intention and experience; ability and requests; particle roles such as a meeting partner | 10 (4 N-level + 6 さん bank) | After N5 acceptance and verified release. |
| N3 | Longer everyday statements; cause and concession; aspect changes; relative clauses; intermediate conversation | 4 | After N4 acceptance and verified release. |
| N2 | Advance preparation; conditions and concession; established arrangements versus personal intent; workplace clauses | 4 | After N3 acceptance and verified release. |
| N1 | Abstract statements; denied inferences; uncertain outcomes; appearance versus actual result; omitted explanation; formal register | 4 | After N2 acceptance and verified release. |

The 20-row bank is pinned at `a24d91502b3d361eec1ff2bf97381d0accbf27d7516ba2188e77b3b4e96695c0`. The 12-row さん bank is pinned at `870e9e759b65d9f75c277517792315f1cbd0b63ae147c5685690f0eab4f62177`. Their public PRs are [#71](https://github.com/faberline/nlp/pull/71) and [#73](https://github.com/faberline/nlp/pull/73). The setup counts are N5 10, N4 10, N3 4, N2 4, and N1 4. These counts do not show acceptance.

The banks are small development samples. Four rows at a level, or ten rows when the さん bank also applies, cannot prove full-level quality. The exact bank and reference rules for any new measurement must be recorded before the measurement starts.

## Acceptance checks for each level

A separate checker must review every designated acceptance item. The checker must keep these questions separate:

1. **Meaning:** Does the Chinese keep the Japanese participants, actions, roles, polarity, time, aspect, scope, and modality?
2. **Completeness:** Does it preserve all important content, names, conditions, and clause relations without adding unsupported people, gender, place, cause, or certainty?
3. **Taiwan Chinese naturalness:** Does it read as natural Traditional Chinese for Taiwan, rather than as a word-for-word Japanese structure?

A score cannot replace these checks. References are provisional. Any wording that preserves meaning and completeness and reads naturally in Taiwan Traditional Chinese may pass; it need not match the primary reference word for word. A primary or alternate string match, or a score by itself, does not pass the three checks. Keeping a proper name in its source script is not an error by itself. Keep unknown content visible and count it under the existing rules. Record uncertain items as open gaps.

Keep all 24 existing evaluation rules and their thresholds. This roadmap changes no evaluation data or baseline. Baseline updates still require a baseline-only PR after the changes they reflect are merged. Never lower a baseline to make a candidate pass. Keep the original ratchet, or no-regression check: overall and set/dev results must not drop, and no dev phenomenon may drop by more than 0.5. Report only the allowed aggregate test number. Never read or display held-out rows.

For a higher level, keep each completed lower level's pinned bank and accepted quality evidence. Recheck that evidence after later changes. A new level cannot make an accepted earlier level worse.

## Workflow and worker contracts

Work on one currently selected issue at a time. The controller breaks work into bounded tasks, assigns and tracks them, then checks the actual diff, required tests, and evidence. The controller owns acceptance, merge, and release. Delegate ordinary implementation to subagents. Choose Terra for complex Rust rules, cross-module work, or deep review. Choose Luna for small fixes, data, documentation, or routine checks. These are task-based choices, not fixed roles.

Every task must state six fields: goal, input version, allowed files, forbidden actions or inputs, deliverables, and acceptance evidence. Each agent may hold one explicit subtask at a time. Parallel work is allowed only when tasks have no dependency and do not write the same file. Follow dependencies in order. Use one writer per file and one exclusive build lease at a time.

Reserve a checker who did not implement the PR. Formal review starts only after the PR head is frozen and all seven required hosted checks show `SUCCESS`. New problems get separate issues. If work is blocked, record the reason and the condition that clears it; choose a ready issue in the next round. Each round reports the issue, result, and next step.

The canonical [ja → zh-Hant development rules](../ja-zh-development-rules.md) remain in force and are not changed by this roadmap. Keep the translator on pure CPU rules and the self-built source dictionary. Data licences stay within MIT, Apache-2.0, Unicode-3.0, and public domain. Do not add an LLM, GPU runtime, or paid runtime.

The discussion check runs every 20 minutes, reads new reports, and routes them to issues. Worker choice follows task difficulty. Field test `CX-JA-20261116-01` has eight cases and remains paused. Keep one outstanding field batch. Await its response or an explicit instruction to resume before changing the field-test plan.

## Backlog and issue state

At roadmap setup, issue #80 was the selected issue. Read live issue records for the current selection and status. The setup record links 32 unique atomic issues and records their verified bodies, milestone assignments, and dependencies.

The latest post-creation REST inventory artifact is timestamped **2026-11-16 17:12:41 UTC**. It contains 115 records: 62 issues (46 open) and 53 pull requests (3 open). This replaces the earlier 16:55:48 snapshot of 30 issues. That earlier count is historical, not current. Refresh live metadata before a later freeze or review.

The snapshot includes issues #80–#116. Issues [#81](https://github.com/faberline/nlp/issues/81), [#82](https://github.com/faberline/nlp/issues/82), [#83](https://github.com/faberline/nlp/issues/83), and [#84](https://github.com/faberline/nlp/issues/84) and milestones #11–#14 cover wider translation planning. Issue #82 is the broader Japanese-to-Taiwan/China parent. It reuses N5–N1 milestones #6–#10 and does not create a competing development schedule. This document remains limited to Japanese → Taiwan Traditional Chinese. Do not change those sibling issues or milestones here.

Legacy architecture milestones #1–#5 remain unchanged. This roadmap does not reopen, close, rename, or mark their issues complete. N5–N1 milestones are [#6](https://github.com/faberline/nlp/milestone/6), [#7](https://github.com/faberline/nlp/milestone/7), [#8](https://github.com/faberline/nlp/milestone/8), [#9](https://github.com/faberline/nlp/milestone/9), and [#10](https://github.com/faberline/nlp/milestone/10). The 2026-11-16 17:16:07 UTC milestone readback showed all five were open and unaccepted, with 17, 6, 4, 3, and 3 open issue entries respectively. This is a dated setup fact. Read live milestone records for current status. This document does not edit milestone descriptions or indexes.

### N5–N1 atomic issue index

The table records Root's approved titles and dependencies. Only one issue may be selected for a development round. Issue creation does not mean an issue is ready to start; milestone gates and dependencies still apply.

### N5 issue index

| Issue | Approved title | Depends on | Scope note |
|---|---|---|---|
| [#80](https://github.com/faberline/nlp/issues/80) | 完成 N5 至 N1 單輪 issue 與依賴盤點 | — | During roadmap setup, this was the selected issue for the roadmap and queue plan. |
| [#85](https://github.com/faberline/nlp/issues/85) | [N5] 建立 N5 採納資料與缺口清單 | [#80](https://github.com/faberline/nlp/issues/80) | Inventory adopted bank hashes, coverage counts, unknowns, and independent-data needs. |
| [#86](https://github.com/faberline/nlp/issues/86) | [N5] 獨立採納 N1–N5 開發資料提案 | [#80](https://github.com/faberline/nlp/issues/80) | Uses open PR #74. Do not execute its 50 rows before separate data adoption and independent review. |
| [#87](https://github.com/faberline/nlp/issues/87) | [N5] 檢查並交付移動終點的助詞處理 | [#80](https://github.com/faberline/nlp/issues/80) | Uses open PR #76. Candidate QA is not main quality or acceptance; recheck seven hosted checks before independent review. |
| [#88](https://github.com/faberline/nlp/issues/88) | [N5] 處理既有英文詞性修正候選 | [#80](https://github.com/faberline/nlp/issues/80) | Uses open PR #79. Preserve exact source, membership, order, and cost evidence. |
| [#89](https://github.com/faberline/nlp/issues/89) | [N5] 審查開發參考譯文與baseline遷移 | [#80](https://github.com/faberline/nlp/issues/80) | Triage confirmed dev-reference issues and baseline migration; keep standing rules and hashes. |
| [#90](https://github.com/faberline/nlp/issues/90) | [N5] 獨立採納開發資料現象分組 | [#85](https://github.com/faberline/nlp/issues/85) | Independently adopt the 32-row, 11-group map; no measurement before review and merge. |
| [#91](https://github.com/faberline/nlp/issues/91) | [N5] 保留姓名與中性敬稱 | [#85](https://github.com/faberline/nlp/issues/85) | Generic honorific/name behavior; preserve explicit gender and ordinary-word controls. |
| [#92](https://github.com/faberline/nlp/issues/92) | [N5] 盤點150人名草稿並只擇已驗證N5必要子集 | [#85](https://github.com/faberline/nlp/issues/85), [#88](https://github.com/faberline/nlp/issues/88) | Triage the 150-name draft; only a source-verified N5-needed subset may advance. Depends on #85 and #88. |
| [#93](https://github.com/faberline/nlp/issues/93) | [N5] 按 N5 驗收缺口拆下一個原子修正 | [#85](https://github.com/faberline/nlp/issues/85) | Split observed N5 blockers into traceable atomic repairs with positive and near-negative controls. |
| [#94](https://github.com/faberline/nlp/issues/94) | [N5] 修正存在句的場所與否定角色 | [#85](https://github.com/faberline/nlp/issues/85) | Existence, location, and negation only; separate from honorifics #91 and names #92. |
| [#95](https://github.com/faberline/nlp/issues/95) | [N5] 補足 N5 一般名詞與地點詞義缺口 | [#85](https://github.com/faberline/nlp/issues/85), [#88](https://github.com/faberline/nlp/issues/88) | Ordinary noun/place meanings only; separate from #91/#92 and do not bundle the name inventory. |
| [#96](https://github.com/faberline/nlp/issues/96) | [N5] 增加已採納資料現象報告 | [#90](https://github.com/faberline/nlp/issues/90) | Report-only phenomenon view after #90; old 24 keys, parser, and standing baseline stay unchanged. |
| [#97](https://github.com/faberline/nlp/issues/97) | [N5] 建立現象基準單檔資料 | [#96](https://github.com/faberline/nlp/issues/96) | Add one separate phenomenon-baseline file after #96; do not change the standing baseline. |
| [#98](https://github.com/faberline/nlp/issues/98) | [N5] 啟用必備現象基準檢查 | [#97](https://github.com/faberline/nlp/issues/97) | Require the new baseline after #97. Missing or mismatched data fails; no silent fallback. No extra owner approval is inferred; only a real rule conflict gets a concrete scoped exception request. |
| [#99](https://github.com/faberline/nlp/issues/99) | [N5] 獨立驗收 N5 語意、完整與自然度 | [#85](https://github.com/faberline/nlp/issues/85), [#87](https://github.com/faberline/nlp/issues/87), [#91](https://github.com/faberline/nlp/issues/91), [#93](https://github.com/faberline/nlp/issues/93), [#98](https://github.com/faberline/nlp/issues/98), [#80](https://github.com/faberline/nlp/issues/80), [#86](https://github.com/faberline/nlp/issues/86), [#90](https://github.com/faberline/nlp/issues/90), [#96](https://github.com/faberline/nlp/issues/96), [#97](https://github.com/faberline/nlp/issues/97), [#89](https://github.com/faberline/nlp/issues/89), [#92](https://github.com/faberline/nlp/issues/92), [#94](https://github.com/faberline/nlp/issues/94), [#95](https://github.com/faberline/nlp/issues/95) | Independent N5 acceptance across meaning, completeness, and Taiwan naturalness. All current and later blockers must be done; unknowns remain open. |
| [#100](https://github.com/faberline/nlp/issues/100) | [N5] N5 milestone 全平台新版本發布 | [#99](https://github.com/faberline/nlp/issues/99) | One new immutable three-platform release after N5 acceptance; do not move or replace v0.1.0. |

### N4 issue index

| Issue | Approved title | Depends on | Scope note |
|---|---|---|---|
| [#101](https://github.com/faberline/nlp/issues/101) | [N4] 盤點 N4 已採納資料及缺口 | [#100](https://github.com/faberline/nlp/issues/100) | Start N4 coverage only after verified N5 release #100. |
| [#102](https://github.com/faberline/nlp/issues/102) | [N4] 辨識見面對象與授受角色 | [#101](https://github.com/faberline/nlp/issues/101) | N4 meeting-partner and giving/receiving roles with motion and recipient controls. Meeting-role draft belongs here, not N5. |
| [#103](https://github.com/faberline/nlp/issues/103) | [N4] 獨立採納 N4 基本意圖開發控制 | [#101](https://github.com/faberline/nlp/issues/101) | Adopt eight basic つもり controls. Provenance pin: private commit 4f03b2a62710b6d307f1a9428d55421934b09498; source SHA-256 60e4e0493c46198f0558a35cca0888ccd685996123105c11cab329565c7a5d32. Not yet adopted. |
| [#104](https://github.com/faberline/nlp/issues/104) | [N4] 一般化 N4 非過去意圖 | [#103](https://github.com/faberline/nlp/issues/103) | Generalize basic N4 nonpast intention only after #103 adoption; keep past-belief examples as controls. |
| [#105](https://github.com/faberline/nlp/issues/105) | [N4] 獨立驗收 N4 與保持 N5 | [#101](https://github.com/faberline/nlp/issues/101), [#102](https://github.com/faberline/nlp/issues/102), [#103](https://github.com/faberline/nlp/issues/103), [#104](https://github.com/faberline/nlp/issues/104) | Independent N4 acceptance and preservation of pinned N5 quality. |
| [#106](https://github.com/faberline/nlp/issues/106) | [N4] N4 milestone 全平台新版本發布 | [#105](https://github.com/faberline/nlp/issues/105) | One new immutable three-platform release after N4 acceptance. |

### N3 issue index

| Issue | Approved title | Depends on | Scope note |
|---|---|---|---|
| [#107](https://github.com/faberline/nlp/issues/107) | [N3] 盤點 N3 已採納資料及缺口 | [#106](https://github.com/faberline/nlp/issues/106) | Start N3 coverage only after verified N4 release #106. |
| [#108](https://github.com/faberline/nlp/issues/108) | [N3] 一般化進階過去意圖與主觀信念結構 | [#107](https://github.com/faberline/nlp/issues/107), [#103](https://github.com/faberline/nlp/issues/103) | Advanced past-belief behavior only; reuse adopted #103 data and do not duplicate or create a dataset. Keep distinct from basic nonpast intention #104. |
| [#109](https://github.com/faberline/nlp/issues/109) | [N3] 獨立驗收 N3 與保持 N5/N4 | [#107](https://github.com/faberline/nlp/issues/107), [#108](https://github.com/faberline/nlp/issues/108) | Independent N3 acceptance and preservation of N5/N4 quality. |
| [#110](https://github.com/faberline/nlp/issues/110) | [N3] N3 milestone 全平台新版本發布 | [#109](https://github.com/faberline/nlp/issues/109) | One new immutable three-platform release after N3 acceptance. |

### N2 issue index

| Issue | Approved title | Depends on | Scope note |
|---|---|---|---|
| [#111](https://github.com/faberline/nlp/issues/111) | [N2] 盤點 N2 已採納資料及缺口 | [#110](https://github.com/faberline/nlp/issues/110) | Start N2 coverage only after verified N3 release #110. |
| [#112](https://github.com/faberline/nlp/issues/112) | [N2] 獨立驗收 N2 與保持 N5–N3 | [#111](https://github.com/faberline/nlp/issues/111) | Independent N2 acceptance and preservation of N5–N3 quality. |
| [#113](https://github.com/faberline/nlp/issues/113) | [N2] N2 milestone 全平台新版本發布 | [#112](https://github.com/faberline/nlp/issues/112) | One new immutable three-platform release after N2 acceptance. |

### N1 issue index

| Issue | Approved title | Depends on | Scope note |
|---|---|---|---|
| [#114](https://github.com/faberline/nlp/issues/114) | [N1] 盤點 N1 已採納資料及缺口 | [#113](https://github.com/faberline/nlp/issues/113) | Start N1 coverage only after verified N2 release #113. |
| [#115](https://github.com/faberline/nlp/issues/115) | [N1] 獨立驗收 N1 與保持 N5–N2 | [#114](https://github.com/faberline/nlp/issues/114) | Independent N1 acceptance and preservation of N5–N2 quality. |
| [#116](https://github.com/faberline/nlp/issues/116) | [N1] N1 milestone 全平台新版本發布 | [#115](https://github.com/faberline/nlp/issues/115) | One new immutable three-platform release after N1 acceptance. |

### Related existing issue outside the N3 milestone

[Issue #50](https://github.com/faberline/nlp/issues/50) remains open with no milestone. Do not move it, close it, or create a duplicate. After N3 coverage issue #107, decide whether its copyright-safe Japanese dialogue evaluation belongs in N3 acceptance. It is related work only until that decision.

### Boundaries between related work

- N5 #91 handles neutral honorific behavior. N5 #92 handles verified personal-name inventory. N5 #94 covers existence, location, and negation. N5 #95 covers ordinary noun and place meanings. Keep these scopes separate.
- Open PR #74 is assigned to N5 [issue #86](https://github.com/faberline/nlp/issues/86) for independent data adoption. Its 50 rows remain unadopted and must not be translated or measured before adoption.
- Open PR #76 is assigned to N5 [issue #87](https://github.com/faberline/nlp/issues/87). Its candidate diagnostic is not evidence about current main.
- Open PR #79 is assigned to N5 [issue #88](https://github.com/faberline/nlp/issues/88). Lexicon work in #92 and #95 depends on this issue where the approved plan says so.
- The meeting-role draft belongs to N4 #102, not N5. N4 #103 adopts eight basic つもり controls; N4 #104 generalizes nonpast intention. N3 #108 handles advanced past-belief behavior only and does not duplicate data.
- The measurement chain is #90 → #96 → #97 → #98. #90 adopts grouping metadata; #96 reports it; #97 adds a separate phenomenon baseline; #98 makes that baseline required. This chain does not change the standing baseline or original 24-rule ratchet.
- The controller's discussion check has a verified 20-minute readback and selects worker models by task difficulty. It tracks work; it does not choose a different active issue, bypass dependencies, or accept work.

## Known evidence and gaps

**SHA labels in the PR table:** these are author/frozen PR heads, not squash-merge commits. The main snapshot for this roadmap is `4398f28052cb3b6f891cc14ed99ca5c5504bf105`. The actual source for historical release v0.1.0 is `b0ad8bdc0477cfc3432d8dc1950d23cee44832d8`.

| Evidence | Current state | Roadmap treatment |
|---|---|---|
| [PR #70, frozen head 14d3c1dedb2c723df55b2375884714bece70d7e6](https://github.com/faberline/nlp/pull/70) | Merged CLI work combines the seven existing commands. | Reuse the existing CLI. Do not treat it as N5 acceptance. |
| [PR #71, frozen head 34583cb2d917af513a9d1724f4ce66e52cca4559](https://github.com/faberline/nlp/pull/71) | Merged independent 20-row N1–N5 approximate-level development bank. | Reuse the pinned visible development rows and hash above. The small counts do not prove level quality. |
| [PR #73, frozen head ae01afb45b73e076bf9d73698b52655972407279](https://github.com/faberline/nlp/pull/73) | Merged independent 12-row gender-neutral さん development bank. | Reuse the pinned visible rows and hash above; retain its ordinary-word controls. |
| [PR #72, frozen head 3c709e239b65d42c7da05eda6c2b43adcbc1bcd9](https://github.com/faberline/nlp/pull/72) and [v0.1.0 release](https://github.com/faberline/nlp/releases/tag/v0.1.0) | Verified historical platform delivery from source `b0ad8bdc0477cfc3432d8dc1950d23cee44832d8`. | Historical delivery only. It does not accept N5. Each accepted milestone needs one new complete release. |
| [PR #75, frozen head 9d17beb5a9ffd73fda2a76f48a05e705c9bf7101](https://github.com/faberline/nlp/pull/75) | Merged lexicon rebuild from allowlisted inputs. | Reuse the first-party embedded dictionary work and its notices. Keep dictionary/source review separate from translation acceptance. |
| [PR #77, frozen head 698673b083e8cc7c293f11a3ce7c5d6a25bb8446](https://github.com/faberline/nlp/pull/77) | Merged aggregate regression check. | Preserve it and all prior thresholds. It cannot accept meaning or naturalness. |
| [PR #78, frozen head 215779b7ce1793b6509b00c3b46c0b649074d647](https://github.com/faberline/nlp/pull/78) | Merged report for adopted development banks by approximate level. | Treat as reporting, not whole-level proof or acceptance. |
| [PR #74, 1e1def64e7dcb694cfbff58570437ceb004f09bc](https://github.com/faberline/nlp/pull/74) | Open. N5 [issue #86](https://github.com/faberline/nlp/issues/86) handles this PR. The metadata records 50 additional rows and an unadopted bank. | Keep outside the adopted counts. Do not execute or use it until its separate data adoption is complete. Its frozen public head is `1e1def64e7dcb694cfbff58570437ceb004f09bc`. |
| [PR #76, frozen head bd96b678de5a428190771fc1894c1e9b69708f1d](https://github.com/faberline/nlp/pull/76) | Open candidate change assigned to N5 [issue #87](https://github.com/faberline/nlp/issues/87). Its provisional model diagnostic on 32 adopted rows reported semantic 3 correct / 14 partial / 15 wrong, and naturalness 3 correct / 4 partial / 25 wrong. This is not human gold or a statistical accuracy estimate. The candidate result is not current-main quality. | Treat it as candidate evidence only. Check exact-head hosted CI and perform independent review before any acceptance. |
| [PR #79, 50fe8acd52a6f53b6ce7688573b01adf3257577e](https://github.com/faberline/nlp/pull/79) | Open lexicon change, assigned to N5 [issue #88](https://github.com/faberline/nlp/issues/88). | No acceptance claim here. Check its exact head and evidence when it is selected for review. |

An independent audit of the standing visible development references found three high-confidence concerns (jz008, jz086, jz142), seven uncertain cases, and three style-only examples. Each concern also has a neutral alternate reference. These rows are separate from the adopted 32-row bank. This audit does not show that the primary references caused the H1 candidate's 0.87 metric drop. Triage them through N5 [issue #89](https://github.com/faberline/nlp/issues/89). Do not change data or baseline to rescue a candidate. If a real rule conflict remains, submit a concrete, scoped owner exception request.

The 32-row phenomenon map remains unadopted; N5 [issue #90](https://github.com/faberline/nlp/issues/90) owns independent adoption. The standing runner uses the better chrF match across each primary and alternate reference. The adopted-32 reporter uses all primary references only. Both use chrF, but their reference policies differ. Do not combine or directly compare their scores. The reporting, separate baseline, and mandatory activation issues are N5 [#96](https://github.com/faberline/nlp/issues/96), [#97](https://github.com/faberline/nlp/issues/97), and [#98](https://github.com/faberline/nlp/issues/98). The sequence is #90 → #96 → #97 → #98. Missing or mismatched mandatory data must fail; no silent fallback is allowed. This does not weaken the original 24 rules. No extra owner approval is inferred; only a real rule conflict gets a concrete, scoped exception request. Keep author drafts out of counts and measurements until a separate data-only PR records review and adoption. The map draft does not create a new gate or weaken the existing 24-rule ratchet.

### Preserved local drafts and evidence

These local fingerprints preserve provenance. They are not public PR heads and do not mean the work is accepted.

| Issue | Preserved draft evidence | State and boundary |
|---|---|---|
| N5 #91 — honorific repair | Private base `6f1a9dc31e2629228badca98e87d2f2a7c42c3c5`; dirty-file SHA-256 `77a2fa022696d90fdb2732f98e1c7d8cc845b9891468773d988489b2c0841b54` | The draft scored 6.97 versus 7.84 on the standing katakana group, a 0.87 drop. It failed, has no PR, and is not accepted. Preserve the dirty file. The three reference concerns above do not prove the cause of this drop. |
| N5 #92 — name inventory | Private commit `9344f4164de2dd28c3bfb8f89da8b53255e5a803` | The 150-row draft has 49 new translation glosses: 17 are tentatively plausible and 32 remain unverified. The other 101 rows are unchanged. No new row or cost change was found. Only a source-verified N5-needed subset may advance. Keep unverified glosses private or unknown. |
| N4 #102 — meeting-role controls | Private commit `aceb6df548743aa2b6fc4d53a83e2725c92a7627` | The draft was flat on the standing 24-rule metric. It is not accepted and does not justify a rule change. |
| N4 #103 — basic つもり controls | Private commit `4f03b2a62710b6d307f1a9428d55421934b09498`; data SHA-256 `60e4e0493c46198f0558a35cca0888ccd685996123105c11cab329565c7a5d32` | Eight basic-intention controls remain unadopted. Do not execute them before independent data review and adoption. |

The local hashes above are retained but not published. These drafts have no accepted PR. Do not present their private fingerprints as public commit links.

## Release rule

After all acceptance issues for one milestone pass, publish **one new immutable full CPU NLP version for that milestone**. Do not publish one product version per issue or per intermediate batch. Each milestone release must include the existing seven commands and embedded resources for macOS ARM, macOS Intel, and Windows. Verify native execution, download, source/version identity, and checksums. Close the milestone only after all three platform packages are verified usable.

The required exact-head hosted checks are seven checks, together with the author's required checks. All must pass on the frozen head. Keep earlier passing evidence, but refresh it after a rebase or source change. Do not infer a pass from missing, stale, or pending checks.

## Public history

The PR links above are the public record for prior product work. Their listed PR SHAs are author/frozen heads, not squash-merge commit SHAs. The main setup source is `4398f28052cb3b6f891cc14ed99ca5c5504bf105`; the actual v0.1.0 release source is `b0ad8bdc0477cfc3432d8dc1950d23cee44832d8`. The milestone pages link the five sequential N-level milestones. The public historical QA record is [Discussion #45, comment 19643372](https://github.com/faberline/nlp/discussions/45#discussioncomment-19643372); it records a prior candidate diagnostic and is not main acceptance. The local draft fingerprints above are retained but not published and have no accepted PR.
