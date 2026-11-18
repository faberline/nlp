# Clock boundary and lock development controls v1

Status: proposed. Independent formal review and adoption are pending. No new case has been run through the engine. See [issue #133](https://github.com/faberline/nlp/issues/133) and the related [issue #130](https://github.com/faberline/nlp/issues/130).

The [bank](../../crates/faberline-nlp-translate/tests/data/ja_zh/clock_boundary_lock_dev_v1.json) uses schema `faberline.nlp.ja_zh.clock_boundary_lock_dev.v1`, version `1.0.0`, SHA-256 `6bc9fc2dd0560ae9305a4a183a727fe80d07a310bdd65a250255a674378742d3`. The B26 reference-drafting policy and independent wording review concern [issue #133](https://github.com/faberline/nlp/issues/133); full-bank adoption remains pending.

The bank contains 34 rows: 24 new sentence intents, six glossary-option derivatives and four markup derivatives. B10 aliases B06. B25 reuses CLOCK-C07 and CLOCK-C08. L01 is the external unlocked CLOCK-C04 baseline. These aliases are not new rows.

Static counts: 28 new raw source strings; 29 raw source strings including C04; 34 distinct source-plus-complete-options units; 35 including C04 with default options. Comparison covers 225 permitted DEV rows. The six glossary derivatives reuse the exact C04 Japanese with different typed options. B08/B09 deliberately become equal under NFKC. No held-out or original challenge rows were read.

## Matrix

- **CB-B01**: 4 AM whole. Contrast: CB-B02. Changed: same event and period; hour 4 to 5. Limit: boundary AM early versus morning; natural references separately reviewed
- **CB-B02**: 5 AM whole. Contrast: CB-B01. Changed: same event and period; hour 4 to 5. Limit: 4/5 policy boundary; no exhaustive hour claim
- **CB-B03**: 4 AM half. Contrast: CB-B01. Changed: same hour, AM and event; add half. Limit: whole versus half at boundary
- **CB-B04**: 5 AM half. Contrast: CB-B02. Changed: same hour, AM and event; add half. Limit: whole versus half at boundary
- **CB-B05**: 4 PM whole. Contrast: CB-B01. Changed: same hour/event, change AM to PM. Limit: same numeral, explicit period
- **CB-B06**: 5 PM whole in natural event-time marked に; serves also as original B10. Contrast: CB-B11. Changed: B11 omits に, moves the clock to the sentence opening and adds a comma.. Limit: The particle, phrase position and punctuation change together; this is a close control, not a strict minimal pair.
- **CB-B07**: 5 PM half kanji numeral. Contrast: CB-B08. Changed: same event/time; numeral script only. Limit: non-2/9 hour and half
- **CB-B08**: 5 PM half ASCII numeral. Contrast: CB-B07. Changed: same event/time; numeral script only. Limit: token shape remains unknown
- **CB-B09**: 5 PM half fullwidth numeral. Contrast: CB-B07. Changed: same event/time; numeral script only. Limit: token shape remains unknown
- **CB-B11**: 5 PM whole in same natural event without に. Contrast: CB-B06. Changed: B11 omits に, moves the clock to the sentence opening and adds a comma.. Limit: The particle, phrase position and punctuation change together; this is a close control, not a strict minimal pair.
- **CB-B12**: 5 PM whole with alternate event predicate. Contrast: CB-B06. Changed: alternate event and predicate. Limit: Event and predicate change together; not a strict minimal pair.
- **CB-B13**: 5 PM whole in a natural alternate position. Contrast: CB-B06. Changed: move the clock phrase to the opening and change the meeting marker from は to が. Limit: both change together; this is not a strict minimal pair.
- **CB-B14**: 00:00 at the beginning of day D+1; both denote the same boundary between D and D+1. Contrast: CB-B15. Changed: 00:00 associated with day D+1 versus 24:00 associated with day D. Limit: The calendar date wording changes necessarily; require complete reference preserving each association. Data guard only; no calendar conversion or date-rule implementation implied.
- **CB-B15**: 24:00 at the end of day D; both denote the same boundary between D and D+1. Contrast: CB-B14. Changed: 00:00 associated with day D+1 versus 24:00 associated with day D. Limit: The calendar date wording changes necessarily; require complete reference preserving each association. Data guard only; no calendar conversion or date-rule implementation implied.
- **CB-B16**: noon with the explicit 昼 cue and unambiguous 12-hour clock. Contrast: CB-B17. Changed: noon cue versus midnight cue and period change. Limit: retain explicit day boundary and clock value
- **CB-B17**: Explicit midnight cue and 0-hour notation, without English AM/PM mapping; preserve any stated date association.. Contrast: CB-B16. Changed: noon vs midnight; cue and numeric form differ. Limit: no arbitrary mapping of 12 AM/PM
- **CB-B18**: 午前12時 denotes noon at the whole hour; preserve original period and number as applicable while rendering intended boundary accurately. Contrast: CB-B21. Changed: Exact period or numeral changes among four distinct forms. Limit: Conventional meaning follows NAOJ; source notes historic legal-definition qualification for 午後0時. Date association and the event context must be preserved, not invented.
- **CB-B19**: 午後12時 denotes midnight at the whole hour; preserve original period and number as applicable while rendering intended boundary accurately. Contrast: CB-B20. Changed: Exact period or numeral changes among four distinct forms. Limit: Conventional meaning follows NAOJ; source notes historic legal-definition qualification for 午後0時. Date association and the event context must be preserved, not invented.
- **CB-B20**: 午前0時 denotes midnight at the whole hour; preserve original period and number as applicable while rendering intended boundary accurately. Contrast: CB-B19. Changed: Exact period or numeral changes among four distinct forms. Limit: Conventional meaning follows NAOJ; source notes historic legal-definition qualification for 午後0時. Date association and the event context must be preserved, not invented.
- **CB-B21**: 午後0時 denotes noon at the whole hour; preserve original period and number as applicable while rendering intended boundary accurately. Contrast: CB-B18. Changed: Exact period or numeral changes among four distinct forms. Limit: Conventional meaning follows NAOJ; source notes historic legal-definition qualification for 午後0時. Date association and the event context must be preserved, not invented.
- **CB-B22**: unqualified 24-hour representative 13:00. Contrast: CB-B23. Changed: hour 13 vs 23, time of day differs. Limit: event context must make notation clear; preserve exact value
- **CB-B23**: unqualified 24-hour representative 23:00. Contrast: CB-B22. Changed: hour 13 vs 23. Limit: same event; high 24-hour value is not grounds for fictional preservation policy
- **CB-B24**: non-clock class-period term near numeric material. Contrast: CLOCK-C04. Both contain a number and 時; their events, referents and constructions differ. Limit: distant negative guard, not a strict minimal pair.
- **CB-B26**: A clock expression using 午前12 with a half-hour, without a clarifying context. Contrast: CB-B18. Changed: whole hour versus attached half-hour. The complete reference explicitly expresses uncertainty between both source-supported readings of one meeting time. Do not force one reading, imply an intentional scheduling choice, or invent context. The policy concerns issue #133; full-bank adoption remains pending.
- **CB-L02**: CLOCK-C04 with whole 午後二時半 glossary lock. Contrast: CLOCK-C04. Changed: same JP, new glossary surface and target. Limit: whole span replacement expected exactly once; lock integrity separately from full translation
- **CB-L03**: CLOCK-C04 with 二 glossary lock. Contrast: CLOCK-C04. Changed: same JP, hour substring glossary option. Limit: partial lock; ASCII boundary not involved; alignment reachability unverified
- **CB-L04**: CLOCK-C04 with 時 glossary lock. Contrast: CLOCK-C04. Changed: same JP, 時 substring glossary option. Limit: partial unit lock; no claim tokenizer reaches a separate atom
- **CB-L05**: CLOCK-C04 with 半 glossary lock. Contrast: CLOCK-C04. Changed: same JP, half substring glossary option. Limit: Partial 半 glossary lock: exact lock integrity and span obligations apply; recognizer behavior and whether any whole-clock rewrite is admissible remain unknown until source-specific independent assessment and permitted runtime observation.
- **CB-L06**: CLOCK-C04 with 午後 glossary lock. Contrast: CLOCK-C04. Changed: same JP, period substring glossary option. Limit: partial period lock; no inferred whole-clock match
- **CB-L07**: CLOCK-C04 with 会議は glossary lock directly adjacent to unlocked clock. Contrast: CLOCK-C04. Changed: same JP; different glossary option. Limit: adjacent non-clock lock, verify placement and intact clock separately
- **CB-L08**: CLOCK-C04 with whole 午後二時半 enclosed in backticks. Contrast: CLOCK-C04. Changed: same original clause, add opaque markup around whole clock. Limit: backticks alter source and visibility; detect actual inline opaque and preserve exact bytes
- **CB-L09**: CLOCK-C04 with 二 enclosed in backticks. Contrast: CLOCK-C04. Changed: same clause, opaque markup around partial hour. Limit: markup alters source; detection and resulting token shape unknown
- **CB-L10**: CLOCK-C04 with 午後 enclosed in backticks. Contrast: CLOCK-C04. Changed: same clause, opaque markup around period. Limit: markup alters source; language segmentation and supported path unknown
- **CB-L11**: CLOCK-C04 with 会議は enclosed in backticks immediately before unlocked clock. Contrast: CLOCK-C04. Changed: same clause plus markup around adjacent non-clock fragment. Limit: source adjacency includes delimiter; check target order and clock separately

## Meaning and time policy

The 4／5 boundary uses provisional Taiwan wording; it is not a universal language law. Whole-hour 午前12 and 午後0 denote noon; 午前0 and 午後12 denote midnight. The historical definition of 午後0 has a qualification noted by the [National Astronomical Observatory of Japan](https://www.nao.ac.jp/faq/a0401.html). B26 alone explicitly preserves the uncertainty of the source’s 12 plus half-hour expression. Its independently reviewed reference-drafting policy permits both source-supported interpretations for one meeting. It does not assert an intentional schedule choice or permit invented context. Full-bank review and adoption remain pending.

The 24:00 case belongs to May 1; the 00:00 case belongs to May 2. They denote the same boundary. These cases protect date association; they do not request calendar-rule repair. The no-に comparison also changes position and punctuation. It is not a strict minimal pair.

## Options and protected spans

Each row embeds the complete translation options, backend, model-directory choice and typed glossary. The glossary schema is `nlp.translate.glossary.v1`. A glossary target is a supplied literal replacement. Inline backticks express a candidate opaque span; callers cannot inject an arbitrary opaque span through these options. Matcher and detector behavior, atom shape, restoration and alignment remain unobserved.

Each protected fixture records half-open Unicode scalar source and target spans and separately derived UTF-8 byte offsets. It requires exact protected content, one corresponding occurrence per source occurrence, position and alignment. A restorer can repair a missing placeholder; this does not prove exact-once behavior. Protection and complete natural sentence quality are separate checks. Unlocked CLOCK-C04 is the contrast.

## Sources and review

- `nlevels_dev_v1.json`: SHA-256 `a24d91502b3d361eec1ff2bf97381d0accbf27d7516ba2188e77b3b4e96695c0`, 20 DEV rows.
- `honorific_san_dev_v1.json`: SHA-256 `280da5014f9069994bce1f23e8cdc10c3cad3200450278c9b4e3618231134229`, 12 DEV rows.
- `nlevels_expansion_dev_v1.json`: SHA-256 `22d0753fd384ae5f07a118c6417c53b213cc66c1637a8369b8c0315e69ab4fcc`, 50 DEV rows.
- `clock_time_dev_v1.json`: SHA-256 `01db148efcc90ca4fe025a65c6e39fb712ce6643e5c27d3018cbc6e691e1bf69`, 13 DEV rows.
- `issue86-standing-dev-export-059c10a.json`: SHA-256 `31776ea83cc2046ef84f905f5ba2bdfee7261aaa68124cc3405cdad05441555d`, 130 DEV rows.

The safe standing DEV export retains source head `059c10a260ec75d92ee0867c1649ddc0a0db10be`; it is not the original challenge file. The JSON binds code and adopted-bank hashes and names each reused control. Source-kara is only a partial start-marker fixture, not a full clock-quality pass.

New Japanese, references and glossary targets are first_party_mit and AI-assisted. References are provisional Taiwan Traditional Chinese, not human-certified gold. Labels are approximate; no official JLPT or N5 acceptance is claimed. This bank is not registered with an evaluator. The existing report remains 82 rows. All seven applicable exact-head CI checks and an independent formal review are required before adoption.
