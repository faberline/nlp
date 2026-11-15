# Japanese to Taiwan Traditional Chinese N1-N5 development bank v1

This bank has 20 original Japanese sentences. It has four items for each approximate level. Every item is development-visible. None is held out.

The bank supplements the existing challenge data and baseline. It does not change either one. It cannot lower the old baseline.

## Authorship and references

Luna (gpt-6-luna), independent QA author, composed the original Japanese sentences with AI assistance. No JLPT exam, textbook, game, or other external sentence text was copied.

The Taiwan Traditional Chinese references and allowed variants are provisional, AI-assisted drafts. They are not model gold. Formal adoption requires independent Codex and Antigravity agent review; those reviews are not complete until recorded. Human calibration remains open. The data is marked `first_party_mit`.

The level labels are sentence-level QA approximations guided by the [official JLPT English summary](https://www.jlpt.jp/e/about/levelsummary.html) and [official Japanese summary](https://www.jlpt.jp/about/levelsummary.html). The JLPT describes broad reading and listening ability. These short written items cannot certify JLPT level.

## Coverage

| Approximate level | IDs | Paired controls |
|---|---|---|
| N5 | L5-01–L5-04 | Habitual positive/negative; past positive/negative |
| N4 | L4-01–L4-04 | Ongoing cause and intention; movement toward/from a place |
| N3 | L3-01–L3-04 | Cause versus concession; gradual change versus result |
| N2 | L2-01–L2-04 | Conditional outcome versus concession; established procedure versus intention |
| N1 | L1-01–L1-04 | Rejected inference versus uncertainty; apparent acknowledgment versus additive disclosure |

Semantic checks focus on participants and omitted subjects, movement direction, negation and tense, causal and concessive links, aspect, necessity and intention, contrastive scope, idiomatic readings, and register. The checks preserve uncertainty when Japanese leaves a participant or exact meeting location unstated.

## Adoption and use

Formal adoption requires a separate data-only pull request before later semantic rule changes are judged with this bank. Root and Antigravity must review it independently. The bank author does not approve their own bank.

Twenty items are only a small smoke set. Passing them is not evidence that all of N1-N5 works well. Human calibration remains open. Broader evidence should include stratified human review across contexts and phenomena, preserve the old baseline and ratchet, and include separate product/package acceptance.

## Open calibration questions

- Are the sentence-level N1-N5 assignments appropriate? Human calibration remains open.
- Are the listed zh-Hant variants natural and equivalent in Taiwan usage, including register and omitted subjects?
- Do any paired items permit more than one materially different reading without added context?
- Is the mix of vocabulary, syntax, discourse scope, and sentence length broad enough for a useful regression signal? It is not intended to represent a full level.
