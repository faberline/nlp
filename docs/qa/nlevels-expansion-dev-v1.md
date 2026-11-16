# N1-N5 development expansion v1

This bank adds 50 original Japanese sentences. Each approximate level has ten sentences and five paired controls. Every row is development-visible and `heldout: false`.

## Authorship and review

Luna (`gpt-6-luna`) wrote the Japanese sentences and provisional Taiwan Traditional Chinese references with AI assistance. No external exam, textbook, game, dictionary or corpus sentence text was copied. The data is `first_party_mit`.

Before the original freeze, Root reviewed all 50 rows. A separate Luna reviewer checked the N5/N4 and N3 drafts. Terra checked the N2/N1 draft. The authors applied the review corrections. These were agent reviews. No human review, gold status or official JLPT difficulty is claimed.

Formal independent adoption is pending. An independent checker must check the current frozen data PR and its evidence on current main. The authors do not accept their own work. Later semantic rule PRs must wait for adoption before measurement with these rows.

## Coverage

| Approximate level | IDs | Paired controls |
|---|---|---|
| N5 | N5X-01–N5X-10 | Living and nonliving existence; clock times; object counters; adjective polarity; floor numbers |
| N4 | N4X-01–N4X-10 | Opposite routes; open and close requests; permission and prohibition; present and past actions; intention and desire |
| N3 | N3X-01–N3X-10 | Relative-clause roles; purpose and cause; lending and borrowing; active and passive voice; not-yet and completed arrival |
| N2 | N2-D01–N2-D10 | Concession and condition; obligation scope; hearsay and inference; person and time focus; appearance and stated facts |
| N1 | N1-D01–N1-D10 | Concession and evidence limits; role duties and prohibition; hearsay and formal inference; content and method focus; incomplete records and uncertain conclusions |

Each row names its paired control. Each pair stays within one approximate level. The semantic checks preserve participant roles, polarity, scope, time and modality. The unknown fields preserve omitted people, counts, causes and certainty.

Review corrections include the scope of yesterday in relative clauses, unspecified counts, repair actions versus successful results, intention versus forecast, reported claims versus confirmed facts, newly installed versus newly made objects, schedule versus date, and handover content versus transmission time.

## Use and limits

The current challenge runner does not load this JSON. This PR reports structure and provenance only. It reports no translation score or pass rate. It does not change the old challenge, runner, baseline or ratchet.

Ten sentences per level are limited probes. Passing them cannot establish full N1-N5 quality. Human calibration and broader review across vocabulary, contexts and sentence forms remain open. CLI and macOS/Windows package acceptance remain separate requirements.
