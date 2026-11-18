# Adopted Japanese to Taiwan Traditional Chinese DEV phenomenon map

This map assigns each of the 32 permitted, already adopted development-visible rows to one primary semantic group. It uses only the exact bank memberships and source text. It does not change either source bank, the evaluator, the report or a baseline.

Source revision: `2c89250b31dba56ac8f02cd55629bf6037d3b179`.

| Bank file | Rows | SHA-256 | Adoption |
|---|---:|---|---|
| `nlevels_dev_v1.json` | 20 | `a24d91502b3d361eec1ff2bf97381d0accbf27d7516ba2188e77b3b4e96695c0` | PR #71 |
| `honorific_san_dev_v1.json` | 12 | `280da5014f9069994bce1f23e8cdc10c3cad3200450278c9b4e3618231134229` | PR #73 |

The rows carry `development-visible` visibility and `heldout=false`. Their original approximate level labels and references remain unchanged. The references are provisional first-party MIT development references, not model gold.

## Primary groups

| Group | Rows | Members | Definition and limit |
|---|---:|---|---|
| `habitual_action_polarity` | 2 | L5-01, L5-02 | L5-01 and L5-02 contrast a habitual morning action with its negated form. Preserve the repeated-time meaning, subject and object; this pair is not a past-tense contrast. |
| `past_destination_polarity` | 2 | L5-03, L5-04 | L5-03 and L5-04 retain the same past trip to school while changing its polarity. The destination remains school in both cases. |
| `weather_cause_and_intention` | 2 | L4-01, L4-02 | L4-01 and L4-02 share ongoing rain as the reason for studying at home. L4-02 adds an explicit intention; the pair therefore does not isolate causality alone, and neither sentence identifies the actor. |
| `movement_origin_destination` | 2 | L4-03, L4-04 | L4-03 and L4-04 contrast walking to a station with walking from it, followed by meeting a friend. They do not state where the meeting occurs or prove that it happens during the walk. |
| `cause_concession_and_outcome` | 2 | L3-01, L3-02 | L3-01 and L3-02 contrast a train delay followed by failure to arrive in time with a delay followed by success. Relation and outcome both change, so these are not connective-only controls; the arriving person is unstated. |
| `gradual_change_vs_completion` | 2 | L3-03, L3-04 | L3-03 and L3-04 contrast coughs gradually improving with a completed recovery after taking medicine. Neither states who took the medicine. |
| `condition_vs_concession_and_necessity` | 2 | L2-01, L2-02 | L2-01 and L2-02 contrast a condition that avoids spending meeting time on detail checks with a concession where checks remain necessary. The relation and result both differ. |
| `standing_procedure_vs_intention` | 2 | L2-03, L2-04 | L2-03 and L2-04 contrast an established arrangement to notify people after confirmation with the speaker's intention to notify them. The Japanese does not identify the notifier in L2-03 or the speaker in L2-04. |
| `inference_rejection_vs_uncertainty` | 2 | L1-01, L1-02 | L1-01 rejects the inference that a delayed review means lower workload. L1-02 says lower workload is not guaranteed given the delay; neither says workload definitely rises. |
| `apparent_acknowledgment_vs_addition` | 2 | L1-03, L1-04 | L1-03 contrasts an appearance of acknowledging facts with omission of the key decision process. L1-04 states that facts are acknowledged and the process is also addressed. This is not a simple positive/negative pair. |
| `personal_name_honorific_gender_neutrality` | 7 | HS-01, HS-02, HS-03, HS-04, HS-05, HS-06, HS-12 | HS-01..06 and HS-12 use さん with personal names. The pairs check presence, time or completion while preserving the name and avoiding unsupported gender or spelling claims. HS-12 also contains a teacher role; it does not make teacher and name honorifics interchangeable. |
| `explicit_gender_and_teacher_title` | 2 | HS-07, HS-08 | HS-07 explicitly says the person is male; HS-08 explicitly describes a teacher. Preserve stated information and do not infer gender from 先生. These rows are contrasting contexts, not matched versions of one sentence. |
| `non_name_san_boundaries` | 3 | HS-09, HS-10, HS-11 | HS-09 preserves the kinship meaning in お母さん; HS-10 preserves the clerk role in 店員さん; HS-11 treats さんま as a fish name, not a name suffix. They cover distinct lexical and role boundaries, not one shared morphology. |

Each row appears in exactly one group. Related controls remain together when they share a useful source-level axis. The definitions call out where a pair also changes a second feature, leaves an actor or location unstated, or represents a different lexical form.

## Use and limits

This map fixes group membership before viewing translation outputs. It is a classification of these 32 source contexts. It does not certify the approximate JLPT labels, set a numeric gate, prove complete N5 coverage, or accept translation quality.

The existing adopted development report remains report-only and keeps its existing bank membership and policy. This map does not change the challenge set, baseline, evaluator, engine or lexicon. The usual independent review and exact-head checks still apply to any repository change.
