# Training data

This record says who labels the training text for layers 4 to 6 of `nlp.doc.v1` (issue #26)
and which texts are labeled. It downloads no corpus and trains nothing. No model file and no
corpus go into git.

## Decision

**Labeler: Claude Sonnet** (a Claude Code sub-session, model `claude-sonnet-5-5`), not a local
model.

- User decision, 2026-11-11, to nlp-service-desk (quoted from the task brief of this issue):
  "我們應該沒有要用LLM 就算是本地小模型也沒有要用？" and "你派一個sonnet去標注就行".
  The product runs no LLM at run time, and no local LLM labels the data either. Claude Sonnet
  sub-sessions do the labeling.
- This **replaces** the rule of 2026-11-10 in issue #26 and #27: "local open-weight LLM", "local
  only", "no hosted model, no upload of text". Labeling now means text goes to Claude. The other
  rules stay: no Universal Dependencies treebank text (CC BY-SA; the label scheme is allowed),
  text only from public domain, CC0, CC BY or self-written sources, no model file and no corpus
  in git, content neutral.
- Why the four local models are not used: on the 30-item probe none gave a usable dependency
  tree (table below). Sonnet gave 30 of 30 structurally valid annotations on the same probe
  (section Probe).
- Sources chosen per cell:

| genre | zh-TW / zh | en |
|---|---|---|
| novel | Project Gutenberg Chinese texts (authors died in or before 1975) | Project Gutenberg English fiction (authors died in or before 1975) |
| paper | DGBAS civil-servant reports (Taiwan official documents) | Europe PMC open-access articles, CC BY |
| casual | moztw `cc0-sentences` (CC0) | Mozilla Common Voice `sentence-collector.txt` (CC0) |

The ToS question for training on Claude output is open and goes to the user (section "To
confirm with the user"). It does not stop this issue.

## Labeling model

The labeler is Claude Sonnet. These local models were probed on 2026-11-10 by the previous
session (30 items, 10 with gold labels, greedy decoding, `mlx_lm` 0.32.2, 4-bit, Apple M1
Max 64 GB, two runs each). The data below is copied from that session's `score.json` and
`meta.json` files. Licence, `output_training` and disk size were not recorded in those files,
so the table does not show them; they do not matter, because no local model is used.

| model | params | labeler | valid_json | valid_tree | uas | las | frame_f1 | coref_f1 | sentences_per_hour | peak_memory_gb |
|---|---|---|---|---|---|---|---|---|---|---|
| mlx-community/Qwen3-14B-4bit | 14B | no | 0.956 | 0.000 | 0.000 | 0.000 | 0.689 | 0.684 | 160.7 | 8.77 |
| mlx-community/Qwen3.5-9B-MLX-4bit | 9B | no | 0.900 | 0.033 | 0.000 | 0.000 | 0.432 | 0.529 | 150.4 | 5.81 |
| mlx-community/Ministral-3-14B-Instruct-2512-4bit | 14B | no | 0.667 | 0.000 | 0.000 | 0.000 | 0.549 | 0.333 | 56.1 | 8.44 |
| mlx-community/gemma-4-e4b-it-4bit | about 4B effective | no | 0.633 | 0.000 | 0.000 | 0.000 | 0.107 | 0.000 | 713.0 | 4.77 |
| Claude Sonnet (`claude-sonnet-5-5`, this session) | not published | **yes** | 1.000 | 1.000 | not scored | not scored | not scored | not scored | not measured | not applicable |

- `valid_tree`: share of the 30 items whose dependency answer is a valid tree over the given
  tokens. Local models: from the previous session's scorer (its tokens were hand-cut). Sonnet:
  from `tests/training_data_probe.rs` on the tokens of the repo's layer 1 and 2 (see Probe).
- `uas`, `las`, `frame_f1`, `coref_f1` for the local models are on the 10 gold items of the
  previous session. Sonnet's annotations have no gold set to compare with (gold is issue #28),
  so they are `not scored`. Sonnet's 1.000 is validity, not accuracy.
- Local models only differ in speed and in frames/coreference; the tree failure is common to all
  four, and only one of 30 trees (Qwen3.5-9B, valid_tree 0.033) was valid.
- Not probed and not used: Llama 3.x and Gemma 3 (licence terms on trained derivatives), and
  models above about 10 GB at 4-bit. This was the previous session's judgement; the licence
  texts were not re-read in this session.

## Text sources

Allowed: public domain (PD), CC0, CC BY, a government licence compatible with CC BY, and
self-written text. Not allowed: share-alike, non-commercial, no-derivatives, GPL-like or an
unclear licence. Universal Dependencies treebank text is out; only its label scheme is used.
Sizes come from each source's own metadata as read by the previous session on 2026-11-10; this
session re-checked only the two CC0 casual sources (the `moztw/cc0-sentences` README says "zh-tw
sentences release under CC0"; the Common Voice `server/data/en/` folder has a `LICENSE` file and
`sentence-collector.txt`). Nothing was downloaded except metadata.

| genre | lang | source | licence | licence text | fetch | size | attribution |
|---|---|---|---|---|---|---|---|
| novel | en | Project Gutenberg, English fiction whose authors died in or before 1975 | PD: US, published before 1931; Taiwan, Copyright Act Art. 30 (life + 50 years) | https://www.gutenberg.org/policy/license.html | filter `pg_catalog.csv` (author birth and death years are in the `Authors` field), then fetch plain text through the robot harvest endpoint (`/robot/harvest?filetypes[]=txt&langs[]=en`, wait 2 s between requests) or a mirror; strip the Gutenberg header and footer | 13,058 texts (of 24,039 English fiction texts, catalog of 2026-11-10) | none required once the Project Gutenberg name and licence text are removed |
| novel | zh | Project Gutenberg, Chinese texts whose authors died in or before 1975 | PD: US, published before 1931; Taiwan, Copyright Act Art. 30 | https://www.gutenberg.org/policy/license.html | same as above with `langs[]=zh` | 208 texts (of 435 Chinese texts) | none required, as above |
| paper | en | Europe PMC open-access articles with licence CC BY | CC BY 4.0 (some articles CC BY 2.0 or 3.0; keep the per-article licence) | https://creativecommons.org/licenses/by/4.0/legalcode | REST search `OPEN_ACCESS:y AND LICENSE:"cc by"`, then `/{PMCID}/fullTextXML` | 5,394,763 articles (search hit count) | per article: authors, title, journal, DOI, licence, and a note that the text was changed |
| paper | zh | Directorate-General of Budget, Accounting and Statistics (DGBAS) overseas trip reports by civil servants | Taiwan official documents: not a subject of copyright (Copyright Act Art. 9); the index is under the Open Government Data License v1, compatible with CC BY 4.0 | https://data.gov.tw/license | download the index XML of https://data.gov.tw/dataset/10979, then each report PDF it links | 201 reports (index of 2026-05-03) | the index: the OGDL v1 attribution statement; skip quoted third-party material |
| paper | zh | Europe PMC open-access Chinese articles with licence CC BY | CC BY 4.0 | https://creativecommons.org/licenses/by/4.0/legalcode | REST search `LANG:"chi" AND OPEN_ACCESS:y AND LICENSE:"cc by"` | 4,191 articles; Simplified Chinese, so a backup only | per article, as for English |
| casual | zh | moztw `cc0-sentences`, zh-TW folder, without `wikipedia.txt` and `gpt-4.txt` | CC0 1.0 | https://creativecommons.org/publicdomain/zero/1.0/legalcode | `git clone https://github.com/moztw/cc0-sentences` | 18 files, 591,021 bytes, 26,393 sentences (README, 2025-06-28) | none required |
| casual | zh | Tatoeba, Mandarin sentences | CC BY 2.0 FR (a CC0 subset is marked per sentence) | https://creativecommons.org/licenses/by/2.0/fr/ | weekly export from https://tatoeba.org/en/downloads | 89,398 sentences, mixed Traditional and Simplified script | per sentence: contributor, sentence id and link |
| casual | en | Mozilla Common Voice, `server/data/en/sentence-collector.txt` | CC0 1.0 (the `LICENSE` file in that folder; the repository code is MPL-2.0) | https://creativecommons.org/publicdomain/zero/1.0/legalcode | download the file from https://github.com/common-voice/common-voice | 2,803,252 bytes | none required |
| casual | en | Tatoeba, English sentences | CC BY 2.0 FR (a CC0 subset is marked per sentence) | https://creativecommons.org/licenses/by/2.0/fr/ | weekly export from https://tatoeba.org/en/downloads | 2,049,378 sentences | per sentence: contributor, sentence id and link |
| any | zh | Wikipedia | rejected: CC BY-SA (share-alike) | https://creativecommons.org/licenses/by-sa/4.0/legalcode | none | none | none |
| novel | zh | Wikisource | rejected: page licences mix PD and CC BY-SA | https://creativecommons.org/licenses/by-sa/4.0/legalcode | none | none | none |
| any | any | Universal Dependencies treebanks | rejected: treebank text excluded by rule (the label scheme is used) | https://universaldependencies.org/ | none | none | none |
| paper | zh | Government Research Bulletin (GRB) research reports | rejected: rights stay with the agency or the researcher; licence unclear | https://www.grb.gov.tw/ | none | none | none |
| paper | zh | Taiwan open-access journals in DOAJ | rejected: the zh-TW journals found use CC BY-NC, CC BY-ND or CC BY-SA | https://doaj.org/ | none | none | none |
| casual | en | Common Voice `wiki.en*.txt` | rejected: sentences taken from Wikipedia; licence unclear | https://github.com/common-voice/common-voice | none | none | none |
| casual | zh | moztw `cc0-sentences` `gpt-4.txt` and `wikipedia.txt` | rejected: model output of another service and Wikipedia text; licence unclear | https://github.com/moztw/cc0-sentences | none | none | none |
| any | zh | Academia Sinica Balanced Corpus | rejected: paid licence | https://asbc.iis.sinica.edu.tw/ | none | none | none |
| casual | zh | PTT bulletin board posts | rejected: posts belong to their authors, no open licence | https://www.ptt.cc/ | none | none | none |

Notes:

- A PD claim uses two rules. US: the work was published before 1931. Taiwan: the author died in
  or before 1975 (Copyright Act Art. 30, life + 50 years). The fetch script must keep only texts
  that pass both rules.
- Gutenberg Chinese texts are mostly classical or 1920s vernacular, not modern Taiwan usage. Modern
  zh-TW fiction and papers are thin in open sources; self-written text can fill this gap, but
  text that Claude writes falls under the same ToS question as labels (see below), so no
  synthetic row is listed.
- The DGBAS reports can quote third-party text; the fetch script must drop quoted blocks.
- `moztw/cc0-sentences` also holds chat lines from a community archive. The sentences have no
  topic filter here; neutrality is checked on the sample that is committed, not by a content rule.

## Probe

- 30 short items, self-written and neutral (the previous session wrote them): 10 per genre
  (novel, paper, casual), per genre 4 zh, 4 en and 2 mixed. Texts: `training/probe/probe_texts.jsonl`.
  They are small enough and neutral, so they are committed.
- Tokens come from the repo, not from the labeler: `training/probe/dump_tokens.py` runs the
  `nlp_doc` binary (layers 1 and 2) over each text. The test also calls `analyze_document` and
  requires the annotation tokens to equal its tokens.
- Annotator: Claude Sonnet (`claude-sonnet-5-5`), this session, by hand, on 2026-11-11. No API,
  no other model, no text left the session. File: `training/probe/annotations.jsonl`. Per item:
  `tokens`, `sentences` (token ranges), `heads` (1-based token ids, 0 = root of that sentence),
  `deprels` (the 37 UD relation names, no subtypes), `frames` (`predicate` and `args` with role
  ARG0, ARG1 or ARGM and a head token), `coref` (clusters of 1-based inclusive `[start, end]`
  token ranges).
- Check: `tests/training_data_probe.rs` (`cargo test --test training_data_probe`). It tests valid
  JSON, token equality with the repo, one root per sentence, a head in the same sentence, no
  self-head, no cycle, labels in the scheme, `root` only on the root, frame and mention indexes
  in range. Negative cases (a cycle, two roots, `nsubj:pass`, an out-of-range head, an unknown
  role, a mention out of range) are rejected.
- **Result: 30 of 30 items valid, 0 invalid** (first run, no repair needed). Counts: 30 items,
  30 dependency trees valid, 0 invalid trees, 0 invalid frames, 0 invalid coreference clusters.
- sha256 of `training/probe/probe_texts.jsonl`: 96e8a4e31daf36cb90024b619cf85fe1686f835f366b92b9a83b7742de5f6d0f
- sha256 of `training/probe/annotations.jsonl`: 0bd254a454a019fbedf3f8e0fe0533f0d3e6144ea9c6d720eef97edb85b4c3f5

Findings from the annotation work:

- Validity is not accuracy. The test cannot tell a wrong but well-formed head. Accuracy needs
  the gold set of #28.
- Layer 2 zh tokens are sometimes not words: `他馬上`, `它會`, `裡了`, `記得剛`, `剛還`, `錯|誤率`,
  `後|者`, `只用|於` merge or split words. The annotation gives such a token the relation of its
  content part, so a model trained on these tokens learns the layer 2 errors. Issue #27 and the
  model issues should decide whether the training tokens are the layer 2 tokens as they are or
  corrected tokens.
- Roles are ARG0, ARG1 and ARGM. Issue #14 lists agent, patient, target, time, place, manner
  and quantity; #28's annotation guide must map them.
- Omitted subjects and objects (for example the missing object of `讀完了`) cannot be pointed at
  with token ids. Layer 6 needs a way to mark a zero mention.

## To confirm with the user

Question: does training our own model on Claude's output match Anthropic's terms? This section
quotes what I read on 2026-11-11. It is not a legal conclusion. The work continues.

Which terms apply depends on how the user uses Claude; I do not know that.

1. Consumer Terms of Service, <https://www.anthropic.com/legal/consumer-terms>, effective
   October 8, 2025. Section 3 introduces a list with "You may not access or use, or help another
   person to access or use, our Services in the following ways". Item 2 of that list reads: "To
   develop any products or services that compete with our Services, including to develop or
   train any artificial intelligence or machine learning algorithms or models or resell the
   Services." It has no exception text at its start. (The API-key exception sits in a different
   item, about automated access.) Section 4 assigns Anthropic's rights in Outputs to the user
   "subject to your compliance with our Terms".
2. Commercial Terms of Service, <https://www.anthropic.com/legal/commercial-terms>, effective
   Jun 17, 2025 (as shown on the page). Section D.4 (Use Restrictions) says Customer may not
   "access the Services to build a competing product or service, including to train competing AI
   models", and not support a third party doing so.

What I could not decide: whether a small syntax, frame and coreference model for a text library
is a "competing" AI model; whether the Claude Code subscription falls under 1 or 2; and whether
there is an agreement with Anthropic that changes these terms. The user, or whoever has the
contract, should answer. If the answer is no, the options are: label with a model whose licence
allows training on its output (the Apache-2.0 local models failed the tree task, so that needs
a larger model or a different method), or write the labels by rule and by the user's own hands
for a smaller gold set.

## Changes issue 27 needs

Issue #27 (silver-label pipeline) still says "local model" and "Local only". I did not edit it.
It needs these changes:

- Goal: "the local model chosen in the data decision issue" becomes "the labeler chosen in
  `docs/training-data.md`: Claude Sonnet sub-sessions".
- Change 1: "asks the local model for labels" becomes "a Claude Sonnet sub-session labels the
  tokens"; the script prepares batches of cut sentences (layers 1 and 2, `nlp_doc`) and
  validates the returned files; it does not call a model itself.
- Rules: "Local only: no network call during labeling. No text leaves the Mac." becomes "Text
  leaves the Mac only to the Claude Sonnet session; no other service; no API key in scripts".
- Change 3, agreement filter: "label each sentence twice with different prompts or seeds"
  becomes two independent Sonnet passes (fresh sessions, different prompt wording), because
  seeds do not exist.
- Change 4: the manifest records the model id `claude-sonnet-5-5`, the prompt version and the
  source sha256 (no seed). Resume still works through finished-batch files.
- Acceptance: "A dry run ... passes in CI without a model (recorded answers)" stays; the CI
  check is the validator pattern of `tests/training_data_probe.rs`. "local run" speed is not
  meaningful; report labeled, invalid and dropped counts per batch.
- Add the ToS answer from the section above as a precondition for a large run.

Issue #28 also says "first labels come from the local labeling model" and "a different local
model or prompt checks"; it needs the same change (Sonnet labels, a second Sonnet pass checks).

## Limits

- The probe has 30 items. Validity of 30 of 30 shows Sonnet can follow the scheme on this
  format; it does not show accuracy. Accuracy comes from the gold set of #28.
- The annotations are one annotator's choices (UD relations without subtypes; the `ba` marker
  as `case`, resultative verbs as `compound`). Other valid analyses exist.
- The local model numbers are from the previous session's offline run and could not be re-run
  here; the 30 items were re-tokenized by the repo rules, so the two probes are not identical
  in tokens.
- Source sizes come from metadata of 2026-11-10 and change over time.
