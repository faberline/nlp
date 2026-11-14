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
- Why the four local models are not used: on the 30-item probe none gave a valid dependency
  tree, even when rescored with the same rule as the Sonnet check (one root per sentence; table
  below). The one tree that passed the original scorer was a flat star over a two-sentence
  item. Sonnet gave 30 of 30 structurally valid annotations on the same probe (section Probe).
  The two probes did not use identical tokens (see Limits), and validity is not accuracy, so
  this is a statement about following the output format, not about label quality.
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

| model | params | labeler | valid_json | valid_tree (1 root per item, original scorer) | valid_tree (1 root per sentence, rescored) | uas | las | frame_f1 | coref_f1 | sentences_per_hour | peak_memory_gb |
|---|---|---|---|---|---|---|---|---|---|---|
| mlx-community/Qwen3-14B-4bit | 14B | no | 0.956 | 0.000 | 0.000 | 0.000 | 0.000 | 0.689 | 0.684 | 160.7 | 8.77 |
| mlx-community/Qwen3.5-9B-MLX-4bit | 9B | no | 0.900 | 0.033 | 0.000 | 0.000 | 0.000 | 0.432 | 0.529 | 150.4 | 5.81 |
| mlx-community/Ministral-3-14B-Instruct-2512-4bit | 14B | no | 0.667 | 0.000 | 0.000 | 0.000 | 0.000 | 0.549 | 0.333 | 56.1 | 8.44 |
| mlx-community/gemma-4-e4b-it-4bit | about 4B effective | no | 0.633 | 0.000 | 0.000 | 0.000 | 0.000 | 0.107 | 0.000 | 713.0 | 4.77 |
| Claude Sonnet (`claude-sonnet-5-5`, this session) | not published | **yes** | 1.000 | not applicable | 1.000 | not scored | not scored | not scored | not scored | not measured | not applicable |

- `valid_tree` has two bases and they must not be compared across columns.
  - *1 root per item* is the previous session's scorer (`tree_ok` in its `score.py`): the whole
    item must have exactly one root. 24 of the 30 items have two or more sentences, so a
    correct per-sentence analysis of those items fails this rule. The numbers 0.000 / 0.033 /
    0.000 / 0.000 are copied from its `score.json` and are on hand-cut tokens.
  - *1 root per sentence* is the rule of `tests/training_data_probe.rs` (Sonnet) and is also
    what the `nlp.doc.v1` layer 4 needs. I rescored the existing raw answers of the four local
    models (`run1.jsonl` and `run2.jsonl` of the previous session, not in git) with that rule,
    sentences cut at `。 . ! ? ！ ？` in the hand-cut tokens, all other checks unchanged (length,
    heads in range, no self-head, no cycle, UD labels, `root` iff head 0). Result: 0 of 30 for
    every model in both runs. The one tree that passed the old scorer (Qwen3.5-9B, item p01)
    attaches every token to token 1, and fails the new rule because p01 has two sentences. I did
    not re-run any local model. The rescoring script is not in git because the raw answers are
    not; the inputs are the previous session's files in worktree `nlp-off26`.
  - Failures of the rescored answers (run 1): most answers have the wrong number of roots per
    sentence; of those with the right root counts, the rest fail on a self-head, a root label
    that disagrees with head 0, or a label outside the scheme (`dobj`, `poss`, `neg`,
    `nmod:tmod`, `acomp`, `ROOT`). So the failure is not only the root-counting rule.
- `uas`, `las`, `frame_f1`, `coref_f1` for the local models are on the 10 gold items of the
  previous session. Sonnet's annotations have no gold set to compare with (gold is issue #28),
  so they are `not scored`. Sonnet's 1.000 is validity, not accuracy.
- Local models only differ in speed and in frames/coreference; the tree failure is common to all
  four.
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
| paper (weak fit) | zh | Directorate-General of Budget, Accounting and Statistics (DGBAS) overseas trip reports ("出國報告", dataset 行政院主計總處出國報告) by civil servants; official reports, not research papers | Taiwan official documents: not a subject of copyright (Copyright Act Art. 9); the index is under the Open Government Data License v1, compatible with CC BY 4.0 | https://data.gov.tw/license | download the XML of https://data.gov.tw/dataset/10979 (fields: link, summary, date, retention period, storage unit), then each linked PDF | the XML holds 201 records (read through a web fetch, not counted locally; about 175 link a PDF, 24 an HTML page, 1 a `.doc`); record dates 1989-06-01 to 2026-05-03; the dataset page shows a metadata update date of 2025-09-16 and "updated irregularly", so the XML may change; count unverified | the dataset: the OGDL v1 attribution statement (OGDL v1 clause 3.2 requires a visible attribution, otherwise the licence is void from the start); skip quoted third-party material |
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
  They are trip reports of civil servants (conference and study-visit reports), a formal
  Traditional Chinese register but not research papers, so they only fill the `paper / zh`
  cell weakly. No better open zh-TW paper source was found (see the rejected GRB and DOAJ rows).
  The claim "Art. 9 covers them" (official documents compiled for public duty) is my reading of
  the Copyright Act, not a legal conclusion.
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
  token ranges). An item may carry `flags` and a `note`; the only flag is `layer2_token_error`
  (p17, see Findings): its frame and coref must not be used as training data.
- Check: `tests/training_data_probe.rs` (`cargo test --test training_data_probe`). It tests valid
  JSON, token equality with the repo, one root per sentence, a head in the same sentence, no
  self-head, no cycle, labels in the scheme, `root` only on the root, frame and mention indexes
  in range, frame arguments in the sentence of their predicate, and known flags. It also compares the
  annotation's `sentences` ranges with the sentence numbers of `analyze_document`, and compares
  the sha256 of both data files with fixed values written in the test. Negative cases (a cycle,
  two roots, `nsubj:pass`, an out-of-range head, an unknown role, a mention out of range, an
  argument in another sentence, an unknown flag) are rejected, and each test asserts the exact
  error text, not only that an error happened.
- **Result: 30 of 30 items valid, 0 invalid** (first run, no repair needed). Counts: 30 items,
  30 dependency trees valid, 0 invalid trees, 0 invalid frames, 0 invalid coreference clusters.
- sha256 of `training/probe/probe_texts.jsonl`: 96e8a4e31daf36cb90024b619cf85fe1686f835f366b92b9a83b7742de5f6d0f
- sha256 of `training/probe/annotations.jsonl`: 7a7ddd82d63965c7a9565612758c9e529dbc3770a1ea3b7a4bfd48e0a3cb4f87

Findings from the annotation work:

- Validity is not accuracy. The test cannot tell a wrong but well-formed head. Accuracy needs
  the gold set of #28.
- Layer 2 zh tokens are sometimes not words: `他馬上`, `它會`, `裡了`, `記得剛`, `剛還`, `錯|誤率`,
  `後|者`, `只用|於`, `阿芳訂`, `我剛`, `很乾淨`, `新開`, `牛肉|麵` merge or split words
  (`好像` in p26 is also a doubtful token; the reviewer listed it, and I did not re-check it). The annotation gives such a token the relation of its
  content part, so a model trained on these tokens learns the layer 2 errors. Issue #27 and the
  model issues should decide whether the training tokens are the layer 2 tokens as they are or
  corrected tokens.
- p17 originally had `阿芳訂` merged as one token under legacy jieba-rs; after switching the
  default Chinese segmenter to the first-party lexicon segmenter (`阿芳` | `訂`), p17 was
  re-aligned with its `ARG0` frame and `coref` restored, and the `layer2_token_error` flag was
  removed (0 flagged items remain). Other items were likewise re-aligned to the first-party
  segmenter tokens.
- Review of the first version (a second Sonnet session, all 30 items read, 14 zh, 11 en and 6
  mixed checked sentence by sentence): head and label errors were found in none of the checked
  sentences. About 9% of the frames (5 of about 58 predicates) were doubtful, mostly as a side
  effect of the layer 2 tokens. That is an estimate of one reviewer, not a measured error rate.
  Fixed after the review: p05 had no frame for `stable` (added: ARG1 `weight`); p26 had a frame
  that made `週末` an ARGM of `轉涼` while the tree makes it an `nmod` of `天氣` (the frame now
  has only ARG1 `天氣`, consistent with the tree). Known and left: p18 has `實驗` as ARG0 of
  `招募` (the agent is the researchers, debatable); p08 has `me` of `texted me` as ARG1 (it is
  the recipient; the role set has no recipient role).
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
- The local model numbers are from the previous session's offline run and were not re-run
  here; the 30 items were re-tokenized by the repo rules, so the two probes are not identical
  in tokens. The rescoring of their answers (section Labeling model) uses their hand-cut
  tokens. The two valid_tree columns have different rules, and Sonnet's number is only on the
  second basis.
- Source sizes come from metadata of 2026-11-10 and change over time.

## Silver pipeline

Issue #27 implements the resumable two-pass silver-label pipeline under `training/silver/` without
embedding API keys or running network calls inside the scripts:

- `training/silver/prepare_batches.py` reads raw UTF-8 text or JSONL items, runs the local
  `nlp_doc` binary (layers 1 and 2) to attach exact `tokens`, `layer2_upos`, and `sentences`
  ranges, slices items into fixed-size `batches/batch_NNNN.input.jsonl` files, and writes
  `<work-dir>/manifest.json` (`schema: "nlp.silver.v1"`) recording `source_sha256`,
  `model_id` (`claude-sonnet-5-5`), `prompt_version` (`silver-v1`), `batch_size`, and per-batch
  status (`pending`, `partial`, `ready_to_merge`, `merged`). Re-running on an existing work
  directory checks `source_sha256`, `model_id`, `prompt_version`, and `batch_size` before
  continuing, preserving completed Pass-A, Pass-B, and merged batch files.
- `training/silver/validate_merge.py` validates independent Pass-A (`batch_NNNN.pass_a.jsonl`)
  and Pass-B (`batch_NNNN.pass_b.jsonl`) annotations against each prepared batch: valid JSON,
  exact Layer 1/2 token and sentence-range match, one root per sentence, no self-heads or cycles,
  the 37 UD v2 relation labels (no subtypes), `ARG0`/`ARG1`/`ARGM` frame arguments inside the
  predicate's sentence, coreference mention bounds, and known flags (`layer2_token_error`). It
  then applies the two-pass agreement filter (requiring Pass A and Pass B to agree on `heads`,
  `deprels`, flags, and canonicalized `frames` and `coref`, while restricting `layer2_token_error`
  items to `usable_layers: ["syntax"]`), records per-batch `kept`, `invalid`, and `dropped`
  counts in `manifest.json`, skips already-merged batches whose SHA-256 hashes match on resume,
  and writes the merged silver output when all batches are finished.
- `training/silver/sample_dryrun.jsonl` and `tests/silver_pipeline.rs` verify the validator,
  agreement filter (4 kept, 3 disagreed/dropped, 3 structurally invalid), and resumable manifest
  checks offline in `cargo test` without Python, network access, or a model.

## Syntax model training and dual export (`doc-syntax-v1`)

Issue #20 trains and dual-exports `doc-syntax-v1` (Layer-4 UD v2 dependency parser and contextual
UPOS upgrader) under `training/syntax/` using `training/requirements.lock.txt`:

- **Multilingual & multivariant coverage**: `training/syntax/corpus_gen.py` builds a neutral,
  `nlp_doc`-verified training and validation corpus in `training/.work/syntax/` (gitignored)
  covering authentic Taiwan Traditional Chinese (`zh-TW`), authentic Mainland Simplified Chinese
  (`zh-CN`), English (`en`), and both `mixed-TW` (`zh-TW/en`) and `mixed-CN` (`zh-CN/en`)
  code-switched sentences across `novel`, `paper`, and `casual` genres, strictly excluding
  `training/probe/`, `training/silver/sample_dryrun.jsonl`, and `tests/data/gold_layers4_6_v1.jsonl`.
- **Encoder & pruned subword vocabulary**: `training/syntax/train_syntax.py` initializes from
  `intfloat/multilingual-e5-small` (MIT licence), pruning the Unigram vocabulary to `32,768` pieces
  (`[PAD]=0, [UNK]=1, [CLS]=2, [SEP]=3`, single-character Traditional and Simplified Han pieces,
  punctuation/symbols, and top English subwords) with pre-tokenizer splitting that prevents
  subword-to-token collisions across both `jieba` and English tokenization.
- **Dual Core ML + ONNX export**: `training/syntax/export_dual.py` exports both `model.onnx`
  (opset 13, `ir_version = 8`, exactly one `Gather` node, fused `[1, 256, 384]` positional +
  token-type constant, internal pad-mask attention bias, packed `[1, 256, 421]` output) and
  single-file `model.mlmodel` (`neuralnetwork` float32) from identical float32 weights, checking
  cross-backend parity (`max |onnx - coreml| <= 1e-4`).
- **Manifest pinning**: `training/syntax/fetch_model.py` pins `doc-syntax-v1` hashes and byte
  counts in `models/manifest.json` while keeping `models/doc-syntax-v1/` binaries out of git.
