# Annotation Guide (Layers 4–6)

This document defines the gold annotation standard for Layers 4–6 of `nlp.doc.v1` (Issue #28):
Universal POS and Universal Dependencies v2 syntax trees (Layer 4), predicate–argument semantic
frames (Layer 5), and entity coreference clusters plus pro-drop zero mentions (Layer 6). It
accompanies the frozen gold evaluation set in `tests/data/gold_layers4_6_v1.jsonl`, the offline
validator in `tests/doc_gold_eval.rs`, and the standalone scorer in `training/eval_gold.py`.

## Corpus & Provenance

- **File**: `tests/data/gold_layers4_6_v1.jsonl`
- **SHA-256**: `cd78c88a1ebfbbcaf7113df0b5ef104402560a177d26d4fea5987850c8e3f149`
- **Provenance & licence**: All 300 items (**450 sentences**, **4,457 tokens**, **621 frames**,
  **144 coref clusters**, **24 zero mentions**) are self-written,
  CC0-clean, topic-neutral evaluation texts authored specifically for Issue #28. No Universal
  Dependencies treebank text, no share-alike/non-commercial source, and no external copyrighted
  excerpt is used. Every item is strictly disjoint from `training/probe/probe_texts.jsonl` and
  `tests/data/pos_gold.json`.
- **Topics**: Neutral everyday and technical subjects only — weather, cooking, lab measurements,
  software/algorithms, public transit, books, gardening, and daily errands.
- **Balanced 3 × 5 design** (100 items / 150 sentences per genre; 50 single-sentence + 50
  two-sentence items per genre):

| genre | `zh-TW` (`lang="zh"`) | `zh-CN` (`lang="zh"`) | `en` (`lang="en"`) | `mixed-TW` (`lang="mixed"`) | `mixed-CN` (`lang="mixed"`) | items | sentences |
|---|---:|---:|---:|---:|---:|---:|---:|
| `novel` (`g001`–`g100`) | 20 (10×1s, 10×2s) | 20 (10×1s, 10×2s) | 40 (20×1s, 20×2s) | 10 (5×1s, 5×2s) | 10 (5×1s, 5×2s) | 100 | 150 |
| `paper` (`g101`–`g200`) | 20 (10×1s, 10×2s) | 20 (10×1s, 10×2s) | 40 (20×1s, 20×2s) | 10 (5×1s, 5×2s) | 10 (5×1s, 5×2s) | 100 | 150 |
| `casual` (`g201`–`g300`) | 20 (10×1s, 10×2s) | 20 (10×1s, 10×2s) | 40 (20×1s, 20×2s) | 10 (5×1s, 5×2s) | 10 (5×1s, 5×2s) | 100 | 150 |
| **Total** | **60** (90 sents) | **60** (90 sents) | **120** (180 sents) | **30** (45 sents) | **30** (45 sents) | **300** | **450** |

### Regional Mandarin Variants (`zh-TW` vs `zh-CN`)

`nlp.doc.v1` emits `lang: "zh"` or `lang: "mixed"` at Layer 1, while the gold set additionally
records `variant` (`zh-TW`, `zh-CN`, `en`, `mixed-TW`, `mixed-CN`) so evaluation can check parity
between Taiwan Mandarin and Mainland Mandarin:

- **`zh-TW` / `mixed-TW`**: Authored natively in Traditional characters with Taiwan lexical usage
  (`捷運`, `公車`, `便利商店`, `隨身碟`, `行動電源`, `螢幕`, `軟體`, `演算法`, `資料庫`, `最佳化`, `番茄`,
  `馬鈴薯`, `優格`).
- **`zh-CN` / `mixed-CN`**: Authored natively in Simplified characters with Mainland lexical usage
  (`地铁`, `公交`, `快递驿站`, `胡同`, `充电宝`, `显示器`, `软件`, `算法`, `数据集`, `优化`, `西红柿`,
  `土豆`, `酸奶`), never produced by mechanical character conversion from `zh-TW`.

## JSONL Schema & Layer 1–2 Alignment

Each line of `tests/data/gold_layers4_6_v1.jsonl` is one UTF-8 JSON object with 13 fields:

```json
{
  "id": "g026",
  "genre": "novel",
  "lang": "zh",
  "variant": "zh-TW",
  "text": "周小姐借了三本歷史書。她在星期天之前讀完了。",
  "tokens": ["周", "小姐", "借", "了", "三本", "歷史", "書", "。", "她", "在", "星期天", "之前", "讀", "完", "了", "。"],
  "sentences": [[0, 8], [8, 16]],
  "upos": ["PROPN", "NOUN", "VERB", "PART", "NUM", "NOUN", "NOUN", "PUNCT", "PRON", "ADP", "NOUN", "ADP", "VERB", "VERB", "PART", "PUNCT"],
  "heads": [2, 3, 0, 3, 7, 7, 3, 3, 13, 11, 13, 11, 0, 13, 13, 13],
  "deprels": ["compound", "nsubj", "root", "aux", "nummod", "compound", "obj", "punct", "nsubj", "case", "obl", "case", "root", "compound", "aux", "punct"],
  "frames": [
    {"predicate": 3, "args": [{"role": "ARG0", "head": 2}, {"role": "ARG1", "head": 7}]},
    {"predicate": 13, "args": [{"role": "ARG0", "head": 9}, {"role": "ARGM", "head": 11}]}
  ],
  "coref": [[[1, 2], [9, 9]]],
  "zero_mentions": [{"zero_at": 13, "role": "obj", "antecedent": [5, 7]}]
}
```

### Indexing & Alignment Rules

1. **Exact Layer 1–2 alignment**: `tokens`, `sentences`, and `lang` must match
   `analyze_document(text, &Options::default())` identically.
   - `sentences` uses **0-based half-open** token index pairs `[start, end)` covering `0..tokens.len()`
     contiguously, matching `training/probe/annotations.jsonl` and `tests/training_data_probe.rs`.
   - `heads`, `frames[].predicate`, `frames[].args[].head`, `coref` mention spans `[start, end]`,
     and `zero_mentions` use **1-based** token indices (`1..=tokens.len()`), with `0` reserved for
     the sentence root in `heads`.
2. **Avoiding cross-constituent `jieba` merges**: Every Chinese and mixed sentence was checked
   against `nlp_doc` before annotation so no item suffers from HMM subject+verb merges like `p17`
   (`阿芳訂` in `docs/training-data.md`). Where `jieba` splits a multi-syllabic word across two
   tokens (`麵|糰`, `陽|台上`, `遙|控器`, `充电|宝`, `上|傳給`), the left piece attaches to the right
   head piece via `compound`. Where `jieba` groups a quantifier+classifier (`三本`, `這雙`, `这张`,
   `兩組`), the combined token receives the tag and relation of its phrasal function (`NUM` +
   `nummod` or `DET` + `det`).
3. **Layer-2 closed-class UPOS preservation**: Whenever Layer 2 (`src/doc/pos.rs`) assigns
   `Some(upos)` to a token (punctuation, numbers, inline code/citations `X`, English closed-class
   words, and Traditional Chinese closed-class entries), the gold `upos` array matches that tag.
   Tokens left `None` by Layer 2 — open-class words, ambiguous words (`can`, `to`, `在`, `會`,
   `到`), and Simplified Chinese function words (`我们`, `他们`, `因为`, `虽然`, `通过`, `已经`, `这`,
   `个`, `着`, `里`, `从`) — receive their contextually resolved UD v2 tag in Layer 4.

## Layer 4: UPOS & Dependency Syntax

### Universal POS Tags (17 UD v2 Tags)

| UPOS | Category | Examples (`zh-TW` / `zh-CN` / `en`) |
|---|---|---|
| `ADJ` | Adjective (attributive & predicative) | `舊`, `沉重`, `乾淨`, `显着`, `清爽`, `warm`, `stable` |
| `ADP` | Adposition (preposition, postposition, disposal/passive marker, English verb particle) | `在`, `從`/`从`, `把`, `將`/`将`, `被`, `上`, `裡`/`里`, `in`, `from`, `up` |
| `ADV` | Adverb (degree, temporal, manner, negation) | `很`, `已經`/`已经`, `慢慢`, `不`, `沒`/`没`, `quickly`, `never` |
| `AUX` | Auxiliary verb & copula | `會`/`会`, `能`, `可以`, `要`, `是`, `will`, `can`, `is`, `was` |
| `CCONJ` | Coordinating conjunction | `和`, `與`/`与`, `跟`, `或`, `或者`, `and`, `or`, `but` |
| `DET` | Determiner (articles & demonstratives) | `這`/`这`, `那`, `這座`, `这张`, `每`, `the`, `a`, `this`, `each` |
| `INTJ` | Interjection | `哇`, `唉`, `Wow`, `Oops`, `Please` |
| `NOUN` | Common noun & split classifier noun | `木匠`, `捷運`, `地铁`, `隻`, `個`/`个`, `款`, `ferry`, `dataset` |
| `NUM` | Numeral & merged numeral-classifier | `三`, `五組`, `百分之十`, `twelve`, `ninety`, `0.05` |
| `PART` | Grammatical particle (structural, aspectual, modal, English `not`/`'s`/infinitival `to`) | `的`, `得`, `了`, `著`/`着`, `過`, `嗎`, `吧`, `not`, `n't`, `to` |
| `PRON` | Pronoun (personal, possessive, interrogative) | `我`, `他`, `我們`/`我们`, `自己`, `哪裡`, `I`, `they`, `my`, `their` |
| `PROPN` | Proper noun | `林`, `小芸`, `老王`, `Python`, `USB-C`, `Clara`, `July` |
| `PUNCT` | Punctuation | `。`, `，`, `？`, `！`, `.`, `,`, `?`, `!` |
| `SCONJ` | Subordinating conjunction | `因為`/`因为`, `雖然`/`虽然`, `如果`, `要是`, `若`, `because`, `once`, `while` |
| `SYM` | Symbol | `%`, `°C`, `+`, `=` |
| `VERB` | Lexical verb | `買`/`买`, `掃進`, `分析`, `收敛`, `measure`, `optimize` |
| `X` | Opaque / non-linguistic span | `` `batch_size` ``, `` `max_len` ``, `[12]` |

### Dependency Relations (37 Base UD v2 Labels, No Subtypes)

Language-specific subtypes (`nsubj:pass`, `aux:pass`, `obl:tmod`, `compound:prt`, `acl:relcl`,
`nmod:poss`, `flat:name`) are **not** used; every subtype collapses to its base UD v2 label:

| Label | Syntactic role | Head → Dependent direction |
|---|---|---|
| `acl` | Clausal modifier of noun (relative / participial clause) | `茶` → `剛泡` (`剛泡好的茶`), `data` → `collected` |
| `advcl` | Adverbial clause modifier (temporal, conditional, manner, `得` degree) | `發車` → `修好`, `煮` → `好` (`煮得剛剛好`) |
| `advmod` | Adverbial modifier | `搖晃` → `慢慢`, `converged` → `rapidly`, `have` → `n't` |
| `amod` | Adjectival modifier of noun | `皮箱` → `舊`, `鐵門` → `沉重` (`沉重的鐵門`) |
| `appos` | Appositional modifier (including inline variable/citation tags) | `參數` → `` `batch_size` ``, `定理` → `[12]` |
| `aux` | Auxiliary verb, aspectual particle (`了`/`著`/`過`), or agentless passive `被` | `籠罩` → `了`, `吹倒` → `被`, `will` → `arrive` (head `arrive` → `will`) |
| `case` | Preposition, postposition, possessive `的`/`'s`, disposal `把`/`將`, or long-passive `被` | `木屑` → `把`, `木橋` → `在` & `上`, `木匠` → `的` |
| `cc` | Coordinating conjunction | `2nd conjunct` → `和` / `and` |
| `ccomp` | Finite/clausal complement without obligatory subject control | `顯示` → `降低`, `覺得` → `痠`, `said` → `would` |
| `clf` | Split numeral/demonstrative classifier | `鴿子` → `隻` (`一 隻 灰 鴿子`), `鼠标` → `款` (`这 款 无线 鼠标`) |
| `compound` | Noun–noun compound, split word piece, Chinese V–V resultative/directional, English verb particle | `燈塔`, `讀` → `完`, `pick` → `up` |
| `conj` | Second or later conjunct (attached to first conjunct) | `門窗` → `座位號`, `tomatoes` → `basil` |
| `cop` | Copula (`是`, `be`) attached to nonverbal predicate | `手藝人` → `是`, `stable` → `was` |
| `csubj` | Clausal subject | `順手` → `修` (`用來修 photo 特別順手`) |
| `dep` | Unspecified dependency (last resort only) | — |
| `det` | Determiner | `山谷` → `整座`, `ferry` → `the` |
| `discourse` | Sentence-final modal particle (`了`, `嗎`, `吧`, `喔`) or interjection (`哇`, `Wow`) | `香` → `哇` & `喔` |
| `dislocated` | Fronted topic with resumptive/core slot in clause | `刷` → `滤网` (`水池的滤网我刚才已经刷干净了`) |
| `expl` | Expletive (`there`, dummy `it`) | `is` / predicate → `there` |
| `fixed` | Fixed multiword grammaticalized expression | `instead` → `of`, `due` → `to`, `as` → `well` |
| `flat` | Flat multiword proper name / date expression | `Clara` → `Lin` |
| `goeswith` | Orthographically split single token | — |
| `iobj` | Core recipient indirect object in ditransitive transfer | `傳給` → `我`, `lent` → `me`, `pass` → `me` |
| `list` | List element | — |
| `mark` | Clausal subordinator (`因為`, `如果`, `once`, infinitival `to`, relative/adjectival `的`, resultative `得`) | `下雨` → `因為`, `剛泡` → `的`, `borrow` → `to` |
| `nmod` | Nominal / pronominal modifier of noun (possessor, Genitive `的`, English PP on noun) | `木屑` → `桌上`, `bike` → `my`, `crates` → `apples` |
| `nsubj` | Nominal subject (active, passive, unaccusative, copular) | `掃進` → `木匠`, `吹倒` → `花盆`, `stable` → `weight` |
| `nummod` | Numeric modifier of noun | `書` → `三本`, `crates` → `twelve` |
| `obj` | Direct object (or existential postverbal theme) | `籠罩` → `山谷`, `放著` → `皮箱`, `carried` → `crates` |
| `obl` | Oblique nominal (PP adjunct, temporal noun, `把`/`將` preposed patient, long-passive `被` agent) | `掃進` → `木屑` (`把`) & `籃裡`, `left` → `dawn` |
| `orphan` | Elliptical conjunct remnant | — |
| `parataxis` | Comma-joined independent clause within one sentence | `到` → `去` (`捷運還要兩站才到，我們去旁邊便利商店買一瓶水。`) |
| `punct` | Punctuation attached to clause/phrase head | `root` → `。` / `.` |
| `reparandum` | Disfluency repair | — |
| `root` | Sentence root (`head == 0`) | `0` → `掃進` |
| `vocative` | Addressee nominal | — |
| `xcomp` | Open clausal/predicative complement (subject/object control or resultative state) | `開始` → `蒸`, `wants` → `borrow`, `變得` → `濕滑`, `wipe` → `dry` |

### Tree Structural Invariants

Every sentence range `[b, e)` in `sentences` forms an independent dependency tree over
1-based token indices `(b + 1)..=e`:
1. `deprels[i] == "root"` if and only if `heads[i] == 0`.
2. Exactly **one** token in `(b + 1)..=e` has `heads[i] == 0`.
3. Every non-root token in `(b + 1)..=e` has `(b + 1) <= heads[i] <= e` and `heads[i] != i + 1`
   (no cross-sentence attachments and no self-loops).
4. Walking `heads` from every token reaches `0` in at most `e - b` steps (acyclic tree).

### Construction-Specific Resolution Rules

1. **Disposal `把` / `將` (`將` / `将`)**:
   - `把` / `將` is `ADP` with `deprel = "case"` attached to the preposed patient noun.
   - The patient noun attaches to the matrix verb as `obl` (`老 木匠 把 木屑[obl] 掃進[root] 竹 籃裡`).
   - At Layer 5, that `把`/`將`-marked `obl` receives semantic role **`ARG1`** (patient).
2. **Passive `被`**:
   - **Short passive (no overt agent)**: `空 花盆[nsubj] 被[ADP, aux] 吹倒[root] 了` — `被` attaches
     directly to the verb as `aux`; the patient subject is `nsubj` (Layer 5: `ARG1`).
   - **Long passive (with overt agent)**: `包裹[nsubj] 被[ADP, case] 管理員[obl] 收進[root] 儲物櫃` —
     `被` attaches to the agent noun as `case`, and the agent noun attaches to the verb as `obl`
     (Layer 5: `ARG0`), while the surface subject remains `nsubj` (`ARG1`).
3. **Chinese V–V / V–Adj resultatives & English phrasal verbs (`compound`)**:
   - Single-token compounds (`掃進`, `吹倒`, `修好`, `滤除`) remain `VERB`.
   - Split resultative or directional complements (`讀[root] + 完[compound]`, `拿[root] + 起[compound]`,
     `刷[root] + 干净[ADJ, compound]`, `彈[root] + 出[compound]`) and English verb particles
     (`pick[root] + up[ADP, compound]`, `turn[root] + down[ADP, compound]`) attach to the main verb
     as `compound`.
4. **Three uses of `的` (`PART`)**:
   - **Verbal relative clause**: `[剛泡 好] 的 茶` → `剛泡` (`VERB`) attaches to `茶` as `acl`; `的`
     attaches to `剛泡` as `mark`.
   - **Modified / disyllabic adjectival modifier**: `沉重 的 鐵門`, `显着 的 相关性` → adjective
     attaches to the noun as `amod`; `的` attaches to the adjective as `mark`. Bare monosyllabic
     adjectives (`舊 皮箱`) attach directly as `amod`.
   - **Nominal / pronominal modifier**: `桌上 的 木屑`, `我 的 雨傘` → modifier noun/pronoun attaches
     to the head noun as `nmod`; `的` attaches to the modifier noun/pronoun as `case`.
   - **Headless `的`-nominalization**: `買 一台 新[ADJ, obj] 的[PART, mark]` (`g291`) → the predicate
     or adjective stands in the NP slot (`obj`/`nsubj`) and `的` attaches to it as `mark`.
5. **Split classifiers (`clf`)**:
   - When a numeral or demonstrative and its classifier are separate tokens (`一 隻 灰 鴿子`,
     `五 組 樣本`, `这 款 无线 鼠标`), **both** `nummod`/`det` **and** `clf` attach directly to the
     **head noun** (`鴿子`, `樣本`, `鼠标`), matching `p19` in `training/probe/annotations.jsonl`
     and keeping `nummod`/`det` leaf modifiers.
6. ** Circumfixal locatives (`在 ... 上/裡/中/下`)**:
   - Both the preposition (`在`, `從`, `沿著`) and the localizer postposition (`上`, `裡`/`里`, `中`,
     `下`, `之前`) attach to the landmark noun as `case`, and the landmark noun attaches to the
     predicate as `obl` (or to a head noun as `nmod` when followed by `的`, as in `屋簷 下 的 枯葉`).
7. **Copula `是` / `be`**:
   - Nonverbal predicates (`手藝人`, `stable`, `clear`, `there`) head their clause (`root`); `是` /
     `is` / `was` (`AUX`) attaches to the predicate as `cop`.
8. **Sentence-initial connectors, fronted adverbials, and interjections**:
   - Commas after sentence-initial connectors (`接著，`, `隨後，`, `然而，`, `其中，`, `However,`,
     `Thankfully,`, `Even so,`) and interjections (`哇，`, `唉，`, `Wow,`, `Oops,`) attach to the
     **clause predicate (`root`)**, matching `p11`, `p19`, and `p21` in `training/probe/annotations.jsonl`.
9. **Mixed Chinese–English code-switching & opaque `X` spans**:
   - Embedded English nouns/acronyms (`app`, `router`, `SSD`, `vlog`, `learning rate`, `p-value`)
     participate directly in the matrix dependency tree (`compound` for multi-word English technical
     terms like `learning[compound] rate[obj]`, `obj`/`nsubj`/`nmod` for NP heads).
   - Inline backtick code spans (`` `batch_size` ``, `` `max_len` ``) and bracketed citations (`[12]`)
     keep Layer-2 UPOS `X` and attach syntactically (`appos`, `obj`, `obl`, or `compound`).

## Layer 5: Semantic Frames & Issue #14 Role Mapping

Each verbal or predicative-adjectival frame in `frames` has:
- `predicate`: 1-based token index of the predicate (`VERB` or predicative `ADJ`; nominal copular
  predicates like `是 手藝人` emit no verbal frame).
- `args`: list of `{"role": "ARG0" | "ARG1" | "ARGM", "head": <1-based token idx>}`, where `head`
  lies in the **same sentence** as `predicate` (`head != predicate`) and points to the syntactic
  head of the argument constituent (never to `把`, `被`, `在`, or a preposition).

### Mapping from Issue #14 Semantic Roles

Issue #14 specifies 7 semantic roles (`agent`, `patient`, `target`, `time`, `place`, `manner`,
`quantity`). Layer 5 maps them deterministically onto the 3 coarse frame roles:

| Issue #14 role | Layer 5 role | Syntactic realization & examples |
|---|---|---|
| `agent` | **`ARG0`** | Volitional actor, causer, or experiencer (`老木匠` in `掃進`, `管理員` in `被管理員收進`, `小貓` in `聽見`) |
| `patient` | **`ARG1`** | Undergoer, affected/created object, disposal `把`/`將` object, passive/unaccusative/adjectival theme (`木屑` in `把木屑掃進`, `花盆` in `花盆被吹倒`, `茶` in `茶好香`, `皮箱` in `角落放著皮箱`) |
| `target` | **`ARG1`** | Core recipient `iobj` of transfer (`傳給我[ARG1]`, `lent me[ARG1]`, `pass me[ARG1]`), resultative product/state object (`剪成 vlog[ARG1]`, `轉為 向量[ARG1]`, `改為 雙線程[ARG1]`), or direct goal object (`進入 系統[ARG1]`, `抵達 車站[ARG1]`) |
| `time` | **`ARGM`** | Temporal point, duration, or frequency (`昨天`, `清晨五點`, `兩小時`, `只需 幾毫秒[ARGM]`) |
| `place` | **`ARGM`** | Static location, source, path, or postverbal locative destination (`在木橋上`, `從煙囪裡`, `放到 驛站[ARGM]`, `發到 群裡[ARGM]`) |
| `manner` | **`ARGM`** | Manner adverbial, degree modifier, instrument/means PP, or causal/conditional `advcl` (`慢慢`, `格外`, `通過 鏈接`, `因為連日下雨`) |
| `quantity` | **`ARGM`** | Numeric margin, extent, or threshold (`增加 百分之五`, `超過 百分之十`, `exceeded ninety percent[ARGM]`, ``超過 `max_len`[ARGM] ``) |

### Linking Summary Across Diatheses

- **Transitive active**: `nsubj -> ARG0`, `obj -> ARG1`.
- **Disposal (`把` / `將`)**: `nsubj -> ARG0`, `把/將`-marked `obl -> ARG1`; if the verb also takes
  a resultative product `obj` (`把 短片 剪成 一支 vlog`), **both** the input patient (`短片`) and
  the output target (`vlog`) are labeled `ARG1`.
- **Ditransitive transfer (`iobj` + `obj`)**: `nsubj -> ARG0`, theme `obj -> ARG1`, recipient
  `iobj -> ARG1` (consistent with `p08` in `docs/training-data.md`).
- **Passive (`被`)**: patient `nsubj -> ARG1`; overt `被`-agent `obl -> ARG0` (if present).
- **Unaccusative & adjectival predicates** (`搖晃`, `升起`, `收斂`, `香`, `stable`, `clear`):
  theme `nsubj -> ARG1`.
- **Serial / `conj` / `xcomp` predicates**: shared overt subjects within the same sentence are
  linked explicitly as `ARG0`/`ARG1` on each controlled predicate (`老人[1] 拿起[2] 鑰匙 打開[4] 鐵門`
  gives `ARG0=1` on both `2` and `4`).

## Layer 6: Coreference Clusters & Pro-Drop Zero Mentions

### Overt Coreference (`coref`)

- **Format**: `Vec<Vec<[usize; 2]>>` — a list of entity clusters, each containing **at least two**
  overt mentions represented as 1-based inclusive `[start, end]` token spans (identical to
  `training/probe/annotations.jsonl` and `tests/training_data_probe.rs`). Singletons are omitted.
- **Maximal constituent span rule**: Every mention `[start, end]` covers the maximal NP projection
  of its syntactic head inside a single sentence (for example `[4, 7]` `一條 藍色 的 毛巾` paired with
  `[11, 11]` `毛巾` in `g022`; `[1, 3]` `The express bus` paired with `[14, 14]` `it` in `g266`).
  Every span is verified to have a single syntactic head inside `[start, end]` whose external head
  lies outside `[start, end]`.
- **Nested mentions**: Possessive pronouns inside larger NPs may participate in separate clusters
  (`[[1, 1], [4, 4]]` for `Ben` / `his` alongside `[[4, 6], [14, 14]]` for `his camping lantern` /
  `it` in `g267`).

### Pro-Drop & Omitted Arguments (`zero_mentions`)

Mandarin (`zh-TW` and `zh-CN`) frequently omits anaphoric subjects and objects across sentence or
parataxis boundaries (noted in `docs/training-data.md` for `讀完了`). Because omitted arguments
occupy no surface token span, inserting synthetic indices into `coref` would break span-based
compatibility with `tests/training_data_probe.rs` and standard CoNLL-2012 mention scoring. Instead,
each item carries an explicit `zero_mentions` array:

```json
"zero_mentions": [
  {"zero_at": 13, "role": "obj", "antecedent": [5, 7]}
]
```

- `zero_at`: 1-based token index of the governing predicate (`VERB` or `ADJ`) where the omitted
  argument is licensed.
- `role`: grammatical function of the omitted argument, restricted to `"nsubj"` or `"obj"`. No
  overt dependent of `zero_at` in the same sentence may already carry `deprel == role`.
- `antecedent`: 1-based inclusive `[start, end]` token span of the closest antecedent NP in the
  item (obeying the same single-sentence constituent rule as `coref` mentions).
- Note on intra-sentence vs cross-sentence control: within a single clause chain (`conj`, `xcomp`),
  subject sharing is already captured directly by Layer 5 `frames`; `zero_mentions` marks genuine
  discourse/parataxis pro-drop where the predicate's clause has no local subject or object token
  (e.g. `g026` `讀完了`, `g031` `開始裁剪`, `g212` `順路拿回來`, `g229` `還給我`, `g298` `發到群裡了`).

## Second-Pass Audit & Resolved Disagreements

After initial annotation of all 300 items, a programmatic and manual second-pass linguistic audit
checked every dependency arc, UPOS tag, semantic role, coreference span, and zero mention against
the probe conventions (`p01`–`p30`). Six systematic disagreements were identified and resolved:

1. **Split classifier (`clf`) attachment (`HeadN` vs `NUM`/`DET`)** — *Items `g121`, `g122`, `g124`,
   `g127`, `g288`, `g296`*:
   - *Disagreement*: Initial drafts of `paper` and `casual` attached split classifiers (`組`, `種`,
     `個`, `款`) to the preceding numeral or determiner (`五 <- 組`, `这 <- 款`), whereas `novel`
     (`g025` `一 隻 灰 鴿子`) and probe item `p19` (`第三 -> 模型 (nummod)`, `個 -> 模型 (clf)`)
     attached both `nummod`/`det` and `clf` to the head noun.
   - *Resolution*: Standardized on **head-noun attachment** (`clf -> HeadN`) across all 300 items so
     `nummod` and `det` remain pure leaves.
2. **Commas after sentence-initial connectors & interjections** — *Items `g122`, `g124`, `g126`,
   `g162`, `g170`, `g204`, `g218`, `g246`, `g258`, `g264`, `g266`*:
   - *Disagreement*: Several items attached the comma after sentence-initial `因此，`, `然而，`,
     `Nevertheless,`, `Thankfully,`, `哇，`, or `Wow,` to the initial adverb/conjunction/interjection
     token (`head = 1`), while `novel` (`g024` `接著，`, `g031` `隨後，`) and probe items `p11`, `p19`,
     and `p21` (`However,`) attached the comma to the **clause `root`**.
   - *Resolution*: Standardized all sentence-initial connector and interjection commas to attach to
     the **clause predicate (`root`)**.
3. **Ditransitive recipient role (`iobj`: `ARG1` vs `ARGM`)** — *Items `g250`, `g267`, `g292`,
   `g295` (vs `g229` and probe `p08`)*:
   - *Disagreement*: Initial `casual` annotations labeled `iobj` recipients (`pass me`, `lent me`,
     `傳給我`, `上傳給大家`) as `ARGM`, whereas `g229` (`還給我`) and `docs/training-data.md` (`p08`
     `texted me`) map recipients (`target` in Issue #14) to `ARG1`.
   - *Resolution*: Mapped all core recipients (`iobj`) to **`ARG1`**. (In `g295`, `大家` after `上傳給`
     was also refined from `obj` to `iobj`, since the theme `影片` is already fronted via `把`.)
4. **Postverbal locative destinations (`放到`/`發到` + place) vs direct objects** — *Items `g212`,
   `g298`*:
   - *Disagreement*: `g298` (`通過鏈接發到群裡了`) initially labeled `群裡` as `obj` while also
     recording an omitted theme `zero_mentions: [{"zero_at": 18, "role": "obj", "antecedent": [10, 11]}]`
     (`一期 vlog`), creating a duplicate-`obj` collision.
   - *Resolution*: Postverbal spatial destinations after `放到`/`發到`/`收到` (`驛站` in `g212`,
     `群裡` in `g298`, `哪裡` in `g203`) are syntactic **`obl`** and semantic **`ARGM`** (`place`),
     leaving the `obj` slot cleanly available for the overt or pro-dropped theme.
5. **Resultative target nominals (`V成` / `改為` / `轉為`) vs numeric extent `obj`s** — *Items `g034`,
   `g090`, `g103`, `g127`, `g130`, `g132`, `g193`, `g295`, `g298` vs `g119`, `g180`, `g186`, `g195`*:
   - *Disagreement*: Resultative target nominals (`捆成 一束`, `捏成 圓餅`, `轉為 稠密向量`, `剪成 vlog`)
     were inconsistently split between `ARG1` and `ARGM`. Meanwhile, transitive verbs (`經過` `g105`,
     `呈現出` `g117`/`g138`, `接近` `g136`, `適合` `g195`, `來自` `g199`) also had `obj` marked `ARGM`.
   - *Resolution*: All resultative target/product nominals and two-argument transitive `obj`s map to
     **`ARG1`** (`target`/`patient`). Only genuine numeric extent, margin, threshold, or duration
     `obj`s (`超過 百分之十` in `g119`, `exceeded ninety percent` / `reached sixty percent` in `g180`,
     ``超過 `max_len` `` in `g186`, and `只需 幾毫秒` in `g195`) carry **`ARGM`** (`quantity`/`time`),
     matching probe item `p20` (`增加 百分之五 -> ARGM`).
6. **English gerund-introducing `from` (`prevent ... from entering`)** — *Item `g169`*:
   - *Disagreement*: `from` before `entering` in `g169` (`prevents invalid records from entering the
     table`) was initially tagged `SCONJ`, conflicting with Layer 2 (`pos.rs`), which deterministically
     tags `from` as `ADP`.
   - *Resolution*: Kept Layer-2 `upos = "ADP"` with `deprel = "mark"` attached to `entering`
     (`advcl`), matching standard UD v2 English EWT treatment of prepositional clausal markers.

## Evaluation Scorer (`training/eval_gold.py`)

`training/eval_gold.py` scores predictions against `tests/data/gold_layers4_6_v1.jsonl` using only
the Python 3 standard library:

```bash
# Score a prediction JSONL file (annotation format or nlp.doc.v1 Document JSONL):
python3 training/eval_gold.py --gold tests/data/gold_layers4_6_v1.jsonl --pred <pred.jsonl>

# Run built-in mathematical & perturbation self-tests:
python3 training/eval_gold.py --self-test
```

### Reported Metrics & Slices

- **`uas`** / **`las`**: Token-level Unlabeled / Labeled Attachment Score across all gold tokens
  (plus `uas_no_punct`, `las_no_punct`, and `syntax_coverage`; undecided `null` predictions count as
  misses against the full gold denominator).
- **`upos_acc`**: Token-level 17-tag UPOS accuracy (plus `upos_coverage`).
- **`frame_f1`**: Micro-averaged exact-tuple `(item_id, predicate_tok, role, arg_head_tok)` F1
  (plus `frame_p`, `frame_r`).
- **`coref_conll_f1`**: Exact CoNLL-2012 average F1 $(\text{MUC} + B^3 + \text{CEAF}_e)/3$ using
  Kuhn–Munkres (Hungarian) maximum-weight bipartite matching for $\phi_4$ $\text{CEAF}_e$ (plus
  individual `muc_f1`, `b3_f1`, `ceafe_f1`, and `zero_mention_f1`).
- **Breakdowns**: `overall`, `by_lang` (`zh`, `en`, `mixed`), `by_variant` (`zh-TW`, `zh-CN`, `en`,
  `mixed-TW`, `mixed-CN`), `by_genre` (`novel`, `paper`, `casual`), and `by_cell` (`genre/lang`).
