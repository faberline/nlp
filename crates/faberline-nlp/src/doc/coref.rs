//! Layer 6 of `nlp.doc.v1`: Multi-Sieve Entity Coreference (`coref`) and Pro-Drop Zero Mention
//! Resolution (`zero_mentions`) across `zh`, `en`, `ja`, `ko`, and `mixed`.
//!
//! Pure-Rust, deterministic, zero-model CPU implementation. See `docs/annotation-guide.md`
//! ("Layer 6: Coreference Clusters & Pro-Drop Zero Mentions").

use crate::Span;
use crate::doc::t2s::to_simplified_char;
use crate::doc::{Document, EntityKind, Lang, SemanticClass, Upos};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorefMention {
    /// 0-based inclusive start token index into `Document.tokens`.
    pub start_token: usize,
    /// 0-based inclusive end token index into `Document.tokens`.
    pub end_token: usize,
    /// 0-based syntactic head token index inside `start_token..=end_token`.
    pub head_token: usize,
    /// 0-based sentence index into `Document.segmentation.sentences`.
    pub sentence: usize,
    /// Exact character span `[tokens[start_token].span.start_char, tokens[end_token].span.end_char)`.
    pub span: Span,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ZeroRole {
    Nsubj,
    Obj,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ZeroMention {
    /// 0-based predicate token index into `Document.tokens` where the omitted argument is licensed.
    pub zero_at: usize,
    /// 0-based sentence index of `zero_at`.
    pub sentence: usize,
    /// Surface span of `tokens[zero_at].span`.
    pub span: Span,
    pub role: ZeroRole,
    pub antecedent: CorefMention,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Number {
    Singular,
    Plural,
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Animacy {
    HumanMale,
    HumanFemale,
    HumanUnknown,
    Animal,
    Inanimate,
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Person {
    First,
    Second,
    Third,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PronounKind {
    None,
    Personal3rd,
    Personal1st,
    Personal2nd,
    Reflexive,
    DiscourseFormer,
    DiscourseLatter,
    DiscourseLoc,
    DiscoursePartitive,
}

#[derive(Clone, Debug)]
struct MentionInfo {
    mention: CorefMention,
    canon: String,
    head_canon: String,
    upos: Option<Upos>,
    deprel: String,
    number: Number,
    animacy: Animacy,
    person: Person,
    pronoun: PronounKind,
    is_definite: bool,
    is_time_or_qty: bool,
    is_proper: bool,
    salience: u8,
    clause_head: Option<usize>,
}

struct SyntaxGraph<'a> {
    heads: Vec<Option<usize>>,
    deprels: Vec<&'a str>,
    children: Vec<Vec<usize>>,
    surfaces: Vec<String>,
    canons: Vec<String>,
}

impl<'a> SyntaxGraph<'a> {
    fn build(doc: &'a Document, text: &str) -> Self {
        let n = doc.tokens.len();
        let chars: Vec<char> = text.chars().collect();
        let mut heads = vec![None; n];
        let mut deprels = vec!["dep"; n];
        let mut children = vec![Vec::new(); n];

        for syn in &doc.syntax {
            if syn.token < n {
                heads[syn.token] = syn.head;
                deprels[syn.token] = syn.deprel.as_str();
                if let Some(h) = syn.head
                    && h < n
                    && h != syn.token
                {
                    children[h].push(syn.token);
                }
            }
        }
        for ch in &mut children {
            ch.sort_unstable();
            ch.dedup();
        }

        let mut surfaces = Vec::with_capacity(n);
        let mut canons = Vec::with_capacity(n);
        for tok in &doc.tokens {
            let s: String =
                if tok.span.end_char <= chars.len() && tok.span.start_char <= tok.span.end_char {
                    chars[tok.span.start_char..tok.span.end_char]
                        .iter()
                        .collect()
                } else {
                    String::new()
                };
            let c = canonicalize_token(&s, tok.canonical_variant.as_deref(), tok.lemma.as_deref());
            surfaces.push(s);
            canons.push(c);
        }

        Self {
            heads,
            deprels,
            children,
            surfaces,
            canons,
        }
    }
}

fn canonicalize_str(s: &str) -> String {
    s.chars()
        .map(to_simplified_char)
        .collect::<String>()
        .to_lowercase()
}

fn canonicalize_token(surface: &str, variant: Option<&str>, lemma: Option<&str>) -> String {
    if let Some(v) = variant {
        return canonicalize_str(v);
    }
    if let Some(l) = lemma
        && surface.is_ascii()
    {
        return l.to_lowercase();
    }
    canonicalize_str(surface)
}

/// Annotate `doc.coref` and `doc.zero_mentions` using `doc.syntax`, `doc.tokens`, `doc.entities`,
/// and `doc.segmentation`.
pub fn annotate(doc: &mut Document, text: &str) {
    doc.coref.clear();
    doc.zero_mentions.clear();

    if doc.syntax.is_empty() || doc.tokens.is_empty() {
        return;
    }

    let (clusters, zero_mentions) = {
        let graph = SyntaxGraph::build(doc, text);
        let mentions = extract_mentions(doc, &graph);
        let clusters = resolve_coref(doc, &graph, &mentions);
        let zero_mentions = resolve_zero_mentions(doc, &graph, &mentions);
        (clusters, zero_mentions)
    };
    doc.coref = clusters;
    doc.zero_mentions = zero_mentions;
}

fn is_discourse_anaphor_surface(canon: &str) -> bool {
    matches!(canon, "前者" | "后者" | "其中" | "上面" | "这里" | "那里")
}

fn is_candidate_head(doc: &Document, graph: &SyntaxGraph<'_>, idx: usize) -> bool {
    let tok = &doc.tokens[idx];
    let rel = graph.deprels[idx];
    if rel == "expl" || rel == "punct" {
        return false;
    }

    let upos = tok.upos;
    let is_nominal = matches!(upos, Some(Upos::Noun | Upos::Propn | Upos::Pron));
    let is_x_nominal = upos == Some(Upos::X)
        && matches!(
            rel,
            "nsubj" | "obj" | "iobj" | "obl" | "nmod" | "conj" | "root"
        );
    let is_discourse_anaphor = is_discourse_anaphor_surface(&graph.canons[idx]);

    if !is_nominal && !is_x_nominal && !is_discourse_anaphor {
        return false;
    }

    // Modifier roles are absorbed by their parent NP, except nested possessive pronouns.
    if matches!(
        rel,
        "compound" | "flat" | "fixed" | "clf" | "nummod" | "det" | "case" | "mark"
    ) {
        return upos == Some(Upos::Pron);
    }

    true
}

fn collect_subtree(
    root: usize,
    sent: usize,
    doc: &Document,
    graph: &SyntaxGraph<'_>,
    included: &mut [bool],
) {
    if included[root] || doc.tokens[root].sentence != sent {
        return;
    }
    included[root] = true;
    for &ch in &graph.children[root] {
        if doc.tokens[ch].sentence == sent && graph.deprels[ch] != "punct" {
            collect_subtree(ch, sent, doc, graph, included);
        }
    }
}

fn extract_np_span(head: usize, doc: &Document, graph: &SyntaxGraph<'_>) -> Option<(usize, usize)> {
    let n = doc.tokens.len();
    let sent = doc.tokens[head].sentence;
    let rel_h = graph.deprels[head];

    // Nested possessive/determiner pronoun stays a single-token span.
    if doc.tokens[head].upos == Some(Upos::Pron)
        && matches!(rel_h, "det" | "nmod" | "compound" | "flat" | "fixed")
    {
        return Some((head, head));
    }

    let mut included = vec![false; n];
    included[head] = true;

    for &ch in &graph.children[head] {
        if doc.tokens[ch].sentence != sent {
            continue;
        }
        let rel = graph.deprels[ch];
        if matches!(
            rel,
            "det" | "nummod" | "clf" | "amod" | "compound" | "flat" | "fixed" | "nmod" | "acl"
        ) {
            collect_subtree(ch, sent, doc, graph, &mut included);
        } else if rel == "conj"
            && matches!(
                doc.tokens[ch].upos,
                Some(Upos::Noun | Upos::Propn | Upos::X)
            )
            && ch > head
            && !graph.children[ch]
                .iter()
                .any(|&gc| doc.tokens[gc].upos == Some(Upos::Pron) || graph.deprels[gc] == "det")
        {
            // Coordinated NP (e.g. `Rust 專案 與 函式 庫`).
            let has_cc_between = graph.children[head]
                .iter()
                .any(|&c| graph.deprels[c] == "cc" && c > head && c < ch)
                || graph.children[ch]
                    .iter()
                    .any(|&c| graph.deprels[c] == "cc" && c > head && c < ch);
            if has_cc_between {
                collect_subtree(ch, sent, doc, graph, &mut included);
                for &c in &graph.children[head] {
                    if graph.deprels[c] == "cc" && c > head && c < ch {
                        included[c] = true;
                    }
                }
            }
        }
    }

    let mut min_idx = head;
    let mut max_idx = head;
    for (i, &inc) in included.iter().enumerate() {
        if inc {
            min_idx = min_idx.min(i);
            max_idx = max_idx.max(i);
        }
    }

    // Re-include any internal `case`/`mark` child of `head` that lies strictly inside [min_idx, max_idx].
    for &ch in &graph.children[head] {
        if ch > min_idx && ch < max_idx && matches!(graph.deprels[ch], "case" | "mark") {
            included[ch] = true;
        }
    }

    // Enforce contiguity and the single-internal-head invariant around `head`.
    let mut start = head;
    while start > min_idx {
        let prev = start - 1;
        if doc.tokens[prev].sentence != sent {
            break;
        }
        let h_prev = graph.heads[prev];
        let in_subtree = included[prev]
            || (h_prev.is_some_and(|p| p >= min_idx && p <= max_idx)
                && doc.tokens[prev].upos != Some(Upos::Punct));
        if !in_subtree {
            break;
        }
        start = prev;
    }

    let mut end = head;
    while end < max_idx {
        let next = end + 1;
        if doc.tokens[next].sentence != sent {
            break;
        }
        let h_next = graph.heads[next];
        let in_subtree = included[next]
            || (h_next.is_some_and(|p| p >= min_idx && p <= max_idx)
                && doc.tokens[next].upos != Some(Upos::Punct));
        if !in_subtree {
            break;
        }
        end = next;
    }

    // Trim leading/trailing punctuation unless it is the head itself.
    while start < head && doc.tokens[start].upos == Some(Upos::Punct) {
        start += 1;
    }
    while end > head && doc.tokens[end].upos == Some(Upos::Punct) {
        end -= 1;
    }

    // Verify strictly one token in start..=end has its head outside start..=end.
    let mut ext_heads = 0usize;
    for &head_opt in &graph.heads[start..=end] {
        let outside = match head_opt {
            None => true,
            Some(p) => p < start || p > end,
        };
        if outside {
            ext_heads += 1;
        }
    }
    if ext_heads != 1 {
        return Some((head, head));
    }

    Some((start, end))
}

fn is_det_prefix(canon: &str) -> bool {
    matches!(
        canon,
        "the"
            | "a"
            | "an"
            | "this"
            | "that"
            | "these"
            | "those"
            | "这"
            | "那"
            | "这个"
            | "那个"
            | "这张"
            | "那张"
            | "这款"
            | "那款"
            | "这台"
            | "那台"
            | "这本"
            | "那本"
            | "这家"
            | "那家"
            | "这些"
            | "那些"
            | "整锅"
            | "その"
            | "この"
            | "あの"
            | "그"
            | "이"
            | "저"
    )
}

fn strip_det_prefix(tokens: &[String]) -> String {
    let mut i = 0;
    while i + 1 < tokens.len() && is_det_prefix(&tokens[i]) {
        i += 1;
    }
    tokens[i..].join(" ")
}

fn infer_pronoun_kind(canon: &str) -> (PronounKind, Person, Number, Animacy) {
    match canon {
        "他" | "him" | "his" | "he" | "彼" | "그" => (
            PronounKind::Personal3rd,
            Person::Third,
            Number::Singular,
            Animacy::HumanMale,
        ),
        "她" | "she" | "her" | "hers" | "彼女" | "그녀" => (
            PronounKind::Personal3rd,
            Person::Third,
            Number::Singular,
            Animacy::HumanFemale,
        ),
        "它" | "it" | "its" | "それ" | "그것" => (
            PronounKind::Personal3rd,
            Person::Third,
            Number::Singular,
            Animacy::Inanimate,
        ),
        "牠" => (
            PronounKind::Personal3rd,
            Person::Third,
            Number::Singular,
            Animacy::Animal,
        ),
        "他们" | "她们" | "它们" | "牠们" | "they" | "them" | "their" | "theirs" | "彼ら"
        | "彼女ら" | "그들" => (
            PronounKind::Personal3rd,
            Person::Third,
            Number::Plural,
            if matches!(canon, "它们" | "牠们") {
                Animacy::Inanimate
            } else if canon == "彼女ら" {
                Animacy::HumanFemale
            } else {
                Animacy::Unknown
            },
        ),
        "あいつ" | "こいつ" | "そいつ" => (
            PronounKind::Personal3rd,
            Person::Third,
            Number::Singular,
            Animacy::HumanUnknown,
        ),
        "其" => (
            PronounKind::Personal3rd,
            Person::Third,
            Number::Unknown,
            Animacy::Unknown,
        ),
        "自己" | "himself" | "herself" | "itself" | "themselves" | "自分" | "자신" => (
            PronounKind::Reflexive,
            Person::Third,
            if canon == "themselves" {
                Number::Plural
            } else {
                Number::Singular
            },
            match canon {
                "himself" => Animacy::HumanMale,
                "herself" => Animacy::HumanFemale,
                "itself" => Animacy::Inanimate,
                _ => Animacy::Unknown,
            },
        ),
        "我" | "i" | "me" | "my" | "mine" | "私" | "わたし" | "あたし" | "わたくし" | "僕"
        | "ぼく" | "俺" | "おれ" | "나" | "저" => (
            PronounKind::Personal1st,
            Person::First,
            Number::Singular,
            Animacy::HumanUnknown,
        ),
        "我们" | "we" | "us" | "our" | "ours" | "私たち" | "僕ら" | "俺たち" | "우리" => {
            (
                PronounKind::Personal1st,
                Person::First,
                Number::Plural,
                Animacy::HumanUnknown,
            )
        }
        "你" | "you" | "your" | "yours" | "あなた" | "あんた" | "お前" | "おまえ" | "きみ"
        | "貴様" | "너" => (
            PronounKind::Personal2nd,
            Person::Second,
            Number::Unknown,
            Animacy::HumanUnknown,
        ),
        "你们" | "여러분" => (
            PronounKind::Personal2nd,
            Person::Second,
            Number::Plural,
            Animacy::HumanUnknown,
        ),
        "前者" => (
            PronounKind::DiscourseFormer,
            Person::Third,
            Number::Singular,
            Animacy::Unknown,
        ),
        "后者" => (
            PronounKind::DiscourseLatter,
            Person::Third,
            Number::Singular,
            Animacy::Unknown,
        ),
        "上面" | "这里" | "那里" => (
            PronounKind::DiscourseLoc,
            Person::Third,
            Number::Singular,
            Animacy::Inanimate,
        ),
        "其中" => (
            PronounKind::DiscoursePartitive,
            Person::Third,
            Number::Plural,
            Animacy::Unknown,
        ),
        _ => (
            PronounKind::None,
            Person::Third,
            Number::Unknown,
            Animacy::Unknown,
        ),
    }
}

fn has_female_hint(s: &str) -> bool {
    const FEMALE_HINTS: &[&str] = &[
        "小姐",
        "女士",
        "阿姨",
        "姐姐",
        "妹妹",
        "妈妈",
        "奶奶",
        "外婆",
        "表妹",
        "表姐",
        "姑姑",
        "mrs.",
        "ms.",
        "clara",
        "elena",
        "hannah",
        "miriam",
        "maya",
        "rachel",
        "claire",
        "alice",
        "小芸",
        "小琳",
        "ちゃん",
        "お嬢様",
        "嬢",
        "少女",
        "姉",
        "妹",
        "妻",
    ];
    let lower = s.to_lowercase();
    if FEMALE_HINTS.iter().any(|h| lower.contains(h)) {
        return true;
    }
    lower.contains("女") && !lower.contains("男女") && !lower.contains("子女")
}

fn has_male_hint(s: &str) -> bool {
    const MALE_HINTS: &[&str] = &[
        "先生",
        "伯伯",
        "伯父",
        "大爷",
        "爸爸",
        "师傅",
        "小伙子",
        "爷爷",
        "外公",
        "叔叔",
        "舅舅",
        "表弟",
        "表哥",
        "哥哥",
        "弟弟",
        "mr.",
        "thomas",
        "arthur",
        "martin",
        "owen",
        "ben",
        "sam",
        "老王",
        "老陈",
        "小林",
        "小宇",
        "小杰",
        "小周",
        "阿文",
        "くん",
        "氏",
        "少年",
        "兄",
        "弟",
        "夫",
    ];
    let lower = s.to_lowercase();
    if MALE_HINTS.iter().any(|h| lower.contains(h)) {
        return true;
    }
    lower.contains("男") && !lower.contains("男女")
}

fn has_human_hint(
    canon_span: &str,
    head_canon: &str,
    upos: Option<Upos>,
    sem: Option<SemanticClass>,
) -> bool {
    const INANIMATE_AGENTIVE: &[&str] = &[
        "solver",
        "encoder",
        "decoder",
        "scheduler",
        "compiler",
        "parser",
        "router",
        "server",
        "browser",
        "printer",
        "timer",
        "notifier",
        "monitor",
        "sensor",
        "adapter",
    ];
    if INANIMATE_AGENTIVE.contains(&head_canon) {
        return false;
    }
    if sem == Some(SemanticClass::Person) {
        return true;
    }
    let is_short_ascii_acronym = head_canon.is_ascii() && head_canon.chars().count() <= 4;
    if upos == Some(Upos::Propn)
        && !matches!(
            sem,
            Some(SemanticClass::Location | SemanticClass::Organization | SemanticClass::Artifact)
        )
        && !canon_span.chars().any(|c| c.is_ascii_digit())
        && !is_short_ascii_acronym
    {
        return true;
    }
    if has_female_hint(canon_span) || has_male_hint(canon_span) {
        return true;
    }
    const HUMAN_NOUNS: &[&str] = &[
        "邮差",
        "裁缝",
        "园丁",
        "保安",
        "琴师",
        "房东",
        "同学",
        "研究员",
        "团队",
        "陶艺家",
        "librarian",
        "shepherd",
        "researcher",
        "researchers",
        "annotator",
        "annotators",
        "老師",
        "老师",
        "医生",
        "教授",
        "老板",
        "朋友",
        "学生",
        "记者",
        "医生",
        "先生",
        "さん",
        "様",
        "先生",
        "씨",
        "님",
        "학생",
        "선생",
    ];
    HUMAN_NOUNS
        .iter()
        .any(|h| head_canon == *h || canon_span.contains(h))
}

fn has_animal_hint(canon_span: &str, head_canon: &str) -> bool {
    const ANIMAL_NOUNS: &[&str] = &[
        "鸽子", "小猫", "猫", "狗", "鸟", "鱼", "马", "heron", "kitten", "cat", "dog", "beagle",
        "robin", "bird",
    ];
    ANIMAL_NOUNS
        .iter()
        .any(|a| head_canon == *a || canon_span.contains(a))
}

fn is_plural_np(
    doc: &Document,
    graph: &SyntaxGraph<'_>,
    start: usize,
    end: usize,
    head: usize,
) -> Number {
    let head_canon = &graph.canons[head];
    let head_surf = &graph.surfaces[head];
    if head_canon.ends_with('们') || head_canon == "这些" || head_canon == "那些" {
        return Number::Plural;
    }
    // Check English plural noun heuristic.
    if doc.tokens[head].lang == Lang::En && doc.tokens[head].upos == Some(Upos::Noun) {
        let lower = head_surf.to_lowercase();
        if let Some(lemma) = doc.tokens[head].lemma.as_deref()
            && lower != lemma.to_lowercase()
            && lower.ends_with('s')
        {
            return Number::Plural;
        }
        if lower.ends_with('s')
            && !lower.ends_with("ss")
            && !lower.ends_with("us")
            && !lower.ends_with("is")
            && !lower.ends_with("as")
            && !lower.ends_with("os")
            && lower.len() > 3
            && !matches!(lower.as_str(), "bus" | "lens" | "gas" | "news" | "series")
        {
            return Number::Plural;
        }
    }
    // Check Japanese / Korean plural suffixes.
    if head_canon.ends_with("たち") || head_canon.ends_with("ら") || head_canon.ends_with("들")
    {
        return Number::Plural;
    }
    // Check plural determiners/numerals/conj inside span.
    for i in start..=end {
        let c = &graph.canons[i];
        let is_one_zh = c.starts_with('一')
            && c.chars().count() <= 3
            && !matches!(c.as_str(), "一些" | "一起" | "一切" | "一般");
        if matches!(
            c.as_str(),
            "these"
                | "those"
                | "all"
                | "many"
                | "several"
                | "both"
                | "two"
                | "three"
                | "four"
                | "five"
                | "这些"
                | "那些"
                | "两"
                | "两个"
                | "两台"
                | "两块"
                | "三"
                | "四"
                | "十二个"
        ) || (doc.tokens[i].upos == Some(Upos::Num)
            && !is_one_zh
            && !matches!(c.as_str(), "1" | "one"))
        {
            return Number::Plural;
        }
        if i != head && graph.deprels[i] == "conj" && graph.heads[i] == Some(head) {
            return Number::Plural;
        }
    }
    // Collective or bare Chinese human nouns can match plural pronouns (`他們`).
    if matches!(
        head_canon.as_str(),
        "团队" | "小组" | "公司" | "committee" | "team" | "研究员" | "学生" | "老师"
    ) {
        return Number::Unknown;
    }
    Number::Singular
}

fn is_time_or_quantity_np(
    doc: &Document,
    graph: &SyntaxGraph<'_>,
    start: usize,
    end: usize,
    head: usize,
) -> bool {
    let span = Span {
        start_char: doc.tokens[start].span.start_char,
        end_char: doc.tokens[end].span.end_char,
    };
    for ent in &doc.entities {
        if ent.span.start_char <= span.start_char
            && ent.span.end_char >= span.end_char
            && matches!(
                ent.kind,
                EntityKind::Date
                    | EntityKind::RelativeDate
                    | EntityKind::Time
                    | EntityKind::Quantity
                    | EntityKind::Number
            )
        {
            return true;
        }
    }
    if matches!(
        doc.tokens[head].sem_class,
        Some(SemanticClass::Time | SemanticClass::Quantity)
    ) {
        return true;
    }
    let hc = &graph.canons[head];
    matches!(
        hc.as_str(),
        "早上"
            | "下午"
            | "晚上"
            | "昨晚"
            | "明天"
            | "今天"
            | "昨天"
            | "刚才"
            | "现在"
            | "星期天"
            | "星期六"
            | "十分钟"
            | "几周"
            | "时候"
            | "每次"
            | "一碗"
            | "一会儿"
    )
}

fn find_clause_head(idx: usize, graph: &SyntaxGraph<'_>) -> Option<usize> {
    let mut cur = idx;
    for _ in 0..12 {
        let p = graph.heads[cur]?;
        let rel = graph.deprels[cur];
        if matches!(
            rel,
            "nsubj" | "csubj" | "obj" | "iobj" | "obl" | "root" | "parataxis" | "ccomp" | "advcl"
        ) {
            return Some(p);
        }
        cur = p;
    }
    None
}

fn extract_mentions(doc: &Document, graph: &SyntaxGraph<'_>) -> Vec<MentionInfo> {
    let mut out = Vec::new();
    let mut seen_spans: HashSet<(usize, usize)> = HashSet::new();

    for h in 0..doc.tokens.len() {
        if !is_candidate_head(doc, graph, h) {
            continue;
        }
        let Some((start, end)) = extract_np_span(h, doc, graph) else {
            continue;
        };
        if !seen_spans.insert((start, end)) {
            continue;
        }

        let sent = doc.tokens[h].sentence;
        let span = Span {
            start_char: doc.tokens[start].span.start_char,
            end_char: doc.tokens[end].span.end_char,
        };
        let canon_slice = &graph.canons[start..=end];
        let canon = strip_det_prefix(canon_slice);
        let head_surface = &graph.surfaces[h];
        let head_canon = graph.canons[h].clone();
        let upos = doc.tokens[h].upos;
        let deprel = graph.deprels[h].to_string();

        let (pronoun, person, p_num, p_anim) =
            if upos == Some(Upos::Pron) || is_discourse_anaphor_surface(&head_canon) {
                infer_pronoun_kind(&head_canon)
            } else {
                (
                    PronounKind::None,
                    Person::Third,
                    Number::Unknown,
                    Animacy::Unknown,
                )
            };

        let number = if pronoun != PronounKind::None {
            p_num
        } else {
            is_plural_np(doc, graph, start, end, h)
        };

        let sem = doc.tokens[h]
            .sem_class
            .or_else(|| match doc.tokens[h].lang {
                Lang::Zh => crate::lexicon::lookup_zh(head_surface).and_then(|l| l.sem_class),
                Lang::En => crate::lexicon::lookup_en(head_surface).and_then(|l| l.sem_class),
                Lang::Ja => crate::lexicon::lookup_ja(head_surface).and_then(|l| l.sem_class),
                Lang::Ko => crate::lexicon::lookup_ko(head_surface).and_then(|l| l.sem_class),
                _ => None,
            });

        let animacy = if pronoun != PronounKind::None {
            p_anim
        } else if has_animal_hint(&canon, &head_canon) {
            Animacy::Animal
        } else if has_female_hint(&canon) {
            Animacy::HumanFemale
        } else if has_male_hint(&canon) {
            Animacy::HumanMale
        } else if has_human_hint(&canon, &head_canon, upos, sem) {
            Animacy::HumanUnknown
        } else {
            Animacy::Inanimate
        };

        let is_definite = start < end && is_det_prefix(&graph.canons[start]);
        let is_time_or_qty = is_time_or_quantity_np(doc, graph, start, end, h);
        let is_proper = upos == Some(Upos::Propn)
            || doc.entities.iter().any(|e| {
                e.kind == EntityKind::Name
                    && e.span.start_char >= span.start_char
                    && e.span.end_char <= span.end_char
            });
        let salience = match deprel.as_str() {
            "nsubj" | "csubj" => 5,
            "obj" => 4,
            "iobj" => 3,
            "obl" => 2,
            _ => 0,
        };

        out.push(MentionInfo {
            mention: CorefMention {
                start_token: start,
                end_token: end,
                head_token: h,
                sentence: sent,
                span,
            },
            canon,
            head_canon,
            upos,
            deprel,
            number,
            animacy,
            person,
            pronoun,
            is_definite,
            is_time_or_qty,
            is_proper,
            salience,
            clause_head: find_clause_head(h, graph),
        });
    }

    out.sort_by_key(|m| (m.mention.start_token, m.mention.end_token));
    out
}

fn numbers_compatible(a: Number, b: Number) -> bool {
    a == Number::Unknown || b == Number::Unknown || a == b
}

fn animacies_compatible(ant: Animacy, pro: Animacy) -> bool {
    matches!(
        (ant, pro),
        (_, Animacy::Unknown)
            | (Animacy::Unknown, _)
            | (
                Animacy::HumanMale,
                Animacy::HumanMale | Animacy::HumanUnknown
            )
            | (
                Animacy::HumanFemale,
                Animacy::HumanFemale | Animacy::HumanUnknown
            )
            | (
                Animacy::HumanUnknown,
                Animacy::HumanMale | Animacy::HumanFemale | Animacy::HumanUnknown
            )
            | (
                Animacy::Animal,
                Animacy::Animal | Animacy::Inanimate | Animacy::HumanMale | Animacy::HumanFemale
            )
            | (Animacy::Inanimate, Animacy::Inanimate)
    )
}

fn modifiers_conflict(m1: &MentionInfo, m2: &MentionInfo, graph: &SyntaxGraph<'_>) -> bool {
    const COLOR_WORDS: &[&str] = &[
        "红", "蓝", "绿", "黄", "黑", "白", "red", "blue", "green", "yellow", "black", "white",
    ];
    let colors1: Vec<&str> = (m1.mention.start_token..m1.mention.head_token)
        .filter_map(|i| {
            COLOR_WORDS
                .iter()
                .copied()
                .find(|&c| graph.canons[i].contains(c))
        })
        .collect();
    let colors2: Vec<&str> = (m2.mention.start_token..m2.mention.head_token)
        .filter_map(|i| {
            COLOR_WORDS
                .iter()
                .copied()
                .find(|&c| graph.canons[i].contains(c))
        })
        .collect();
    !colors1.is_empty() && !colors2.is_empty() && colors1 != colors2
}

fn is_generic_head_stopword(head_canon: &str) -> bool {
    matches!(
        head_canon,
        "人" | "事"
            | "东西"
            | "时候"
            | "地方"
            | "问题"
            | "结果"
            | "情况"
            | "方法"
            | "thing"
            | "way"
            | "time"
            | "part"
            | "place"
            | "case"
    )
}

struct UnionFind {
    parent: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
        }
    }
    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }
    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra != rb {
            self.parent[rb] = ra;
        }
    }
}

fn resolve_coref(
    doc: &Document,
    graph: &SyntaxGraph<'_>,
    mentions: &[MentionInfo],
) -> Vec<Vec<CorefMention>> {
    let n = mentions.len();
    if n < 2 {
        return Vec::new();
    }
    let mut uf = UnionFind::new(n);

    // Sieve 1: Exact & Canonical Surface / Definite & Bare Head-Noun Repetition.
    for j in 1..n {
        let m2 = &mentions[j];
        if m2.pronoun != PronounKind::None || m2.is_time_or_qty {
            continue;
        }
        for i in (0..j).rev() {
            let m1 = &mentions[i];
            if m1.pronoun != PronounKind::None || m1.is_time_or_qty {
                continue;
            }
            if m1.mention.sentence + 3 < m2.mention.sentence {
                break;
            }
            // Do not merge nested span with its enclosing NP.
            if m1.mention.start_token <= m2.mention.start_token
                && m1.mention.end_token >= m2.mention.end_token
            {
                continue;
            }
            if !m1.canon.is_empty()
                && m1.canon == m2.canon
                && !is_generic_head_stopword(&m2.head_canon)
            {
                uf.union(i, j);
                break;
            }
            let m2_is_bare_or_def = m2.is_definite
                || (m2.mention.start_token == m2.mention.end_token
                    && m2.upos == Some(Upos::Noun)
                    && m2.head_canon.chars().count() >= 2);
            if m2_is_bare_or_def
                && !m2.head_canon.is_empty()
                && m1.head_canon == m2.head_canon
                && !is_generic_head_stopword(&m2.head_canon)
                && !modifiers_conflict(m1, m2, graph)
            {
                uf.union(i, j);
                break;
            }
        }
    }

    // Sieve 2: Proper Name & Title/Surname Alias Match.
    for j in 1..n {
        let m2 = &mentions[j];
        if !m2.is_proper || m2.pronoun != PronounKind::None {
            continue;
        }
        for i in (0..j).rev() {
            let m1 = &mentions[i];
            if !m1.is_proper || m1.pronoun != PronounKind::None {
                continue;
            }
            if m1.mention.start_token <= m2.mention.start_token
                && m1.mention.end_token >= m2.mention.end_token
            {
                continue;
            }
            let tokens1: HashSet<&str> = m1.canon.split_whitespace().collect();
            let tokens2: HashSet<&str> = m2.canon.split_whitespace().collect();
            if !tokens1.is_empty()
                && !tokens2.is_empty()
                && tokens1
                    .intersection(&tokens2)
                    .any(|t| t.chars().count() >= 2)
            {
                uf.union(i, j);
                break;
            }
        }
    }

    // Sieve 3: Anaphoric Pronoun & Discourse Anaphor Resolution.
    for j in 0..n {
        let m2 = &mentions[j];
        match m2.pronoun {
            PronounKind::None => continue,
            PronounKind::Personal1st | PronounKind::Personal2nd => {
                for i in (0..j).rev() {
                    let m1 = &mentions[i];
                    if m1.person == m2.person && numbers_compatible(m1.number, m2.number) {
                        uf.union(i, j);
                        break;
                    }
                }
            }
            PronounKind::Reflexive => {
                if m2.deprel == "appos" {
                    continue;
                }
                for i in (0..j).rev() {
                    let m1 = &mentions[i];
                    if m1.mention.sentence != m2.mention.sentence {
                        break;
                    }
                    if m1.deprel == "nsubj"
                        && numbers_compatible(m1.number, m2.number)
                        && animacies_compatible(m1.animacy, m2.animacy)
                    {
                        uf.union(i, j);
                        break;
                    }
                }
            }
            PronounKind::DiscourseLatter | PronounKind::DiscourseFormer => {
                let mut candidates = Vec::new();
                for i in (0..j).rev() {
                    let m1 = &mentions[i];
                    if m1.mention.sentence + 1 < m2.mention.sentence {
                        break;
                    }
                    if m1.mention.sentence < m2.mention.sentence
                        && m1.pronoun == PronounKind::None
                        && !m1.is_time_or_qty
                    {
                        candidates.push(i);
                    }
                }
                if m2.pronoun == PronounKind::DiscourseLatter
                    && let Some(&idx) = candidates.first()
                {
                    uf.union(idx, j);
                } else if m2.pronoun == PronounKind::DiscourseFormer
                    && let Some(&idx) = candidates.get(1)
                {
                    uf.union(idx, j);
                }
            }
            PronounKind::DiscourseLoc => {
                for i in (0..j).rev() {
                    let m1 = &mentions[i];
                    if m1.mention.sentence + 1 < m2.mention.sentence {
                        break;
                    }
                    if m1.pronoun == PronounKind::None
                        && !m1.is_time_or_qty
                        && m1.animacy == Animacy::Inanimate
                    {
                        uf.union(i, j);
                        break;
                    }
                }
            }
            PronounKind::DiscoursePartitive => {
                for i in (0..j).rev() {
                    let m1 = &mentions[i];
                    if m1.mention.sentence + 1 < m2.mention.sentence {
                        break;
                    }
                    if m1.pronoun == PronounKind::None && m1.number == Number::Plural {
                        uf.union(i, j);
                        break;
                    }
                }
            }
            PronounKind::Personal3rd => {
                let mut best: Option<(i32, usize)> = None;
                let m2_is_poss = matches!(m2.deprel.as_str(), "nmod" | "det" | "compound");
                for i in (0..j).rev() {
                    let m1 = &mentions[i];
                    if m1.mention.sentence + 2 < m2.mention.sentence {
                        break;
                    }
                    // Skip nested mention if m1 encloses m2.
                    if m1.mention.start_token <= m2.mention.start_token
                        && m1.mention.end_token >= m2.mention.end_token
                    {
                        continue;
                    }
                    if m1.is_time_or_qty {
                        continue;
                    }
                    if m1.person != Person::Third {
                        continue;
                    }
                    // Binding Condition B: non-reflexive non-possessive pronoun cannot bind to
                    // co-argument nsubj of the same clause.
                    if !m2_is_poss
                        && m1.mention.sentence == m2.mention.sentence
                        && m1.deprel == "nsubj"
                        && m1.clause_head.is_some()
                        && m1.clause_head == m2.clause_head
                    {
                        continue;
                    }
                    if !numbers_compatible(m1.number, m2.number) {
                        continue;
                    }
                    if !animacies_compatible(m1.animacy, m2.animacy) {
                        continue;
                    }

                    let sent_dist = m2.mention.sentence.saturating_sub(m1.mention.sentence) as i32;
                    let mut score = (m1.salience as i32) * 10 - sent_dist * 10;
                    if m1.mention.sentence == m2.mention.sentence
                        && m2_is_poss
                        && m1.deprel == "nsubj"
                    {
                        score += 25;
                    }
                    if m1.pronoun == PronounKind::None {
                        score += 5;
                    }
                    if doc.tokens[m1.mention.head_token].upos == Some(Upos::Propn) {
                        score += 3;
                    }
                    if best.is_none_or(|(bs, _)| score > bs) {
                        best = Some((score, i));
                    }
                }
                if let Some((_, ant_idx)) = best {
                    uf.union(ant_idx, j);
                }
            }
        }
    }

    let mut groups: HashMap<usize, Vec<CorefMention>> = HashMap::new();
    for (idx, m) in mentions.iter().enumerate() {
        let root = uf.find(idx);
        groups.entry(root).or_default().push(m.mention.clone());
    }

    let mut clusters: Vec<Vec<CorefMention>> = groups
        .into_values()
        .filter(|cl| cl.len() >= 2)
        .map(|mut cl| {
            cl.sort_by_key(|m| (m.start_token, m.end_token));
            cl.dedup_by_key(|m| (m.start_token, m.end_token));
            cl
        })
        .filter(|cl| cl.len() >= 2)
        .collect();

    clusters.sort_by_key(|cl| (cl[0].start_token, cl[0].end_token));
    clusters
}

fn is_prodrop_lang(lang: Lang) -> bool {
    matches!(lang, Lang::Zh | Lang::Ja | Lang::Ko | Lang::Mixed)
}

fn is_strong_zero_obj_verb(canon: &str) -> bool {
    const EXACT_VERBS: &[&str] = &[
        "吃", "拿", "找", "去", "发", "看", "戴", "裁剪", "更新", "照着", "尝尝",
    ];
    const SUFFIX_COMPOUNDS: &[&str] = &[
        "读完",
        "看完",
        "吃完",
        "写完",
        "做完",
        "修好",
        "洗好",
        "切好",
        "切开",
        "拿回来",
        "带回来",
        "还给",
        "交给",
        "发到",
        "放到",
        "寄出",
        "上传",
        "买了",
        "卖了",
        "買った",
        "読んだ",
        "食べた",
        "見た",
        "送った",
        "返した",
        "읽었다",
        "샀다",
        "보냈다",
        "먹었다",
        "봤다",
        "돌려주었다",
    ];
    EXACT_VERBS.contains(&canon)
        || SUFFIX_COMPOUNDS
            .iter()
            .any(|v| canon == *v || canon.ends_with(v))
}

fn is_ba_obl_head(idx: usize, graph: &SyntaxGraph<'_>) -> bool {
    graph.deprels[idx] == "obl"
        && graph.children[idx].iter().any(|&gc| {
            graph.deprels[gc] == "case" && matches!(graph.canons[gc].as_str(), "把" | "将")
        })
}

fn has_ba_disposal_obl(pred: usize, graph: &SyntaxGraph<'_>) -> bool {
    graph.children[pred]
        .iter()
        .any(|&c| is_ba_obl_head(c, graph))
}

fn has_passive_marker(pred: usize, graph: &SyntaxGraph<'_>) -> bool {
    graph.children[pred]
        .iter()
        .any(|&c| matches!(graph.canons[c].as_str(), "被" | "られる" | "れる"))
}

fn has_comma_between(a: usize, b: usize, graph: &SyntaxGraph<'_>) -> bool {
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    (lo + 1..hi).any(|i| matches!(graph.surfaces[i].as_str(), "，" | ","))
}

fn resolve_zero_mentions(
    doc: &Document,
    graph: &SyntaxGraph<'_>,
    mentions: &[MentionInfo],
) -> Vec<ZeroMention> {
    let mut out = Vec::new();
    let n_sents = doc.segmentation.sentences.len();
    if n_sents == 0 {
        return out;
    }

    let mut mentions_by_sent: Vec<Vec<&MentionInfo>> = vec![Vec::new(); n_sents];
    for m in mentions {
        if m.mention.sentence < n_sents {
            mentions_by_sent[m.mention.sentence].push(m);
        }
    }

    for pred in 0..doc.tokens.len() {
        let tok = &doc.tokens[pred];
        if !is_prodrop_lang(tok.lang) {
            continue;
        }
        if !matches!(tok.upos, Some(Upos::Verb | Upos::Adj)) {
            continue;
        }
        let p_rel = graph.deprels[pred];
        let p_canon = &graph.canons[pred];
        if matches!(
            p_canon.as_str(),
            "有" | "是" | "いる" | "ある" | "있다" | "없다" | "이다"
        ) {
            continue;
        }
        if matches!(p_rel, "acl" | "compound" | "aux" | "conj") {
            continue;
        }

        let sent = tok.sentence;
        let has_nsubj = graph.children[pred]
            .iter()
            .any(|&c| matches!(graph.deprels[c], "nsubj" | "csubj"));
        let has_obj = graph.children[pred]
            .iter()
            .any(|&c| matches!(graph.deprels[c], "obj" | "ccomp" | "xcomp"));

        // ZeroRole::Nsubj: finite matrix/parataxis/ccomp predicates (and advcl only when separated
        // from matrix nsubj by a comma or when matrix has no nsubj).
        if !has_nsubj
            && matches!(p_rel, "root" | "parataxis" | "ccomp" | "advcl")
            && !has_passive_marker(pred, graph)
        {
            let mut ant: Option<&MentionInfo> = None;
            let same_sent_nsubj = mentions_by_sent[sent]
                .iter()
                .rev()
                .find(|m| m.mention.end_token < pred && !m.is_time_or_qty && m.deprel == "nsubj")
                .copied();

            if p_rel == "advcl" {
                let matrix_has_nsubj = graph.heads[pred].is_some_and(|h| {
                    graph.children[h]
                        .iter()
                        .any(|&c| matches!(graph.deprels[c], "nsubj" | "csubj"))
                });
                if !matrix_has_nsubj
                    && graph.heads[pred].is_some_and(|h| graph.deprels[h] == "root")
                {
                    // e.g. `该算法 ... 利用 ... 完成` / `构建脚本 每次 运行 都 会 校验`
                    ant = same_sent_nsubj;
                }
            } else if let Some(m) = same_sent_nsubj {
                if (p_rel == "parataxis"
                    || (p_rel == "root" && has_comma_between(m.mention.end_token, pred, graph)))
                    && !(m.animacy == Animacy::Inanimate
                        && matches!(p_canon.as_str(), "记得" | "下班" | "打算"))
                {
                    ant = Some(m);
                }
            } else if sent > 0 {
                // Cross-sentence pro-drop: prefer Human antecedent if previous sentence's nsubj is inanimate.
                ant = mentions_by_sent[sent - 1]
                    .iter()
                    .find(|m| {
                        !m.is_time_or_qty && m.deprel == "nsubj" && m.animacy != Animacy::Inanimate
                    })
                    .copied()
                    .or_else(|| {
                        mentions_by_sent[sent - 1]
                            .iter()
                            .find(|m| !m.is_time_or_qty && m.animacy != Animacy::Inanimate)
                            .copied()
                    })
                    .or_else(|| {
                        mentions_by_sent[sent - 1]
                            .iter()
                            .find(|m| !m.is_time_or_qty && m.deprel == "nsubj")
                            .copied()
                    });
            }

            if let Some(a) = ant {
                out.push(ZeroMention {
                    zero_at: pred,
                    sentence: sent,
                    span: tok.span,
                    role: ZeroRole::Nsubj,
                    antecedent: a.mention.clone(),
                });
            }
        }

        // ZeroRole::Obj: strongly transitive/completive/transfer predicate with no overt obj/ba-obl.
        let is_non_initial_clause = sent > 0
            || mentions_by_sent[sent].iter().any(|m| {
                m.mention.end_token < pred && has_comma_between(m.mention.end_token, pred, graph)
            });
        if !has_obj
            && is_non_initial_clause
            && is_strong_zero_obj_verb(p_canon)
            && !has_ba_disposal_obl(pred, graph)
            && !has_passive_marker(pred, graph)
        {
            let mut ant: Option<&MentionInfo> = None;
            if sent > 0 {
                // Priority 1: `obj` in previous sentence.
                ant = mentions_by_sent[sent - 1]
                    .iter()
                    .rev()
                    .find(|m| {
                        !m.is_time_or_qty && m.animacy == Animacy::Inanimate && m.deprel == "obj"
                    })
                    .copied()
                    // Priority 2: `把`/`將` disposal `obl` in previous sentence.
                    .or_else(|| {
                        mentions_by_sent[sent - 1]
                            .iter()
                            .rev()
                            .find(|m| {
                                !m.is_time_or_qty && is_ba_obl_head(m.mention.head_token, graph)
                            })
                            .copied()
                    })
                    // Priority 3: topicalized inanimate `nsubj` in previous sentence.
                    .or_else(|| {
                        mentions_by_sent[sent - 1]
                            .iter()
                            .find(|m| {
                                !m.is_time_or_qty
                                    && m.animacy == Animacy::Inanimate
                                    && m.deprel == "nsubj"
                            })
                            .copied()
                    });
            }
            if ant.is_none() {
                ant = mentions_by_sent[sent]
                    .iter()
                    .rev()
                    .find(|m| {
                        m.mention.end_token < pred
                            && !m.is_time_or_qty
                            && m.animacy == Animacy::Inanimate
                            && matches!(m.deprel.as_str(), "obj" | "nsubj")
                    })
                    .copied();
            }

            if let Some(a) = ant {
                out.push(ZeroMention {
                    zero_at: pred,
                    sentence: sent,
                    span: tok.span,
                    role: ZeroRole::Obj,
                    antecedent: a.mention.clone(),
                });
            }
        }
    }

    out.sort_by_key(|z| (z.zero_at, z.role));
    out.dedup_by_key(|z| (z.zero_at, z.role));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::doc::{
        Clause, Options, Paragraph, Segmentation, Sentence, SyntaxToken, Token, analyze_document,
    };
    use serde_json::Value;
    use std::collections::{BTreeMap, BTreeSet};
    use std::path::Path;

    fn upos_from_str(s: &str) -> Upos {
        match s {
            "ADJ" => Upos::Adj,
            "ADP" => Upos::Adp,
            "ADV" => Upos::Adv,
            "AUX" => Upos::Aux,
            "CCONJ" => Upos::Cconj,
            "DET" => Upos::Det,
            "INTJ" => Upos::Intj,
            "NOUN" => Upos::Noun,
            "NUM" => Upos::Num,
            "PART" => Upos::Part,
            "PRON" => Upos::Pron,
            "PROPN" => Upos::Propn,
            "PUNCT" => Upos::Punct,
            "SCONJ" => Upos::Sconj,
            "SYM" => Upos::Sym,
            "VERB" => Upos::Verb,
            _ => Upos::X,
        }
    }

    fn lang_from_str(s: &str) -> Lang {
        match s {
            "zh" => Lang::Zh,
            "en" => Lang::En,
            "ja" => Lang::Ja,
            "ko" => Lang::Ko,
            "mixed" => Lang::Mixed,
            _ => Lang::Und,
        }
    }

    fn build_gold_doc(item: &Value) -> (Document, String) {
        let lang = lang_from_str(item["lang"].as_str().unwrap());
        let toks_json = item["tokens"].as_array().unwrap();
        let upos_json = item["upos"].as_array().unwrap();
        let heads_json = item["heads"].as_array().unwrap();
        let deprels_json = item["deprels"].as_array().unwrap();
        let sents_json = item["sentences"].as_array().unwrap();

        let mut text = String::new();
        let mut token_spans = Vec::with_capacity(toks_json.len());
        let mut char_pos = 0usize;
        for (i, t) in toks_json.iter().enumerate() {
            if i > 0 {
                text.push(' ');
                char_pos += 1;
            }
            let s = t.as_str().unwrap();
            let len = s.chars().count();
            text.push_str(s);
            token_spans.push(Span {
                start_char: char_pos,
                end_char: char_pos + len,
            });
            char_pos += len;
        }

        let mut tok_sent = vec![0usize; toks_json.len()];
        let mut sentences = Vec::new();
        let mut clauses = Vec::new();
        for (si, sr) in sents_json.iter().enumerate() {
            let s_start = sr[0].as_u64().unwrap() as usize;
            let s_end = sr[1].as_u64().unwrap() as usize;
            for slot in tok_sent.iter_mut().take(s_end).skip(s_start) {
                *slot = si;
            }
            let span = Span {
                start_char: token_spans[s_start].start_char,
                end_char: token_spans[s_end - 1].end_char,
            };
            sentences.push(Sentence {
                span,
                lang,
                paragraph: 0,
            });
            clauses.push(Clause {
                span,
                lang,
                sentence: si,
            });
        }

        let mut tokens = Vec::with_capacity(toks_json.len());
        let mut syntax = Vec::with_capacity(toks_json.len());
        for i in 0..toks_json.len() {
            let s = toks_json[i].as_str().unwrap();
            let u = upos_from_str(upos_json[i].as_str().unwrap());
            let h_raw = heads_json[i].as_u64().unwrap() as usize;
            let head = if h_raw == 0 { None } else { Some(h_raw - 1) };
            let deprel = deprels_json[i].as_str().unwrap().to_string();
            tokens.push(Token {
                span: token_spans[i],
                lang,
                sentence: tok_sent[i],
                upos: Some(u),
                lemma: Some(s.to_lowercase()),
                canonical_variant: None,
                upos_candidates: Vec::new(),
                sem_class: None,
                morphemes: Vec::new(),
            });
            syntax.push(SyntaxToken {
                token: i,
                sentence: tok_sent[i],
                span: token_spans[i],
                head,
                deprel,
                score: crate::doc::Score(1000),
            });
        }

        let doc = Document {
            schema: crate::doc::SCHEMA.to_owned(),
            text_chars: char_pos,
            text_sha256: String::new(),
            options: Options::default(),
            segmentation: Segmentation {
                paragraphs: vec![Paragraph {
                    span: Span {
                        start_char: 0,
                        end_char: char_pos,
                    },
                    lang,
                }],
                sentences,
                clauses,
                opaque: Vec::new(),
            },
            tokens,
            entities: Vec::new(),
            syntax,
            frames: Vec::new(),
            coref: Vec::new(),
            zero_mentions: Vec::new(),
            relations: Vec::new(),
            unknown: Vec::new(),
        };
        (doc, text)
    }

    #[test]
    fn empty_syntax_leaves_layer6_empty() {
        let mut doc = analyze_document("周小姐借了三本歷史書。", &Options::default());
        annotate(&mut doc, "周小姐借了三本歷史書。");
        assert!(doc.coref.is_empty());
        assert!(doc.zero_mentions.is_empty());
    }

    #[test]
    fn japanese_coref_and_prodrop() {
        // 田中さんは本を買った。彼は読んだ。
        let text = "田中 さん は 本 を 買った 。 彼 は 読んだ 。";
        let toks = [
            "田中",
            "さん",
            "は",
            "本",
            "を",
            "買った",
            "。",
            "彼",
            "は",
            "読んだ",
            "。",
        ];
        let upos = [
            "PROPN", "NOUN", "ADP", "NOUN", "ADP", "VERB", "PUNCT", "PRON", "ADP", "VERB", "PUNCT",
        ];
        let heads = [2, 6, 2, 6, 4, 0, 6, 10, 8, 0, 10];
        let deprels = [
            "compound", "nsubj", "case", "obj", "case", "root", "punct", "nsubj", "case", "root",
            "punct",
        ];
        let item = serde_json::json!({
            "lang": "ja",
            "tokens": toks,
            "upos": upos,
            "heads": heads,
            "deprels": deprels,
            "sentences": [[0, 7], [7, 11]]
        });
        let (mut doc, joined) = build_gold_doc(&item);
        assert!(!joined.is_empty() && !text.is_empty());
        annotate(&mut doc, &joined);
        assert!(
            doc.coref.iter().any(|cl| {
                cl.iter().any(|m| m.start_token == 0 && m.end_token == 1)
                    && cl.iter().any(|m| m.start_token == 7 && m.end_token == 7)
            }),
            "expected [田中 さん] <-> [彼], got {:?}",
            doc.coref
        );
        assert!(
            doc.zero_mentions.iter().any(|z| {
                z.zero_at == 9 && z.role == ZeroRole::Obj && z.antecedent.start_token == 3
            }),
            "expected zero obj at 読んだ -> 本, got {:?}",
            doc.zero_mentions
        );
    }

    #[test]
    fn korean_coref_and_prodrop() {
        // 지수씨는 책을 샀다. 그녀는 읽었다.
        let toks = [
            "지수씨",
            "는",
            "책",
            "을",
            "샀다",
            ".",
            "그녀",
            "는",
            "읽었다",
            ".",
        ];
        let upos = [
            "PROPN", "ADP", "NOUN", "ADP", "VERB", "PUNCT", "PRON", "ADP", "VERB", "PUNCT",
        ];
        let heads = [5, 1, 5, 3, 0, 5, 9, 7, 0, 9];
        let deprels = [
            "nsubj", "case", "obj", "case", "root", "punct", "nsubj", "case", "root", "punct",
        ];
        let item = serde_json::json!({
            "lang": "ko",
            "tokens": toks,
            "upos": upos,
            "heads": heads,
            "deprels": deprels,
            "sentences": [[0, 6], [6, 10]]
        });
        let (mut doc, joined) = build_gold_doc(&item);
        annotate(&mut doc, &joined);
        assert!(
            doc.coref.iter().any(|cl| {
                cl.iter().any(|m| m.start_token == 0 && m.end_token == 0)
                    && cl.iter().any(|m| m.start_token == 6 && m.end_token == 6)
            }),
            "expected [지수씨] <-> [그녀], got {:?}",
            doc.coref
        );
        assert!(
            doc.zero_mentions.iter().any(|z| {
                z.zero_at == 8 && z.role == ZeroRole::Obj && z.antecedent.start_token == 2
            }),
            "expected zero obj at 읽었다 -> 책, got {:?}",
            doc.zero_mentions
        );
    }

    #[test]
    fn gold_300_calibration_metrics() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/gold_layers4_6_v1.jsonl");
        let raw = std::fs::read_to_string(&path).unwrap();
        let items: Vec<Value> = raw
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(items.len(), 300);

        let mut muc_r_num = 0.0f64;
        let mut muc_r_den = 0.0f64;
        let mut muc_p_num = 0.0f64;
        let mut muc_p_den = 0.0f64;
        let mut b3_r_num = 0.0f64;
        let mut b3_r_den = 0.0f64;
        let mut b3_p_num = 0.0f64;
        let mut b3_p_den = 0.0f64;
        let mut ceaf_num = 0.0f64;
        let mut ceaf_r_den = 0.0f64;
        let mut ceaf_p_den = 0.0f64;

        let mut zero_tp = 0usize;
        let mut zero_pred = 0usize;
        let mut zero_gold = 0usize;

        for item in &items {
            let (mut doc, text) = build_gold_doc(item);
            annotate(&mut doc, &text);

            let gold_clusters: Vec<BTreeSet<(usize, usize)>> = item["coref"]
                .as_array()
                .unwrap()
                .iter()
                .map(|cl| {
                    cl.as_array()
                        .unwrap()
                        .iter()
                        .map(|m| {
                            (
                                m[0].as_u64().unwrap() as usize,
                                m[1].as_u64().unwrap() as usize,
                            )
                        })
                        .collect()
                })
                .collect();

            let pred_clusters: Vec<BTreeSet<(usize, usize)>> = doc
                .coref
                .iter()
                .map(|cl| {
                    cl.iter()
                        .map(|m| (m.start_token + 1, m.end_token + 1))
                        .collect()
                })
                .collect();

            // MUC
            let mut g_map: BTreeMap<(usize, usize), usize> = BTreeMap::new();
            for (gi, g) in gold_clusters.iter().enumerate() {
                for &m in g {
                    g_map.insert(m, gi);
                }
            }
            let mut p_map: BTreeMap<(usize, usize), usize> = BTreeMap::new();
            for (pi, p) in pred_clusters.iter().enumerate() {
                for &m in p {
                    p_map.insert(m, pi);
                }
            }
            for g in &gold_clusters {
                muc_r_den += (g.len() - 1) as f64;
                let mut seen_p = HashSet::new();
                let mut unaligned = 0usize;
                for m in g {
                    if let Some(&pi) = p_map.get(m) {
                        seen_p.insert(pi);
                    } else {
                        unaligned += 1;
                    }
                }
                muc_r_num += (g.len() - (seen_p.len() + unaligned)) as f64;
            }
            for p in &pred_clusters {
                muc_p_den += (p.len() - 1) as f64;
                let mut seen_g = HashSet::new();
                let mut unaligned = 0usize;
                for m in p {
                    if let Some(&gi) = g_map.get(m) {
                        seen_g.insert(gi);
                    } else {
                        unaligned += 1;
                    }
                }
                muc_p_num += (p.len() - (seen_g.len() + unaligned)) as f64;
            }

            // B^3
            for g in &gold_clusters {
                for m in g {
                    b3_r_den += 1.0;
                    if let Some(&pi) = p_map.get(m) {
                        let inter = g.intersection(&pred_clusters[pi]).count() as f64;
                        b3_r_num += inter / g.len() as f64;
                    }
                }
            }
            for p in &pred_clusters {
                for m in p {
                    b3_p_den += 1.0;
                    if let Some(&gi) = g_map.get(m) {
                        let inter = p.intersection(&gold_clusters[gi]).count() as f64;
                        b3_p_num += inter / p.len() as f64;
                    }
                }
            }

            // Greedy/exact CEAFe for small per-item cluster counts (usually 0..3)
            ceaf_r_den += gold_clusters.len() as f64;
            ceaf_p_den += pred_clusters.len() as f64;
            if !gold_clusters.is_empty() && !pred_clusters.is_empty() {
                let mut best = 0.0f64;
                if gold_clusters.len() <= 4 && pred_clusters.len() <= 4 {
                    let mut pairs = Vec::new();
                    for (gi, g) in gold_clusters.iter().enumerate() {
                        for (pi, p) in pred_clusters.iter().enumerate() {
                            let inter = g.intersection(p).count() as f64;
                            let sim = (2.0 * inter) / (g.len() + p.len()) as f64;
                            pairs.push((sim, gi, pi));
                        }
                    }
                    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                    let mut ug = HashSet::new();
                    let mut up = HashSet::new();
                    for (sim, gi, pi) in pairs {
                        if ug.insert(gi) && up.insert(pi) {
                            best += sim;
                        }
                    }
                }
                ceaf_num += best;
            }

            // Zero mentions
            let g_zeros: BTreeSet<(usize, String, usize, usize)> = item["zero_mentions"]
                .as_array()
                .unwrap()
                .iter()
                .map(|z| {
                    (
                        z["zero_at"].as_u64().unwrap() as usize,
                        z["role"].as_str().unwrap().to_string(),
                        z["antecedent"][0].as_u64().unwrap() as usize,
                        z["antecedent"][1].as_u64().unwrap() as usize,
                    )
                })
                .collect();
            let p_zeros: BTreeSet<(usize, String, usize, usize)> = doc
                .zero_mentions
                .iter()
                .map(|z| {
                    (
                        z.zero_at + 1,
                        match z.role {
                            ZeroRole::Nsubj => "nsubj".to_string(),
                            ZeroRole::Obj => "obj".to_string(),
                        },
                        z.antecedent.start_token + 1,
                        z.antecedent.end_token + 1,
                    )
                })
                .collect();
            zero_gold += g_zeros.len();
            zero_pred += p_zeros.len();
            zero_tp += g_zeros.intersection(&p_zeros).count();
        }

        let f1 = |p: f64, r: f64| {
            if p + r > 0.0 {
                2.0 * p * r / (p + r)
            } else {
                0.0
            }
        };
        let muc_f1 = f1(muc_p_num / muc_p_den, muc_r_num / muc_r_den);
        let b3_f1 = f1(b3_p_num / b3_p_den, b3_r_num / b3_r_den);
        let ceafe_f1 = f1(ceaf_num / ceaf_p_den, ceaf_num / ceaf_r_den);
        let conll_f1 = (muc_f1 + b3_f1 + ceafe_f1) / 3.0;
        let zero_p = zero_tp as f64 / zero_pred.max(1) as f64;
        let zero_r = zero_tp as f64 / zero_gold.max(1) as f64;
        let zero_f1 = f1(zero_p, zero_r);

        eprintln!(
            "CALIBRATION: MUC={:.4} B3={:.4} CEAFe={:.4} CoNLL_F1={:.4} | Zero P={:.4} R={:.4} F1={:.4} (tp={} pred={} gold={})",
            muc_f1,
            b3_f1,
            ceafe_f1,
            conll_f1,
            zero_p,
            zero_r,
            zero_f1,
            zero_tp,
            zero_pred,
            zero_gold
        );
        assert!(conll_f1 > 0.50, "CoNLL F1 too low: {conll_f1}");
        assert!(zero_f1 > 0.40, "Zero F1 too low: {zero_f1}");
    }
}
