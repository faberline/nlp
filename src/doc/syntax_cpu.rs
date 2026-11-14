//! Pure-Rust zero-model CPU Layer-4 UPOS disambiguator and UD v2 dependency parser
//! for Chinese (`zh-Hant` + `zh-Hans`), English (`en`), and `mixed` sentences.

use std::collections::{HashMap, HashSet};

use crate::Span;
use crate::doc::syntax::{DEPRELS, SYNTAX_THRESHOLD_MILLI, Score, SyntaxToken, chu_liu_edmonds};
use crate::doc::{Document, EntityKind, Lang, Reason, Upos, sort_unknown};
use crate::lexicon::{LexTokenAnalysis, SemanticClass, en, zh};

const MAX_WINDOW_TOKENS: usize = 96;

#[inline]
const fn upos_bit(u: Upos) -> u32 {
    match u {
        Upos::Adj => 1 << 0,
        Upos::Adp => 1 << 1,
        Upos::Adv => 1 << 2,
        Upos::Aux => 1 << 3,
        Upos::Cconj => 1 << 4,
        Upos::Det => 1 << 5,
        Upos::Intj => 1 << 6,
        Upos::Noun => 1 << 7,
        Upos::Num => 1 << 8,
        Upos::Part => 1 << 9,
        Upos::Pron => 1 << 10,
        Upos::Propn => 1 << 11,
        Upos::Punct => 1 << 12,
        Upos::Sconj => 1 << 13,
        Upos::Sym => 1 << 14,
        Upos::Verb => 1 << 15,
        Upos::X => 1 << 16,
    }
}

#[inline]
const fn has_upos(mask: u32, u: Upos) -> bool {
    (mask & upos_bit(u)) != 0
}

fn char_to_byte_table(text: &str) -> Vec<usize> {
    let mut bytes = Vec::with_capacity(text.chars().count() + 1);
    for (byte, _) in text.char_indices() {
        bytes.push(byte);
    }
    bytes.push(text.len());
    bytes
}

#[inline]
fn span_text<'a>(text: &'a str, char_to_byte: &[usize], span: Span) -> &'a str {
    let s = char_to_byte
        .get(span.start_char)
        .copied()
        .unwrap_or(text.len());
    let e = char_to_byte
        .get(span.end_char)
        .copied()
        .unwrap_or(text.len());
    if s <= e && e <= text.len() {
        &text[s..e]
    } else {
        ""
    }
}

fn is_unsupported_sentence(doc: &Document, span: Span) -> bool {
    doc.unknown.iter().any(|u| {
        u.layer == "segmentation" && u.reason == Reason::UnsupportedLanguage && u.span == span
    })
}

fn is_kana_or_hangul(ch: char) -> bool {
    matches!(
        ch,
        '\u{3040}'..='\u{309F}'
            | '\u{30A0}'..='\u{30FF}'
            | '\u{31F0}'..='\u{31FF}'
            | '\u{FF66}'..='\u{FF9F}'
            | '\u{1100}'..='\u{11FF}'
            | '\u{3130}'..='\u{318F}'
            | '\u{AC00}'..='\u{D7AF}'
    )
}

fn is_ja_or_ko_sentence(
    doc: &Document,
    sent_text: &str,
    sent_idx: usize,
    tok_indices: &[usize],
) -> bool {
    if let Some(sent) = doc.segmentation.sentences.get(sent_idx)
        && matches!(sent.lang, Lang::Ja | Lang::Ko)
    {
        return true;
    }
    tok_indices
        .iter()
        .any(|&idx| matches!(doc.tokens[idx].lang, Lang::Ja | Lang::Ko))
        || sent_text.chars().any(is_kana_or_hangul)
}

/// Annotate Layer 4 (`Token.upos` upgrade + `Document.syntax`) for all supported sentences
/// on CPU without any external neural model.
pub fn annotate(doc: &mut Document, text: &str) {
    let char_to_byte = char_to_byte_table(text);
    doc.syntax.clear();

    let n_sents = doc.segmentation.sentences.len();
    let mut tokens_by_sent: Vec<Vec<usize>> = vec![Vec::new(); n_sents];
    for (idx, tok) in doc.tokens.iter().enumerate() {
        if let Some(bucket) = tokens_by_sent.get_mut(tok.sentence) {
            bucket.push(idx);
        }
    }

    let mut clauses_by_sent: Vec<Vec<Span>> = vec![Vec::new(); n_sents];
    for clause in &doc.segmentation.clauses {
        if let Some(bucket) = clauses_by_sent.get_mut(clause.sentence) {
            bucket.push(clause.span);
        }
    }

    let mut names_by_sent: Vec<Vec<Span>> = vec![Vec::new(); n_sents];
    for ent in &doc.entities {
        if ent.kind == EntityKind::Name
            && let Some(bucket) = names_by_sent.get_mut(ent.sentence)
        {
            bucket.push(ent.span);
        }
    }

    let mut cleared_pos_spans: HashSet<(usize, usize)> = HashSet::new();
    for sent_idx in 0..n_sents {
        let sent_span = doc.segmentation.sentences[sent_idx].span;
        if is_unsupported_sentence(doc, sent_span) {
            continue;
        }
        let tok_indices = &tokens_by_sent[sent_idx];
        if tok_indices.is_empty() {
            continue;
        }
        let sent_text = span_text(text, &char_to_byte, sent_span);
        if is_ja_or_ko_sentence(doc, sent_text, sent_idx, tok_indices) {
            crate::doc::syntax_cpu_jako::annotate_sentence(doc, text, sent_idx);
            continue;
        }
        annotate_sentence_zhen_inner(
            doc,
            text,
            &char_to_byte,
            sent_idx,
            tok_indices,
            &clauses_by_sent[sent_idx],
            &names_by_sent[sent_idx],
            &mut cleared_pos_spans,
        );
    }

    doc.syntax.sort_by_key(|s| s.token);
    if !cleared_pos_spans.is_empty() {
        doc.unknown.retain(|u| {
            !(u.layer == "pos" && cleared_pos_spans.contains(&(u.span.start_char, u.span.end_char)))
        });
    }
    sort_unknown(&mut doc.unknown);
}

/// Standalone sentence entry point for `zh`, `en`, and `mixed` sentences.
#[allow(dead_code)]
pub(crate) fn annotate_sentence_zhen(doc: &mut Document, text: &str, sent_idx: usize) {
    let Some(sent) = doc.segmentation.sentences.get(sent_idx) else {
        return;
    };
    if is_unsupported_sentence(doc, sent.span) {
        return;
    }
    let char_to_byte = char_to_byte_table(text);
    let tok_indices: Vec<usize> = doc
        .tokens
        .iter()
        .enumerate()
        .filter_map(|(i, t)| (t.sentence == sent_idx).then_some(i))
        .collect();
    if tok_indices.is_empty() {
        return;
    }
    let clause_spans: Vec<Span> = doc
        .segmentation
        .clauses
        .iter()
        .filter(|c| c.sentence == sent_idx)
        .map(|c| c.span)
        .collect();
    let name_spans: Vec<Span> = doc
        .entities
        .iter()
        .filter(|e| e.sentence == sent_idx && e.kind == EntityKind::Name)
        .map(|e| e.span)
        .collect();
    let mut cleared = HashSet::new();
    annotate_sentence_zhen_inner(
        doc,
        text,
        &char_to_byte,
        sent_idx,
        &tok_indices,
        &clause_spans,
        &name_spans,
        &mut cleared,
    );
    if !cleared.is_empty() {
        doc.unknown.retain(|u| {
            !(u.layer == "pos" && cleared.contains(&(u.span.start_char, u.span.end_char)))
        });
        sort_unknown(&mut doc.unknown);
    }
}

#[derive(Clone, Debug)]
struct TokInfo<'a> {
    global_idx: usize,
    span: Span,
    surf: &'a str,
    lower: String,
    lang: Lang,
    is_name: bool,
    lex: LexTokenAnalysis,
    upos: Upos,
}

fn is_zh_classifier(s: &str) -> bool {
    matches!(
        s,
        "個" | "个"
            | "本"
            | "隻"
            | "只"
            | "張"
            | "张"
            | "位"
            | "名"
            | "次"
            | "種"
            | "种"
            | "組"
            | "组"
            | "台"
            | "輛"
            | "辆"
            | "雙"
            | "双"
            | "條"
            | "条"
            | "件"
            | "款"
            | "支"
            | "座"
            | "篇"
            | "封"
            | "杯"
            | "碗"
            | "瓶"
            | "盒"
            | "袋"
            | "束"
            | "朵"
            | "棵"
            | "顆"
            | "颗"
            | "塊"
            | "块"
            | "片"
            | "段"
            | "場"
            | "场"
            | "陣"
            | "阵"
            | "批"
            | "套"
            | "把"
            | "家"
            | "間"
            | "间"
            | "所"
            | "棟"
            | "栋"
            | "層"
            | "层"
            | "頁"
            | "页"
            | "行"
            | "句"
            | "首"
            | "門"
            | "门"
            | "項"
            | "项"
            | "輪"
            | "轮"
            | "班"
            | "趟"
            | "回"
            | "遍"
            | "頓"
            | "顿"
            | "口"
            | "頭"
            | "头"
            | "匹"
            | "尾"
            | "架"
            | "艘"
            | "列"
            | "節"
            | "节"
            | "步"
            | "份"
            | "副"
            | "對"
            | "对"
            | "群"
            | "堆"
            | "排"
            | "串"
            | "滴"
            | "根"
            | "粒"
            | "枚"
            | "盞"
            | "盏"
            | "頂"
            | "顶"
            | "扇"
            | "面"
            | "幅"
            | "道"
    )
}

fn is_zh_localizer(s: &str) -> bool {
    matches!(
        s,
        "上" | "下"
            | "裡"
            | "里"
            | "中"
            | "內"
            | "内"
            | "外"
            | "前"
            | "後"
            | "后"
            | "旁"
            | "邊"
            | "边"
            | "之前"
            | "之後"
            | "之后"
            | "以前"
            | "以後"
            | "以后"
            | "之間"
            | "之间"
            | "之上"
            | "之下"
            | "之中"
            | "以外"
            | "以內"
            | "以内"
    )
}

fn is_zh_num_det_merged(s: &str) -> Option<Upos> {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() < 2 {
        return None;
    }
    let first = chars[0];
    let rest: String = chars[1..].iter().collect();
    if is_zh_classifier(&rest) {
        if matches!(
            first,
            '這' | '这' | '那' | '每' | '各' | '某' | '哪' | '該' | '本' | '此'
        ) {
            return Some(Upos::Det);
        }
        if matches!(
            first,
            '一' | '二'
                | '兩'
                | '两'
                | '三'
                | '四'
                | '五'
                | '六'
                | '七'
                | '八'
                | '九'
                | '十'
                | '幾'
                | '几'
                | '數'
                | '数'
                | '半'
        ) {
            return Some(Upos::Num);
        }
    }
    if chars.iter().all(|c| {
        matches!(
            c,
            '〇' | '零'
                | '一'
                | '二'
                | '三'
                | '四'
                | '五'
                | '六'
                | '七'
                | '八'
                | '九'
                | '十'
                | '百'
                | '千'
                | '萬'
                | '万'
                | '億'
                | '亿'
                | '兩'
                | '两'
                | '第'
        )
    }) {
        return Some(Upos::Num);
    }
    None
}

fn is_zh_char(c: char) -> bool {
    ('\u{4E00}'..='\u{9FFF}').contains(&c)
        || ('\u{3400}'..='\u{4DBF}').contains(&c)
        || ('\u{F900}'..='\u{FAFF}').contains(&c)
}

fn token_is_cjk(surf: &str, lang: Lang) -> bool {
    lang == Lang::Zh || surf.chars().any(is_zh_char)
}

fn is_en_poss_pron(lower: &str) -> bool {
    matches!(
        lower,
        "my" | "your" | "his" | "her" | "its" | "our" | "their"
    )
}

fn is_en_irregular_verb(lower: &str) -> bool {
    matches!(
        lower,
        "sat"
            | "found"
            | "led"
            | "let"
            | "kept"
            | "held"
            | "fell"
            | "felt"
            | "left"
            | "met"
            | "built"
            | "sent"
            | "spent"
            | "lent"
            | "bent"
            | "slept"
            | "swept"
            | "wept"
            | "crept"
            | "brought"
            | "bought"
            | "caught"
            | "taught"
            | "thought"
            | "fought"
            | "sought"
            | "wrote"
            | "spoke"
            | "broke"
            | "chose"
            | "froze"
            | "stole"
            | "woke"
            | "drove"
            | "rode"
            | "rose"
            | "shone"
            | "wore"
            | "tore"
            | "bore"
            | "swore"
            | "drew"
            | "blew"
            | "flew"
            | "grew"
            | "knew"
            | "threw"
            | "gave"
            | "forgave"
            | "took"
            | "shook"
            | "saw"
            | "came"
            | "became"
            | "ran"
            | "began"
            | "drank"
            | "sang"
            | "sank"
            | "rang"
            | "swam"
            | "ate"
            | "bit"
            | "hid"
            | "slid"
            | "struck"
            | "dug"
            | "hung"
            | "spun"
            | "stuck"
            | "stung"
            | "swung"
            | "won"
            | "lost"
            | "sold"
            | "told"
            | "paid"
            | "said"
            | "laid"
            | "made"
            | "stood"
            | "understood"
            | "heard"
            | "spread"
            | "shut"
            | "hurt"
    )
}

fn en_morph_guess(surf: &str, lower: &str) -> Option<Upos> {
    if surf.len() >= 2
        && surf.chars().next().is_some_and(|c| c.is_ascii_uppercase())
        && surf.chars().skip(1).any(|c| c.is_ascii_lowercase())
    {
        // Might be proper noun or sentence-initial capitalized word; handled with context.
    }
    if lower.ends_with("ly") && lower.len() > 4 {
        return Some(Upos::Adv);
    }
    if lower.ends_with("tion")
        || lower.ends_with("sion")
        || lower.ends_with("ment")
        || lower.ends_with("ness")
        || lower.ends_with("ity")
        || lower.ends_with("ance")
        || lower.ends_with("ence")
    {
        return Some(Upos::Noun);
    }
    if lower.ends_with("able")
        || lower.ends_with("ible")
        || lower.ends_with("ous")
        || lower.ends_with("ive")
        || lower.ends_with("ful")
        || lower.ends_with("less")
    {
        return Some(Upos::Adj);
    }
    if lower.ends_with("ize") || lower.ends_with("ise") || lower.ends_with("ate") {
        return Some(Upos::Verb);
    }
    if lower.ends_with("ed") && lower.len() > 4 {
        return Some(Upos::Verb);
    }
    if lower.ends_with("ing") && lower.len() > 5 {
        return Some(Upos::Verb);
    }
    None
}

fn lexical_prior_upos(
    surf: &str,
    lower: &str,
    lang: Lang,
    lex: &LexTokenAnalysis,
    is_name: bool,
) -> Upos {
    if is_name {
        return Upos::Propn;
    }
    if let Some(u) = lex.decided_upos {
        return u;
    }
    if token_is_cjk(surf, lang) {
        if let Some(u) = is_zh_num_det_merged(surf) {
            return u;
        }
        // Simplified Chinese function words left None by Layer 2
        match surf {
            "我们" | "你们" | "他们" | "她们" | "它们" | "咱们" | "谁" | "这里" | "那里" =>
            {
                return Upos::Pron;
            }
            "因为" | "虽然" | "尽管" | "假如" | "如果" | "只要" | "除非" | "既然" | "一旦" =>
            {
                return Upos::Sconj;
            }
            "并且" | "而且" | "或者" | "但是" | "然而" | "可是" => return Upos::Cconj,
            "该" | "本" | "此" | "每" | "各" | "某" => return Upos::Det,
            "已经" | "曾经" | "刚刚" | "马上" | "终于" | "一直" | "非常" | "十分" | "格外"
            | "渐渐" => {
                return Upos::Adv;
            }
            "对于" | "关于" | "为了" | "按照" | "根据" | "除了" | "沿着" | "沿著" | "通过"
            | "通過" | "透過" => {
                return Upos::Adp;
            }
            "能够" | "应该" | "必须" | "可以" | "将要" | "將要" => return Upos::Aux,
            "吗" | "呢" | "吧" | "啊" | "啦" | "喔" | "呀" => return Upos::Part,
            "个" | "個" | "只" | "隻" | "张" | "張" | "组" | "組" | "种" | "種" | "辆" | "輛"
            | "双" | "雙" | "条" | "條" => {
                return Upos::Noun;
            }
            _ => {}
        }
        if has_upos(lex.upos_mask, Upos::Verb) && !has_upos(lex.upos_mask, Upos::Noun) {
            return Upos::Verb;
        }
        if has_upos(lex.upos_mask, Upos::Noun) && !has_upos(lex.upos_mask, Upos::Verb) {
            return Upos::Noun;
        }
        if has_upos(lex.upos_mask, Upos::Adj)
            && !has_upos(lex.upos_mask, Upos::Verb)
            && !has_upos(lex.upos_mask, Upos::Noun)
        {
            return Upos::Adj;
        }
        if has_upos(lex.upos_mask, Upos::Adv)
            && !has_upos(lex.upos_mask, Upos::Verb)
            && !has_upos(lex.upos_mask, Upos::Noun)
        {
            return Upos::Adv;
        }
        if matches!(
            lex.sem_class,
            Some(SemanticClass::Person | SemanticClass::Location | SemanticClass::Organization)
        ) && has_upos(lex.upos_mask, Upos::Propn)
        {
            return Upos::Propn;
        }
        if has_upos(lex.upos_mask, Upos::Noun) {
            return Upos::Noun;
        }
        if has_upos(lex.upos_mask, Upos::Verb) {
            return Upos::Verb;
        }
        if has_upos(lex.upos_mask, Upos::Adj) {
            return Upos::Adj;
        }
        if has_upos(lex.upos_mask, Upos::Adv) {
            return Upos::Adv;
        }
        return Upos::Noun;
    }

    // English / Latin
    if matches!(
        lower,
        "for"
            | "near"
            | "up"
            | "down"
            | "behind"
            | "above"
            | "along"
            | "off"
            | "out"
            | "around"
            | "over"
            | "under"
            | "through"
            | "across"
            | "against"
            | "between"
            | "during"
            | "without"
            | "within"
            | "upon"
            | "toward"
            | "towards"
            | "about"
            | "into"
            | "onto"
            | "by"
            | "with"
            | "from"
            | "at"
            | "on"
            | "in"
            | "of"
            | "via"
            | "per"
            | "despite"
            | "among"
            | "beside"
            | "besides"
            | "beyond"
            | "inside"
            | "outside"
            | "beneath"
            | "underneath"
            | "throughout"
            | "amid"
    ) {
        return Upos::Adp;
    }
    if matches!(
        lower,
        "only"
            | "just"
            | "even"
            | "already"
            | "still"
            | "also"
            | "soon"
            | "never"
            | "always"
            | "often"
            | "sometimes"
    ) {
        return Upos::Adv;
    }
    if matches!(
        lower,
        "this"
            | "that"
            | "these"
            | "those"
            | "some"
            | "any"
            | "no"
            | "all"
            | "both"
            | "either"
            | "neither"
            | "another"
    ) {
        return Upos::Det;
    }
    if matches!(lower, "who" | "whom" | "whose" | "which" | "what") {
        return Upos::Pron;
    }
    if matches!(lower, "do" | "does" | "did" | "have" | "has" | "had") {
        return Upos::Aux;
    }
    if matches!(lower, "to") {
        return Upos::Part;
    }
    if matches!(
        lower,
        "as" | "than" | "before" | "after" | "until" | "since" | "while"
    ) {
        return Upos::Adp;
    }
    if matches!(lower, "when" | "where" | "why" | "how" | "so" | "yet") {
        return Upos::Adv;
    }
    if lower == "'s" {
        return Upos::Part;
    }
    if lower == "'d" {
        return Upos::Aux;
    }

    if is_en_irregular_verb(lower) {
        return Upos::Verb;
    }
    // Check lexicon mask
    if has_upos(lex.upos_mask, Upos::Adp)
        && !has_upos(lex.upos_mask, Upos::Verb)
        && !has_upos(lex.upos_mask, Upos::Noun)
    {
        return Upos::Adp;
    }
    if has_upos(lex.upos_mask, Upos::Adj)
        && !has_upos(lex.upos_mask, Upos::Noun)
        && !has_upos(lex.upos_mask, Upos::Verb)
    {
        return Upos::Adj;
    }
    if has_upos(lex.upos_mask, Upos::Noun)
        && !has_upos(lex.upos_mask, Upos::Verb)
        && !has_upos(lex.upos_mask, Upos::Adj)
    {
        return Upos::Noun;
    }
    if has_upos(lex.upos_mask, Upos::Verb)
        && !has_upos(lex.upos_mask, Upos::Noun)
        && !has_upos(lex.upos_mask, Upos::Adj)
    {
        return Upos::Verb;
    }
    if let Some(u) = en_morph_guess(surf, lower) {
        return u;
    }
    if has_upos(lex.upos_mask, Upos::Adj)
        && matches!(
            lower,
            "old"
                | "last"
                | "narrow"
                | "quiet"
                | "new"
                | "small"
                | "large"
                | "long"
                | "short"
                | "high"
                | "low"
                | "good"
                | "bad"
                | "warm"
                | "cold"
                | "hot"
                | "cool"
                | "clean"
                | "dry"
                | "wet"
                | "heavy"
                | "light"
                | "dark"
                | "bright"
                | "soft"
                | "hard"
                | "fast"
                | "slow"
                | "early"
                | "late"
                | "full"
                | "empty"
                | "open"
                | "clear"
                | "stable"
                | "simple"
                | "valid"
                | "invalid"
                | "raw"
                | "dense"
                | "fresh"
                | "local"
                | "final"
                | "total"
                | "exact"
                | "main"
                | "wide"
                | "deep"
                | "thin"
                | "thick"
                | "flat"
                | "sharp"
                | "smooth"
                | "rough"
                | "loud"
                | "silent"
                | "busy"
                | "free"
                | "ready"
                | "safe"
                | "tall"
                | "young"
        )
    {
        return Upos::Adj;
    }
    if has_upos(lex.upos_mask, Upos::Noun) {
        return Upos::Noun;
    }
    if has_upos(lex.upos_mask, Upos::Verb) {
        return Upos::Verb;
    }
    if has_upos(lex.upos_mask, Upos::Adj) {
        return Upos::Adj;
    }
    if has_upos(lex.upos_mask, Upos::Adv) {
        return Upos::Adv;
    }
    if surf.chars().next().is_some_and(|c| c.is_ascii_uppercase())
        && surf
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        && surf.len() >= 2
        && surf.chars().all(|c| !c.is_ascii_lowercase())
    {
        // All-caps acronym like SSD, USB-C, API -> NOUN or PROPN
        return Upos::Noun;
    }
    Upos::Noun
}

fn span_overlaps_any(span: Span, spans: &[Span]) -> bool {
    spans
        .iter()
        .any(|s| span.start_char < s.end_char && s.start_char < span.end_char)
}

#[allow(clippy::too_many_arguments)]
fn annotate_sentence_zhen_inner(
    doc: &mut Document,
    text: &str,
    char_to_byte: &[usize],
    sent_idx: usize,
    tok_indices: &[usize],
    clause_spans: &[Span],
    name_spans: &[Span],
    cleared_pos_spans: &mut HashSet<(usize, usize)>,
) {
    let mut infos: Vec<TokInfo<'_>> = Vec::with_capacity(tok_indices.len());
    for &idx in tok_indices {
        let tok = &doc.tokens[idx];
        let surf = span_text(text, char_to_byte, tok.span);
        let lower = surf.to_ascii_lowercase().replace('\u{2019}', "'");
        let is_name = span_overlaps_any(tok.span, name_spans);
        let lex = if tok.upos.is_none() {
            if token_is_cjk(surf, tok.lang) {
                zh::analyze_zh_token(surf)
            } else {
                en::analyze_en_token(surf)
            }
        } else {
            LexTokenAnalysis::default()
        };
        let initial_upos = tok
            .upos
            .unwrap_or_else(|| lexical_prior_upos(surf, &lower, tok.lang, &lex, is_name));
        infos.push(TokInfo {
            global_idx: idx,
            span: tok.span,
            surf,
            lower,
            lang: tok.lang,
            is_name,
            lex,
            upos: initial_upos,
        });
    }

    // Step A: Contextual UPOS disambiguation (only for tokens where Layer 2 had upos == None)
    disambiguate_upos(doc, &mut infos);
    // Second pass for Aux/Adp/Adv whose right context was just disambiguated to Verb/Noun
    disambiguate_upos(doc, &mut infos);

    // Write back UPOS and record cleared pos unknowns
    for info in &infos {
        let tok = &mut doc.tokens[info.global_idx];
        if tok.upos.is_none() {
            tok.upos = Some(info.upos);
            cleared_pos_spans.insert((info.span.start_char, info.span.end_char));
        }
    }

    // Step B: Dependency decoding
    let n = infos.len();
    if n == 0 {
        return;
    }

    let syntax_tokens = if n <= MAX_WINDOW_TOKENS {
        decode_window(&infos, 0, n, sent_idx)
    } else {
        decode_windowed(&infos, clause_spans, sent_idx)
    };

    doc.syntax.extend(syntax_tokens);
}

fn disambiguate_upos(doc: &Document, infos: &mut [TokInfo<'_>]) {
    let n = infos.len();

    // Pass 1: Clear lexical / morphological / local Chinese & English disambiguation
    for i in 0..n {
        if doc.tokens[infos[i].global_idx].upos.is_some() {
            continue;
        }
        let surf = infos[i].surf;
        let lower = infos[i].lower.as_str();
        let mask = infos[i].lex.upos_mask;

        // Proper noun heuristic: capitalized non-sentence-initial English word not in closed class
        if !token_is_cjk(surf, infos[i].lang)
            && !infos[i].is_name
            && i > 0
            && infos[i - 1].upos != Upos::Punct
            && surf.chars().next().is_some_and(|c| c.is_ascii_uppercase())
            && surf.chars().skip(1).any(|c| c.is_ascii_lowercase())
            && infos[i].lex.decided_upos.is_none()
            && (mask == 0 || has_upos(mask, Upos::Propn))
        {
            infos[i].upos = Upos::Propn;
            continue;
        }

        // Chinese contextual rules
        if token_is_cjk(surf, infos[i].lang) {
            match surf {
                "這" | "这" | "那" | "每" | "各" | "某" => {
                    let next_is_np_or_clf = i + 1 < n
                        && (is_zh_classifier(infos[i + 1].surf)
                            || matches!(
                                infos[i + 1].upos,
                                Upos::Noun | Upos::Num | Upos::Adj | Upos::Propn
                            ));
                    infos[i].upos = if next_is_np_or_clf {
                        Upos::Det
                    } else {
                        Upos::Pron
                    };
                }
                "許多" | "许多" => {
                    infos[i].upos = Upos::Num;
                }
                "這個" | "这个" | "那個" | "那个" => {
                    let next_is_noun = i + 1 < n
                        && matches!(infos[i + 1].upos, Upos::Noun | Upos::Adj | Upos::Propn);
                    infos[i].upos = if next_is_noun { Upos::Det } else { Upos::Pron };
                }
                "什麼" | "什么" | "甚麼" => {
                    let next_is_noun = i + 1 < n
                        && matches!(infos[i + 1].upos, Upos::Noun | Upos::Propn | Upos::Adj);
                    infos[i].upos = if next_is_noun { Upos::Det } else { Upos::Pron };
                }
                "著" | "着" | "過" | "过" => {
                    if i > 0 && infos[i - 1].upos == Upos::Verb {
                        infos[i].upos = Upos::Part;
                    }
                }
                "得" => {
                    if i > 0 && infos[i - 1].upos == Upos::Verb {
                        infos[i].upos = Upos::Part;
                    } else if i + 1 < n && infos[i + 1].upos == Upos::Verb {
                        infos[i].upos = Upos::Aux;
                    }
                }
                "地" => {
                    if i > 0 && matches!(infos[i - 1].upos, Upos::Adj | Upos::Adv) {
                        infos[i].upos = Upos::Part;
                    }
                }
                "是" => {
                    infos[i].upos = Upos::Aux;
                }
                "會" | "会" | "要" | "將要" | "将要" | "能" | "可以" | "應該" | "应该" =>
                {
                    let has_later_vp = (i + 1..n).take(4).any(|j| {
                        matches!(infos[j].upos, Upos::Verb | Upos::Adj | Upos::Aux)
                            || (has_upos(infos[j].lex.upos_mask, Upos::Verb)
                                && j + 1 < n
                                && matches!(
                                    infos[j + 1].upos,
                                    Upos::Noun | Upos::Pron | Upos::Det | Upos::Num | Upos::X
                                ))
                    });
                    if has_later_vp {
                        infos[i].upos = Upos::Aux;
                    } else if surf == "會" || surf == "会" || surf == "要" {
                        infos[i].upos = Upos::Verb;
                    }
                }
                "把" | "將" | "将" => {
                    let has_np_then_vp = has_following_np_and_vp(infos, i);
                    infos[i].upos = if has_np_then_vp {
                        Upos::Adp
                    } else {
                        Upos::Verb
                    };
                }
                "被" => {
                    infos[i].upos = Upos::Adp;
                }
                "在" => {
                    let prev_is_verb = i > 0 && infos[i - 1].upos == Upos::Verb;
                    let has_np_then_vp = has_following_np_and_vp(infos, i);
                    let next_is_np = i + 1 < n
                        && matches!(
                            infos[i + 1].upos,
                            Upos::Noun
                                | Upos::Propn
                                | Upos::Pron
                                | Upos::Det
                                | Upos::Num
                                | Upos::Adj
                                | Upos::X
                        );
                    if has_np_then_vp || (prev_is_verb && next_is_np) {
                        infos[i].upos = Upos::Adp;
                    } else if i + 1 < n && infos[i + 1].upos == Upos::Verb {
                        infos[i].upos = Upos::Adv;
                    } else if next_is_np {
                        // Check if there is a localizer like 上/裡/里/中/下 later in clause
                        let has_loc = (i + 1..n)
                            .take_while(|&k| infos[k].upos != Upos::Punct)
                            .any(|k| is_zh_localizer(infos[k].surf));
                        infos[i].upos = if has_loc { Upos::Adp } else { Upos::Verb };
                    } else {
                        infos[i].upos = Upos::Verb;
                    }
                }
                "從" | "从" | "對" | "对" | "向" | "往" | "用" | "以" | "按" | "通過" | "通过"
                | "透過" | "沿著" | "沿着" | "到" | "給" | "给" | "比" | "經過" | "经过" | "由" => {
                    if has_following_np_and_vp(infos, i)
                        || matches!(
                            surf,
                            "從" | "从"
                                | "沿著"
                                | "沿着"
                                | "通過"
                                | "通过"
                                | "透過"
                                | "對"
                                | "对"
                                | "向"
                                | "往"
                                | "經過"
                                | "经过"
                                | "由"
                        )
                    {
                        infos[i].upos = Upos::Adp;
                    }
                }
                "大於" | "大于" | "小於" | "小于" | "等於" | "等于" | "高於" | "高于" | "低於"
                | "低于" | "多於" | "多于" | "少於" | "少于" => {
                    infos[i].upos = Upos::Adj;
                }
                "和" | "與" | "与" | "跟" => {
                    if i > 0
                        && i + 1 < n
                        && matches!(
                            infos[i - 1].upos,
                            Upos::Noun | Upos::Propn | Upos::Pron | Upos::X
                        )
                        && matches!(
                            infos[i + 1].upos,
                            Upos::Noun
                                | Upos::Propn
                                | Upos::Pron
                                | Upos::Det
                                | Upos::Num
                                | Upos::Adj
                                | Upos::X
                        )
                    {
                        infos[i].upos = Upos::Cconj;
                    } else {
                        infos[i].upos = Upos::Adp;
                    }
                }
                "沒" | "没" | "沒有" | "没有" => {
                    // 沒/沒有 is ADV before VERB, or before predicative ADJ (not ADJ + NOUN).
                    // Before NP (including ADJ + NOUN), 沒有 is VERB!
                    let next_is_verb = i + 1 < n && infos[i + 1].upos == Upos::Verb;
                    let next_is_pred_adj = i + 1 < n
                        && infos[i + 1].upos == Upos::Adj
                        && (i + 2 >= n
                            || !matches!(infos[i + 2].upos, Upos::Noun | Upos::Propn | Upos::Adj));
                    infos[i].upos = if next_is_verb || next_is_pred_adj {
                        Upos::Adv
                    } else {
                        Upos::Verb
                    };
                }
                "還" | "还" => {
                    let next_is_vp = i + 1 < n
                        && matches!(
                            infos[i + 1].upos,
                            Upos::Verb | Upos::Adj | Upos::Adv | Upos::Aux
                        );
                    infos[i].upos = if next_is_vp { Upos::Adv } else { Upos::Verb };
                }
                "所以" | "因此" | "於是" | "于是" => {
                    infos[i].upos = Upos::Sconj;
                }
                "幾" | "几" => {
                    infos[i].upos = Upos::Num;
                }
                _ => {
                    // Localizer check: 上/下/裡/里/中/內/外/前/後/后/之前/之後/之后
                    let next_is_aspect = i + 1 < n
                        && matches!(
                            infos[i + 1].surf,
                            "了" | "著" | "着" | "過" | "过" | "去" | "來" | "来"
                        );
                    let next_is_noun =
                        i + 1 < n && matches!(infos[i + 1].upos, Upos::Noun | Upos::Propn);
                    let is_temp_loc = matches!(
                        surf,
                        "後" | "后" | "前" | "之前" | "之後" | "之后" | "以前" | "以後" | "以后"
                    );
                    if is_zh_localizer(surf)
                        && !next_is_aspect
                        && !next_is_noun
                        && i > 0
                        && (matches!(
                            infos[i - 1].upos,
                            Upos::Noun | Upos::Propn | Upos::Pron | Upos::X
                        ) || (is_temp_loc && infos[i - 1].upos == Upos::Verb))
                    {
                        infos[i].upos = Upos::Adp;
                    } else if has_upos(mask, Upos::Adj)
                        && i + 1 < n
                        && (infos[i + 1].upos == Upos::Verb
                            || has_upos(infos[i + 1].lex.upos_mask, Upos::Verb))
                        && matches!(
                            surf,
                            "完全"
                                | "多"
                                | "仔細"
                                | "仔细"
                                | "明顯"
                                | "明显"
                                | "快速"
                                | "迅速"
                                | "輕輕"
                                | "轻轻"
                                | "慢慢"
                                | "隨機"
                                | "随机"
                                | "統一"
                                | "统一"
                        )
                    {
                        infos[i].upos = Upos::Adv;
                    } else if has_upos(mask, Upos::Verb) && has_upos(mask, Upos::Noun) {
                        // Look back up to 2 tokens within the same clause (stop at PUNCT)
                        let prev_is_verb_cue = (1..=2usize).any(|back| {
                            i >= back
                                && !(back == 2 && infos[i - 1].upos == Upos::Punct)
                                && (matches!(infos[i - back].upos, Upos::Aux | Upos::Adv)
                                    || (back == 1 && infos[i - 1].upos == Upos::Pron)
                                    || matches!(infos[i - back].surf, "來" | "来" | "去"))
                        });
                        let next_is_aspect = i + 1 < n
                            && matches!(
                                infos[i + 1].surf,
                                "了" | "著" | "着" | "過" | "过" | "完" | "好" | "到" | "出" | "起"
                            );
                        let next_is_verb_or_aspect = next_is_aspect
                            || (i + 1 < n
                                && (infos[i + 1].surf == "在" || infos[i + 1].upos == Upos::Verb));
                        let is_nominal_compound_head = matches!(
                            surf,
                            "實驗"
                                | "实验"
                                | "研究"
                                | "觀測"
                                | "观测"
                                | "變化"
                                | "变化"
                                | "反應"
                                | "反应"
                                | "結果"
                                | "结果"
                                | "測試"
                                | "测试"
                                | "記錄"
                                | "记录"
                                | "迴歸"
                                | "回归"
                        ) && i + 1 < n
                            && matches!(infos[i + 1].upos, Upos::Noun | Upos::Propn);
                        let prev_is_det_or_de = i > 0
                            && (matches!(infos[i - 1].upos, Upos::Det | Upos::Num | Upos::Adj)
                                || infos[i - 1].surf == "的"
                                || (matches!(
                                    infos[i - 1].surf,
                                    "把" | "將" | "将" | "被" | "在" | "由" | "經過" | "经过"
                                ) && i + 1 < n
                                    && matches!(infos[i + 1].upos, Upos::Noun | Upos::Propn)));
                        let is_strong_zh_verb = matches!(
                            surf,
                            "使用"
                                | "利用"
                                | "採用"
                                | "采用"
                                | "介紹"
                                | "介绍"
                                | "比較"
                                | "比较"
                                | "檢查"
                                | "检查"
                                | "完成"
                                | "沖洗"
                                | "冲洗"
                                | "生成"
                                | "處理"
                                | "处理"
                                | "測量"
                                | "测量"
                                | "保存"
                                | "結束"
                                | "结束"
                        ) && !prev_is_det_or_de
                            && (i + 1 < n
                                && matches!(
                                    infos[i + 1].upos,
                                    Upos::Noun
                                        | Upos::Propn
                                        | Upos::Pron
                                        | Upos::Det
                                        | Upos::Num
                                        | Upos::Adj
                                        | Upos::X
                                )
                                || (i + 1 == n || infos[i + 1].upos == Upos::Punct));
                        if next_is_aspect {
                            infos[i].upos = Upos::Verb;
                        } else if prev_is_det_or_de {
                            infos[i].upos = Upos::Noun;
                        } else if prev_is_verb_cue || next_is_verb_or_aspect || is_strong_zh_verb {
                            infos[i].upos = Upos::Verb;
                        } else if is_nominal_compound_head {
                            infos[i].upos = Upos::Noun;
                        }
                    }
                }
            }
        } else {
            // English contextual disambiguation
            match lower {
                "to" => {
                    let next_is_verb = i + 1 < n
                        && (infos[i + 1].upos == Upos::Verb
                            || has_upos(infos[i + 1].lex.upos_mask, Upos::Verb))
                        && !matches!(infos[i + 1].upos, Upos::Det | Upos::Pron | Upos::Propn);
                    infos[i].upos = if next_is_verb { Upos::Part } else { Upos::Adp };
                }
                "that" => {
                    let prev_is_verb = i > 0 && matches!(infos[i - 1].upos, Upos::Verb | Upos::Adj);
                    let prev_is_noun =
                        i > 0 && matches!(infos[i - 1].upos, Upos::Noun | Upos::Propn | Upos::Pron);
                    let next_is_np = i + 1 < n
                        && matches!(infos[i + 1].upos, Upos::Noun | Upos::Adj)
                        && !prev_is_verb;
                    if prev_is_verb {
                        infos[i].upos = Upos::Sconj;
                    } else if prev_is_noun
                        && i + 1 < n
                        && matches!(
                            infos[i + 1].upos,
                            Upos::Verb | Upos::Aux | Upos::Pron | Upos::Det | Upos::Noun
                        )
                    {
                        infos[i].upos = Upos::Pron;
                    } else if next_is_np {
                        infos[i].upos = Upos::Det;
                    } else {
                        infos[i].upos = Upos::Sconj;
                    }
                }
                "'s" => {
                    if i > 0 && infos[i - 1].lower == "let" {
                        infos[i].upos = Upos::Pron;
                    } else {
                        infos[i].upos = Upos::Part;
                    }
                }
                "before" | "after" | "until" | "since" | "while" => {
                    // SCONJ if followed by a clause with a verb before punct
                    let has_clause_verb = {
                        let mut saw_subj = false;
                        let mut found_v = false;
                        for tok in infos.iter().take(n).skip(i + 1) {
                            if tok.upos == Upos::Punct {
                                break;
                            }
                            if matches!(tok.upos, Upos::Pron | Upos::Noun | Upos::Propn | Upos::Det)
                            {
                                saw_subj = true;
                            } else if saw_subj && matches!(tok.upos, Upos::Verb | Upos::Aux) {
                                found_v = true;
                                break;
                            }
                        }
                        found_v
                    };
                    infos[i].upos = if has_clause_verb {
                        Upos::Sconj
                    } else {
                        Upos::Adp
                    };
                }
                "can" | "will" | "would" | "could" | "should" | "may" | "might" | "must"
                | "shall" => {
                    let next_is_vp = i + 1 < n
                        && matches!(infos[i + 1].upos, Upos::Verb | Upos::Adv | Upos::Part);
                    if next_is_vp {
                        infos[i].upos = Upos::Aux;
                    }
                }
                _ => {
                    let prev_is_det_or_poss = i > 0
                        && (matches!(
                            infos[i - 1].upos,
                            Upos::Det | Upos::Adj | Upos::Num | Upos::Adp
                        ) || (infos[i - 1].upos == Upos::Pron
                            && is_en_poss_pron(&infos[i - 1].lower)));
                    let next_is_noun =
                        i + 1 < n && matches!(infos[i + 1].upos, Upos::Noun | Upos::Propn);
                    let is_material_noun = matches!(
                        lower,
                        "stone"
                            | "brick"
                            | "wood"
                            | "glass"
                            | "paper"
                            | "iron"
                            | "steel"
                            | "metal"
                            | "plastic"
                            | "clay"
                            | "bamboo"
                            | "olive"
                    );
                    // Between DET/PossPron and NOUN: prefer ADJ if ADJ is in mask, else NOUN
                    if prev_is_det_or_poss
                        && next_is_noun
                        && has_upos(mask, Upos::Adj)
                        && !is_material_noun
                        && !matches!(infos[i].upos, Upos::Det | Upos::Num)
                    {
                        infos[i].upos = Upos::Adj;
                    } else if has_upos(mask, Upos::Noun) && has_upos(mask, Upos::Verb) {
                        if prev_is_det_or_poss {
                            infos[i].upos = Upos::Noun;
                        } else if i > 0
                            && (infos[i - 1].upos == Upos::Aux
                                || (infos[i - 1].upos == Upos::Pron
                                    && !is_en_poss_pron(&infos[i - 1].lower))
                                || (infos[i - 1].lower == "to" && infos[i - 1].upos == Upos::Part))
                        {
                            infos[i].upos = Upos::Verb;
                        }
                    }
                }
            }
        }
    }

    // Pass 3: If a clause has NO Verb and NO copula (Aux), promote a Noun|Verb token
    // that is preceded by a subject NP and followed by an object NP/Num/Det to VERB.
    let mut c_start = 0usize;
    for end in 0..=n {
        let is_boundary = end == n
            || (infos[end].upos == Upos::Punct
                && matches!(
                    infos[end].surf,
                    "," | "，" | ";" | "；" | "。" | "." | "!" | "！" | "?" | "？"
                ));
        if !is_boundary {
            continue;
        }
        let has_pred =
            (c_start..end).any(|k| infos[k].upos == Upos::Verb || infos[k].upos == Upos::Aux);
        if !has_pred && end - c_start >= 3 {
            // Find candidate Noun|Verb with a nominal before it and (nominal/num/det or end-of-clause) after it
            let mut promoted = false;
            for k in (c_start + 1)..(end - 1) {
                if doc.tokens[infos[k].global_idx].upos.is_some() {
                    continue;
                }
                if infos[k].upos == Upos::Noun
                    && has_upos(infos[k].lex.upos_mask, Upos::Verb)
                    && matches!(
                        infos[k - 1].upos,
                        Upos::Noun | Upos::Propn | Upos::Pron | Upos::X
                    )
                    && matches!(
                        infos[k + 1].upos,
                        Upos::Noun
                            | Upos::Propn
                            | Upos::Pron
                            | Upos::Num
                            | Upos::Det
                            | Upos::Adj
                            | Upos::X
                    )
                {
                    infos[k].upos = Upos::Verb;
                    promoted = true;
                    break;
                }
            }
            if !promoted && end > c_start + 1 {
                let last = end - 1;
                if doc.tokens[infos[last].global_idx].upos.is_none()
                    && infos[last].upos == Upos::Noun
                    && has_upos(infos[last].lex.upos_mask, Upos::Verb)
                {
                    infos[last].upos = Upos::Verb;
                }
            }
        }
        c_start = end + 1;
    }
}

fn has_following_np_and_vp(infos: &[TokInfo<'_>], start: usize) -> bool {
    let mut saw_np = false;
    for tok in infos.iter().skip(start + 1) {
        let u = tok.upos;
        if u == Upos::Punct {
            break;
        }
        if matches!(u, Upos::Noun | Upos::Propn | Upos::Pron | Upos::X) {
            saw_np = true;
        } else if saw_np && matches!(u, Upos::Verb | Upos::Adj) {
            return true;
        }
    }
    false
}

fn is_temporal_noun(info: &TokInfo<'_>) -> bool {
    info.lex.sem_class == Some(SemanticClass::Time)
        || matches!(
            info.lower.as_str(),
            "今天"
                | "昨天"
                | "明天"
                | "今晚"
                | "昨晚"
                | "明晚"
                | "清晨"
                | "早上"
                | "上午"
                | "中午"
                | "下午"
                | "傍晚"
                | "晚上"
                | "半夜"
                | "星期天"
                | "星期一"
                | "星期二"
                | "星期三"
                | "星期四"
                | "星期五"
                | "星期六"
                | "周末"
                | "週末"
                | "today"
                | "yesterday"
                | "tomorrow"
                | "tonight"
                | "morning"
                | "afternoon"
                | "evening"
                | "night"
                | "dawn"
                | "dusk"
                | "noon"
                | "midnight"
                | "monday"
                | "tuesday"
                | "wednesday"
                | "thursday"
                | "friday"
                | "saturday"
                | "sunday"
        )
}

fn decode_windowed(
    infos: &[TokInfo<'_>],
    clause_spans: &[Span],
    sent_idx: usize,
) -> Vec<SyntaxToken> {
    // Partition `infos` into windows of <= MAX_WINDOW_TOKENS aligned to clause_spans where possible.
    let n = infos.len();
    let mut windows: Vec<(usize, usize)> = Vec::new();
    let mut start = 0usize;
    while start < n {
        let max_end = (start + MAX_WINDOW_TOKENS).min(n);
        if max_end == n {
            windows.push((start, n));
            break;
        }
        // Find last clause end_char inside (start..=max_end)
        let mut best_end = max_end;
        for j in (start + 1..max_end).rev() {
            let end_c = infos[j - 1].span.end_char;
            if clause_spans.iter().any(|c| c.end_char == end_c) {
                best_end = j;
                break;
            }
        }
        windows.push((start, best_end));
        start = best_end;
    }

    let mut all_tokens = Vec::with_capacity(n);
    let mut global_root_tok: Option<usize> = None;

    for (w_idx, &(ws, we)) in windows.iter().enumerate() {
        let mut w_res = decode_window(infos, ws, we, sent_idx);
        if w_idx == 0 {
            if let Some(r) = w_res.iter().find(|st| st.head.is_none()) {
                global_root_tok = Some(r.token);
            }
        } else if let Some(groot) = global_root_tok {
            for st in &mut w_res {
                if st.head.is_none() {
                    st.head = Some(groot);
                    st.deprel = "parataxis".to_owned();
                }
            }
        }
        all_tokens.extend(w_res);
    }

    all_tokens
}

fn decode_window(infos: &[TokInfo<'_>], ws: usize, we: usize, sent_idx: usize) -> Vec<SyntaxToken> {
    let slice = &infos[ws..we];
    let m = slice.len();
    if m == 0 {
        return Vec::new();
    }
    if m == 1 {
        let tok = &slice[0];
        return vec![SyntaxToken {
            token: tok.global_idx,
            sentence: sent_idx,
            span: tok.span,
            head: None,
            deprel: "root".to_owned(),
            score: Score(950),
        }];
    }

    // Identify copulas and their nonverbal predicates
    let mut cop_pred: HashMap<usize, usize> = HashMap::new(); // cop_idx -> pred_idx
    let mut is_cop_pred = vec![false; m];
    for i in 0..m {
        if slice[i].upos == Upos::Aux
            && matches!(
                slice[i].lower.as_str(),
                "是" | "is" | "am" | "are" | "was" | "were" | "be" | "been" | "being"
            )
        {
            // Find the head of the following nonverbal predicate before punct/verb/sconj.
            // Skip past `... 的` modifiers so `是 胡同 里 最 细心 的 手艺人` picks `手艺人`!
            let mut cand: Option<usize> = None;
            let mut saw_adj_or_noun = false;
            for (j, tok) in slice.iter().enumerate().skip(i + 1) {
                if tok.upos == Upos::Punct || tok.upos == Upos::Verb || tok.upos == Upos::Sconj {
                    break;
                }
                if matches!(
                    tok.upos,
                    Upos::Noun | Upos::Propn | Upos::Adj | Upos::Pron | Upos::Num
                ) {
                    cand = Some(j);
                    saw_adj_or_noun = true;
                } else if tok.upos == Upos::Adp || tok.surf == "的" {
                    // Continue past prepositional/的 modifier inside predicate NP
                    continue;
                } else if !saw_adj_or_noun && matches!(tok.upos, Upos::Det | Upos::Adv) {
                    cand = Some(j);
                }
            }
            if let Some(head_j) = cand {
                cop_pred.insert(i, head_j);
                is_cop_pred[head_j] = true;
            }
        }
    }

    // Identify prenominal relative clause verbs (V ... 的 N) and non-finite (to V)
    let mut is_rel_verb = vec![false; m];
    for i in 0..m {
        if slice[i].surf == "的" && i > 0 && i + 1 < m {
            // Only mark a verb as relative if 的 directly follows Verb (or Verb + 了/過/著/好/完/到)
            // and is NOT preceded immediately by Adj/Noun/Pron (which own 的 themselves).
            let mut k = i - 1;
            if matches!(
                slice[k].surf,
                "了" | "過" | "过" | "著" | "着" | "後" | "后" | "前" | "之前" | "之後" | "之后"
            ) && k > 0
            {
                k -= 1;
            }
            if slice[k].upos == Upos::Verb
                && (i + 1..m).any(|r| matches!(slice[r].upos, Upos::Noun | Upos::Propn))
            {
                is_rel_verb[k] = true;
                if k > 0 && slice[k - 1].upos == Upos::Verb {
                    is_rel_verb[k - 1] = true;
                }
            } else if k > 0 && matches!(slice[k].upos, Upos::Noun | Upos::Propn) {
                // Check for `在 ... Verb ... 的 Noun 下/中/上` (must have closing localizer after Noun)
                let has_closing_loc = (i + 1..m)
                    .find(|&r| !matches!(slice[r].upos, Upos::Noun | Upos::Propn | Upos::Adj))
                    .is_some_and(|r| slice[r].upos == Upos::Adp && is_zh_localizer(slice[r].surf));
                if has_closing_loc {
                    for v in (0..k).rev() {
                        if slice[v].upos == Upos::Punct {
                            break;
                        }
                        if slice[v].upos == Upos::Verb {
                            if (0..v).rev().any(|p| {
                                slice[p].upos == Upos::Adp
                                    && matches!(slice[p].surf, "在" | "於" | "于")
                            }) {
                                is_rel_verb[v] = true;
                            }
                            break;
                        }
                    }
                }
            }
        }
        if slice[i].upos == Upos::Verb
            && i > 0
            && slice[i - 1].lower == "to"
            && slice[i - 1].upos == Upos::Part
        {
            is_rel_verb[i] = true;
        }
    }
    // Also mark a verb immediately following an intra-clause SCONJ (e.g. 因为 下雨 变得) as subordinate
    // when another non-rel verb follows it in the same clause.
    let mut is_subord_verb = vec![false; m];
    for i in 0..m {
        if slice[i].upos == Upos::Sconj {
            let mut first_v: Option<usize> = None;
            let mut second_v: Option<usize> = None;
            for j in (i + 1)..m {
                if slice[j].upos == Upos::Punct {
                    break;
                }
                if (slice[j].upos == Upos::Verb && !is_rel_verb[j]) || is_cop_pred[j] {
                    if first_v.is_none() {
                        first_v = Some(j);
                    } else {
                        second_v = Some(j);
                        break;
                    }
                }
            }
            if let (Some(v1), Some(_)) = (first_v, second_v) {
                is_subord_verb[v1] = true;
            }
        }
    }

    // Clause segmentation by comma/semicolon (excluding sentence-initial connector commas)
    let mut clause_id = vec![0usize; m];
    let mut cur_clause = 0usize;
    let mut clause_starts: Vec<usize> = vec![0];
    for i in 0..m {
        clause_id[i] = cur_clause;
        if slice[i].upos == Upos::Punct
            && matches!(slice[i].surf, "," | "，" | ";" | "；")
            && i + 1 < m
        {
            // If tokens in current clause before comma are just 1-2 connector/interjection tokens (no Verb/cop_pred),
            // don't split into a separate main clause!
            let c_start = clause_starts[cur_clause];
            let has_pred = (c_start..i)
                .any(|k| (slice[k].upos == Upos::Verb && !is_rel_verb[k]) || is_cop_pred[k]);
            if has_pred {
                cur_clause += 1;
                clause_starts.push(i + 1);
            }
        }
    }
    let num_clauses = cur_clause + 1;
    let mut clause_has_sconj = vec![false; num_clauses];
    for c in 0..num_clauses {
        let cs = clause_starts[c];
        let ce = if c + 1 < num_clauses {
            clause_starts[c + 1]
        } else {
            m
        };
        if (cs..ce).take(3).any(|k| slice[k].upos == Upos::Sconj) {
            clause_has_sconj[c] = true;
        }
    }
    // Primary clause: first clause without SCONJ (or last clause if all have SCONJ)
    let primary_clause = (0..num_clauses)
        .find(|&c| !clause_has_sconj[c])
        .unwrap_or(num_clauses - 1);

    // Find each clause's main predicate candidate
    let mut clause_pred: Vec<Option<usize>> = vec![None; num_clauses];
    for c in 0..num_clauses {
        let cs = clause_starts[c];
        let ce = if c + 1 < num_clauses {
            clause_starts[c + 1]
        } else {
            m
        };
        // Prefer cop_pred or first non-rel, non-subord Verb, or last predicative Adj
        clause_pred[c] = (cs..ce)
            .find(|&k| is_cop_pred[k])
            .or_else(|| {
                (cs..ce)
                    .find(|&k| slice[k].upos == Upos::Verb && !is_rel_verb[k] && !is_subord_verb[k])
            })
            .or_else(|| (cs..ce).find(|&k| slice[k].upos == Upos::Verb && !is_rel_verb[k]))
            .or_else(|| {
                (cs..ce).rev().find(|&k| {
                    slice[k].upos == Upos::Adj
                        && (k + 1 >= ce
                            || !matches!(slice[k + 1].upos, Upos::Noun | Upos::Propn | Upos::Adj))
                })
            })
            .or_else(|| (cs..ce).rev().find(|&k| slice[k].upos == Upos::Adj));
    }

    // Contiguous NP head table: for each token i, np_head[i] is the rightmost NOUN/PROPN/X
    // in the same contiguous NP (crossing Det, Num, Adj, clf, Noun, Propn, X).
    let mut np_head: Vec<Option<usize>> = vec![None; m];
    let mut i = 0usize;
    while i < m {
        let is_np_elem = matches!(
            slice[i].upos,
            Upos::Det | Upos::Num | Upos::Adj | Upos::Noun | Upos::Propn | Upos::X
        ) && !is_cop_pred[i]
            && !matches!(
                slice[i].surf,
                "大於"
                    | "大于"
                    | "小於"
                    | "小于"
                    | "等於"
                    | "等于"
                    | "高於"
                    | "高于"
                    | "低於"
                    | "低于"
                    | "多於"
                    | "多于"
                    | "少於"
                    | "少于"
            );
        if is_np_elem {
            let start = i;
            let mut last_noun: Option<usize> = None;
            while i < m {
                let u = slice[i].upos;
                if is_cop_pred[i] {
                    if matches!(u, Upos::Noun | Upos::Propn | Upos::X) {
                        last_noun = Some(i);
                    }
                    i += 1;
                    break;
                }
                if matches!(u, Upos::Noun | Upos::Propn | Upos::X) {
                    last_noun = Some(i);
                    i += 1;
                } else if matches!(u, Upos::Det | Upos::Num | Upos::Adj) {
                    // Stop if Adj is followed by nothing or punct/verb
                    if u == Upos::Adj
                        && (i + 1 >= m
                            || !matches!(
                                slice[i + 1].upos,
                                Upos::Noun | Upos::Propn | Upos::Adj | Upos::X
                            ))
                    {
                        break;
                    }
                    i += 1;
                } else {
                    break;
                }
            }
            if let Some(hn) = last_noun {
                for slot in &mut np_head[start..i] {
                    *slot = Some(hn);
                }
            }
            if i == start {
                i += 1;
            }
        } else {
            i += 1;
        }
    }

    // Build arc_scores[dep_idx][head] where head=0 is root, head=1..=m is 1-based token in slice
    let mut arc_scores: Vec<Vec<f32>> = vec![vec![-100.0; m + 1]; m];

    for dep in 0..m {
        let u = slice[dep].upos;
        let dc = clause_id[dep];
        let is_primary_pred = clause_pred[primary_clause] == Some(dep);

        let root_score = if is_primary_pred {
            12.0
        } else if (u == Upos::Verb && !is_rel_verb[dep]) || is_cop_pred[dep] {
            if dc == primary_clause { 6.0 } else { 2.0 }
        } else if u == Upos::Adj && !is_rel_verb[dep] {
            1.0
        } else if matches!(u, Upos::Noun | Upos::Propn | Upos::Pron) {
            -4.0
        } else {
            -8.0
        };
        arc_scores[dep][0] = root_score;

        for head in 0..m {
            if head == dep {
                continue;
            }
            let h_upos = slice[head].upos;
            let hc = clause_id[head];
            let same_clause = dc == hc;
            let dist = dep.abs_diff(head) as f32;
            let mut sc = -15.0 - 0.1 * dist;
            if !same_clause && u != Upos::Punct && clause_pred[dc] != Some(dep) {
                sc -= 6.0;
            }

            if cop_pred.get(&dep) == Some(&head) {
                sc = 20.0;
            } else if u == Upos::Punct {
                if matches!(slice[dep].surf, "," | "，" | ";" | "；") {
                    // Clause-ending comma attaches to its own clause's predicate;
                    // sentence-initial connector comma attaches to primary_clause's predicate!
                    if same_clause && clause_pred[dc] == Some(head) {
                        sc = 16.0;
                    } else if same_clause
                        && ((h_upos == Upos::Verb && !is_rel_verb[head]) || is_cop_pred[head])
                    {
                        sc = 12.0 - 0.1 * dist;
                    }
                } else {
                    // Sentence-final punctuation attaches to primary_clause's predicate (global root)
                    if clause_pred[primary_clause] == Some(head) {
                        sc = 16.0;
                    } else if (h_upos == Upos::Verb && !is_rel_verb[head]) || is_cop_pred[head] {
                        sc = 10.0 - 0.05 * dist;
                    }
                }
            } else if matches!(u, Upos::Det | Upos::Num)
                || (token_is_cjk(slice[dep].surf, slice[dep].lang)
                    && is_zh_classifier(slice[dep].surf)
                    && u == Upos::Noun)
            {
                if np_head[dep] == Some(head) {
                    sc = 17.0;
                } else if head > dep
                    && matches!(h_upos, Upos::Noun | Upos::Propn | Upos::X)
                    && same_clause
                {
                    sc = 12.0 - 0.5 * dist;
                } else if u == Upos::Num
                    && head < dep
                    && matches!(h_upos, Upos::Verb | Upos::Adj)
                    && same_clause
                {
                    // Postverbal duration/extent NUM (e.g. 靜置 兩小時, 漲高 了 許多)
                    sc = 12.0 - 0.3 * dist;
                }
            } else if u == Upos::Adj && !is_cop_pred[dep] {
                if np_head[dep] == Some(head) {
                    sc = 16.5;
                } else if head > dep
                    && matches!(h_upos, Upos::Noun | Upos::Propn | Upos::X)
                    && same_clause
                {
                    sc = 11.5 - 0.4 * dist;
                } else if head < dep && h_upos == Upos::Verb && same_clause {
                    sc = 10.0 - 0.5 * dist;
                }
            } else if u == Upos::Part {
                if slice[dep].surf == "的" {
                    if head < dep
                        && matches!(
                            h_upos,
                            Upos::Verb | Upos::Adj | Upos::Noun | Upos::Pron | Upos::Propn
                        )
                    {
                        sc = 16.0 - 1.5 * dist;
                    }
                } else if matches!(slice[dep].surf, "了" | "著" | "着" | "過" | "过" | "得") {
                    if head < dep && matches!(h_upos, Upos::Verb | Upos::Adj) {
                        sc = 16.0 - 1.5 * dist;
                    }
                } else if slice[dep].lower == "to"
                    || slice[dep].lower == "not"
                    || slice[dep].lower == "n't"
                {
                    if head > dep && matches!(h_upos, Upos::Verb | Upos::Aux | Upos::Adj) {
                        sc = 15.0 - 1.0 * dist;
                    }
                } else if slice[dep].lower == "'s" {
                    if head < dep && matches!(h_upos, Upos::Noun | Upos::Propn | Upos::Pron) {
                        sc = 16.0 - 1.5 * dist;
                    }
                } else if matches!(
                    slice[dep].surf,
                    "嗎" | "吗" | "吧" | "呢" | "啊" | "啦" | "喔" | "呀"
                ) && head < dep
                    && matches!(h_upos, Upos::Verb | Upos::Adj)
                {
                    sc = 14.0 - 0.3 * dist;
                }
            } else if u == Upos::Adp {
                if slice[dep].surf == "被" {
                    let has_agent = ((dep + 1)..head)
                        .any(|k| matches!(slice[k].upos, Upos::Noun | Upos::Propn | Upos::Pron));
                    if head > dep
                        && matches!(h_upos, Upos::Noun | Upos::Propn | Upos::Pron)
                        && same_clause
                    {
                        if dep + 1 < m && np_head[dep + 1] == Some(head) {
                            sc = 16.0;
                        } else {
                            sc = 13.0 - 0.8 * dist;
                        }
                    } else if head > dep && h_upos == Upos::Verb && !has_agent && same_clause {
                        sc = 14.5 - 0.5 * dist;
                    }
                } else if is_zh_localizer(slice[dep].surf)
                    && head < dep
                    && (matches!(h_upos, Upos::Noun | Upos::Propn | Upos::Pron | Upos::X)
                        || (h_upos == Upos::Verb
                            && matches!(
                                slice[dep].surf,
                                "後" | "后"
                                    | "前"
                                    | "之前"
                                    | "之後"
                                    | "之后"
                                    | "以前"
                                    | "以後"
                                    | "以后"
                            )))
                    && same_clause
                {
                    sc = 16.5 - 1.0 * dist;
                } else if head > dep
                    && matches!(h_upos, Upos::Noun | Upos::Propn | Upos::Pron | Upos::X)
                    && same_clause
                {
                    if dep + 1 < m && np_head[dep + 1] == Some(head) {
                        sc = 16.5;
                    } else {
                        sc = 13.5 - 0.6 * dist;
                    }
                } else if head < dep && h_upos == Upos::Verb && same_clause {
                    // English phrasal verb particle (pick up)
                    sc = 12.0 - 1.0 * dist;
                }
            } else if u == Upos::Aux {
                if head > dep
                    && (h_upos == Upos::Verb || is_cop_pred[head] || h_upos == Upos::Adj)
                    && same_clause
                {
                    sc = 14.5 - 0.5 * dist;
                }
            } else if u == Upos::Adv {
                if head > dep
                    && (matches!(h_upos, Upos::Verb | Upos::Adj) || is_cop_pred[head])
                    && same_clause
                {
                    sc = 13.5 - 0.4 * dist;
                } else if head < dep
                    && (matches!(h_upos, Upos::Verb | Upos::Adj) || is_cop_pred[head])
                    && same_clause
                {
                    sc = 10.0 - 0.4 * dist;
                }
            } else if u == Upos::Sconj {
                if head > dep
                    && (h_upos == Upos::Verb || is_cop_pred[head] || h_upos == Upos::Adj)
                    && same_clause
                {
                    sc = 14.0 - 0.3 * dist;
                }
            } else if u == Upos::Cconj {
                if head > dep
                    && matches!(
                        h_upos,
                        Upos::Noun | Upos::Propn | Upos::Verb | Upos::Adj | Upos::X
                    )
                    && same_clause
                {
                    sc = 13.5 - 0.6 * dist;
                }
            } else if matches!(u, Upos::Noun | Upos::Propn | Upos::Pron | Upos::X) {
                let is_poss = u == Upos::Pron && is_en_poss_pron(&slice[dep].lower);
                if is_poss
                    && head > dep
                    && matches!(h_upos, Upos::Noun | Upos::Propn | Upos::X)
                    && same_clause
                {
                    // Possessive pronoun attaches to rightmost head noun of following NP
                    if dep + 1 < m && np_head[dep + 1] == Some(head) {
                        sc = 17.0;
                    } else {
                        sc = 13.0 - 0.5 * dist;
                    }
                } else if np_head[dep] == Some(head) && head > dep && u != Upos::Pron {
                    sc = 15.5 - 0.2 * dist;
                } else if head > dep
                    && matches!(h_upos, Upos::Noun | Upos::Propn | Upos::X)
                    && !is_cop_pred[head]
                    && same_clause
                    && (slice[dep + 1..head]
                        .iter()
                        .any(|t| t.surf == "的" || t.lower == "'s")
                        || (u != Upos::Pron
                            && dist <= 2.0
                            && !slice[dep + 1..head].iter().any(|t| {
                                matches!(
                                    t.upos,
                                    Upos::Adp
                                        | Upos::Verb
                                        | Upos::Aux
                                        | Upos::Punct
                                        | Upos::Cconj
                                        | Upos::Sconj
                                )
                            })))
                {
                    sc = 12.5 - 0.5 * dist;
                } else if head < dep
                    && matches!(h_upos, Upos::Noun | Upos::Propn | Upos::X)
                    && !is_cop_pred[head]
                    && same_clause
                    && (dep > head + 1 && slice[head + 1].upos == Upos::Adp)
                {
                    // English PP modifying noun (crates of apples)
                    sc = 11.5 - 0.5 * dist;
                } else if (matches!(h_upos, Upos::Verb) || is_cop_pred[head]) && same_clause {
                    sc = 11.5 - 0.15 * dist;
                } else if h_upos == Upos::Adj && head > dep && same_clause {
                    sc = 9.5 - 0.2 * dist;
                }
            } else if u == Upos::Verb || is_cop_pred[dep] {
                if is_rel_verb[dep]
                    && head > dep
                    && matches!(h_upos, Upos::Noun | Upos::Propn)
                    && same_clause
                {
                    sc = 14.5 - 0.5 * dist;
                } else if clause_pred[dc] == Some(dep)
                    && !same_clause
                    && clause_pred[primary_clause] == Some(head)
                {
                    // Subordinate or paratactic clause predicate attaches to primary clause root
                    sc = 11.0 - 0.02 * dist;
                } else if matches!(h_upos, Upos::Verb) && !is_rel_verb[head] && same_clause {
                    if dep == head + 1 && token_is_cjk(slice[dep].surf, slice[dep].lang) {
                        sc = 13.5;
                    } else {
                        sc = 10.0 - 0.2 * dist;
                    }
                }
            }

            arc_scores[dep][head + 1] = sc;
        }
    }

    let raw_heads = chu_liu_edmonds(&arc_scores);

    // Compute deprels and scores
    let mut out = Vec::with_capacity(m);
    for dep in 0..m {
        let h = raw_heads[dep];
        let deprel = if h == 0 {
            "root"
        } else {
            select_deprel(
                slice,
                dep,
                h - 1,
                &cop_pred,
                &is_cop_pred,
                &is_rel_verb,
                &clause_id,
                &clause_has_sconj,
            )
        };
        let chosen_score = arc_scores[dep][h];
        let second_best = arc_scores[dep]
            .iter()
            .enumerate()
            .filter(|&(k, _)| k != h && (k == 0 || k - 1 != dep))
            .map(|(_, &v)| v)
            .fold(f32::NEG_INFINITY, f32::max);
        let margin = (chosen_score - second_best).max(0.0);
        let conf_milli =
            ((850.0 + margin * 10.0).round() as u16).clamp(SYNTAX_THRESHOLD_MILLI.max(850), 980);

        debug_assert!(DEPRELS.contains(&deprel));
        out.push(SyntaxToken {
            token: slice[dep].global_idx,
            sentence: sent_idx,
            span: slice[dep].span,
            head: if h == 0 {
                None
            } else {
                Some(slice[h - 1].global_idx)
            },
            deprel: deprel.to_owned(),
            score: Score(conf_milli),
        });
    }

    out
}

#[allow(clippy::too_many_arguments)]
fn select_deprel(
    slice: &[TokInfo<'_>],
    dep: usize,
    head: usize,
    cop_pred: &HashMap<usize, usize>,
    is_cop_pred: &[bool],
    is_rel_verb: &[bool],
    clause_id: &[usize],
    clause_has_sconj: &[bool],
) -> &'static str {
    let du = slice[dep].upos;
    let hu = slice[head].upos;
    let dsurf = slice[dep].surf;
    let dlower = slice[dep].lower.as_str();

    if du == Upos::Punct {
        return "punct";
    }
    if cop_pred.get(&dep) == Some(&head) {
        return "cop";
    }
    if du == Upos::Det {
        return "det";
    }
    if du == Upos::Num {
        if matches!(hu, Upos::Verb | Upos::Adj) || is_cop_pred[head] {
            return "obl";
        }
        return "nummod";
    }
    if du == Upos::Adp {
        if dsurf == "被" && matches!(hu, Upos::Verb | Upos::Adj) {
            return "aux";
        }
        if head < dep && hu == Upos::Verb {
            if !token_is_cjk(dsurf, slice[dep].lang) {
                return "compound";
            }
            return "mark";
        }
        return "case";
    }
    if token_is_cjk(dsurf, slice[dep].lang)
        && is_zh_classifier(dsurf)
        && matches!(hu, Upos::Noun | Upos::Propn)
        && !is_cop_pred[head]
    {
        return "clf";
    }
    if du == Upos::Cconj {
        return "cc";
    }
    if du == Upos::Sconj {
        return "mark";
    }
    if du == Upos::Intj {
        return "discourse";
    }
    if du == Upos::Aux {
        return "aux";
    }
    if du == Upos::Adv {
        return "advmod";
    }
    if du == Upos::Part {
        if dsurf == "的" {
            if matches!(hu, Upos::Noun | Upos::Pron | Upos::Propn) {
                return "case";
            }
            return "mark";
        }
        if matches!(dsurf, "了" | "著" | "着" | "過" | "过") {
            return "aux";
        }
        if matches!(dsurf, "嗎" | "吗" | "吧" | "呢" | "啊" | "啦" | "喔" | "呀") {
            return "discourse";
        }
        if dlower == "to" || dsurf == "得" || dsurf == "地" {
            return "mark";
        }
        if dlower == "'s" {
            return "case";
        }
        if dlower == "not" || dlower == "n't" {
            return "advmod";
        }
        return "mark";
    }
    if du == Upos::Adj {
        if matches!(hu, Upos::Noun | Upos::Propn | Upos::Pron) && dep < head && !is_cop_pred[head] {
            return "amod";
        }
        if hu == Upos::Verb && dep > head {
            return "compound";
        }
        return "amod";
    }
    if matches!(du, Upos::Noun | Upos::Propn | Upos::Pron | Upos::X) {
        if matches!(hu, Upos::Noun | Upos::Propn | Upos::X) && !is_cop_pred[head] {
            if dep < head {
                if (dep + 1..head).any(|k| slice[k].surf == "的" || slice[k].lower == "'s") {
                    return "nmod";
                }
                if du == Upos::Pron {
                    return "nmod";
                }
                return "compound";
            }
            if du == Upos::X {
                return "appos";
            }
            return "nmod";
        }
        if matches!(hu, Upos::Verb | Upos::Adj) || is_cop_pred[head] {
            let has_case_before = (0..dep)
                .rev()
                .take_while(|&k| {
                    slice[k].upos != Upos::Punct && slice[k].upos != Upos::Verb && !is_cop_pred[k]
                })
                .any(|k| slice[k].upos == Upos::Adp);
            let has_localizer_after = dep + 1 < slice.len()
                && slice[dep + 1].upos == Upos::Adp
                && is_zh_localizer(slice[dep + 1].surf);
            if has_case_before || has_localizer_after || is_temporal_noun(&slice[dep]) {
                return "obl";
            }
            if dep < head {
                return "nsubj";
            }
            return "obj";
        }
    }
    if du == Upos::Verb || is_cop_pred[dep] {
        if is_rel_verb[dep] && matches!(hu, Upos::Noun | Upos::Propn) {
            return "acl";
        }
        if clause_id[dep] != clause_id[head] {
            if clause_has_sconj[clause_id[dep]] {
                return "advcl";
            }
            return "parataxis";
        }
        if hu == Upos::Verb {
            if dep == head + 1 && token_is_cjk(dsurf, slice[dep].lang) {
                return "compound";
            }
            if dep < head {
                return "advcl";
            }
            let is_ctrl = matches!(
                slice[head].lower.as_str(),
                "開始"
                    | "开始"
                    | "決定"
                    | "决定"
                    | "準備"
                    | "准备"
                    | "打算"
                    | "試著"
                    | "试着"
                    | "想"
                    | "希望"
                    | "繼續"
                    | "继续"
                    | "學會"
                    | "学会"
                    | "幫"
                    | "帮"
                    | "讓"
                    | "让"
                    | "使"
                    | "請"
                    | "请"
                    | "變得"
                    | "变得"
                    | "停止"
                    | "用來"
                    | "用来"
                    | "want"
                    | "wants"
                    | "wanted"
                    | "try"
                    | "tried"
                    | "begin"
                    | "began"
                    | "start"
                    | "started"
                    | "continue"
                    | "continued"
                    | "decide"
                    | "decided"
                    | "need"
                    | "needed"
                    | "seem"
                    | "seemed"
                    | "help"
                    | "helped"
                    | "make"
                    | "made"
                    | "let"
                    | "ask"
                    | "asked"
                    | "tell"
                    | "told"
                    | "allow"
                    | "allowed"
                    | "enable"
                    | "enabled"
            );
            let has_to_before =
                dep > 0 && slice[dep - 1].lower == "to" && slice[dep - 1].upos == Upos::Part;
            if is_ctrl || has_to_before {
                return "xcomp";
            }
            return "conj";
        }
    }
    "dep"
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::doc::{Options, analyze_document};
    use serde_json::Value;
    use std::fs;

    #[test]
    fn standalone_sentence_and_windowing_invariants() {
        let text = "老木匠把桌上的木屑掃進竹籃裡。她買了三本書。";
        let mut doc = analyze_document(text, &Options::default());
        annotate_sentence_zhen(&mut doc, text, 0);
        assert!(doc.syntax.iter().all(|s| s.sentence == 0));
        assert_eq!(doc.syntax.iter().filter(|s| s.head.is_none()).count(), 1);
        assert!(
            !doc.unknown.iter().any(|u| u.layer == "pos"
                && u.span.end_char <= doc.segmentation.sentences[0].span.end_char)
        );

        // Long sentence (> 96 tokens) exercises decode_windowed
        let long_clause = "The engineer checked the calibration log on the desk, ";
        let long_text = format!("{}and the sensor remained stable.", long_clause.repeat(12));
        let mut long_doc = analyze_document(&long_text, &Options::default());
        assert!(long_doc.tokens.len() > MAX_WINDOW_TOKENS);
        annotate(&mut long_doc, &long_text);
        assert_eq!(long_doc.syntax.len(), long_doc.tokens.len());
        assert_eq!(
            long_doc.syntax.iter().filter(|s| s.head.is_none()).count(),
            1
        );
        assert!(long_doc.syntax.iter().any(|s| s.deprel == "parataxis"));
        for st in &long_doc.syntax {
            assert_ne!(st.head, Some(st.token), "self-loop at token {}", st.token);
            assert!(st.score.milli() >= SYNTAX_THRESHOLD_MILLI);
        }
    }

    #[test]
    fn skips_unsupported_and_jako_sentences() {
        let text = "Привет мир. 老木匠把桌上的木屑掃進竹籃裡。";
        let mut doc = analyze_document(text, &Options::default());
        annotate(&mut doc, text);
        assert!(!doc.syntax.is_empty());
        assert!(doc.syntax.iter().all(|s| s.sentence == 1));
    }

    #[test]
    fn eval_gold_layers4_6_v1() {
        let raw = fs::read_to_string("tests/data/gold_layers4_6_v1.jsonl").unwrap();
        let mut total_tok = 0usize;
        let mut upos_ok = 0usize;
        let mut uas_ok = 0usize;
        let mut las_ok = 0usize;

        let mut by_lang: HashMap<String, [usize; 4]> = HashMap::new();
        let mut upos_errs: HashMap<(String, String, String), usize> = HashMap::new();
        let mut deprel_errs: HashMap<(String, String), usize> = HashMap::new();
        let mut head_err_by_gold_rel: HashMap<String, usize> = HashMap::new();

        for line in raw.lines().filter(|l| !l.trim().is_empty()) {
            let v: Value = serde_json::from_str(line).unwrap();
            let text = v["text"].as_str().unwrap();
            let lang = v["lang"].as_str().unwrap().to_owned();
            let gold_upos: Vec<&str> = v["upos"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_str().unwrap())
                .collect();
            let gold_heads: Vec<usize> = v["heads"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as usize)
                .collect();
            let gold_deprels: Vec<&str> = v["deprels"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_str().unwrap())
                .collect();

            let mut doc = analyze_document(text, &Options::default());
            annotate(&mut doc, text);

            assert_eq!(doc.tokens.len(), gold_upos.len());
            assert_eq!(doc.syntax.len(), gold_heads.len());

            // Check structural invariants per sentence
            for (s_idx, _) in doc.segmentation.sentences.iter().enumerate() {
                let st_sent: Vec<&SyntaxToken> = doc
                    .syntax
                    .iter()
                    .filter(|st| st.sentence == s_idx)
                    .collect();
                let roots = st_sent.iter().filter(|st| st.head.is_none()).count();
                assert_eq!(roots, 1, "sentence {s_idx} has {roots} roots");
            }

            let entry = by_lang.entry(lang.clone()).or_insert([0; 4]);
            let char_to_byte = char_to_byte_table(text);
            for i in 0..gold_upos.len() {
                total_tok += 1;
                entry[0] += 1;

                let pred_u = serde_json::to_value(doc.tokens[i].upos.unwrap())
                    .unwrap()
                    .as_str()
                    .unwrap()
                    .to_owned();
                if pred_u == gold_upos[i] {
                    upos_ok += 1;
                    entry[1] += 1;
                } else {
                    let surf = span_text(text, &char_to_byte, doc.tokens[i].span).to_owned();
                    *upos_errs
                        .entry((
                            lang.clone(),
                            format!("{surf}:{pred_u}"),
                            gold_upos[i].to_owned(),
                        ))
                        .or_insert(0) += 1;
                }

                let st = &doc.syntax[i];
                let pred_h = st.head.map_or(0, |h| h + 1);
                if pred_h == gold_heads[i] {
                    uas_ok += 1;
                    entry[2] += 1;
                    if st.deprel == gold_deprels[i] {
                        las_ok += 1;
                        entry[3] += 1;
                    } else {
                        *deprel_errs
                            .entry((st.deprel.clone(), gold_deprels[i].to_owned()))
                            .or_insert(0) += 1;
                    }
                } else {
                    *head_err_by_gold_rel
                        .entry(gold_deprels[i].to_owned())
                        .or_insert(0) += 1;
                }
            }
        }

        eprintln!(
            "OVERALL ({total_tok} tokens): UPOS={:.2}% UAS={:.2}% LAS={:.2}%",
            100.0 * upos_ok as f64 / total_tok as f64,
            100.0 * uas_ok as f64 / total_tok as f64,
            100.0 * las_ok as f64 / total_tok as f64,
        );
        for (l, c) in &by_lang {
            eprintln!(
                "  {l}: UPOS={:.2}% UAS={:.2}% LAS={:.2}% ({} tok)",
                100.0 * c[1] as f64 / c[0] as f64,
                100.0 * c[2] as f64 / c[0] as f64,
                100.0 * c[3] as f64 / c[0] as f64,
                c[0]
            );
        }
        let mut ue: Vec<_> = upos_errs.into_iter().collect();
        ue.sort_by_key(|b| std::cmp::Reverse(b.1));
        eprintln!("Top UPOS errors:");
        for ((l, sp, g), c) in ue.into_iter().take(20) {
            eprintln!("  {c:3} [{l}] {sp} -> gold {g}");
        }
        let mut he: Vec<_> = head_err_by_gold_rel.into_iter().collect();
        he.sort_by_key(|b| std::cmp::Reverse(b.1));
        eprintln!("Top head errors by gold deprel:");
        for (g, c) in he.into_iter().take(15) {
            eprintln!("  {c:3} {g}");
        }
        let mut de: Vec<_> = deprel_errs.into_iter().collect();
        de.sort_by_key(|b| std::cmp::Reverse(b.1));
        eprintln!("Top deprel errors (when head ok):");
        for ((p, g), c) in de.into_iter().take(15) {
            eprintln!("  {c:3} pred {p} -> gold {g}");
        }
    }
}
