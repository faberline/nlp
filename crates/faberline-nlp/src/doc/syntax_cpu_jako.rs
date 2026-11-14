//! Pure-Rust zero-model CPU Layer-4 UPOS disambiguator and UD v2 SOV head-final
//! dependency parser for Japanese (`ja`) and Korean (`ko`).

use std::collections::HashSet;

use crate::Span;
use crate::doc::syntax::{DEPRELS, SOFT_WINDOW_MAX_LEN, Score, SyntaxToken, chu_liu_edmonds};
use crate::doc::{Document, EntityKind, Lang, Upos, sort_unknown};
use crate::lexicon::ko::analyze_ko_eojeol;
use crate::lexicon::lookup_ja;

const PUNCT_CHARS: &str = "!\"'(),-./:;?[\\]_{}，。．、；：？！…‥⋯「」『』（）《》〈〉“”‘’—–―─～〜·‧・｡｢｣､･«»【】〔〕［］｛｝〘〙〖〗〝〟※";
const JA_KANJI_NUMERALS: &str = "〇零一二三四五六七八九十百千萬万億兆";

const SCORE_PRIMARY: f64 = 0.90;
const SCORE_FALLBACK: f64 = 0.75;
const ARC_PREFERRED: f32 = 10.0;
const ARC_FALLBACK: f32 = 1.0;
const ARC_FORBIDDEN: f32 = -10.0;

/// Upgrades undecided `upos` tokens in sentence `sent_idx` (clearing resolved `"pos"`
/// entries from `doc.unknown`) and appends valid single-root UD v2 [`SyntaxToken`]s
/// using only the 37 base [`DEPRELS`] labels.
#[allow(dead_code)]
pub(crate) fn annotate_sentence(doc: &mut Document, text: &str, sent_idx: usize) {
    let sent_tok_indices: Vec<usize> = doc
        .tokens
        .iter()
        .enumerate()
        .filter(|(_, t)| t.sentence == sent_idx)
        .map(|(i, _)| i)
        .collect();
    if sent_tok_indices.is_empty() {
        return;
    }

    let text_chars: Vec<char> = text.chars().collect();
    let surfaces: Vec<String> = sent_tok_indices
        .iter()
        .map(|&idx| slice_chars(&text_chars, doc.tokens[idx].span))
        .collect();

    let sent_lang = detect_sentence_lang(doc, sent_idx, &sent_tok_indices);

    // 1. Upgrade undecided UPOS tokens and clear resolved `"pos"` unknown entries.
    match sent_lang {
        Lang::Ja => disambiguate_ja_upos(doc, sent_idx, &sent_tok_indices, &surfaces),
        Lang::Ko => disambiguate_ko_upos(doc, sent_idx, &sent_tok_indices, &surfaces),
        _ => disambiguate_ja_upos(doc, sent_idx, &sent_tok_indices, &surfaces),
    }

    // 2. Partition into <= SOFT_WINDOW_MAX_LEN windows along clause boundaries.
    let windows = build_windows(doc, sent_idx, &sent_tok_indices);

    // 3. Parse each window with SOV rules + Chu-Liu/Edmonds single-root enforcement.
    let mut window_trees: Vec<WindowTree> = Vec::with_capacity(windows.len());
    for win in &windows {
        let win_surfaces: Vec<&str> = win
            .iter()
            .map(|&local_i| surfaces[local_i].as_str())
            .collect();
        let win_tok_indices: Vec<usize> = win
            .iter()
            .map(|&local_i| sent_tok_indices[local_i])
            .collect();
        let tree = match sent_lang {
            Lang::Ko => parse_ko_window(doc, &win_tok_indices, &win_surfaces),
            _ => parse_ja_window(doc, &win_tok_indices, &win_surfaces),
        };
        window_trees.push(tree);
    }

    // 4. Link multiple windows via `parataxis` onto the final window's root (SOV head-final).
    let global_root_tok = window_trees
        .last()
        .map(|w| w.tok_indices[w.root_local])
        .expect("non-empty sentence has at least one window");

    for (w_idx, tree) in window_trees.iter().enumerate() {
        let is_last_win = w_idx + 1 == window_trees.len();
        for u in 0..tree.tok_indices.len() {
            let tok_idx = tree.tok_indices[u];
            let (head, deprel, score) = if u == tree.root_local {
                if is_last_win {
                    (None, "root", tree.scores[u])
                } else {
                    (
                        Some(global_root_tok),
                        "parataxis",
                        Score::from_confidence(SCORE_FALLBACK),
                    )
                }
            } else {
                let h_local = tree.heads[u];
                (
                    Some(tree.tok_indices[h_local]),
                    tree.deprels[u],
                    tree.scores[u],
                )
            };
            debug_assert!(
                DEPRELS.contains(&deprel),
                "invalid UD v2 deprel emitted: {deprel}"
            );
            doc.syntax.push(SyntaxToken {
                token: tok_idx,
                sentence: sent_idx,
                span: doc.tokens[tok_idx].span,
                head,
                deprel: deprel.to_owned(),
                score,
            });
        }
    }

    sort_unknown(&mut doc.unknown);
}

struct WindowTree {
    tok_indices: Vec<usize>,
    heads: Vec<usize>,
    deprels: Vec<&'static str>,
    scores: Vec<Score>,
    root_local: usize,
}

fn slice_chars(text_chars: &[char], span: Span) -> String {
    let start = span.start_char.min(text_chars.len());
    let end = span.end_char.min(text_chars.len());
    if start >= end {
        String::new()
    } else {
        text_chars[start..end].iter().collect()
    }
}

fn detect_sentence_lang(doc: &Document, sent_idx: usize, sent_tok_indices: &[usize]) -> Lang {
    if let Some(sent) = doc.segmentation.sentences.get(sent_idx)
        && matches!(sent.lang, Lang::Ja | Lang::Ko)
    {
        return sent.lang;
    }
    let mut ja_count = 0usize;
    let mut ko_count = 0usize;
    for &idx in sent_tok_indices {
        match doc.tokens[idx].lang {
            Lang::Ja => ja_count += 1,
            Lang::Ko => ko_count += 1,
            _ => {}
        }
    }
    if ko_count > ja_count {
        Lang::Ko
    } else {
        Lang::Ja
    }
}

fn clear_pos_unknown(doc: &mut Document, span: Span) {
    doc.unknown
        .retain(|u| !(u.layer == "pos" && u.span == span));
}

fn is_punct_str(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| PUNCT_CHARS.contains(c))
}

fn is_ascii_digits_str(s: &str) -> bool {
    !s.is_empty()
        && s.chars().next().is_some_and(|c| c.is_ascii_digit())
        && s.chars().last().is_some_and(|c| c.is_ascii_digit())
        && s.chars()
            .all(|c| c.is_ascii_digit() || matches!(c, '.' | ','))
}

fn is_ja_kanji_numeral(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| JA_KANJI_NUMERALS.contains(c))
}

fn is_katakana_run(s: &str) -> bool {
    !s.is_empty()
        && s.chars()
            .all(|c| matches!(c, '\u{30A1}'..='\u{30FA}' | '\u{30FC}' | '\u{30FD}'..='\u{30FE}'))
}

fn is_kanji(c: char) -> bool {
    matches!(c, '\u{4E00}'..='\u{9FFF}' | '\u{3400}'..='\u{4DBF}' | '々' | '〆')
}

fn is_hiragana(c: char) -> bool {
    matches!(c, '\u{3041}'..='\u{309F}' | 'ー')
}

fn entity_kind_for_span(doc: &Document, sent_idx: usize, span: Span) -> Option<EntityKind> {
    doc.entities
        .iter()
        .find(|e| {
            e.sentence == sent_idx
                && e.span.start_char <= span.start_char
                && e.span.end_char >= span.end_char
        })
        .map(|e| e.kind)
}

fn build_windows(doc: &Document, sent_idx: usize, sent_tok_indices: &[usize]) -> Vec<Vec<usize>> {
    let n = sent_tok_indices.len();
    if n <= SOFT_WINDOW_MAX_LEN {
        return vec![(0..n).collect()];
    }

    let clause_starts: HashSet<usize> = doc
        .segmentation
        .clauses
        .iter()
        .filter(|c| c.sentence == sent_idx)
        .map(|c| c.span.start_char)
        .collect();

    let mut windows = Vec::new();
    let mut start = 0usize;
    while start < n {
        let max_end = (start + SOFT_WINDOW_MAX_LEN).min(n);
        if max_end == n {
            windows.push((start..n).collect());
            break;
        }
        // Prefer splitting at the latest clause boundary within the window.
        let mut split_at = max_end;
        for pos in ((start + 1)..=max_end).rev() {
            let tok_start = doc.tokens[sent_tok_indices[pos]].span.start_char;
            if clause_starts.contains(&tok_start) {
                split_at = pos;
                break;
            }
        }
        windows.push((start..split_at).collect());
        start = split_at;
    }
    windows
}

fn solve_window_with_cle(
    preferred_heads: &[Option<usize>],
    preferred_deprels: &[&'static str],
    root_local: usize,
) -> (Vec<usize>, Vec<&'static str>, Vec<Score>) {
    let m = preferred_heads.len();
    if m == 0 {
        return (Vec::new(), Vec::new(), Vec::new());
    }
    if m == 1 {
        return (
            vec![0],
            vec!["root"],
            vec![Score::from_confidence(SCORE_PRIMARY)],
        );
    }

    let mut arc_scores = vec![vec![ARC_FORBIDDEN; m + 1]; m];
    for u in 0..m {
        if u == root_local {
            arc_scores[u][0] = ARC_PREFERRED;
        } else {
            arc_scores[u][0] = ARC_FORBIDDEN - 10.0;
            arc_scores[u][root_local + 1] = ARC_FALLBACK;
            if let Some(h) = preferred_heads[u]
                && h != u
                && h < m
            {
                arc_scores[u][h + 1] = ARC_PREFERRED;
            }
        }
    }

    let raw_heads = chu_liu_edmonds(&arc_scores);
    let mut heads = Vec::with_capacity(m);
    let mut deprels = Vec::with_capacity(m);
    let mut scores = Vec::with_capacity(m);

    for u in 0..m {
        let h_raw = raw_heads[u];
        if h_raw == 0 {
            heads.push(u);
            deprels.push("root");
            scores.push(Score::from_confidence(if u == root_local {
                SCORE_PRIMARY
            } else {
                SCORE_FALLBACK
            }));
        } else {
            let h = h_raw - 1;
            heads.push(h);
            if preferred_heads[u] == Some(h) {
                deprels.push(preferred_deprels[u]);
                scores.push(Score::from_confidence(SCORE_PRIMARY));
            } else {
                deprels.push("dep");
                scores.push(Score::from_confidence(SCORE_FALLBACK));
            }
        }
    }

    (heads, deprels, scores)
}

// ============================================================================
// Japanese (Lang::Ja) UPOS Disambiguator + Bunsetsu SOV Parser
// ============================================================================

const JA_CASE_PARTICLES: &[&str] = &[
    "が",
    "は",
    "を",
    "に",
    "で",
    "と",
    "から",
    "まで",
    "より",
    "へ",
    "の",
    "や",
    "も",
    "って",
    "にて",
    "には",
    "では",
    "とは",
    "へは",
    "からも",
    "までも",
    "よりも",
    "だけ",
    "ばかり",
    "など",
    "とか",
];

const JA_DISCOURSE_PARTS: &[&str] = &[
    "か", "よ", "ね", "な", "わ", "さ", "ぞ", "ぜ", "かな", "よね",
];

const JA_SCONJ_MARKS: &[&str] = &[
    "て",
    "で",
    "ば",
    "たら",
    "なら",
    "ので",
    "のに",
    "けれど",
    "けれども",
    "けど",
    "ながら",
    "つつ",
    "たり",
    "だり",
    "し",
    "ちゃ",
    "じゃ",
    "ても",
    "でも",
    "まま",
];

const JA_AUX_WORDS: &[&str] = &[
    "た",
    "だ",
    "ます",
    "ました",
    "ません",
    "ない",
    "なかった",
    "ず",
    "ぬ",
    "ん",
    "れる",
    "られる",
    "せる",
    "させる",
    "たい",
    "たかっ",
    "よう",
    "そう",
    "らしい",
    "べき",
    "です",
    "でした",
    "でし",
    "である",
    "であり",
    "だっ",
    "てる",
    "てた",
    "ちゃう",
    "ちゃっ",
    "じゃう",
    "じゃっ",
    "とく",
    "とい",
    "なきゃ",
    "なくちゃ",
    "すぎ",
    "ほしい",
];

const JA_COPULA_WORDS: &[&str] = &["だ", "です", "である", "でした", "でし", "だっ", "であり"];

const JA_PRONOUNS: &[&str] = &[
    "私",
    "わたし",
    "あたし",
    "僕",
    "ぼく",
    "俺",
    "おれ",
    "あなた",
    "あんた",
    "お前",
    "おまえ",
    "きみ",
    "彼",
    "彼女",
    "彼ら",
    "あいつ",
    "こいつ",
    "そいつ",
    "貴様",
    "俺たち",
    "僕ら",
    "私たち",
    "これ",
    "それ",
    "あれ",
    "どれ",
    "ここ",
    "そこ",
    "あそこ",
    "どこ",
    "誰",
    "だれ",
    "何",
    "なに",
    "なん",
    "自分",
    "みんな",
    "皆さん",
];

const JA_DETS: &[&str] = &[
    "この",
    "その",
    "あの",
    "どの",
    "こんな",
    "そんな",
    "あんな",
    "どんな",
    "ある",
    "あらゆる",
    "すべての",
    "各",
    "同じ",
];

const JA_ADVS: &[&str] = &[
    "とても",
    "非常に",
    "たいへん",
    "かなり",
    "もっと",
    "ずっと",
    "もう",
    "まだ",
    "すでに",
    "いつも",
    "よく",
    "すぐ",
    "すぐに",
    "ゆっくり",
    "はっきり",
    "しっかり",
    "やはり",
    "やっぱり",
    "なぜ",
    "どうして",
    "どう",
    "どのように",
    "まず",
    "さらに",
    "特に",
    "実は",
    "たぶん",
    "おそらく",
    "きっと",
    "必ず",
    "決して",
    "全く",
    "少し",
    "ちょっと",
    "はい",
    "いいえ",
    "静かに",
    "急に",
    "突然",
    "そっと",
    "ぎゅっと",
    "めっちゃ",
    "すごく",
    "激しく",
    "深く",
    "優しく",
];

const JA_CCONJS: &[&str] = &[
    "しかし",
    "だが",
    "けれども",
    "そして",
    "それから",
    "また",
    "それで",
    "だから",
    "したがって",
    "つまり",
    "すなわち",
    "あるいは",
    "または",
    "および",
    "ならびに",
    "ただし",
    "なお",
];

const JA_TRANSFER_VERBS: &[&str] = &[
    "あげる",
    "あげ",
    "渡す",
    "渡し",
    "教える",
    "教え",
    "送る",
    "送り",
    "送っ",
    "貸す",
    "貸し",
    "見せる",
    "見せ",
    "言う",
    "言い",
    "言っ",
    "聞く",
    "聞き",
    "頼む",
    "頼み",
    "くれる",
    "くれ",
    "もらう",
    "もらい",
    "もらっ",
    "与える",
    "与え",
    "渡っ",
];

fn disambiguate_ja_upos(
    doc: &mut Document,
    sent_idx: usize,
    sent_tok_indices: &[usize],
    surfaces: &[String],
) {
    let n = sent_tok_indices.len();
    for i in 0..n {
        let tok_idx = sent_tok_indices[i];
        if doc.tokens[tok_idx].upos.is_some() {
            continue;
        }
        let surf = surfaces[i].as_str();
        let prev_upos = if i > 0 {
            doc.tokens[sent_tok_indices[i - 1]].upos
        } else {
            None
        };
        let next_surf = surfaces.get(i + 1).map(|s| s.as_str());
        let span = doc.tokens[tok_idx].span;
        let ent = entity_kind_for_span(doc, sent_idx, span);

        let decided = infer_ja_upos(
            surf,
            ent,
            prev_upos,
            next_surf,
            &doc.tokens[tok_idx].upos_candidates,
        );
        doc.tokens[tok_idx].upos = Some(decided);
        clear_pos_unknown(doc, span);
    }
}

fn infer_ja_upos(
    surf: &str,
    ent: Option<EntityKind>,
    prev_upos: Option<Upos>,
    next_surf: Option<&str>,
    candidates: &[Upos],
) -> Upos {
    if is_punct_str(surf) {
        return Upos::Punct;
    }
    if is_ascii_digits_str(surf) || is_ja_kanji_numeral(surf) {
        return Upos::Num;
    }
    if ent == Some(EntityKind::Name) {
        return Upos::Propn;
    }
    if ent == Some(EntityKind::Number) {
        return Upos::Num;
    }

    if JA_PRONOUNS.contains(&surf) {
        return Upos::Pron;
    }
    if JA_DETS.contains(&surf) {
        if surf == "ある"
            && (prev_upos == Some(Upos::Adp)
                || next_surf.is_some_and(|ns| {
                    JA_SCONJ_MARKS.contains(&ns)
                        || JA_AUX_WORDS.contains(&ns)
                        || matches!(
                            ns,
                            "からといって"
                                | "ため"
                                | "はず"
                                | "わけ"
                                | "ので"
                                | "から"
                                | "けど"
                                | "けれど"
                                | "けれども"
                                | "と"
                                | "。"
                                | "、"
                                | "が"
                                | "し"
                        )
                }))
        {
            return Upos::Verb;
        }
        return Upos::Det;
    }
    if JA_CCONJS.contains(&surf) {
        return Upos::Cconj;
    }
    if JA_ADVS.contains(&surf) {
        return Upos::Adv;
    }

    // Disambiguate `の` (nominalizer before copula vs genitive/case particle)
    if surf == "の" {
        if matches!(next_surf, Some("だ" | "です" | "である"))
            && matches!(prev_upos, Some(Upos::Verb | Upos::Adj | Upos::Aux))
        {
            return Upos::Sconj;
        }
        return Upos::Adp;
    }
    if surf == "だって" {
        if prev_upos.is_none() || prev_upos == Some(Upos::Punct) {
            return Upos::Cconj;
        }
        return Upos::Part;
    }

    if JA_DISCOURSE_PARTS.contains(&surf) {
        return Upos::Part;
    }

    if matches!(surf, "て" | "で") {
        if matches!(prev_upos, Some(Upos::Verb | Upos::Adj | Upos::Aux)) {
            return Upos::Sconj;
        }
        return Upos::Adp;
    }

    if JA_SCONJ_MARKS.contains(&surf) {
        return Upos::Sconj;
    }

    if JA_CASE_PARTICLES.contains(&surf) {
        return Upos::Adp;
    }

    if JA_AUX_WORDS.contains(&surf) {
        // `ない` after case particle `が`/`は`/`も` is an adjective predicate ("not exist").
        if surf == "ない" && prev_upos == Some(Upos::Adp) {
            return Upos::Adj;
        }
        return Upos::Aux;
    }

    if surf == "し"
        && matches!(prev_upos, Some(Upos::Verb | Upos::Adj | Upos::Aux))
        && !next_surf.is_some_and(|ns| {
            JA_AUX_WORDS.contains(&ns)
                || matches!(
                    ns,
                    "て" | "た"
                        | "ない"
                        | "なかっ"
                        | "なかった"
                        | "ます"
                        | "まし"
                        | "ません"
                        | "よう"
                        | "しろ"
                        | "せよ"
                        | "に"
                        | "ながら"
                )
        })
    {
        return Upos::Sconj;
    }

    if matches!(
        surf,
        "する"
            | "し"
            | "さ"
            | "せ"
            | "すれ"
            | "しろ"
            | "せよ"
            | "できる"
            | "でき"
            | "いる"
            | "い"
            | "ある"
            | "あり"
            | "あっ"
            | "なる"
            | "なり"
            | "なっ"
    ) {
        if matches!(prev_upos, Some(Upos::Sconj)) {
            return Upos::Aux;
        }
        if matches!(surf, "する" | "し" | "できる" | "でき") && prev_upos == Some(Upos::Noun)
        {
            return Upos::Verb;
        }
        return Upos::Verb;
    }

    let lex_hit = lookup_ja(surf);
    if let Some(hit) = lex_hit.as_ref()
        && let Some(u) = hit.decided_upos
    {
        return u;
    }

    if is_katakana_run(surf) {
        return Upos::Noun;
    }

    // Mixed English / ASCII technical term in Japanese.
    if surf
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        return Upos::Noun;
    }

    // Context-guided selection from lexicon candidates (falling back to lexicon lookup when L2 morph was disabled).
    let lex_cands;
    let cands: &[Upos] = if !candidates.is_empty() {
        candidates
    } else if let Some(hit) = lex_hit {
        lex_cands = hit.upos_candidates;
        &lex_cands
    } else {
        &[]
    };
    if !cands.is_empty() {
        if cands.contains(&Upos::Adv)
            && !next_surf.is_some_and(|ns| {
                matches!(
                    ns,
                    "な" | "だって" | "だけ" | "ばかり" | "など" | "くらい" | "ぐらい"
                ) || JA_CASE_PARTICLES.contains(&ns)
                    || JA_COPULA_WORDS.contains(&ns)
            })
        {
            return Upos::Adv;
        }
        if next_surf.is_some_and(|ns| {
            matches!(
                ns,
                "だって" | "だけ" | "ばかり" | "など" | "くらい" | "ぐらい"
            ) || JA_CASE_PARTICLES.contains(&ns)
        }) && cands.contains(&Upos::Noun)
        {
            return Upos::Noun;
        }
        if next_surf.is_some_and(|ns| JA_AUX_WORDS.contains(&ns) || JA_SCONJ_MARKS.contains(&ns)) {
            if cands.contains(&Upos::Verb) {
                return Upos::Verb;
            }
            if cands.contains(&Upos::Adj) {
                return Upos::Adj;
            }
        }
        return cands[0];
    }

    // Okurigana / script heuristics.
    let chars: Vec<char> = surf.chars().collect();
    if chars.iter().all(|&c| is_kanji(c)) {
        return Upos::Noun;
    }
    if chars.iter().any(|&c| is_kanji(c)) && chars.last().is_some_and(|&c| is_hiragana(c)) {
        if surf.ends_with('い') || surf.ends_with("かっ") {
            return Upos::Adj;
        }
        return Upos::Verb;
    }

    Upos::Noun
}

#[derive(Clone, Debug)]
struct Bunsetsu {
    head_local: usize,
    head_upos: Upos,
    has_topic_subj: bool,
    has_obj: bool,
    has_genitive: bool,
    has_dative_ni: bool,
    has_oblique: bool,
    has_coord_part: bool,
    has_copula: bool,
    is_nonfinal_clause: bool,
    is_coord_clause: bool,
}

fn is_ja_content_upos(u: Upos) -> bool {
    !matches!(
        u,
        Upos::Adp | Upos::Part | Upos::Aux | Upos::Sconj | Upos::Punct
    )
}

fn is_ja_nominal_upos(u: Upos) -> bool {
    matches!(u, Upos::Noun | Upos::Propn | Upos::Pron | Upos::Num)
}

fn is_ja_predicate_bunsetsu(b: &Bunsetsu) -> bool {
    matches!(b.head_upos, Upos::Verb | Upos::Adj) || b.has_copula
}

fn parse_ja_window(doc: &Document, tok_indices: &[usize], surfaces: &[&str]) -> WindowTree {
    let m = tok_indices.len();
    let upos: Vec<Upos> = tok_indices
        .iter()
        .map(|&idx| doc.tokens[idx].upos.unwrap_or(Upos::Noun))
        .collect();

    // If all tokens are punctuation, attach everything to the last token.
    if upos.iter().all(|&u| u == Upos::Punct) {
        let root_local = m.saturating_sub(1);
        let mut pref_heads = vec![Some(root_local); m];
        let pref_deprels = vec!["punct"; m];
        pref_heads[root_local] = None;
        let (heads, deprels, scores) =
            solve_window_with_cle(&pref_heads, &pref_deprels, root_local);
        return WindowTree {
            tok_indices: tok_indices.to_vec(),
            heads,
            deprels,
            scores,
            root_local,
        };
    }

    let mut pref_heads: Vec<Option<usize>> = vec![None; m];
    let mut pref_deprels: Vec<&'static str> = vec!["dep"; m];

    // Step 1: Build bunsetsu chunks (grouping compound nominals/numerals + trailing function words).
    let mut bunsetsus: Vec<Bunsetsu> = Vec::new();
    let mut i = 0usize;
    while i < m {
        if upos[i] == Upos::Punct {
            i += 1;
            continue;
        }
        // Collect contiguous content tokens that form one nominal/verbal compound head.
        let start = i;
        let mut head_local = i;
        if is_ja_content_upos(upos[i]) {
            let mut j = i + 1;
            while j < m && is_ja_content_upos(upos[j]) {
                let prev_u = upos[j - 1];
                let cur_u = upos[j];
                // Merge adjacent nominals (`NOUN`/`PROPN`/`NUM` + `NOUN`/`PROPN`)
                // or `NOUN` + light verb `する`/`できる`.
                if (is_ja_nominal_upos(prev_u) && matches!(cur_u, Upos::Noun | Upos::Propn))
                    || (prev_u == Upos::Noun
                        && cur_u == Upos::Verb
                        && matches!(surfaces[j], "する" | "し" | "できる" | "でき"))
                {
                    head_local = j;
                    j += 1;
                } else {
                    break;
                }
            }
            i = j;
        } else {
            // Leading orphan function token: make it a minimal bunsetsu.
            i += 1;
        }

        // Absorb trailing function suffixes (`ADP`, `PART`, `AUX`, `SCONJ`).
        while i < m && matches!(upos[i], Upos::Adp | Upos::Part | Upos::Aux | Upos::Sconj) {
            i += 1;
        }

        let chunk_tokens: Vec<usize> = (start..i).collect();
        let mut b = Bunsetsu {
            head_local,
            head_upos: upos[head_local],
            has_topic_subj: false,
            has_obj: false,
            has_genitive: false,
            has_dative_ni: false,
            has_oblique: false,
            has_coord_part: false,
            has_copula: false,
            is_nonfinal_clause: false,
            is_coord_clause: false,
        };

        // Attach intra-bunsetsu dependents leftward to `head_local`.
        for &t in &chunk_tokens {
            if t == head_local {
                continue;
            }
            if t < head_local {
                let rel = if upos[t] == Upos::Num {
                    "nummod"
                } else {
                    "compound"
                };
                pref_heads[t] = Some(head_local);
                pref_deprels[t] = rel;
            } else {
                let s = surfaces[t];
                let rel = match upos[t] {
                    Upos::Adp => {
                        if matches!(s, "が" | "は" | "も") {
                            b.has_topic_subj = true;
                        } else if s == "を" {
                            b.has_obj = true;
                        } else if s == "の" {
                            b.has_genitive = true;
                        } else if s == "に" {
                            b.has_dative_ni = true;
                            b.has_oblique = true;
                        } else if matches!(s, "や" | "とか") {
                            b.has_coord_part = true;
                        } else {
                            b.has_oblique = true;
                        }
                        "case"
                    }
                    Upos::Part => "discourse",
                    Upos::Sconj => {
                        b.is_nonfinal_clause = true;
                        if matches!(s, "し" | "たり" | "だり") {
                            b.is_coord_clause = true;
                        }
                        "mark"
                    }
                    Upos::Aux => {
                        if JA_COPULA_WORDS.contains(&s) && !matches!(b.head_upos, Upos::Verb) {
                            b.has_copula = true;
                            "cop"
                        } else {
                            "aux"
                        }
                    }
                    _ => "dep",
                };
                pref_heads[t] = Some(head_local);
                pref_deprels[t] = rel;
            }
        }

        // Check if a trailing comma immediately follows a predicate bunsetsu.
        if i < m && surfaces[i] == "、" && is_ja_predicate_bunsetsu(&b) {
            b.is_nonfinal_clause = true;
        }

        bunsetsus.push(b);
    }

    // Step 2: Choose global root bunsetsu (rightmost predicate, else rightmost nominal).
    let root_b_idx = bunsetsus
        .iter()
        .rposition(is_ja_predicate_bunsetsu)
        .or_else(|| {
            bunsetsus
                .iter()
                .rposition(|b| is_ja_nominal_upos(b.head_upos))
        })
        .unwrap_or_else(|| bunsetsus.len().saturating_sub(1));

    let root_local = bunsetsus
        .get(root_b_idx)
        .map(|b| b.head_local)
        .unwrap_or(m - 1);

    // Step 3: Link each non-root bunsetsu head rightward (head-final SOV).
    for bi in 0..bunsetsus.len() {
        if bi == root_b_idx {
            continue;
        }
        let b = &bunsetsus[bi];
        let mut target_b: Option<usize> = None;
        let mut deprel: &'static str = "dep";

        if b.has_genitive {
            target_b =
                find_right_bunsetsu(&bunsetsus, bi, |cand| is_ja_nominal_upos(cand.head_upos));
            deprel = "nmod";
        } else if b.has_coord_part {
            target_b =
                find_right_bunsetsu(&bunsetsus, bi, |cand| is_ja_nominal_upos(cand.head_upos));
            deprel = "conj";
        } else if b.head_upos == Upos::Det {
            target_b =
                find_right_bunsetsu(&bunsetsus, bi, |cand| is_ja_nominal_upos(cand.head_upos));
            deprel = "det";
        } else if b.head_upos == Upos::Cconj {
            target_b =
                find_right_bunsetsu(&bunsetsus, bi, is_ja_predicate_bunsetsu).or(Some(root_b_idx));
            deprel = "cc";
        } else if b.head_upos == Upos::Adv {
            target_b = find_right_bunsetsu(&bunsetsus, bi, |cand| {
                is_ja_predicate_bunsetsu(cand) || cand.head_upos == Upos::Adv
            });
            deprel = "advmod";
        } else if b.head_upos == Upos::Adj && !b.is_nonfinal_clause && !b.has_copula {
            // Adnominal adjective directly modifying a following noun, else adverbial/clausal.
            if let Some(nb) =
                find_right_bunsetsu(&bunsetsus, bi, |cand| is_ja_nominal_upos(cand.head_upos))
                && nb == bi + 1
            {
                target_b = Some(nb);
                deprel = "amod";
            } else {
                target_b = find_right_bunsetsu(&bunsetsus, bi, is_ja_predicate_bunsetsu);
                deprel = "advcl";
            }
        } else if b.has_topic_subj {
            target_b = find_right_bunsetsu(&bunsetsus, bi, is_ja_predicate_bunsetsu);
            deprel = "nsubj";
        } else if b.has_obj {
            target_b = find_right_bunsetsu(&bunsetsus, bi, |cand| cand.head_upos == Upos::Verb);
            deprel = "obj";
        } else if b.has_oblique {
            target_b = find_right_bunsetsu(&bunsetsus, bi, is_ja_predicate_bunsetsu);
            if b.has_dative_ni
                && let Some(tb) = target_b
                && JA_TRANSFER_VERBS.contains(&surfaces[bunsetsus[tb].head_local])
            {
                deprel = "iobj";
            } else {
                deprel = "obl";
            }
        } else if is_ja_predicate_bunsetsu(b) {
            // Relative clause before an immediate nominal if not marked as non-final clause.
            if !b.is_nonfinal_clause
                && let Some(nb) = bunsetsus.get(bi + 1)
                && is_ja_nominal_upos(nb.head_upos)
            {
                target_b = Some(bi + 1);
                deprel = "acl";
            } else {
                target_b = find_right_bunsetsu(&bunsetsus, bi, is_ja_predicate_bunsetsu);
                deprel = if b.is_coord_clause { "conj" } else { "advcl" };
            }
        } else if is_ja_nominal_upos(b.head_upos) {
            if let Some(nb) = bunsetsus.get(bi + 1)
                && is_ja_nominal_upos(nb.head_upos)
            {
                target_b = Some(bi + 1);
                deprel = "nmod";
            } else {
                target_b = find_right_bunsetsu(&bunsetsus, bi, is_ja_predicate_bunsetsu);
                deprel = "obl";
            }
        }

        let target_head = target_b
            .and_then(|tb| bunsetsus.get(tb))
            .map(|tb| tb.head_local)
            .unwrap_or(root_local);
        let target_head = if target_head == b.head_local {
            root_local
        } else {
            target_head
        };

        pref_heads[b.head_local] = Some(target_head);
        pref_deprels[b.head_local] = deprel;
    }

    // Step 4: Attach punctuation to nearest left/right clause head or global root.
    for u in 0..m {
        if upos[u] != Upos::Punct {
            continue;
        }
        let h = if matches!(surfaces[u], "、" | ",") {
            bunsetsus
                .iter()
                .rev()
                .find(|b| b.head_local < u && is_ja_predicate_bunsetsu(b))
                .map(|b| b.head_local)
                .unwrap_or(root_local)
        } else {
            root_local
        };
        pref_heads[u] = Some(if h == u { root_local } else { h });
        pref_deprels[u] = "punct";
    }

    let (heads, deprels, scores) = solve_window_with_cle(&pref_heads, &pref_deprels, root_local);
    WindowTree {
        tok_indices: tok_indices.to_vec(),
        heads,
        deprels,
        scores,
        root_local,
    }
}

fn find_right_bunsetsu<F>(bunsetsus: &[Bunsetsu], from: usize, pred: F) -> Option<usize>
where
    F: Fn(&Bunsetsu) -> bool,
{
    ((from + 1)..bunsetsus.len()).find(|&j| pred(&bunsetsus[j]))
}

// ============================================================================
// Korean (Lang::Ko) UPOS Disambiguator + Eojeol SOV Parser
// ============================================================================

const KO_PRONOUNS: &[&str] = &[
    "나",
    "나는",
    "내가",
    "나를",
    "내",
    "나의",
    "저",
    "저는",
    "제가",
    "저를",
    "저의",
    "너",
    "너는",
    "네가",
    "너를",
    "우리",
    "우리는",
    "우리가",
    "우리를",
    "우리의",
    "그",
    "그는",
    "그가",
    "그를",
    "그녀",
    "그녀는",
    "그녀가",
    "그녀를",
    "그들",
    "그들은",
    "이것",
    "그것",
    "저것",
    "무엇",
    "누구",
    "어디",
    "언제",
    "자기",
    "자신",
];

const KO_DETS: &[&str] = &[
    "이", "그", "저", "어느", "무슨", "어떤", "모든", "각", "새", "헌", "옛", "온갖", "한", "두",
    "세", "네",
];

const KO_ADVS: &[&str] = &[
    "매우",
    "아주",
    "너무",
    "정말",
    "진짜",
    "참",
    "가장",
    "제일",
    "더",
    "덜",
    "잘",
    "못",
    "안",
    "이미",
    "벌써",
    "아직",
    "항상",
    "늘",
    "자주",
    "가끔",
    "다시",
    "또",
    "곧",
    "바로",
    "빨리",
    "천천히",
    "함께",
    "같이",
    "혼자",
    "다",
    "모두",
    "많이",
    "조금",
    "약간",
    "어떻게",
    "왜",
    "조용히",
    "열심히",
];

const KO_CCONJS: &[&str] = &[
    "그리고",
    "그러나",
    "하지만",
    "그런데",
    "그래서",
    "따라서",
    "그러므로",
    "또는",
    "혹은",
    "및",
];

const KO_ADNOMINAL_ADJ: &[&str] = &[
    "큰",
    "작은",
    "좋은",
    "나쁜",
    "새로운",
    "오래된",
    "아름다운",
    "차가운",
    "따뜻한",
    "많은",
    "적은",
    "높은",
    "낮은",
    "긴",
    "짧은",
    "예쁜",
    "빠른",
    "느린",
    "어려운",
    "쉬운",
];

const KO_JOSA_SUFFIXES: &[&str] = &[
    "에서는",
    "에서도",
    "으로는",
    "으로도",
    "에게서",
    "에게는",
    "한테서",
    "까지는",
    "부터는",
    "께서",
    "에서",
    "으로",
    "에게",
    "한테",
    "부터",
    "까지",
    "보다",
    "처럼",
    "마다",
    "하고",
    "이랑",
    "이",
    "가",
    "은",
    "는",
    "을",
    "를",
    "에",
    "로",
    "의",
    "와",
    "과",
    "도",
    "만",
    "께",
    "랑",
];

const KO_EOMI_FINAL: &[&str] = &[
    "습니다",
    "ㅂ니다",
    "습니까",
    "ㅂ니까",
    "었다",
    "았다",
    "였다",
    "겠다",
    "는다",
    "ㄴ다",
    "다",
    "요",
    "까",
    "지",
];

const KO_EOMI_COORD: &[&str] = &["지만", "거나", "고", "며", "으며"];

const KO_EOMI_SUBORD: &[&str] = &[
    "으니까",
    "어서",
    "아서",
    "여서",
    "는데",
    "은데",
    "ㄴ데",
    "려고",
    "으려고",
    "도록",
    "니까",
    "으면",
    "면",
    "자",
    "러",
    "게",
];

const KO_TRANSFER_STEMS: &[&str] = &[
    "주",
    "보내",
    "가르치",
    "전하",
    "드리",
    "건네",
    "빌리",
    "물어보",
    "말하",
];

fn disambiguate_ko_upos(
    doc: &mut Document,
    sent_idx: usize,
    sent_tok_indices: &[usize],
    surfaces: &[String],
) {
    let n = sent_tok_indices.len();
    for i in 0..n {
        let tok_idx = sent_tok_indices[i];
        if doc.tokens[tok_idx].upos.is_some() {
            continue;
        }
        let surf = surfaces[i].as_str();
        let next_upos = if i + 1 < n {
            doc.tokens[sent_tok_indices[i + 1]].upos
        } else {
            None
        };
        let span = doc.tokens[tok_idx].span;
        let ent = entity_kind_for_span(doc, sent_idx, span);

        let decided = infer_ko_upos(
            surf,
            span,
            ent,
            next_upos,
            &doc.tokens[tok_idx].upos_candidates,
        );
        doc.tokens[tok_idx].upos = Some(decided);
        clear_pos_unknown(doc, span);
    }
}

fn infer_ko_upos(
    surf: &str,
    span: Span,
    ent: Option<EntityKind>,
    next_upos: Option<Upos>,
    candidates: &[Upos],
) -> Upos {
    if is_punct_str(surf) {
        return Upos::Punct;
    }
    if is_ascii_digits_str(surf) {
        return Upos::Num;
    }
    if ent == Some(EntityKind::Name) {
        return Upos::Propn;
    }
    if ent == Some(EntityKind::Number) {
        return Upos::Num;
    }

    if KO_PRONOUNS.contains(&surf) {
        return Upos::Pron;
    }
    if KO_CCONJS.contains(&surf) {
        return Upos::Cconj;
    }
    if KO_ADVS.contains(&surf) {
        return Upos::Adv;
    }
    if KO_DETS.contains(&surf) {
        return Upos::Det;
    }
    if KO_ADNOMINAL_ADJ.contains(&surf) {
        return Upos::Adj;
    }
    if KO_JOSA_SUFFIXES.contains(&surf) {
        return Upos::Adp;
    }

    // Consult morphological analyzer on the eojeol.
    let chars: Vec<char> = surf.chars().collect();
    let lex = analyze_ko_eojeol(&chars, span.start_char);
    if let Some(u) = lex.decided_upos {
        return u;
    }

    // Mixed English / Latin technical term with or without Korean josa.
    if surf.chars().next().is_some_and(|c| c.is_ascii_alphabetic()) {
        return Upos::Noun;
    }

    // Check explicit verbal / adjectival endings.
    if KO_EOMI_FINAL
        .iter()
        .chain(KO_EOMI_COORD.iter())
        .chain(KO_EOMI_SUBORD.iter())
        .any(|suf| surf.len() > suf.len() && surf.ends_with(suf))
    {
        if candidates.contains(&Upos::Adj) && !candidates.contains(&Upos::Verb) {
            return Upos::Adj;
        }
        return Upos::Verb;
    }

    // Adnominal endings before a nominal.
    if matches!(next_upos, Some(Upos::Noun | Upos::Propn | Upos::Pron))
        && (surf.ends_with('은')
            || surf.ends_with('는')
            || surf.ends_with('을')
            || surf.ends_with('ㄹ')
            || surf.ends_with('던'))
    {
        if candidates.contains(&Upos::Adj) {
            return Upos::Adj;
        }
        if candidates.contains(&Upos::Verb) {
            return Upos::Verb;
        }
    }

    // Adverbial suffix `-게`, `-히`, `-리`.
    if (surf.ends_with('히') || surf.ends_with('리')) && surf.chars().count() >= 2 {
        return Upos::Adv;
    }

    // Nominal with josa.
    if KO_JOSA_SUFFIXES
        .iter()
        .any(|suf| surf.len() > suf.len() && surf.ends_with(suf))
    {
        return Upos::Noun;
    }

    if !candidates.is_empty() {
        return candidates[0];
    }

    Upos::Noun
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum KoRole {
    Genitive,
    SubjTopic,
    Object,
    Dative,
    ConjNominal,
    Oblique,
    AdnominalAdj,
    AdnominalVerb,
    CoordPred,
    SubordPred,
    FinalPred,
    Plain,
}

fn classify_ko_eojeol(surf: &str, upos: Upos, has_right_nominal: bool) -> KoRole {
    if matches!(upos, Upos::Verb | Upos::Adj) {
        if KO_EOMI_COORD
            .iter()
            .any(|s| surf.len() > s.len() && surf.ends_with(s))
        {
            return KoRole::CoordPred;
        }
        if KO_EOMI_SUBORD
            .iter()
            .any(|s| surf.len() > s.len() && surf.ends_with(s))
        {
            return KoRole::SubordPred;
        }
        if KO_EOMI_FINAL
            .iter()
            .any(|s| surf.len() > s.len() && surf.ends_with(s))
        {
            return KoRole::FinalPred;
        }
        if has_right_nominal
            && (KO_ADNOMINAL_ADJ.contains(&surf)
                || surf.ends_with('은')
                || surf.ends_with('는')
                || surf.ends_with('을')
                || surf.ends_with('던')
                || last_char_has_nieun_or_rieul(surf))
        {
            return if upos == Upos::Adj {
                KoRole::AdnominalAdj
            } else {
                KoRole::AdnominalVerb
            };
        }
        return KoRole::FinalPred;
    }

    if is_ja_nominal_upos(upos) {
        if surf.ends_with("의") && surf.chars().count() >= 2 {
            return KoRole::Genitive;
        }
        if ["께서", "이", "가", "은", "는"]
            .iter()
            .any(|s| surf.len() > s.len() && surf.ends_with(s))
            || matches!(
                surf,
                "나는"
                    | "내가"
                    | "저는"
                    | "제가"
                    | "우리는"
                    | "우리가"
                    | "그는"
                    | "그가"
                    | "그녀는"
                    | "그녀가"
                    | "너는"
                    | "네가"
            )
        {
            return KoRole::SubjTopic;
        }
        if ["을", "를"]
            .iter()
            .any(|s| surf.len() > s.len() && surf.ends_with(s))
            || matches!(
                surf,
                "나를" | "저를" | "우리를" | "그를" | "그녀를" | "너를"
            )
        {
            return KoRole::Object;
        }
        if ["에게", "한테", "께"]
            .iter()
            .any(|s| surf.len() > s.len() && surf.ends_with(s))
        {
            return KoRole::Dative;
        }
        if ["와", "과"]
            .iter()
            .any(|s| surf.len() > s.len() && surf.ends_with(s))
        {
            return KoRole::ConjNominal;
        }
        if KO_JOSA_SUFFIXES
            .iter()
            .any(|s| surf.len() > s.len() && surf.ends_with(s))
        {
            return KoRole::Oblique;
        }
        // Copular predicate eojeol like `학생이다` / `책입니다`.
        if surf.ends_with("이다") || surf.ends_with("입니다") || surf.ends_with("였다") {
            return KoRole::FinalPred;
        }
    }

    KoRole::Plain
}

fn last_char_has_nieun_or_rieul(surf: &str) -> bool {
    let Some(c) = surf.chars().last() else {
        return false;
    };
    let code = c as u32;
    if !(0xAC00..=0xD7A3).contains(&code) {
        return false;
    }
    let jong = (code - 0xAC00) % 28;
    // 4 = ㄴ, 8 = ㄹ
    jong == 4 || jong == 8
}

fn is_ko_transfer_verb(surf: &str) -> bool {
    KO_TRANSFER_STEMS.iter().any(|stem| surf.starts_with(stem))
}

fn is_ko_pred(upos: Upos, role: KoRole) -> bool {
    matches!(upos, Upos::Verb | Upos::Adj) || role == KoRole::FinalPred
}

fn parse_ko_window(doc: &Document, tok_indices: &[usize], surfaces: &[&str]) -> WindowTree {
    let m = tok_indices.len();
    let upos: Vec<Upos> = tok_indices
        .iter()
        .map(|&idx| doc.tokens[idx].upos.unwrap_or(Upos::Noun))
        .collect();

    if upos.iter().all(|&u| u == Upos::Punct) {
        let root_local = m.saturating_sub(1);
        let mut pref_heads = vec![Some(root_local); m];
        let pref_deprels = vec!["punct"; m];
        pref_heads[root_local] = None;
        let (heads, deprels, scores) =
            solve_window_with_cle(&pref_heads, &pref_deprels, root_local);
        return WindowTree {
            tok_indices: tok_indices.to_vec(),
            heads,
            deprels,
            scores,
            root_local,
        };
    }

    let mut roles = Vec::with_capacity(m);
    for i in 0..m {
        let has_right_nominal = i + 1 < m && is_ja_nominal_upos(upos[i + 1]);
        let mut role = classify_ko_eojeol(surfaces[i], upos[i], has_right_nominal);
        // Propagate standalone post-nominal josa (e.g., `Rust` + `로`, `API` + `를`).
        if role == KoRole::Plain
            && is_ja_nominal_upos(upos[i])
            && i + 1 < m
            && upos[i + 1] == Upos::Adp
        {
            role = match surfaces[i + 1] {
                "의" => KoRole::Genitive,
                "께서" | "이" | "가" | "은" | "는" => KoRole::SubjTopic,
                "을" | "를" => KoRole::Object,
                "에게" | "한테" | "께" => KoRole::Dative,
                "와" | "과" => KoRole::ConjNominal,
                _ => KoRole::Oblique,
            };
        }
        roles.push(role);
    }

    let root_local = (0..m)
        .rfind(|&i| is_ko_pred(upos[i], roles[i]))
        .or_else(|| (0..m).rfind(|&i| is_ja_nominal_upos(upos[i])))
        .unwrap_or_else(|| m.saturating_sub(1));

    let mut pref_heads: Vec<Option<usize>> = vec![None; m];
    let mut pref_deprels: Vec<&'static str> = vec!["dep"; m];

    for i in 0..m {
        if i == root_local {
            continue;
        }
        if upos[i] == Upos::Adp && i > 0 && is_ja_nominal_upos(upos[i - 1]) {
            pref_heads[i] = Some(i - 1);
            pref_deprels[i] = "case";
            continue;
        }
        if upos[i] == Upos::Punct {
            let h = if matches!(surfaces[i], "," | "、") {
                (0..i)
                    .rfind(|&j| is_ko_pred(upos[j], roles[j]))
                    .unwrap_or(root_local)
            } else {
                root_local
            };
            pref_heads[i] = Some(if h == i { root_local } else { h });
            pref_deprels[i] = "punct";
            continue;
        }

        let target: Option<usize>;
        let deprel: &'static str;

        match roles[i] {
            KoRole::Genitive => {
                target = ((i + 1)..m).find(|&j| is_ja_nominal_upos(upos[j]));
                deprel = "nmod";
            }
            KoRole::AdnominalAdj => {
                target = ((i + 1)..m).find(|&j| is_ja_nominal_upos(upos[j]));
                deprel = "amod";
            }
            KoRole::AdnominalVerb => {
                target = ((i + 1)..m).find(|&j| is_ja_nominal_upos(upos[j]));
                deprel = "acl";
            }
            KoRole::SubjTopic => {
                target = ((i + 1)..m).find(|&j| is_ko_pred(upos[j], roles[j]));
                deprel = "nsubj";
            }
            KoRole::Object => {
                target = ((i + 1)..m).find(|&j| upos[j] == Upos::Verb);
                deprel = "obj";
            }
            KoRole::Dative => {
                target = ((i + 1)..m).find(|&j| upos[j] == Upos::Verb);
                if let Some(t) = target
                    && is_ko_transfer_verb(surfaces[t])
                {
                    deprel = "iobj";
                } else {
                    deprel = "obl";
                }
            }
            KoRole::ConjNominal => {
                if let Some(j) = ((i + 1)..m).find(|&j| is_ja_nominal_upos(upos[j])) {
                    target = Some(j);
                    deprel = "conj";
                } else {
                    target = ((i + 1)..m).find(|&j| is_ko_pred(upos[j], roles[j]));
                    deprel = "obl";
                }
            }
            KoRole::Oblique => {
                target = ((i + 1)..m).find(|&j| is_ko_pred(upos[j], roles[j]));
                deprel = "obl";
            }
            KoRole::CoordPred => {
                target = ((i + 1)..m).find(|&j| is_ko_pred(upos[j], roles[j]));
                deprel = "conj";
            }
            KoRole::SubordPred | KoRole::FinalPred => {
                target = ((i + 1)..m).find(|&j| is_ko_pred(upos[j], roles[j]));
                deprel = "advcl";
            }
            KoRole::Plain => {
                if upos[i] == Upos::Det {
                    target = ((i + 1)..m).find(|&j| is_ja_nominal_upos(upos[j]));
                    deprel = "det";
                } else if upos[i] == Upos::Num {
                    target = ((i + 1)..m).find(|&j| is_ja_nominal_upos(upos[j]));
                    deprel = "nummod";
                } else if upos[i] == Upos::Adv {
                    target = ((i + 1)..m)
                        .find(|&j| is_ko_pred(upos[j], roles[j]) || upos[j] == Upos::Adv);
                    deprel = "advmod";
                } else if upos[i] == Upos::Cconj {
                    target = ((i + 1)..m)
                        .find(|&j| is_ko_pred(upos[j], roles[j]))
                        .or(Some(root_local));
                    deprel = "cc";
                } else if is_ja_nominal_upos(upos[i])
                    && i + 1 < m
                    && is_ja_nominal_upos(upos[i + 1])
                {
                    target = Some(i + 1);
                    deprel = "compound";
                } else {
                    target = ((i + 1)..m).find(|&j| is_ko_pred(upos[j], roles[j]));
                    deprel = "obl";
                }
            }
        }

        let h = target.unwrap_or(root_local);
        pref_heads[i] = Some(if h == i { root_local } else { h });
        pref_deprels[i] = deprel;
    }

    let (heads, deprels, scores) = solve_window_with_cle(&pref_heads, &pref_deprels, root_local);
    WindowTree {
        tok_indices: tok_indices.to_vec(),
        heads,
        deprels,
        scores,
        root_local,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::doc::syntax::SYNTAX_THRESHOLD_MILLI;
    use crate::doc::{Options, analyze_document};

    fn run_jako(text: &str) -> Document {
        let mut doc = analyze_document(text, &Options::default());
        for s_idx in 0..doc.segmentation.sentences.len() {
            annotate_sentence(&mut doc, text, s_idx);
        }
        doc
    }

    fn assert_valid_single_root_trees(doc: &Document) {
        for s_idx in 0..doc.segmentation.sentences.len() {
            let sent_toks: Vec<usize> = doc
                .tokens
                .iter()
                .enumerate()
                .filter(|(_, t)| t.sentence == s_idx)
                .map(|(i, _)| i)
                .collect();
            if sent_toks.is_empty() {
                continue;
            }
            for &ti in &sent_toks {
                assert!(
                    doc.tokens[ti].upos.is_some(),
                    "sentence {s_idx} token {ti} still has upos == None"
                );
                let sp = doc.tokens[ti].span;
                assert!(
                    !doc.unknown.iter().any(|u| u.layer == "pos" && u.span == sp),
                    "sentence {s_idx} token {ti} still has unresolved pos unknown"
                );
            }

            let emitted: Vec<&SyntaxToken> = doc
                .syntax
                .iter()
                .filter(|st| st.sentence == s_idx)
                .collect();
            assert_eq!(
                emitted.len(),
                sent_toks.len(),
                "sentence {s_idx} emitted syntax count mismatch"
            );

            let roots: Vec<&&SyntaxToken> = emitted.iter().filter(|st| st.head.is_none()).collect();
            assert_eq!(roots.len(), 1, "sentence {s_idx} must have exactly 1 root");
            assert_eq!(roots[0].deprel, "root");

            for st in &emitted {
                assert!(
                    DEPRELS.contains(&st.deprel.as_str()),
                    "invalid deprel {}",
                    st.deprel
                );
                assert!(
                    st.score.milli() >= SYNTAX_THRESHOLD_MILLI,
                    "score below threshold: {:?}",
                    st.score
                );
                if let Some(h) = st.head {
                    assert_ne!(h, st.token, "self-loop at token {}", st.token);
                    assert!(
                        emitted.iter().any(|p| p.token == h),
                        "orphan head {h} for token {}",
                        st.token
                    );
                }
            }

            // Acyclicity check.
            for st in &emitted {
                let mut cur = *st;
                let mut hops = 0usize;
                while let Some(h) = cur.head {
                    hops += 1;
                    assert!(
                        hops <= emitted.len(),
                        "cycle detected at token {}",
                        st.token
                    );
                    cur = *emitted.iter().find(|p| p.token == h).unwrap();
                }
            }
        }
    }

    fn syn_for<'a>(doc: &'a Document, text: &str, surface: &str) -> &'a SyntaxToken {
        let chars: Vec<char> = text.chars().collect();
        doc.syntax
            .iter()
            .find(|st| slice_chars(&chars, st.span) == surface)
            .unwrap_or_else(|| panic!("missing syntax token for surface `{surface}` in `{text}`"))
    }

    #[test]
    fn ja_novel_sov_and_bunsetsu() {
        let text = "私は昨日図書館で新しい本を三冊借りた。";
        let doc = run_jako(text);
        assert_valid_single_root_trees(&doc);

        let root = doc.syntax.iter().find(|s| s.head.is_none()).unwrap();
        assert_eq!(root.deprel, "root");

        let watashi = syn_for(&doc, text, "私");
        assert_eq!(watashi.deprel, "nsubj");

        let hon = syn_for(&doc, text, "本");
        assert_eq!(hon.deprel, "obj");

        let wo = syn_for(&doc, text, "を");
        assert_eq!(wo.deprel, "case");
        assert_eq!(wo.head, Some(hon.token));
    }

    #[test]
    fn ja_academic_prose_and_genitive() {
        let text = "本研究では大規模言語モデルの構文解析精度を評価する。";
        let doc = run_jako(text);
        assert_valid_single_root_trees(&doc);

        let no = syn_for(&doc, text, "の");
        assert_eq!(no.deprel, "case");
    }

    #[test]
    fn ja_dialogue_and_particles() {
        let text = "「明日、東京駅に行きますか？」「はい、行きますよ。」";
        let doc = run_jako(text);
        assert_valid_single_root_trees(&doc);

        let ka = syn_for(&doc, text, "か");
        assert_eq!(ka.deprel, "discourse");
        let yo = syn_for(&doc, text, "よ");
        assert_eq!(yo.deprel, "discourse");
    }

    #[test]
    fn ja_passive_causative_and_transfer() {
        let passive = "彼は先生に褒められた。";
        let doc_p = run_jako(passive);
        assert_valid_single_root_trees(&doc_p);
        assert_eq!(syn_for(&doc_p, passive, "彼").deprel, "nsubj");
        assert_eq!(syn_for(&doc_p, passive, "先生").deprel, "obl");

        let transfer = "私は友達に手紙を送った。";
        let doc_t = run_jako(transfer);
        assert_valid_single_root_trees(&doc_t);
        assert_eq!(syn_for(&doc_t, transfer, "友達").deprel, "iobj");
        assert_eq!(syn_for(&doc_t, transfer, "手紙").deprel, "obj");
    }

    #[test]
    fn ja_multiclause_sentence() {
        let text = "彼女が静かに窓を開けると、冷たい風が入ってきた。";
        let doc = run_jako(text);
        assert_valid_single_root_trees(&doc);
        assert!(doc.syntax.iter().any(|s| s.deprel == "advcl"));
    }

    #[test]
    fn ko_novel_eojeol_sov() {
        let text = "나는 어제 서울역에서 역사 책 3권을 샀다.";
        let doc = run_jako(text);
        assert_valid_single_root_trees(&doc);

        assert_eq!(syn_for(&doc, text, "나는").deprel, "nsubj");
        assert_eq!(syn_for(&doc, text, "서울역에서").deprel, "obl");
        assert_eq!(syn_for(&doc, text, "3").deprel, "nummod");
        assert_eq!(syn_for(&doc, text, "권을").deprel, "obj");
        assert_eq!(syn_for(&doc, text, "샀다").deprel, "root");
    }

    #[test]
    fn ko_academic_genitive_and_conj() {
        let text = "본 연구에서는 한국어 형태소 분석과 구문 분석의 정확도를 평가한다.";
        let doc = run_jako(text);
        assert_valid_single_root_trees(&doc);

        assert_eq!(syn_for(&doc, text, "분석의").deprel, "nmod");
        assert_eq!(syn_for(&doc, text, "정확도를").deprel, "obj");
        assert_eq!(syn_for(&doc, text, "평가한다").deprel, "root");
    }

    #[test]
    fn ko_dialogue_transfer_iobj() {
        let text = "선생님께서 학생에게 책을 주셨습니다.";
        let doc = run_jako(text);
        assert_valid_single_root_trees(&doc);

        assert_eq!(syn_for(&doc, text, "선생님께서").deprel, "nsubj");
        assert_eq!(syn_for(&doc, text, "학생에게").deprel, "iobj");
        assert_eq!(syn_for(&doc, text, "책을").deprel, "obj");
    }

    #[test]
    fn ko_multiclause_sentence() {
        let text = "비가 많이 와서 우리는 집에 머물렀고 영화를 봤다.";
        let doc = run_jako(text);
        assert_valid_single_root_trees(&doc);

        assert_eq!(syn_for(&doc, text, "와서").deprel, "advcl");
        assert_eq!(syn_for(&doc, text, "머물렀고").deprel, "conj");
        assert_eq!(syn_for(&doc, text, "봤다").deprel, "root");
    }

    #[test]
    fn mixed_english_terms_in_ja_and_ko() {
        let ja_text = "Rustで新しいAPIを実装した。";
        let doc_ja = run_jako(ja_text);
        assert_valid_single_root_trees(&doc_ja);
        assert_eq!(syn_for(&doc_ja, ja_text, "Rust").deprel, "obl");
        assert_eq!(syn_for(&doc_ja, ja_text, "API").deprel, "obj");

        let ko_text = "우리는 Rust로 새로운 API를 구현했다.";
        let doc_ko = run_jako(ko_text);
        assert_valid_single_root_trees(&doc_ko);
        assert_eq!(syn_for(&doc_ko, ko_text, "우리는").deprel, "nsubj");
        assert_eq!(syn_for(&doc_ko, ko_text, "Rust").deprel, "obl");
        assert_eq!(syn_for(&doc_ko, ko_text, "로").deprel, "case");
        assert_eq!(syn_for(&doc_ko, ko_text, "API").deprel, "obj");
        assert_eq!(syn_for(&doc_ko, ko_text, "를").deprel, "case");
    }

    #[test]
    fn edge_cases_single_token_punct_only_and_long_sentence() {
        let doc_1_ja = run_jako("走る");
        assert_valid_single_root_trees(&doc_1_ja);

        let doc_1_ko = run_jako("간다");
        assert_valid_single_root_trees(&doc_1_ko);

        let doc_punct = run_jako("。。。");
        assert_valid_single_root_trees(&doc_punct);

        // > 96 tokens sentence across clauses.
        let mut long_ja = String::new();
        for _ in 0..25 {
            long_ja.push_str("私は本を読み、");
        }
        long_ja.push_str("最後に寝た。");
        let doc_long = run_jako(&long_ja);
        assert!(doc_long.tokens.len() > SOFT_WINDOW_MAX_LEN);
        assert_valid_single_root_trees(&doc_long);
        assert!(doc_long.syntax.iter().any(|s| s.deprel == "parataxis"));
    }
}
