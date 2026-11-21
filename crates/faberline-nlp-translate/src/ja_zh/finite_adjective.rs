//! Taiwanese Mandarin rendering of a Japanese adjective used as a finite predicate.

use super::*;

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Context {
    question_mark: bool,
    pub(super) before_comma: bool,
    time_head: Option<usize>,
    preceding_until: Option<usize>,
    reported_predicate: Option<usize>,
    coordinate: Option<Coordinate>,
}

#[derive(Clone, Copy, Debug)]
enum Coordinate {
    First,
    Next,
}

// The splitter gives each side of 重くて大きい its own clause. Only share
// the subject when て links the two adjectives and neither has another head.
fn coordinated(left: &Segment, right: &Segment) -> bool {
    let (
        Segment::Clause { atoms: first, .. },
        Segment::Clause {
            atoms: second,
            trailing_punct,
            ..
        },
    ) = (left, right)
    else {
        return false;
    };
    let [subject @ .., first_adj, te] = first.as_slice() else {
        return false;
    };
    let [second_adj, tail @ ..] = second.as_slice() else {
        return false;
    };
    let a = first_adj.lemma.as_deref().unwrap_or(&first_adj.surface);
    let b = second_adj.lemma.as_deref().unwrap_or(&second_adj.surface);
    !first_adj.is_locked
        && !second_adj.is_locked
        && !te.is_locked
        && first_adj.upos == Some(Upos::Adj)
        && second_adj.upos == Some(Upos::Adj)
        && first_adj.surface.ends_with('く')
        && first_adj.head == second_adj.doc_tok_idx
        && first_adj.deprel == "advcl"
        && second_adj.deprel == "root"
        && te.surface == "て"
        && te.head == first_adj.doc_tok_idx
        && te.deprel == "mark"
        && gradable(a)
        && gradable(b)
        && inflection(second_adj, tail) == Some((false, false))
        && !matches!(trailing_punct.as_deref(), Some("?" | "？"))
        && subject.iter().any(|n| {
            n.deprel == "nsubj"
                && n.head == first_adj.doc_tok_idx
                && !uncertain_sense(a, &n.surface)
                && !uncertain_sense(b, &n.surface)
        })
        && !subject.iter().any(|n| {
            matches!(n.deprel.as_str(), "advmod" | "obl") && n.head == first_adj.doc_tok_idx
        })
}

impl Context {
    pub(super) fn in_quote(
        segments: &[Segment],
        idx: usize,
        sentence: &[Atom],
        doc: &Document,
    ) -> Self {
        let mut ctx = Self::at(segments, idx, doc);
        let Some(Segment::Clause { atoms, .. }) = segments.get(idx) else {
            return ctx;
        };
        ctx.reported_predicate = atoms
            .iter()
            .rfind(|a| {
                !a.is_locked
                    && (a.upos == Some(Upos::Adj)
                        || a.upos == Some(Upos::Verb) && a.lemma.as_deref() == Some("ある"))
                    && a.deprel == "advcl"
                    && a.head.is_some()
                    && !atoms.iter().any(|inside| inside.doc_tok_idx == a.head)
                    && sentence.iter().any(|report| {
                        !report.is_locked
                            && report.upos == Some(Upos::Verb)
                            && report.doc_tok_idx == a.head
                            && matches!(
                                report.lemma.as_deref().unwrap_or(&report.surface),
                                "言う"
                                    | "話す"
                                    | "述べる"
                                    | "答える"
                                    | "伝える"
                                    | "書く"
                                    | "聞く"
                                    | "尋ねる"
                                    | "思う"
                            )
                    })
            })
            .and_then(|a| a.doc_tok_idx);
        ctx
    }

    pub(super) fn at(segments: &[Segment], idx: usize, doc: &Document) -> Self {
        let question_mark = matches!(segments.get(idx), Some(Segment::Clause { trailing_punct: Some(p), .. }) if matches!(p.as_str(), "？" | "?"));
        let previous = match idx.checked_sub(1).and_then(|i| segments.get(i)) {
            Some(Segment::Clause {
                atoms,
                trailing_punct: Some(p),
                ..
            }) if matches!(p.as_str(), "、" | "，") => Some(atoms.as_slice()),
            _ => None,
        };
        let preceding_until = previous.and_then(|atoms| {
            let (case, rest) = core_non_punct(atoms).split_last()?;
            let time = rest.last()?;
            if case.surface != "まで" || case.is_locked || render_temporal_made_atom(time).is_none()
            {
                return None;
            }
            if time.lemma.as_deref() == Some(CLOCK_ATOM) {
                return clock_case_governor(time, case, doc);
            }
            if case.head == time.doc_tok_idx && time.deprel == "obl" {
                time.head
            } else {
                None
            }
        });
        let time_head = preceding_until.or_else(|| {
            previous.and_then(|atoms| {
                atoms
                    .iter()
                    .find(|a| {
                        past_anchor(a, atoms) && matches!(a.deprel.as_str(), "advmod" | "obl")
                    })
                    .and_then(|a| a.head)
            })
        });
        let before_comma = matches!(segments.get(idx), Some(Segment::Clause { trailing_punct: Some(p), .. }) if matches!(p.as_str(), "、" | "，"))
            && segments.get(idx + 1).is_some();
        Self {
            question_mark,
            time_head,
            preceding_until,
            reported_predicate: None,
            before_comma,
            coordinate: if segments
                .get(idx + 1)
                .is_some_and(|next| coordinated(&segments[idx], next))
            {
                Some(Coordinate::First)
            } else if idx
                .checked_sub(1)
                .is_some_and(|i| coordinated(&segments[i], &segments[idx]))
            {
                Some(Coordinate::Next)
            } else {
                None
            },
        }
    }
}

fn temporal(a: &Atom) -> bool {
    !a.is_locked
        && matches!(
            a.surface.as_str(),
            "昨日"
                | "きのう"
                | "一昨日"
                | "先日"
                | "先週"
                | "昨年"
                | "去年"
                | "当時"
                | "昔"
                | "以前"
                | "さっき"
        )
}

// 前 can also mean a spatial position. A time quantity or 少し attached
// to its adverbial head supplies the elapsed-time reading.
fn past_anchor(a: &Atom, clause: &[Atom]) -> bool {
    temporal(a)
        || !a.is_locked
            && a.surface == "前"
            && a.deprel == "advmod"
            && clause.iter().any(|amount| {
                !amount.is_locked
                    && amount.head == a.doc_tok_idx
                    && matches!(
                        amount.surface.as_str(),
                        "少し" | "すこし" | "ちょっと" | "分" | "時間" | "日" | "週間" | "年"
                    )
            })
}

fn uncertain_sense(lemma: &str, subject: &str) -> bool {
    match lemma {
        "広い" => matches!(subject, "顔" | "心"),
        "重い" | "軽い" => matches!(subject, "腰" | "口" | "気" | "足取り"),
        "高い" | "低い" => !matches!(
            subject,
            "山" | "背"
                | "天井"
                | "身長"
                | "気温"
                | "温度"
                | "値段"
                | "価格"
                | "建物"
                | "棚"
                | "壁"
        ),
        _ => false,
    }
}

fn weather(lemma: &str) -> bool {
    matches!(lemma, "暑い" | "寒い" | "涼しい" | "暖かい")
}

// Before a following clause, が can be tagged as a subject particle despite
// following a complete i-adjective predicate. Keep the contrast on that clause.
pub(super) fn concessive(source: &[Atom], ctx: &ClauseDiscourseCtx) -> Option<Vec<Atom>> {
    let (link, body) = source.split_last()?;
    if !ctx.adjective.before_comma
        || link.is_locked
        || !matches!(link.surface.as_str(), "が" | "けど" | "けれど" | "けれども")
    {
        return None;
    }
    let i = body
        .iter()
        .rposition(|a| !a.is_locked && a.upos == Some(Upos::Adj))?;
    let adj = &body[i];
    let auxiliary = polite_auxiliary(adj, &body[i + 1..]);
    let governor = auxiliary.unwrap_or(adj);
    if governor.doc_tok_idx.is_none() || governor.doc_tok_idx != link.head {
        return None;
    }
    let lemma = adj.lemma.as_deref().unwrap_or(&adj.surface);
    let subject = body[..i]
        .iter()
        .rfind(|a| a.head == adj.doc_tok_idx && a.deprel == "nsubj");
    let standalone_weather = weather(lemma)
        && body[..i].windows(2).any(|p| {
            p[1].surface == "まで"
                && !p[1].is_locked
                && p[1].head == p[0].doc_tok_idx
                && p[0].head == adj.doc_tok_idx
                && render_temporal_made_atom(&p[0]).is_some()
        });
    if subject.is_none() && !standalone_weather
        || hue(lemma).is_none() && !gradable(lemma)
        || uncertain_sense(lemma, subject.map_or("", |s| &s.surface))
    {
        return None;
    }
    inflection(adj, &body[i + 1..])?;
    let mut clause = body.to_vec();
    let root_index = i + usize::from(auxiliary.is_some());
    clause[root_index].deprel = "root".to_owned();
    clause[root_index].head = None;
    Some(clause)
}

// With a negative predicate より denies exceeding the reference (equality
// is possible). ほど…ない says the subject falls short of the reference.
pub(super) fn comparison(
    source: &[Atom],
    ctx: &ClauseDiscourseCtx,
    target: TargetVariant,
) -> Option<String> {
    let i = source
        .iter()
        .position(|a| !a.is_locked && matches!(a.surface.as_str(), "ほど" | "より"))?;
    let (left, right) = source.split_at(i);
    let [marker, adj, tail @ ..] = right else {
        return None;
    };
    let governor = polite_auxiliary(adj, tail).unwrap_or(adj);
    if adj.is_locked || adj.upos != Some(Upos::Adj) || governor.deprel != "root" {
        return None;
    }
    let lemma = adj.lemma.as_deref().unwrap_or(&adj.surface);
    let (ending, suffix) = match tail.split_last() {
        Some((p, body))
            if !p.is_locked && p.upos == Some(Upos::Part) && p.head == governor.doc_tok_idx =>
        {
            (
                body,
                match p.surface.as_str() {
                    "か" => "嗎",
                    "ね" => "呢",
                    "よ" => "喔",
                    _ => return None,
                },
            )
        }
        _ => (tail, ""),
    };
    let (past, true) = inflection(adj, ending)? else {
        return None;
    };
    let case = left
        .iter()
        .position(|a| !a.is_locked && matches!(a.surface.as_str(), "は" | "が"))?;
    let subj = left.get(case.checked_sub(1)?)?;
    let other = left.last()?;
    if subj.doc_tok_idx != left[case].head
        || subj.head != adj.doc_tok_idx
        || subj.deprel != "nsubj"
        || other.doc_tok_idx != marker.head
        || other.deprel != "obl"
        || other.head != adj.doc_tok_idx
    {
        return None;
    }
    let word = if let Some(c) = hue(lemma) {
        if !has_surface_colour(subj, &left[..case], c)
            || !has_surface_colour(other, &left[case + 1..], c)
        {
            return None;
        }
        c.to_owned()
    } else if gradable(lemma) && !uncertain_sense(lemma, &subj.surface) {
        translate_single_ja_term(lemma, Some(lemma), Some(Upos::Adj))
    } else {
        return None;
    };
    let nominal = |np: &[Atom]| {
        if np.is_empty()
            || !np.iter().all(|a| {
                a.is_locked
                    || matches!(
                        a.upos,
                        Some(Upos::Noun | Upos::Propn | Upos::Pron | Upos::Det)
                    )
                    || a.surface == "の"
            })
        {
            return None;
        }
        let group = Bunsetsu {
            atoms: normalize_ja_atoms(&apply_phrase_matches(np)),
            role: BunsetsuRole::Other,
            case_particle: None,
            sconj_tail: None,
            sfp_tail: None,
        };
        Some(render_bunsetsu_np(&group, target))
    };
    let subject = nominal(&left[..case])?;
    let reference = nominal(&left[case + 1..])?;
    let time = if past && ctx.adjective.time_head != adj.doc_tok_idx {
        "當時"
    } else {
        ""
    };
    let relation = if marker.surface == "より" {
        format!("不比{reference}")
    } else {
        format!("沒有{reference}那麼")
    };
    Some(format!("{subject}{time}{relation}{word}{suffix}"))
}

// The arguments and degree adverb attach to 白く; あり hosts the polite
// negation, tense, and sentence particle in 白くありませんでしたか.
fn polite_auxiliary<'a>(adjective: &Atom, tail: &'a [Atom]) -> Option<&'a Atom> {
    tail.first().filter(|a| {
        !a.is_locked
            && a.upos == Some(Upos::Verb)
            && a.lemma.as_deref() == Some("ある")
            && adjective.surface.ends_with('く')
            && a.doc_tok_idx.is_some()
            && adjective.head == a.doc_tok_idx
    })
}

/// An i-adjective describes a past state with かった. Rendering that ending
/// as resultative 了 instead asserts that the quality changed.
fn inflection(predicate: &Atom, tail: &[Atom]) -> Option<(bool, bool)> {
    let lexical_tail: String = tail.iter().map(|a| a.surface.as_str()).collect();
    if tail.iter().any(|a| a.is_locked) {
        return None;
    }
    let inflection = lexical_tail
        .strip_suffix("のです")
        .or_else(|| lexical_tail.strip_suffix("です"))
        .unwrap_or(&lexical_tail);
    match inflection {
        "" if predicate.surface.ends_with('い') => Some((false, false)),
        "た" if predicate.surface.ends_with("かっ") => Some((true, false)),
        "ない" | "ありません" if predicate.surface.ends_with('く') => Some((false, true)),
        "なかった" | "ありませんでした" if predicate.surface.ends_with('く') => {
            Some((true, true))
        }
        _ => None,
    }
}

fn hue(lemma: &str) -> Option<&'static str> {
    Some(match lemma {
        "白い" => "白",
        "黒い" => "黑",
        "赤い" => "紅",
        "青い" => "藍",
        "黄色い" => "黃",
        "茶色い" => "棕",
        _ => return None,
    })
}

fn gradable(lemma: &str) -> bool {
    const PHYSICAL_QUALITIES: &[&str] = &[
        "重い",
        "軽い",
        "長い",
        "短い",
        "大きい",
        "小さい",
        "広い",
        "狭い",
        "厚い",
        "薄い",
        "太い",
        "細い",
        "高い",
        "低い",
        "深い",
        "浅い",
        "硬い",
        "柔らかい",
        "暗い",
        "明るい",
        "暑い",
        "寒い",
        "暖かい",
        "冷たい",
        "熱い",
        "涼しい",
    ];
    PHYSICAL_QUALITIES.contains(&lemma)
}

// Use only the head and its contiguous compound. A colour inside a dependent
// phrase cannot supply an unrelated noun's sense, e.g. 信号の箱 vs 箱の信号.
fn compound(atoms: &[Atom], end: usize) -> String {
    let head = &atoms[end];
    let start = atoms[..end]
        .iter()
        .rposition(|a| a.is_locked || a.deprel != "compound" || a.head != head.doc_tok_idx)
        .map_or(0, |i| i + 1);
    atoms[start..=end]
        .iter()
        .map(|a| a.surface.as_str())
        .collect()
}

fn has_surface_colour(noun: &Atom, previous: &[Atom], colour: &str) -> bool {
    // Character, health, ripeness and traffic indications can also use colour
    // adjectives. Only choose a hue when the source subject supports it.
    if noun.is_locked || noun.upos != Some(Upos::Noun) {
        return false;
    }
    let Some(end) = previous
        .iter()
        .position(|a| a.doc_tok_idx == noun.doc_tok_idx)
    else {
        return false;
    };
    let name = compound(previous, end);
    if name == "色"
        && colour == "藍"
        && previous.iter().enumerate().any(|(i, a)| {
            a.head == noun.doc_tok_idx
                && a.deprel == "nmod"
                && green_or_unripe(&compound(previous, i))
        })
    {
        return false;
    }
    literal_colour(&name, colour)
}

// Recover the head of a whole, simple Object subject for sense selection.
// The rendered phrase still uses the caller's placeholder.
fn locked_colour_noun(subject: &Atom, doc: &Document, chars: &[char]) -> Option<Atom> {
    if !subject.is_locked
        || subject.lemma.as_deref() != Some(LOCKED_OBJECT_ATOM)
        || subject.deprel != "nsubj"
    {
        return None;
    }
    let start = doc
        .tokens
        .partition_point(|token| token.span.start_char < subject.span.start_char);
    let tokens = doc.tokens.get(start..=subject.doc_tok_idx?)?;
    let last = tokens.last()?;
    if tokens.first()?.span.start_char != subject.span.start_char
        || last.span.end_char != subject.span.end_char
        || last.upos != Some(Upos::Noun)
        || tokens
            .windows(2)
            .any(|pair| pair[0].span.end_char != pair[1].span.start_char)
        || tokens.iter().any(|token| {
            !matches!(
                token.upos,
                Some(Upos::Noun | Upos::Propn | Upos::Pron | Upos::Det)
            ) && !(token.upos == Some(Upos::Adp)
                && chars.get(token.span.start_char..token.span.end_char) == Some(&['の']))
        })
    {
        return None;
    }
    let spelling: String = chars
        .get(last.span.start_char..last.span.end_char)?
        .iter()
        .collect();
    // A caller lexicon entry can join a demonstrative or genitive noun phrase
    // into one token. Take the noun at its end for the literal-colour check.
    let nominal = ["この", "その", "あの"]
        .into_iter()
        .find_map(|prefix| spelling.strip_prefix(prefix))
        .unwrap_or(&spelling);
    let word = nominal.rsplit_once('の').map_or(nominal, |(_, noun)| noun);
    let mut original = subject.clone();
    original.surface = word.to_owned();
    original.lemma = last.lemma.clone();
    original.upos = last.upos;
    original.span = Span {
        start_char: last.span.end_char - word.chars().count(),
        end_char: last.span.end_char,
    };
    original.is_locked = false;
    Some(original)
}

pub(super) fn bind_locked_colour_subject(
    atoms: &mut Vec<Atom>,
    doc: &Document,
    sentence: usize,
    chars: &[char],
) {
    for index in (1..atoms.len()).rev() {
        let subject = &atoms[index];
        let Some(original) = locked_colour_noun(subject, doc, chars) else {
            continue;
        };
        if original.span != subject.span {
            continue;
        }
        let det = &atoms[index - 1];
        if attached_demonstrative(det, &original).is_none() {
            continue;
        }
        if !atoms.iter().any(|predicate| {
            !predicate.is_locked
                && predicate.upos == Some(Upos::Adj)
                && predicate.doc_tok_idx.is_some()
                && predicate.doc_tok_idx == subject.head
                && hue(predicate.lemma.as_deref().unwrap_or(&predicate.surface)).is_some_and(
                    |colour| original.surface != "色" && literal_colour(&original.surface, colour),
                )
        }) {
            continue;
        }
        let Some(label) = nominal_possession::choice_phrase(
            &atoms[index - 1..=index],
            doc,
            sentence,
            TargetVariant::ZhHant,
            chars,
        ) else {
            continue;
        };
        let mut phrase = subject.clone();
        phrase.span.start_char = det.span.start_char;
        phrase.surface = label;
        atoms.splice(index - 1..=index, [phrase]);
    }
}

fn green_or_unripe(name: &str) -> bool {
    const GREEN_OR_UNRIPE: &[&str] = &["実", "葉", "果物", "信号", "信号機", "りんご", "リンゴ"];
    GREEN_OR_UNRIPE.contains(&name)
}

fn literal_colour(name: &str, colour: &str) -> bool {
    const VISIBLE_SURFACES: &[&str] = &[
        "色",
        "空",
        "海",
        "雲",
        "雪",
        "花",
        "壁",
        "床",
        "屋根",
        "門",
        "扉",
        "ドア",
        "窓",
        "車",
        "自転車",
        "電車",
        "バス",
        "トラック",
        "船",
        "箱",
        "紙",
        "布",
        "糸",
        "靴",
        "服",
        "傘",
        "かばん",
        "鞄",
        "シャツ",
        "スカート",
        "コート",
        "帽子",
        "タオル",
        "スカーフ",
        "靴下",
        "ペン",
        "鉛筆",
        "ノート",
        "封筒",
        "本",
        "机",
        "椅子",
        "皿",
        "カップ",
        "ボタン",
        "ボール",
    ];
    VISIBLE_SURFACES.contains(&name) || colour != "藍" && green_or_unripe(name)
}

// These are the concrete animals also recognized as possible matrix actors.
fn animal_noun(raw: &str) -> bool {
    matches!(raw, "犬" | "猫" | "鳥" | "馬")
}

// が links a subject to the adjective. In attributives the same subject can
// take の, which the parser attaches as a noun modifier of the following noun.
// Only take that reading for a compatible part/whole pair.
fn attributive_part(
    prefix: &[Atom],
    adj: &Atom,
    whole: &str,
    chars: &[char],
) -> Option<(usize, String)> {
    let [.., part, case] = prefix else {
        return None;
    };
    let part_id = part.doc_tok_idx?;
    if case.is_locked || case.deprel != "case" || case.head != Some(part_id) {
        return None;
    }
    match (case.surface.as_str(), part.deprel.as_str()) {
        ("が", "nsubj") if part.head == adj.doc_tok_idx => {}
        ("の", "nmod") if part.head == adj.head => {}
        _ => return None,
    }
    let end = prefix.len() - 2;
    let start = prefix[..end]
        .iter()
        .rposition(|a| a.is_locked || a.deprel != "compound" || a.head != Some(part_id))
        .map_or(0, |i| i + 1);
    // A possessor or another modifier belongs to the part itself.
    if (part.is_locked && start != end) || prefix[..start].iter().any(|a| a.head == Some(part_id)) {
        return None;
    }
    let raw: String = chars
        .get(prefix[start].span.start_char..part.span.end_char)?
        .iter()
        .collect();
    let label = match whole {
        "車" | "バス" | "トラック" | "電車" => match raw.as_str() {
            "屋根" => "車頂",
            "窓" => "車窗",
            "ドア" => "車門",
            _ => return None,
        },
        "家" | "建物" => match raw.as_str() {
            "屋根" => "屋頂",
            "窓" => "窗戶",
            "ドア" => "門",
            _ => return None,
        },
        animal if animal_noun(animal) => match raw.as_str() {
            "目" | "眼" => "眼睛",
            "耳" => "耳朵",
            "鼻" => "鼻子",
            "羽" | "羽根" if animal == "鳥" => "羽毛",
            "毛" if animal != "鳥" => "毛",
            _ => return None,
        },
        _ => return None,
    };
    Some((
        start,
        if part.is_locked {
            part.surface.clone()
        } else {
            label.to_owned()
        },
    ))
}

// Bind an inflected adjective and its own auxiliaries to the exact noun that
// it modifies. Leave the original clause source available to the matrix verb.
pub(super) fn bind_attributives<'a>(
    atoms: &'a [Atom],
    doc: &Document,
    chars: &[char],
) -> std::borrow::Cow<'a, [Atom]> {
    let mut result = std::borrow::Cow::Borrowed(atoms);
    let mut next_phrase = atoms.len();
    for (i, adj) in atoms.iter().enumerate().rev() {
        if adj.is_locked || adj.upos != Some(Upos::Adj) || adj.deprel != "amod" {
            continue;
        }
        if let Some((range, noun)) = attributive_at(atoms, i, doc, chars)
            && *range.end() < next_phrase
        {
            next_phrase = *range.start();
            result.to_mut().splice(range, [noun]);
        }
    }
    if let std::borrow::Cow::Owned(bound) = &mut result
        && atoms
            .iter()
            .any(|a| !a.is_locked && a.deprel == "case" && a.surface == "と")
    {
        restore_list_noun_roles(bound, atoms, doc, chars);
    }
    result
}

pub(super) const ATTRIBUTIVE_PERSON: &str = "__ja_attributive_person";

// A glossary placeholder is provisionally tagged PROPN. Before deciding
// whether と joins things or introduces a companion, consult the actual noun.
// The protected text is still used verbatim, including in the second item.
fn restore_list_noun_roles(bound: &mut [Atom], source: &[Atom], doc: &Document, chars: &[char]) {
    for term in bound.iter_mut().filter(|t| t.is_locked) {
        let Some(id) = term.doc_tok_idx else { continue };
        if !source.iter().any(|p| {
            !p.is_locked
                && p.head == Some(id)
                && p.deprel == "case"
                && matches!(p.surface.as_str(), "と" | "を")
        }) {
            continue;
        }
        let Some(noun) = source.iter().find(|a| a.doc_tok_idx == Some(id)) else {
            continue;
        };
        let Some(token) = doc
            .tokens
            .get(id)
            .filter(|t| t.span == noun.span && t.upos == Some(Upos::Noun))
        else {
            continue;
        };
        let Some(raw) = chars.get(token.span.start_char..token.span.end_char) else {
            continue;
        };
        let mut original = noun.clone();
        original.surface = raw.iter().collect();
        original.upos = token.upos;
        let person = token.sem_class == Some(SemanticClass::Person)
            || original.lemma.as_deref() == Some(LOCKED_RESIDENT_ATOM)
            || is_demonstrative_person_noun(&original);
        term.upos = token.upos;
        if person && (term.lemma.is_none() || term.span != noun.span) {
            term.lemma = Some(ATTRIBUTIVE_PERSON.to_owned());
        }
    }
}

fn attributive_at(
    atoms: &[Atom],
    i: usize,
    doc: &Document,
    chars: &[char],
) -> Option<(std::ops::RangeInclusive<usize>, Atom)> {
    let adj = &atoms[i];
    let id = adj.doc_tok_idx?;
    let aux = atoms[i + 1..]
        .iter()
        .take_while(|a| {
            !a.is_locked && a.upos == Some(Upos::Aux) && a.deprel == "aux" && a.head == Some(id)
        })
        .count();
    let n = i + aux + 1;
    let noun = atoms.get(n)?;
    if noun.doc_tok_idx != Some(adj.head?)
        || nominal_possession::original_pos(noun, doc) != Some(Upos::Noun)
    {
        return None;
    }
    let (past, negative) = inflection(adj, &atoms[i + 1..n])?;
    let lemma = adj.lemma.as_deref().unwrap_or(&adj.surface);
    let raw: String = chars
        .get(noun.span.start_char..noun.span.end_char)?
        .iter()
        .collect();
    let mut start = i;
    if let [.., degree] = &atoms[..i]
        && degree.deprel == "advmod"
        && degree.head == Some(id)
        && !degree.is_locked
        && matches!(
            degree.surface.as_str(),
            "とても" | "すごく" | "非常に" | "かなり" | "あまり" | "それほど"
        )
    {
        if negative != matches!(degree.surface.as_str(), "あまり" | "それほど") {
            return None;
        }
        start -= 1;
    }
    let description_start = start;
    let mut when = if past {
        "當時".to_owned()
    } else {
        String::new()
    };
    let linked_time = |t: &Atom, p: &Atom| {
        if p.is_locked
            || p.head != t.doc_tok_idx
            || p.deprel != "case"
            || t.head != Some(id)
            || !matches!(t.deprel.as_str(), "obl" | "nsubj" | "advmod")
        {
            return None;
        }
        render_temporal_made_atom(t).or_else(|| {
            temporal(t).then(|| translate_single_ja_term(&t.surface, t.lemma.as_deref(), t.upos))
        })
    };
    let time = match &atoms[..start] {
        [.., from, kara, to, made] if kara.surface == "から" && made.surface == "まで" => {
            linked_time(from, kara)
                .zip(linked_time(to, made))
                .map(|(a, b)| (4, format!("從{a}到{b}都")))
        }
        [.., t, p] if matches!(p.surface.as_str(), "まで" | "は" | "に") => linked_time(t, p)
            .map(|label| {
                (
                    2,
                    if p.surface == "まで" {
                        format!("直到{label}都")
                    } else {
                        label
                    },
                )
            }),
        [.., t] if t.head == Some(id) && temporal(t) => Some((
            1,
            translate_single_ja_term(&t.surface, t.lemma.as_deref(), t.upos),
        )),
        _ => None,
    };
    if let Some((len, text)) = time {
        start -= len;
        when = text;
    }
    let part = attributive_part(&atoms[..start], adj, &raw, chars);
    if part.is_some()
        && hue(lemma).is_none()
        && !matches!(
            lemma,
            "大きい" | "小さい" | "広い" | "狭い" | "厚い" | "薄い" | "長い" | "短い"
        )
    {
        return None;
    }
    // A linked degree must stay inside the noun phrase, even without auxiliaries.
    if aux == 0 && part.is_none() && description_start == i {
        return None;
    }
    if let Some(colour) = hue(lemma) {
        if part.is_none() && !literal_colour(&raw, colour) && !animal_noun(&raw) {
            return None;
        }
    } else if !(gradable(lemma) || matches!(lemma, "新しい" | "古い" | "安い"))
        || uncertain_sense(lemma, &raw)
    {
        return None;
    }
    if let Some((part_start, label)) = part {
        start = part_start;
        when.insert_str(0, &label);
    }
    // A glossary entry can make Xの a single token. There is no longer a
    // separate case arc to establish the holder. Keep the opaque term whole.
    if atoms[..start].last().is_some_and(|a| {
        a.is_locked
            && matches!(
                chars
                    .get(a.span.start_char..a.span.end_char)
                    .and_then(|s| s.last()),
                Some('の' | 'が')
            )
    }) {
        return None;
    }
    // An unbound Xの beside an animal may be the holder of the colour.
    // A verified human possessor (母の白かった猫) can remain outside the phrase.
    if animal_noun(&raw)
        && atoms[..start].windows(2).any(|pair| {
            let [owner, case] = pair else { return false };
            owner.deprel == "nmod"
                && owner.head == noun.doc_tok_idx
                && !case.is_locked
                && case.surface == "の"
                && case.deprel == "case"
                && case.head == owner.doc_tok_idx
                && !(owner.lemma.as_deref() == Some(LOCKED_RESIDENT_ATOM)
                    || !owner.is_locked && is_demonstrative_person_noun(owner))
        })
    {
        return None;
    }
    // A separate comparison, unbound subject, or other
    // adverb can describe a part of the noun or change the degree of the
    // property. Do not move that predicate onto the whole noun.
    if atoms[..start].iter().enumerate().any(|(at, a)| {
        if a.head != Some(id) {
            return false;
        }
        let case = atoms
            .get(at + 1)
            .filter(|p| !p.is_locked && p.deprel == "case" && p.head == a.doc_tok_idx);
        let restricts = a.deprel == "obl"
            || a.deprel == "nsubj"
                && atoms[..start]
                    .iter()
                    .any(|p| p.head == a.doc_tok_idx && p.surface == "が");
        a.deprel == "advmod"
            || restricts
                && !case.is_some_and(|p| separate_modified_argument(atoms, at, p, noun, doc, chars))
    }) {
        return None;
    }
    let mut phrase = Vec::new();
    if let [.., det] = &atoms[..start]
        && det.upos == Some(Upos::Det)
        && det.deprel == "det"
        && det.head == noun.doc_tok_idx
        && !det.is_locked
        && matches!(det.surface.as_str(), "この" | "その" | "あの" | "どの")
    {
        start -= 1;
        phrase.push(det.clone());
    }
    phrase.extend_from_slice(&atoms[description_start..i]);
    let mut adjective = adj.clone();
    if let Some(colour) = hue(lemma) {
        // The noun renderer takes pretranslated atoms verbatim. Keep the hue
        // selected from the source lemma, including 茶色い.
        // The linked Japanese auxiliaries still supply its tense and negation.
        adjective.surface = colour.to_owned();
        adjective.is_locked = true;
    }
    phrase.push(adjective);
    phrase.extend_from_slice(&atoms[i + 1..=n]);
    let mut noun = nominal_possession::modified_noun(&phrase, doc, chars, Some(&when))?.pop()?;
    noun.span.start_char = atoms[start].span.start_char;
    Some((start..=n, noun))
}

// In 白かった車と黒かった車, the analyzer attaches the first noun to the
// next adjective as an oblique. Each adjective still has its own nominal head.
// より supplies a comparison; が can identify a property holder (屋根が白い車).
fn separate_modified_argument(
    atoms: &[Atom],
    at: usize,
    case: &Atom,
    noun: &Atom,
    doc: &Document,
    chars: &[char],
) -> bool {
    let earlier = &atoms[at];
    let Some(id) = earlier.doc_tok_idx else {
        return false;
    };
    if case.surface == "と" {
        return matches!(
            nominal_possession::original_pos(earlier, doc),
            Some(Upos::Noun | Upos::Propn | Upos::Pron)
        );
    }
    if !atoms[..at].iter().any(|modifier| {
        modifier.head == Some(id) && matches!(modifier.deprel.as_str(), "amod" | "acl")
    }) {
        return false;
    }
    if noun.deprel != "obj" {
        return false;
    }
    let Some(verb) = atoms
        .iter()
        .find(|a| !a.is_locked && a.doc_tok_idx == noun.head && a.upos == Some(Upos::Verb))
    else {
        return false;
    };
    match case.surface.as_str() {
        "に" => matches!(
            verb.lemma.as_deref(),
            Some("入れる" | "置く" | "運ぶ" | "届ける" | "載せる" | "乗せる")
        ),
        "から" => matches!(
            verb.lemma.as_deref(),
            Some("出す" | "取り出す" | "運ぶ" | "送る")
        ),
        "が" => {
            let Some(token) = doc.tokens.get(id).filter(|t| t.span == earlier.span) else {
                return false;
            };
            let raw: String = chars
                .get(token.span.start_char..token.span.end_char)
                .unwrap_or_default()
                .iter()
                .collect();
            let actor = token.sem_class == Some(SemanticClass::Person)
                || earlier.lemma.as_deref() == Some(LOCKED_RESIDENT_ATOM)
                || !earlier.is_locked
                    && earlier.upos != Some(Upos::Propn)
                    && is_demonstrative_person_noun(earlier)
                || matches!(raw.as_str(), "子" | "子供" | "子ども")
                || animal_noun(&raw);
            actor
                && matches!(
                    verb.lemma.as_deref(),
                    Some("見る" | "買う" | "運ぶ" | "持つ" | "開ける" | "出す" | "入れる")
                )
        }
        _ => false,
    }
}

// The past of a lexical verb such as 運ぶ is 運んだ. Here だ is a verbal
// auxiliary attached to the relative verb, rather than a nominal copula.
pub(super) fn voiced_past_modifier(group: &Bunsetsu, head: &Bunsetsu, source: &[Atom]) -> bool {
    matches!(group.atoms.as_slice(), [verb, past]
        if !verb.is_locked && !past.is_locked && past.surface == "だ" && past.deprel == "aux"
        && past.head == verb.doc_tok_idx && verb.upos == Some(Upos::Verb) && verb.deprel == "acl"
        && head.atoms.iter().any(|noun| noun.doc_tok_idx == verb.head && noun.deprel == "nsubj"
            && source.iter().any(|adj| adj.doc_tok_idx == noun.head && adj.upos == Some(Upos::Adj) && adj.deprel == "root")))
}

// In 子が白いと言った布, と marks the reported property. Its head is
// 言う, whose relative-clause target is 布. Removing the report together
// keeps the ordinary conditional-と renderer out of this speech complement.
pub(super) fn fold_reported_modifier(
    groups: &mut Vec<Bunsetsu>,
    source: &[Atom],
    target: TargetVariant,
) {
    let Some((start, noun, text)) = reported_modifier(source, target) else {
        return;
    };
    let Some(first) = groups
        .iter()
        .position(|g| g.atoms.first().is_some_and(|a| a.span.start_char == start))
    else {
        return;
    };
    let Some(last) = groups.iter().position(|g| {
        g.atoms
            .first()
            .is_some_and(|a| a.doc_tok_idx == noun.doc_tok_idx)
    }) else {
        return;
    };
    if first < last
        && groups[first..last]
            .iter()
            .flat_map(|g| &g.atoms)
            .all(|a| a.span.end_char <= noun.span.start_char)
    {
        prepend_locked_atom(&mut groups[last], text);
        groups.drain(first..last);
    }
}

fn reported_modifier(source: &[Atom], target: TargetVariant) -> Option<(usize, &Atom, String)> {
    let r = source
        .iter()
        .position(|a| !a.is_locked && a.deprel == "acl" && a.lemma.as_deref() == Some("言う"))?;
    let (prefix, rest) = source.split_at(r);
    let (report, after) = rest.split_first()?;
    let noun_index = after
        .iter()
        .position(|a| a.doc_tok_idx == report.head && a.deprel == "nsubj")?;
    let noun = &after[noun_index];
    let matrix = source
        .iter()
        .find(|a| a.doc_tok_idx == noun.head && a.upos == Some(Upos::Adj) && a.deprel == "root")?;
    let matrix_lemma = matrix.lemma.as_deref().unwrap_or(&matrix.surface);
    if hue(matrix_lemma).is_none() && !gradable(matrix_lemma) {
        return None;
    }
    let past_negative = &after[..noun_index];
    if !past_negative
        .iter()
        .all(|a| !a.is_locked && a.head == report.doc_tok_idx && a.upos == Some(Upos::Aux))
    {
        return None;
    }
    let ending: String = past_negative.iter().map(|a| a.surface.as_str()).collect();
    let speech = match ending.as_str() {
        "" | "た" => "說",
        "なかった" => "沒有說",
        "ない" => "不說",
        _ => return None,
    };
    let (to, content) = prefix.split_last()?;
    let idx = content
        .iter()
        .rposition(|a| !a.is_locked && a.upos == Some(Upos::Adj) && a.head == report.doc_tok_idx)?;
    let inner = &content[idx];
    if to.is_locked || to.surface != "と" || to.head != inner.doc_tok_idx {
        return None;
    }
    let (past, neg) = inflection(inner, &content[idx + 1..])?;
    let lemma = inner.lemma.as_deref().unwrap_or(&inner.surface);
    let property = if let Some(c) = hue(lemma) {
        if !has_surface_colour(noun, source, c) {
            return None;
        }
        format!("{}{c}色", if neg { "不是" } else { "是" })
    } else if gradable(lemma) && !uncertain_sense(lemma, &noun.surface) {
        format!(
            "{}{}",
            if neg { "不" } else { "很" },
            translate_single_ja_term(lemma, Some(lemma), Some(Upos::Adj))
        )
    } else {
        return None;
    };
    let (ga, speaker_np) = content[..idx].split_last()?;
    let (speaker, preceding) = speaker_np.split_last()?;
    if ga.is_locked
        || ga.surface != "が"
        || ga.head != speaker.doc_tok_idx
        || speaker.is_locked
        || speaker.deprel != "nsubj"
        || (speaker.head != inner.doc_tok_idx && speaker.head != report.doc_tok_idx)
        || !(is_demonstrative_person_noun(speaker) || speaker.surface == "子")
    {
        return None;
    }
    let det = preceding
        .last()
        .filter(|a| attached_demonstrative(a, speaker).is_some());
    let mut start_index = speaker_np.len() - 1 - usize::from(det.is_some());
    let nominal = |atoms: &[Atom]| {
        render_bunsetsu_np(
            &Bunsetsu {
                atoms: atoms.to_vec(),
                role: BunsetsuRole::Other,
                case_particle: None,
                sconj_tail: None,
                sfp_tail: None,
            },
            target,
        )
    };
    let mut who = if speaker.surface == "子" {
        let demonstrative = det.and_then(|a| attached_demonstrative(a, speaker));
        demonstrative.map_or_else(|| "孩子".to_owned(), |d| format!("{d}個孩子"))
    } else {
        nominal(&speaker_np[start_index..])
    };
    let mut owned = speaker;
    while let [earlier @ .., owner, no] = &speaker_np[..start_index] {
        if no.surface != "の" {
            break;
        }
        if owner.is_locked
            || no.is_locked
            || owner.head != owned.doc_tok_idx
            || owner.deprel != "nmod"
            || no.head != owner.doc_tok_idx
        {
            return None;
        }
        let end = start_index - 1;
        start_index = end
            - 1
            - usize::from(
                earlier
                    .last()
                    .is_some_and(|d| attached_demonstrative(d, owner).is_some()),
            );
        who = format!("{}的{who}", nominal(&speaker_np[start_index..end]));
        owned = owner;
    }
    Some((
        speaker_np[start_index].span.start_char,
        noun,
        format!(
            "{who}{speech}{}{property}的",
            if past { "當時" } else { "" }
        ),
    ))
}

pub(super) struct TimeInterval {
    head: usize,
    pub(super) text: String,
    pub(super) question: bool,
}

// A bare まで after a time bounds the whole state. Keep any paired から
// on the same predicate. An endpoint belonging to a relative verb, a place,
// or a までに deadline cannot supply this interval.
pub(super) fn take_time_interval(
    groups: &mut Vec<Bunsetsu>,
    source: &[Atom],
    doc: &Document,
    ctx: &Context,
) -> Option<TimeInterval> {
    let source = core_non_punct(source);
    let i = source
        .iter()
        .rposition(|a| !a.is_locked && a.upos == Some(Upos::Adj))?;
    let (predicate, tail) = source[i..].split_first()?;
    let governor = polite_auxiliary(predicate, tail).unwrap_or(predicate);
    if governor.deprel != "root"
        && (ctx.reported_predicate.is_none() || governor.doc_tok_idx != ctx.reported_predicate)
    {
        return None;
    }
    let ending = match tail {
        [body @ .., a] if a.upos == Some(Upos::Part) && a.head == governor.doc_tok_idx => body,
        a => a,
    };
    inflection(predicate, ending)?;
    let head = predicate.doc_tok_idx?;
    if !groups.iter().any(|g| {
        g.role == BunsetsuRole::Predicate && g.atoms.iter().any(|a| a.doc_tok_idx == Some(head))
    }) {
        return None;
    }
    let endpoint = |g: &Bunsetsu, marker: &str| -> Option<(String, bool)> {
        let atoms = match g.atoms.as_slice() {
            [body @ .., topic] if topic.surface == "は" && !topic.is_locked => body,
            a => a,
        };
        let [time, case] = atoms else { return None };
        if case.is_locked || case.surface != marker || case.span.start_char != time.span.end_char {
            return None;
        }
        // The normalized clock has no token index. Its case marker still
        // points to the original time token within that clock's source span.
        if time.lemma.as_deref() == Some(CLOCK_ATOM) {
            if clock_case_governor(time, case, doc) != Some(head) {
                return None;
            }
            return Some((render_temporal_made_atom(time)?, false));
        }
        let original = source.iter().find(|a| {
            a.doc_tok_idx == case.head
                && a.head == Some(head)
                && matches!(a.deprel.as_str(), "obl" | "nsubj")
        })?;
        if original.is_locked {
            return None;
        }
        if let Some(wh) = deadline_time_question(original) {
            return Some((wh.to_owned(), true));
        }
        let label = match original.surface.as_str() {
            "夜" | "晩" => "晚上".to_owned(),
            "春" | "夏" | "秋" | "冬" => translate_single_ja_term(
                &original.surface,
                original.lemma.as_deref(),
                original.upos,
            ),
            _ => render_temporal_made_atom(time)?,
        };
        Some((label, false))
    };
    let mut ends = groups
        .iter()
        .enumerate()
        .filter_map(|(i, g)| endpoint(g, "まで").map(|v| (i, v)));
    let (end, (text, mut question)) = ends.next()?;
    if ends.next().is_some() {
        return None;
    }
    let mut starts = groups
        .iter()
        .enumerate()
        .filter_map(|(i, g)| endpoint(g, "から").map(|v| (i, v)));
    let start = starts.next();
    if starts.next().is_some() {
        return None;
    }
    let text = match start {
        Some((i, (from, wh))) if i < end => {
            question |= wh;
            groups.remove(end);
            groups.remove(i);
            format!("從{from}到{text}都")
        }
        Some(_) => return None,
        None => {
            groups.remove(end);
            if question {
                format!("到{text}為止都")
            } else {
                format!("直到{text}都")
            }
        }
    };
    Some(TimeInterval {
        head,
        text,
        question,
    })
}

pub(super) fn render(
    source: &[Atom],
    predicates: &[Bunsetsu],
    doc: &Document,
    ctx: &ClauseDiscourseCtx,
    adverbs: &mut Vec<String>,
    times: &mut Vec<String>,
    interval: Option<&TimeInterval>,
) -> Option<String> {
    let body = core_non_punct(source);
    let idx = body
        .iter()
        .rposition(|a| !a.is_locked && a.upos == Some(Upos::Adj))?;
    let (before, predicate_and_tail) = body.split_at(idx);
    let (predicate, tail) = predicate_and_tail.split_first()?;
    let id = predicate.doc_tok_idx?;
    // Recursive rendering of relative clauses also receives the sentence source.
    // Only its own predicate groups may select a particular adjective head.
    let active = predicates.iter().flat_map(|b| &b.atoms);
    if !active.clone().any(|a| a.doc_tok_idx == Some(id))
        || active.clone().any(|a| {
            a.upos == Some(Upos::Adj) && a.doc_tok_idx != Some(id)
                || a.upos == Some(Upos::Verb)
                    && !(a.lemma.as_deref() == Some("ある") && predicate.head == a.doc_tok_idx)
        })
    {
        return None;
    }
    if let Some(side) = ctx.adjective.coordinate {
        let lemma = predicate.lemma.as_deref().unwrap_or(&predicate.surface);
        let word = translate_single_ja_term(lemma, Some(lemma), Some(Upos::Adj));
        return Some(format!(
            "{}{word}",
            match side {
                Coordinate::First => "很",
                Coordinate::Next => "也很",
            }
        ));
    }
    let aux_head = polite_auxiliary(predicate, tail).and_then(|a| a.doc_tok_idx);
    let (tail, particle) = match tail.split_last() {
        Some((p, body))
            if !p.is_locked
                && p.upos == Some(Upos::Part)
                && (p.head == Some(id) || aux_head.is_some() && p.head == aux_head)
                && matches!(p.surface.as_str(), "か" | "よ" | "ね" | "よね" | "わ") =>
        {
            (body, p.surface.as_str())
        }
        _ => (tail, ""),
    };
    let (past, negative) = inflection(predicate, tail)?;
    if !(predicate.deprel == "root"
        || ctx.inside_quote && predicate.deprel == "advcl"
        || aux_head.is_some())
    {
        return None;
    }
    let subject = before
        .iter()
        .rfind(|a| a.deprel == "nsubj" && a.head == Some(id));
    let previous_range = ctx.adjective.preceding_until == Some(id);
    let bounded = previous_range || interval.is_some_and(|period| period.head == id);
    if subject.is_none() && !bounded {
        return None;
    }
    let lemma = predicate.lemma.as_deref().unwrap_or(&predicate.surface);
    let colour = hue(lemma);
    if let Some(c) = colour {
        let subject = subject?;
        let has_hue = if subject.is_locked {
            locked_colour_noun(subject, doc, ctx.source_chars)
                .is_some_and(|noun| noun.surface != "色" && literal_colour(&noun.surface, c))
        } else {
            has_surface_colour(subject, before, c)
        };
        if !has_hue {
            return None;
        }
    } else if !(gradable(lemma) || bounded && matches!(lemma, "強い" | "弱い" | "忙しい"))
        || uncertain_sense(lemma, subject.map_or("", |s| s.surface.as_str()))
    {
        return None;
    }
    let comparative = subject.is_some_and(|s| matches!(s.surface.as_str(), "方" | "ほう"))
        || before.iter().any(|a| {
            !a.is_locked
                && a.surface == "より"
                && before
                    .iter()
                    .any(|n| n.doc_tok_idx == a.head && n.head == Some(id))
        });
    let linked_adv = before
        .iter()
        .filter(|a| a.head == Some(id) && a.deprel == "advmod");
    // A time or aspect adverb such as まだ does not put the hue on a scale.
    // Keep unknown modifiers on the normal path rather than choosing a new reading.
    let contextual = |a: &Atom| {
        past_anchor(a, before)
            || matches!(
                a.surface.as_str(),
                "まだ" | "今も" | "いつも" | "もう" | "やはり" | "やっぱり" | "今" | "今日"
            )
    };
    let scale = |a: &Atom| {
        !a.is_locked
            && matches!(
                a.surface.as_str(),
                "少し"
                    | "すこし"
                    | "ちょっと"
                    | "とても"
                    | "とっても"
                    | "ずいぶん"
                    | "大変"
                    | "非常に"
                    | "かなり"
                    | "ずっと"
                    | "もっと"
                    | "一番"
                    | "いちばん"
                    | "最も"
                    | "あまり"
                    | "それほど"
                    | "特に"
                    | "やや"
                    | "すごく"
                    | "たいへん"
            )
    };
    if linked_adv.clone().any(|a| !contextual(a) && !scale(a)) {
        return None;
    }
    let modified = linked_adv.clone().any(scale) || comparative;
    if !comparative && before.iter().any(|a| !a.is_locked && a.surface == "より") {
        return None;
    }
    let weather_time = weather(lemma)
        && subject.is_some_and(|s| {
            is_daypart_surface(&s.surface)
                && !before
                    .iter()
                    .any(|owner| owner.head == s.doc_tok_idx && owner.deprel == "nmod")
        });
    let has_time = bounded
        || weather_time
        || ctx.adjective.time_head == Some(id)
        || before.iter().any(|a| {
            past_anchor(a, before)
                && (a.head == Some(id)
                    || linked_adv
                        .clone()
                        .any(|modifier| a.head == modifier.doc_tok_idx))
        });
    let mut word = if let Some(c) = colour {
        c.to_owned()
    } else {
        translate_single_ja_term(lemma, Some(lemma), Some(Upos::Adj))
    };
    if modified && negative {
        // あまり…ない and それほど…ない each express one degree of negation.
        let modifier = linked_adv.clone().find(|a| scale(a))?;
        let (existing, negative_degree) = match modifier.surface.as_str() {
            "あまり" => ("不太", "不太"),
            "それほど" => ("那麼", "沒那麼"),
            _ => return None,
        };
        if comparative || linked_adv.clone().filter(|a| scale(a)).count() != 1 {
            return None;
        }
        let degree = adverbs.iter_mut().find(|s| s.as_str() == existing)?;
        *degree = negative_degree.to_owned();
        if previous_range {
            times.push("都".to_owned());
        }
        if past && !has_time {
            times.push("當時".to_owned());
        }
        return Some(word);
    }
    if linked_adv
        .clone()
        .any(|a| !a.is_locked && matches!(a.surface.as_str(), "少し" | "すこし" | "ちょっと"))
        && let Some(amount) = adverbs.iter_mut().find(|s| s.as_str() == "一點")
    {
        *amount = "有點".to_owned();
    }
    if linked_adv
        .clone()
        .any(|a| !a.is_locked && a.surface == "一番")
        && let Some(most) = adverbs.iter_mut().find(|s| s.as_str() == "一番")
    {
        *most = "最".to_owned();
    }
    if previous_range {
        times.push("都".to_owned());
    }
    if comparative
        && matches!(before, [.., marker, degree] if !marker.is_locked && marker.surface == "より"
            && !degree.is_locked && degree.surface == "ずっと" && degree.head == Some(id))
        && let Some(i) = adverbs.iter().position(|s| s == "一直")
    {
        adverbs.remove(i);
        word.push_str("得多");
    }
    if past && !has_time {
        times.push("當時".to_owned());
    }
    if colour.is_some() && !modified {
        return Some(format!(
            "{}{word}色的",
            if negative { "不是" } else { "是" }
        ));
    }
    let degree = if negative {
        "不"
    } else if modified || particle == "か" || ctx.adjective.question_mark {
        ""
    } else {
        "很"
    };
    Some(format!("{degree}{word}"))
}

#[cfg(test)]
mod tests {
    #[test]
    fn caller_object_terms_keep_colour_subjects_and_ordinary_adjectives() {
        use crate::{TargetVariant, TranslateOptions, TranslationGlossary, analyze_and_translate};
        use faberline_nlp::doc::model::Backend;

        let mut options = TranslateOptions::new(TargetVariant::ZhHant);
        options.source_hint = Some(TargetVariant::Ja);
        for (source, term, translation, expected) in [
            ("この車は白いです。", "車", "測試車", "這輛測試車是白色的。"),
            (
                "この車は白いです。",
                "この車",
                "這輛測試車",
                "這輛測試車是白色的。",
            ),
            (
                "先生の車は白いです。",
                "車",
                "測試車",
                "老師的測試車是白色的。",
            ),
            (
                "先生の車は白いです。",
                "先生の車",
                "老師的配車",
                "老師的配車是白色的。",
            ),
            (
                "昨日買った本はとても面白い。",
                "本",
                "書冊",
                "昨天買的書冊非常有趣。",
            ),
        ] {
            let glossary: TranslationGlossary = serde_json::from_value(serde_json::json!({
                "schema": "nlp.translate.glossary.v1",
                "objects": [{
                    "canonical": term,
                    "surfaces": [term],
                    "targets": {"zh-Hant": translation}
                }]
            }))
            .unwrap();
            let translated =
                analyze_and_translate(source, &options, &glossary, None, Backend::Onnx).unwrap();
            assert_eq!(translated.target_text, expected, "{source}");
            assert_eq!(translated.glossary_hits.len(), 1, "{source}");
        }
    }

    use crate::{TargetVariant, TranslateOptions, TranslationGlossary, analyze_and_translate};
    use faberline_nlp::doc::model::Backend;

    fn translate(jp: &str) -> String {
        let mut options = TranslateOptions::new(TargetVariant::ZhHant);
        options.source_hint = Some(TargetVariant::Ja);
        analyze_and_translate(
            jp,
            &options,
            &TranslationGlossary::new(),
            None,
            Backend::Onnx,
        )
        .unwrap()
        .target_text
    }

    #[test]
    fn inflected_modifiers_keep_state_inside_their_noun_phrase() {
        for (jp, expected) in [
            ("白かった車を見ました。", "看了當時是白色的車。"),
            ("白くない車を見ました。", "看了不是白色的車。"),
            ("白くなかった車を見ました。", "看了當時不是白色的車。"),
            ("重くなかった箱を運びました。", "搬運了當時不重的箱子。"),
            ("その白かった車を見ましたか。", "看了那輛當時是白色的車嗎？"),
            (
                "昨日まで白かった車を見ました。",
                "看了直到昨天都是白色的車。",
            ),
            ("白かった車は青いです。", "當時是白色的車是藍色的。"),
            (
                "父は「白かった車を見た」と言いました。",
                "父親說「看了當時是白色的車」。",
            ),
            ("白くなった車を見ました。", "看了變白的車。"),
            ("車は白かったです。", "車當時是白色的。"),
        ] {
            assert_eq!(translate(jp), expected, "{jp}");
        }
    }

    #[test]
    fn each_modified_noun_keeps_its_own_state_in_lists_and_clause_arguments() {
        for (jp, expected) in [
            (
                "白かった車と黒かった車を見ました。",
                "看了當時是白色的車和當時是黑色的車。",
            ),
            (
                "黒かった車と白かった車を見ました。",
                "看了當時是黑色的車和當時是白色的車。",
            ),
            (
                "重かった箱と軽かった箱を見ました。",
                "看了當時很重的箱子和當時很輕的箱子。",
            ),
            (
                "軽かった箱と重かった箱を見ました。",
                "看了當時很輕的箱子和當時很重的箱子。",
            ),
            (
                "白かった車と赤かった傘と黒かった箱を見ました。",
                "看了當時是白色的車和當時是紅色的傘和當時是黑色的箱子。",
            ),
            (
                "黒かった箱から白かった布を出しました。",
                "從當時是黑色的箱子拿出了當時是白色的布。",
            ),
            (
                "重かった箱や軽かった箱を見ました。",
                "看了當時很重的箱子與當時很輕的箱子。",
            ),
            (
                "小さかった子供と赤かった傘を買いました。",
                "跟當時很小的孩子買了當時是紅色的傘。",
            ),
            (
                "父は「白かった車と黒かった車を見た」と言いました。",
                "父親說「看了當時是白色的車和當時是黑色的車」。",
            ),
        ] {
            assert_eq!(translate(jp), expected, "{jp}");
        }
        let comparison = translate("白かった車より軽かった箱を運びました。");
        assert!(!comparison.contains("當時很輕的箱子"), "{comparison}");
    }

    #[test]
    fn a_degree_stays_with_the_adjective_that_owns_it() {
        assert_eq!(
            translate("とても黒い猫がゆっくり歩きました。"),
            "非常黑的貓慢慢地走了。"
        );
        assert_eq!(
            translate("この犬はとても速く走ります。"),
            "這狗非常快快地跑。"
        );
    }

    #[test]
    fn animal_colour_and_named_part_stay_with_their_source_noun() {
        for (source, expected) in [
            ("黒かった犬を見ました。", "看了當時是黑色的狗。"),
            ("目の青かった猫を見ました。", "看了眼睛當時是藍色的貓。"),
            (
                "耳の白くなかった犬を見ました。",
                "看了耳朵當時不是白色的狗。",
            ),
            ("茶色かった馬を見ました。", "看了當時是棕色的馬。"),
            ("茶色くなかった猫を見ました。", "看了當時不是棕色的貓。"),
            ("母の黒かった犬を見ました。", "看了母親的當時是黑色的狗。"),
            ("黒くなかった犬を見ました。", "看了當時不是黑色的狗。"),
            ("屋根が白かった車を買いました。", "買了車頂當時是白色的車。"),
            (
                "屋根が黒くなかった家を見ました。",
                "看了屋頂當時不是黑色的家。",
            ),
            ("屋根が白い車を見ました。", "看了車頂是白色的車。"),
            ("窓が赤かった車を見ました。", "看了車窗當時是紅色的車。"),
            (
                "白かった車と黒かった犬を見ました。",
                "看了當時是白色的車和當時是黑色的狗。",
            ),
        ] {
            assert_eq!(translate(source), expected, "{source}");
        }
        // が alone cannot establish that an observer is part of the car.
        let observed = translate("母が白かった車を見ました。");
        assert!(observed.starts_with("母親看了"), "{observed}");
    }

    #[test]
    fn a_lock_across_the_part_case_cannot_give_its_colour_to_the_whole_animal() {
        for locked in ["目の", "の"] {
            let mut glossary = TranslationGlossary::new();
            glossary.domain_terms.push(crate::GlossaryEntry {
                canonical: locked.into(),
                surfaces: vec![locked.into()],
                targets: [(TargetVariant::ZhHant, "指定詞甲".into())]
                    .into_iter()
                    .collect(),
                ..Default::default()
            });
            let text = analyze_and_translate(
                "目の青かった猫を見ました。",
                &TranslateOptions::new(TargetVariant::ZhHant),
                &glossary,
                None,
                Backend::Onnx,
            )
            .unwrap()
            .target_text;
            assert!(
                text.contains("指定詞甲") && !text.contains("當時是藍色的貓"),
                "{text}"
            );
        }
    }

    #[test]
    fn a_whole_noun_lock_preserves_its_own_adjective() {
        let mut glossary = TranslationGlossary::new();
        glossary.domain_terms.push(crate::GlossaryEntry {
            canonical: "車".into(),
            surfaces: vec!["車".into()],
            targets: [(TargetVariant::ZhHant, "轎車甲".into())]
                .into_iter()
                .collect(),
            ..Default::default()
        });
        for (source, expected) in [
            ("白かった車を見ました。", "看了當時是白色的轎車甲。"),
            (
                "白くなかった車と赤かった傘を買いました。",
                "買了當時不是白色的轎車甲和當時是紅色的傘。",
            ),
            (
                "赤かった傘と白くなかった車を買いました。",
                "買了當時是紅色的傘和當時不是白色的轎車甲。",
            ),
            (
                "友達と白かった車を見ました。",
                "跟朋友看了當時是白色的轎車甲。",
            ),
            (
                "白くなった車と赤かった傘を買いました。",
                "買了變白的轎車甲和當時是紅色的傘。",
            ),
            (
                "白かった車と黒くなかった車を見ました。",
                "看了當時是白色的轎車甲和當時不是黑色的轎車甲。",
            ),
            ("白くなかった車を見ました。", "看了當時不是白色的轎車甲。"),
            (
                "昨日から今日まで白かった車を見ました。",
                "看了從昨天到今天都是白色的轎車甲。",
            ),
        ] {
            let result = analyze_and_translate(
                source,
                &TranslateOptions::new(TargetVariant::ZhHant),
                &glossary,
                None,
                Backend::Onnx,
            )
            .unwrap();
            assert_eq!(result.target_text, expected, "{source}");
        }
    }

    #[test]
    fn finite_states_keep_their_arguments_and_mood() {
        for (jp, expected) in [
            ("この車は白いです。", "這輛車是白色的。"),
            ("先生の車は白いです。", "老師的車是白色的。"),
            ("この箱は重いです。", "這個箱子很重。"),
            ("この橋は長いです。", "這座橋很長。"),
            ("車は白かったです。", "車當時是白色的。"),
            ("車は白くなかったです。", "車當時不是白色的。"),
            ("先生の車は白いですか。", "老師的車是白色的嗎？"),
            ("その車は赤くないです。", "那輛車不是紅色的。"),
            ("この箱は重くないです。", "這個箱子不重。"),
            ("箱は重いですか。", "箱重嗎？"),
            ("この車は少し赤いです。", "這輛車有點紅。"),
            ("このかばんはとても重いです。", "這包包非常重。"),
            ("この箱はあの箱より軽いです。", "這個箱子比那個箱子輕。"),
            (
                "店員は「この車は白い」と言いました。",
                "店員說「這輛車是白色的」。",
            ),
            ("この箱は重くなりました。", "這個箱子變重了。"),
            ("この箱は重そうです。", "這個箱子看起來很重。"),
            ("この橋はあの橋ほど長くない。", "這座橋沒有那座橋那麼長。"),
            ("車が壁より白かったです。", "車當時比壁白。"),
            ("この箱はあの箱より重かった。", "這個箱子當時比那個箱子重。"),
            (
                "この箱はあの箱よりずっと重かった。",
                "這個箱子當時比那個箱子重得多。",
            ),
            (
                "この箱はあの箱より重くなかった。",
                "這個箱子當時不比那個箱子重。",
            ),
            (
                "昔、この箱はあの箱ほど重くなかった。",
                "以前，這個箱子沒有那個箱子那麼重。",
            ),
            ("車は白いが、箱は重いです。", "車是白色的，但是箱很重。"),
            ("この箱は重くて大きいです。", "這個箱子很重，也很大。"),
            ("当時、箱は重かった。", "當時，箱很重。"),
            ("昨日の車は白かった。", "昨天的車當時是白色的。"),
            ("この箱は昨日とても重かった。", "這個箱子昨天非常重。"),
            (
                "その車は黒くありませんでしたか。",
                "那輛車當時不是黑色的嗎？",
            ),
            ("この箱は重い？", "這個箱子重？"),
            ("この箱は一番重い。", "這個箱子最重。"),
            ("この箱はあまり重くないです。", "這個箱子不太重。"),
            ("この車はあまり赤くなかったです。", "這輛車當時不太紅。"),
            ("この箱はそれほど重くないです。", "這個箱子沒那麼重。"),
        ] {
            assert_eq!(translate(jp), expected, "{jp}");
        }
        assert!(!translate("少し前、この箱は軽かった。").contains("當時"));
    }

    #[test]
    fn a_clock_interval_belongs_to_the_state_inside_a_quote() {
        for (jp, expected) in [
            (
                "先生は「教室は三時まで寒かったです」と言いました。",
                "老師說「教室直到三點都很冷」。",
            ),
            (
                "先生は「三時まで教室は寒かったです」と言いました。",
                "老師說「教室直到三點都很冷」。",
            ),
            (
                "先生は「三時まで、教室は寒かった」と言いました。",
                "老師說「到三點，教室都很冷」。",
            ),
            (
                "先生は「教室は午後三時半まで寒かったです」と言いました。",
                "老師說「教室直到下午三點半都很冷」。",
            ),
            (
                "店員は八時に「この部屋は六時から七時まで寒かった」と言いました。",
                "店員在八點說「這房間從六點到七點都很冷」。",
            ),
        ] {
            assert_eq!(translate(jp), expected, "{jp}");
        }
    }

    #[test]
    fn a_time_interval_precedes_the_state_it_modifies() {
        for (jp, expected) in [
            ("昨日は夕方まで暑かったです。", "昨天直到傍晚都很熱。"),
            ("部屋は夜まで寒かったです。", "房間直到晚上都很冷。"),
            (
                "先生の車は昨日まで白かったです。",
                "老師的車直到昨天都是白色的。",
            ),
            (
                "朝から夕方までとても暑かったです。",
                "從早上到傍晚都非常熱。",
            ),
            ("夜まで部屋は寒くなかったです。", "房間直到晚上都不冷。"),
            (
                "母の車は昨日まで少し赤かったです。",
                "母親的車直到昨天都有點紅。",
            ),
            ("部屋は夜まで寒いですか。", "房間直到晚上都冷嗎？"),
            ("車は何時まで白かったですか。", "車到幾點為止都是白色的？"),
            (
                "先生の車は先週まで白くありませんでした。",
                "老師的車直到上週都不是白色的。",
            ),
            (
                "部屋は昨日まであまり寒くありませんでした。",
                "房間直到昨天都不太冷。",
            ),
            (
                "昨日まで部屋はあまり寒くありませんでした。",
                "房間直到昨天都不太冷。",
            ),
            (
                "部屋は昨日まであまり寒くなかった。",
                "房間直到昨天都不太冷。",
            ),
            (
                "朝から晩までこの部屋は寒くありませんでした。",
                "這房間從早上到晚上都不冷。",
            ),
            ("箱は昼まで重くありませんでしたか。", "箱直到中午都不重嗎？"),
            (
                "部屋は昨日まで寒くありませんでしたが、今日は寒いです。",
                "房間直到昨天都不冷，但是今天很冷。",
            ),
            (
                "先生は「この部屋は夜まで寒かった」と言いました。",
                "老師說「這房間直到晚上都很冷」。",
            ),
            (
                "店員は「この箱は昼まで重かった」と言いました。",
                "店員說「這個箱子直到中午都很重」。",
            ),
            (
                "夕方まで暑かったが、夜は涼しかった。",
                "直到傍晚都很熱，但是夜晚很涼爽。",
            ),
            (
                "今週の金曜日まで、その車は赤かったです。",
                "到這週五，那輛車都是紅色的。",
            ),
            (
                "この箱は昨日まであまり重くなかった。",
                "這個箱子直到昨天都不太重。",
            ),
            (
                "母は朝から晩まで忙しかったです。",
                "母親從早上到晚上都很忙碌。",
            ),
            ("風は昨日まで強かったです。", "風直到昨天都很強。"),
            (
                "この箱は昨日まであの箱よりずっと重かった。",
                "這個箱子直到昨天都比那個箱子重得多。",
            ),
        ] {
            assert_eq!(translate(jp), expected, "{jp}");
        }
    }

    #[test]
    fn relative_verbs_and_reported_colours_keep_their_own_arguments() {
        for (jp, expected) in [
            ("先生が買った車は白いです。", "老師買的車是白色的。"),
            ("店員が運んだ箱は軽いです。", "店員搬運的箱很輕。"),
            (
                "先生が買わなかった車は白くないです。",
                "老師沒有買的車不是白色的。",
            ),
            (
                "あの子が白いと言った布は青いです。",
                "那個孩子說是白色的布是藍色的。",
            ),
            (
                "弟が赤いと言った布は白いです。",
                "弟弟說是紅色的布是白色的。",
            ),
            (
                "先生の弟が赤いと言った布は白いです。",
                "老師的弟弟說是紅色的布是白色的。",
            ),
            (
                "弟が白くないと言った布は青いです。",
                "弟弟說不是白色的布是藍色的。",
            ),
            (
                "弟が黒いと言わなかった布は白いです。",
                "弟弟沒有說是黑色的布是白色的。",
            ),
        ] {
            assert_eq!(translate(jp), expected, "{jp}");
        }
    }

    #[test]
    fn a_modifier_a_cited_word_and_an_uncertain_colour_are_not_colour_assertions() {
        for jp in [
            "白い車が来ました。",
            "彼の腹は黒い。",
            "彼の顔は青い。",
            "彼はまだ青い。",
            "信号の色は青い。",
            "果物の色は青い。",
            "「白い」の意味を調べました。",
        ] {
            let text = translate(jp);
            assert!(!text.contains("色的"), "{jp} -> {text}");
        }
    }
}
