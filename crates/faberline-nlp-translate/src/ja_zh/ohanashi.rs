//! Contextual readings for nominal お話.
use super::{
    Atom, Document, SpanLockTable, Upos, normalized_sentence_atoms, translate_single_ja_term,
};

const PREFIX: &str = "__ja_ohanashi:";
pub(super) const CONVERSE: &str = "__ja_ohanashi_converse";
pub(super) const LATER: &str = "__ja_ohanashi_later";

// Merge only a visibly marked nominal, preserving the original head token.
pub(super) fn normalize(atoms: &mut Vec<Atom>) {
    if protected(atoms) {
        return;
    }
    let mut i = 0;
    while i + 2 < atoms.len() {
        let prefix = &atoms[i];
        let head = &atoms[i + 1];
        let marker = &atoms[i + 2];
        let nominal = match marker.upos {
            Some(Upos::Adp) if marker.deprel == "case" => {
                matches!(marker.surface.as_str(), "を" | "が" | "の")
                    || (marker.surface == "は"
                        && atoms
                            .get(i + 3)
                            .is_some_and(|a| matches!(a.surface.as_str(), "、" | "，")))
            }
            Some(Upos::Aux) => matches!(marker.surface.as_str(), "です" | "だ"),
            _ => false,
        };
        if prefix.surface != "お"
            || prefix.upos != Some(Upos::Intj)
            || head.surface != "話"
            || !matches!(head.upos, Some(Upos::Noun | Upos::Verb))
            || prefix.doc_tok_idx.is_none()
            || head.doc_tok_idx.is_none()
            || (marker.head != prefix.doc_tok_idx && marker.head != head.doc_tok_idx)
            || prefix.span.end_char != head.span.start_char
            || head.span.end_char != marker.span.start_char
            || !nominal
        {
            i += 1;
            continue;
        }
        let old = prefix.doc_tok_idx;
        let mut joined = head.clone();
        joined.surface = "お話".to_owned();
        joined.span.start_char = prefix.span.start_char;
        joined.upos = Some(Upos::Noun);
        joined.lemma = None;
        let new = joined.doc_tok_idx;
        atoms.splice(i..=i + 1, [joined]);
        for atom in atoms.iter_mut() {
            if old.is_some() && atom.head == old {
                atom.head = new;
            }
        }
        i += 1;
    }
}

fn protected(atoms: &[Atom]) -> bool {
    atoms.iter().any(|a| {
        a.is_locked
            || matches!(
                a.surface.as_str(),
                "「" | "」" | "『" | "』" | "\"" | "“" | "”"
            )
    })
}

fn like(a: &Atom, words: &[&str]) -> bool {
    words.contains(&a.surface.as_str()) || a.lemma.as_deref().is_some_and(|l| words.contains(&l))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Reading {
    Story,
    Explanation,
}

fn belongs_to(atom: &Atom, head: &Atom) -> bool {
    head.doc_tok_idx.is_some() && atom.head == head.doc_tok_idx
}

fn copula(atom: &Atom, head: &Atom) -> bool {
    atom.upos == Some(Upos::Aux)
        && matches!(atom.surface.as_str(), "です" | "だ")
        && belongs_to(atom, head)
}

fn opening_topic(atoms: &[Atom]) -> Option<&[Atom]> {
    let [topic @ .., modifier, noun, ending, punctuation] = atoms else {
        return None;
    };
    let head = topic.last()?;
    (noun.surface == "お話"
        && noun.upos == Some(Upos::Noun)
        && noun.deprel == "root"
        && modifier.surface == "の"
        && modifier.deprel == "case"
        && belongs_to(modifier, head)
        && head.deprel == "nmod"
        && belongs_to(head, noun)
        && topic
            .iter()
            .all(|token| matches!(token.upos, Some(Upos::Noun | Upos::Propn)))
        && topic
            .windows(2)
            .all(|pair| pair[0].deprel == "compound" && belongs_to(&pair[0], &pair[1]))
        && copula(ending, noun)
        && matches!(punctuation.surface.as_str(), "。" | "！" | "!"))
    .then_some(topic)
}

fn positive_root(atoms: &[Atom]) -> Option<&Atom> {
    let root = atoms.iter().find(|atom| atom.deprel == "root")?;
    (!protected(atoms)
        && !atoms.iter().any(|atom| {
            matches!(atom.surface.as_str(), "？" | "?" | "別" | "別の" | "別な")
                || (belongs_to(atom, root)
                    && atom.upos == Some(Upos::Aux)
                    && like(atom, &["ない", "ぬ", "ず", "ません", "ん"]))
        }))
    .then_some(root)
}

fn same_topic(words: &[Atom], topic: &[Atom]) -> bool {
    words.iter().zip(topic).all(|(a, b)| a.surface == b.surface)
}

fn topic_link<'a>(atoms: &'a [Atom], topic: &[Atom], object: &Atom) -> Option<&'a Atom> {
    atoms.windows(topic.len() + 1).find_map(|candidate| {
        let (modifier, words) = candidate.split_last()?;
        let head = words.last()?;
        (modifier.surface == "の"
            && modifier.deprel == "case"
            && belongs_to(modifier, head)
            && head.deprel == "nmod"
            && belongs_to(head, object)
            && same_topic(words, topic))
        .then_some(head)
    })
}

fn linked_object(
    atoms: &[Atom],
    topic: &[Atom],
    root: &Atom,
    accept: impl Fn(&Atom) -> bool,
) -> bool {
    atoms.iter().any(|object| {
        object.deprel == "obj"
            && belongs_to(object, root)
            && accept(object)
            && topic_link(atoms, topic, object).is_some()
    })
}

fn prior_opening_cue(atoms: &[Atom], topic: &[Atom]) -> Option<Reading> {
    let root = positive_root(atoms)?;
    if like(root, &["説明", "説明する", "説明し"]) && linked_object(atoms, topic, root, |_| true)
    {
        Some(Reading::Explanation)
    } else if like(root, &["読む", "読み", "読み聞かせる", "読み聞かせ"])
        && linked_object(atoms, topic, root, |atom| {
            like(atom, &["絵本", "物語", "童話"])
        })
    {
        Some(Reading::Story)
    } else {
        None
    }
}

fn following_opening_cue(atoms: &[Atom], topic: &[Atom], adjacent: bool) -> Option<Reading> {
    let root = positive_root(atoms)?;
    // The demonstrative needs a topic match as well as an adjacent sentence.
    // Otherwise it could introduce a different subject.
    let this_subject = adjacent
        && matches!(atoms, [this, marker, ..]
        if this.surface == "これ" && marker.surface == "は" && belongs_to(marker, this));
    let refers_back = this_subject
        && atoms
            .windows(topic.len())
            .any(|words| same_topic(words, topic));
    let story_about_this_topic = topic_link(atoms, topic, root).is_some_and(|head| {
        atoms
            .iter()
            .any(|atom| atom.surface == "この" && belongs_to(atom, head))
            && atoms
                .iter()
                .filter(|atom| atom.deprel == "nsubj" && belongs_to(atom, root))
                .all(|atom| like(atom, &["絵本", "本", "読み物"]))
    });
    if like(root, &["物語", "童話"])
        && atoms.iter().any(|atom| copula(atom, root))
        && (refers_back || story_about_this_topic)
    {
        return Some(Reading::Story);
    }
    let this_talk = match atoms {
        [this, noun, from, marker, ..] if noun.surface == "お話" => {
            this.surface == "この"
                && from.surface == "で"
                && marker.surface == "は"
                && belongs_to(this, noun)
                && belongs_to(from, noun)
                && belongs_to(marker, noun)
        }
        [this, prefix, noun, from, marker, ..] => {
            this.surface == "この"
                && prefix.surface == "お"
                && noun.surface == "話"
                && prefix.span.end_char == noun.span.start_char
                && from.surface == "で"
                && marker.surface == "は"
                && belongs_to(this, noun)
                && belongs_to(from, noun)
                && belongs_to(marker, noun)
        }
        _ => false,
    };
    (this_talk
        && like(root, &["説明", "説明する", "説明し"])
        && linked_object(atoms, topic, root, |_| true))
    .then_some(Reading::Explanation)
}

pub(super) fn render_opening(
    atoms: &[Atom],
    doc: &Document,
    sentence: usize,
    chars: &[char],
    locks: &SpanLockTable,
) -> Option<String> {
    let topic = opening_topic(atoms)?;
    if protected(atoms) {
        return None;
    }
    let sentences = &doc.segmentation.sentences;
    let sentence_atoms = |index: usize| {
        let start = doc.tokens.partition_point(|token| token.sentence < index);
        let end = doc.tokens.partition_point(|token| token.sentence <= index);
        let tokens: Vec<_> = doc.tokens[start..end].iter().collect();
        normalized_sentence_atoms(index, chars, doc, &tokens, locks)
    };
    let continuous = |before: usize, after: usize| {
        sentences.get(after).is_some_and(|next| {
            let prior = &sentences[before];
            if prior.paragraph == next.paragraph {
                return true;
            }
            // Japanese sentence-final punctuation also splits wrapped lines
            // into analysis paragraphs. A lone, unindented LF or CRLF is
            // still a continuation for explicitly linked discourse.
            let gap = &chars[prior.span.end_char..next.span.start_char];
            let gap = gap.strip_prefix(&['\r']).unwrap_or(gap);
            gap == ['\n']
        })
    };
    let mut reading = sentence
        .checked_sub(1)
        .filter(|&index| continuous(index, sentence))
        .and_then(|index| prior_opening_cue(&sentence_atoms(index), topic));
    for next in (sentence + 1..=sentence + 3).take_while(|&index| continuous(index - 1, index)) {
        let mut context = sentence_atoms(next);
        normalize(&mut context);
        if opening_topic(&context).is_some() {
            break;
        }
        if let Some(cue) = following_opening_cue(&context, topic, next == sentence + 1) {
            if reading.is_some_and(|previous| previous != cue) {
                reading = None;
                break;
            }
            reading = Some(cue);
        }
    }
    let target: String = topic
        .iter()
        .map(|atom| translate_single_ja_term(&atom.surface, atom.lemma.as_deref(), atom.upos))
        .collect();
    let punctuation = if atoms.last()?.surface == "。" {
        "。"
    } else {
        "！"
    };
    Some(match reading {
        Some(Reading::Story) => format!("這是一則關於{target}的故事{punctuation}"),
        Some(Reading::Explanation) => format!("這是關於{target}的說明{punctuation}"),
        None => format!("這是關於{target}的內容{punctuation}"),
    })
}

fn nominal_before(atoms: &[Atom], particle: usize) -> Option<String> {
    let end = particle.checked_sub(1)?;
    let last = atoms.get(end)?;
    if !matches!(last.upos, Some(Upos::Noun | Upos::Propn | Upos::Pron)) {
        return None;
    }
    let mut start = end;
    while start > 0
        && atoms[start - 1].head == atoms[start].doc_tok_idx
        && atoms[start - 1].deprel == "compound"
        && atoms[start - 1].span.end_char == atoms[start].span.start_char
    {
        start -= 1;
    }
    Some(
        atoms[start..=end]
            .iter()
            .map(|a| a.surface.as_str())
            .collect(),
    )
}

// In a long object phrase, the parser can attach the preceding marked
// subject to one of the object's nominal modifiers.
fn reaches_object(atoms: &[Atom], mut head: Option<usize>, object: &Atom) -> bool {
    for _ in atoms {
        let Some(index) = head else { return false };
        if Some(index) == object.doc_tok_idx {
            return true;
        }
        let Some(modifier) = atoms.iter().find(|atom| atom.doc_tok_idx == Some(index)) else {
            return false;
        };
        if !matches!(
            modifier.deprel.as_str(),
            "amod" | "det" | "compound" | "nmod"
        ) {
            return false;
        }
        head = modifier.head;
    }
    false
}

fn prior_reading(atoms: &[Atom], speaker: &str) -> Option<&'static str> {
    if protected(atoms) {
        return None;
    }
    let predicate = atoms
        .iter()
        .find(|a| a.deprel == "root" && a.upos == Some(Upos::Verb))?;
    let predicate_token = predicate.doc_tok_idx?;
    let story_object = atoms.iter().find(|a| {
        like(predicate, &["読む", "読み", "読み聞かせる", "読み聞かせ"])
            && like(a, &["絵本", "物語", "童話"])
            && a.deprel == "obj"
            && a.head == Some(predicate_token)
    });
    let mut subjects = atoms
        .iter()
        .enumerate()
        .filter(|(_, a)| a.deprel == "case" && matches!(a.surface.as_str(), "は" | "が"))
        .filter_map(|(i, a)| {
            let topic = atoms.get(i.checked_sub(1)?)?;
            let linked = topic.head == Some(predicate_token)
                || story_object.is_some_and(|object| {
                    topic.span.end_char <= object.span.start_char
                        && reaches_object(atoms, topic.head, object)
                });
            (linked && topic.deprel == "nsubj" && a.head == topic.doc_tok_idx)
                .then(|| nominal_before(atoms, i))
                .flatten()
        });
    if subjects.next().as_deref() != Some(speaker) || subjects.next().is_some() {
        return None;
    }
    let explanation = like(predicate, &["説明", "説明する", "説明し"])
        || (like(predicate, &["する", "し"])
            && atoms
                .iter()
                .any(|a| like(a, &["説明"]) && a.head == predicate.doc_tok_idx));
    match (story_object.is_some(), explanation) {
        (true, false) => Some("故事"),
        (false, true) => Some("說明"),
        _ => None,
    }
}

pub(super) fn bind(
    atoms: &mut [Atom],
    doc: &Document,
    sentence: usize,
    chars: &[char],
    locks: &SpanLockTable,
) {
    if !atoms.iter().any(|a| a.surface == "お話") || protected(atoms) {
        return;
    }
    let previous = sentence
        .checked_sub(1)
        .filter(|&p| {
            doc.segmentation.sentences[p].paragraph
                == doc.segmentation.sentences[sentence].paragraph
        })
        .map(|p| {
            let start = doc.tokens.partition_point(|token| token.sentence < p);
            let end = doc.tokens.partition_point(|token| token.sentence <= p);
            let tokens: Vec<_> = doc.tokens[start..end].iter().collect();
            normalized_sentence_atoms(p, chars, doc, &tokens, locks)
        });
    let readings: Vec<_> = atoms
        .iter()
        .enumerate()
        .filter_map(|(i, a)| {
            if a.surface != "お話" || a.upos != Some(Upos::Noun) || a.is_locked {
                return None;
            }
            let original = doc.tokens.get(a.doc_tok_idx?)?;
            let exact = original.span == a.span && original.upos == a.upos;
            let split = original.span.start_char == a.span.start_char + 1
                && original.span.end_char == a.span.end_char
                && chars.get(a.span.start_char) == Some(&'お');
            if !(exact || split)
                || !chars
                    .get(a.span.start_char..a.span.end_char)?
                    .iter()
                    .copied()
                    .eq(a.surface.chars())
            {
                return None;
            }
            let governor = atoms
                .iter()
                .find(|v| v.doc_tok_idx.is_some() && v.doc_tok_idx == a.head);
            let reciprocal = a.deprel == "obj"
                && governor.is_some_and(|v| {
                    like(v, &["する", "し"])
                        && atoms.iter().any(|peer| {
                            peer.head == v.doc_tok_idx
                                && peer.deprel == "obl"
                                && atoms.iter().any(|case| {
                                    case.head == peer.doc_tok_idx && case.surface == "と"
                                })
                        })
                });
            let speaker = i
                .checked_sub(1)
                .filter(|&j| atoms[j].surface == "の")
                .and_then(|j| nominal_before(atoms, j));
            let reading = if reciprocal {
                "交談"
            } else {
                speaker
                    .as_deref()
                    .and_then(|s| previous.as_deref().and_then(|p| prior_reading(p, s)))
                    .unwrap_or_else(|| {
                        if i > 0 && atoms[i - 1].surface == "その" {
                            "段內容"
                        } else {
                            "內容"
                        }
                    })
            };
            Some((i, reading))
        })
        .collect();
    for (i, reading) in readings {
        atoms[i].lemma = Some(format!("{PREFIX}{reading}"));
    }
    if let Some(head) = atoms.iter().find_map(|a| {
        (a.lemma.as_deref() == Some("__ja_ohanashi:段內容"))
            .then_some(a.head)
            .flatten()
    }) && atoms
        .iter()
        .any(|a| a.doc_tok_idx == Some(head) && like(a, &["聞く", "聞き"]))
    {
        for time in atoms.iter_mut().filter(|a| {
            a.head == Some(head)
                && a.deprel == "advmod"
                && a.surface == "あとで"
                && a.upos == Some(Upos::Adv)
        }) {
            time.lemma = Some(LATER.to_owned());
        }
    }
}

pub(super) fn render<'a>(surface: &str, lemma: Option<&'a str>) -> Option<&'a str> {
    if surface != "お話" {
        return None;
    }
    lemma?.strip_prefix(PREFIX)
}

// Keep the finite verb and its auxiliaries, but treat this light-verb
// construction as one action. Its other arguments remain attached to it.
pub(super) fn join_light_verb(atoms: &mut Vec<Atom>) {
    let mut i = 0;
    while i + 2 < atoms.len() {
        let noun = &atoms[i];
        let case = &atoms[i + 1];
        let verb = &atoms[i + 2];
        if noun.lemma.as_deref() != Some("__ja_ohanashi:交談")
            || case.surface != "を"
            || case.head != noun.doc_tok_idx
            || noun.head != verb.doc_tok_idx
            || !like(verb, &["する", "し"])
            || noun.span.end_char != case.span.start_char
            || case.span.end_char != verb.span.start_char
        {
            i += 1;
            continue;
        }
        let mut action = verb.clone();
        action.span.start_char = noun.span.start_char;
        action.surface = "話す".to_owned();
        action.lemma = Some(CONVERSE.to_owned());
        atoms.splice(i..=i + 2, [action]);
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TargetVariant, TranslateOptions, TranslationGlossary, analyze_and_translate};
    use faberline_nlp::{Span, doc::model::Backend};

    #[test]
    fn a_story_cue_must_refer_to_the_opening_in_the_same_paragraph() {
        let mut options = TranslateOptions::new(TargetVariant::ZhHant);
        options.source_hint = Some(TargetVariant::Ja);
        for (source, sentence, expected) in [
            (
                "山のお話です。これは、山へ向かう人の物語です。",
                0,
                "這是一則關於山的故事。",
            ),
            (
                "山のお話です。\nこれは、山へ向かう人の物語です。",
                0,
                "這是一則關於山的故事。",
            ),
            (
                "山のお話です。\n\nこれは、山へ向かう人の物語です。",
                0,
                "這是關於山的內容。",
            ),
            (
                "山のお話です。\n これは、山へ向かう人の物語です。",
                0,
                "這是關於山的內容。",
            ),
            (
                "山のお話です。これは、電車の物語です。",
                0,
                "這是關於山的內容。",
            ),
            (
                "機械の名前を説明しました。機械のお話です。これは、機械の物語です。",
                1,
                "這是關於機械的內容。",
            ),
        ] {
            let output = analyze_and_translate(
                source,
                &options,
                &TranslationGlossary::new(),
                None,
                Backend::Onnx,
            )
            .unwrap();
            assert_eq!(output.sentences[sentence].target_text, expected, "{source}");
        }
    }

    fn atom(
        surface: &str,
        start: usize,
        index: usize,
        upos: Upos,
        head: Option<usize>,
        relation: &str,
    ) -> Atom {
        Atom {
            span: Span {
                start_char: start,
                end_char: start + surface.chars().count(),
            },
            surface: surface.to_owned(),
            lemma: None,
            upos: Some(upos),
            doc_tok_idx: Some(index),
            head,
            deprel: relation.to_owned(),
            is_locked: false,
        }
    }

    #[test]
    fn split_nominal_preserves_head_and_retargets_case() {
        let mut atoms = vec![
            atom("お", 0, 0, Upos::Intj, Some(2), "dep"),
            atom("話", 1, 1, Upos::Noun, Some(3), "obj"),
            atom("を", 2, 2, Upos::Adp, Some(0), "case"),
        ];
        normalize(&mut atoms);
        assert_eq!(atoms.len(), 2);
        assert_eq!(atoms[0].surface, "お話");
        assert_eq!(
            atoms[0].span,
            Span {
                start_char: 0,
                end_char: 2
            }
        );
        assert_eq!(atoms[0].head, Some(3));
        assert_eq!(atoms[1].head, Some(1));
    }

    #[test]
    fn conditional_and_protected_prefix_are_not_joined() {
        let mut atoms = vec![
            atom("お", 0, 0, Upos::Intj, Some(1), "dep"),
            atom("話", 1, 1, Upos::Verb, None, "root"),
            atom("は", 2, 2, Upos::Adp, Some(1), "case"),
            atom("続く", 3, 3, Upos::Verb, None, "root"),
        ];
        normalize(&mut atoms);
        assert_eq!(atoms.len(), 4);
        atoms[2].surface = "を".to_owned();
        atoms[0].is_locked = true;
        normalize(&mut atoms);
        assert_eq!(atoms.len(), 4);
    }

    #[test]
    fn prior_speaker_and_story_object_must_match() {
        let atoms = vec![
            atom("先生", 0, 0, Upos::Noun, Some(4), "nsubj"),
            atom("は", 2, 1, Upos::Adp, Some(0), "case"),
            atom("絵本", 3, 2, Upos::Noun, Some(4), "obj"),
            atom("を", 5, 3, Upos::Adp, Some(2), "case"),
            atom("読む", 6, 4, Upos::Verb, None, "root"),
        ];
        assert_eq!(prior_reading(&atoms, "先生"), Some("故事"));
        assert_eq!(prior_reading(&atoms, "母"), None);
        let mut other = atoms.clone();
        other[2].surface = "手紙".to_owned();
        assert_eq!(prior_reading(&other, "先生"), None);
    }

    #[test]
    fn same_clause_nominal_misattachment_can_reach_the_story_object() {
        let atoms = vec![
            atom("先生", 0, 0, Upos::Noun, Some(2), "nsubj"),
            atom("は", 2, 1, Upos::Adp, Some(0), "case"),
            atom("古い", 3, 2, Upos::Adj, Some(3), "amod"),
            atom("絵本", 5, 3, Upos::Noun, Some(5), "obj"),
            atom("を", 7, 4, Upos::Adp, Some(3), "case"),
            atom("読む", 8, 5, Upos::Verb, None, "root"),
        ];
        assert_eq!(prior_reading(&atoms, "先生"), Some("故事"));
        let mut other = atoms;
        other[2].deprel = "acl".to_owned();
        assert_eq!(prior_reading(&other, "先生"), None);
    }
}
