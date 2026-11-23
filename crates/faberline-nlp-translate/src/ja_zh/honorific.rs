use std::collections::{HashMap, hash_map::Entry};

use super::{
    Atom, ClauseAlignment, Document, GlossaryKind, LockOrigin, Span, SpanLockTable, TargetVariant,
    Upos, convert_zh_text_to_target, is_case_or_sfp_to_strip, is_demonstrative_person_noun,
    is_kana, localize_close_quote, localize_open_quote, lookup_curated_ja_word,
    lookup_curated_phrase, translate_single_ja_term,
};
use crate::{TranslateUnknown, TranslateUnknownKind};
use faberline_nlp::doc::{EntityKind, SemanticClass};

const HONORIFIC_NAME: &str = "__ja_honorific_name";
const PERSON_WITH_SUFFIX: &str = "__ja_person_with_suffix";
pub(super) const DEFINED_ALIAS: &str = "__ja_defined_alias";

pub(super) fn has_name_suffix(atom: &Atom) -> bool {
    matches!(
        atom.lemma.as_deref(),
        Some(PERSON_WITH_SUFFIX | HONORIFIC_NAME)
    )
}

pub(super) fn is_bound_person(atom: &Atom) -> bool {
    atom.is_locked
        && matches!(
            atom.lemma.as_deref(),
            Some(super::LOCKED_RESIDENT_ATOM | PERSON_WITH_SUFFIX)
        )
}

pub(super) fn is_glossary_person_with_suffix(atom: &Atom) -> bool {
    atom.is_locked
        && atom.lemma.as_deref() == Some(PERSON_WITH_SUFFIX)
        && atom.surface.starts_with("[[PH")
}

#[derive(Default)]
pub(crate) struct HonorificContext {
    paragraph: Option<usize>,
    closing_quotes: Vec<char>,
    quoted_ranges: Vec<Span>,
    starts_inside_quote: bool,
    // A conflicting or qualified definition leaves the reading unresolved.
    bindings: HashMap<String, Option<String>>,
    pub(crate) unknown: Vec<TranslateUnknown>,
}

pub(super) struct Definition {
    source_pivot: usize,
    subject: String,
    open: char,
    close: char,
    predicate: String,
}

impl Definition {
    pub(super) fn render(
        self,
        doc: &Document,
        sentence: usize,
        target: TargetVariant,
    ) -> (String, Vec<ClauseAlignment>) {
        let subject = format!(
            "{}{}{}",
            localize_open_quote(self.open, target),
            convert_zh_text_to_target(&self.subject, target),
            localize_close_quote(self.close, target)
        );
        let predicate = convert_zh_text_to_target(&self.predicate, target);
        let split = subject.chars().count();
        let end = split + predicate.chars().count();
        let clauses = doc
            .segmentation
            .clauses
            .iter()
            .enumerate()
            .filter(|(_, clause)| clause.sentence == sentence)
            .map(|(idx, clause)| ClauseAlignment {
                sentence,
                clause: idx,
                source_span: clause.span,
                target_span: Span {
                    start_char: if clause.span.start_char >= self.source_pivot {
                        split
                    } else {
                        0
                    },
                    end_char: if clause.span.end_char <= self.source_pivot {
                        split
                    } else {
                        end
                    },
                },
            })
            .collect();
        (subject + &predicate, clauses)
    }
}

fn joined(atoms: &[Atom]) -> String {
    atoms.iter().map(|atom| atom.surface.as_str()).collect()
}

fn lexical_np(surface: &str) -> Option<String> {
    let entry = faberline_nlp::lexicon::lookup_ja(surface)?;
    if !matches!(
        entry.decided_upos,
        Some(Upos::Noun | Upos::Propn | Upos::Pron)
    ) {
        return None;
    }
    let text = translate_single_ja_term(surface, None, entry.decided_upos);
    (!text.is_empty() && !text.chars().any(is_kana)).then_some(text)
}

struct Nominal {
    text: String,
    person: bool,
}

fn nominal(words: &[Atom]) -> Option<Nominal> {
    // Only split at an independent genitive particle. The head may be
    // fragmented by morphology even when the full noun has a lexical reading.
    let mut groups = words.split(|a| a.surface == "の" && a.upos == Some(Upos::Adp));
    let first = groups.next()?;
    let mut result = nominal_word(first)?;
    let mut previous_pronoun = matches!(first, [word] if word.upos == Some(Upos::Pron));
    for group in groups {
        let next = nominal_word(group)?;
        if !(previous_pronoun && next.person) {
            result.text.push('的');
        }
        result.text.push_str(&next.text);
        result.person = next.person;
        previous_pronoun = matches!(group, [word] if word.upos == Some(Upos::Pron));
    }
    Some(result)
}

fn nominal_word(words: &[Atom]) -> Option<Nominal> {
    let source = joined(words);
    let (surface, text) = lexical_np(&source)
        .map(|text| (source.clone(), text))
        .or_else(|| {
            let stem = source.strip_suffix("だち")?;
            let spelling = format!("{stem}達");
            lexical_np(&spelling).map(|text| (spelling, text))
        })?;
    let entry = faberline_nlp::lexicon::lookup_ja(&surface)?;
    let mut noun = words.first()?.clone();
    noun.surface = surface;
    noun.upos = entry.decided_upos;
    let person = matches!(noun.upos, Some(Upos::Noun)) && is_demonstrative_person_noun(&noun)
        || entry.sem_class == Some(SemanticClass::Person)
        || noun.upos == Some(Upos::Pron)
            && matches!(
                noun.surface.as_str(),
                "私" | "僕" | "彼" | "彼女" | "あなた" | "君"
            );
    Some(Nominal { text, person })
}

fn definition_word(tail: &[Atom]) -> Option<(usize, bool)> {
    const NICKNAMES: &[&str] = &["あだ名", "渾名", "綽名", "ニックネーム"];
    let mut form = String::new();
    for (i, word) in tail.iter().enumerate() {
        form.push_str(&word.surface);
        if NICKNAMES.contains(&form.as_str()) {
            return Some((i + 1, true));
        }
        if form == "こと" {
            return Some((i + 1, false));
        }
        if !NICKNAMES.iter().any(|term| term.starts_with(&form)) && !"こと".starts_with(&form) {
            break;
        }
    }
    None
}

fn explained_noun(words: &[Atom]) -> Option<(String, String)> {
    if let [adjective, noun] = words {
        if adjective.head != noun.doc_tok_idx
            || noun.doc_tok_idx.is_none()
            || !matches!(adjective.deprel.as_str(), "amod" | "det" | "acl")
        {
            return None;
        }
        let modifier = match adjective.upos {
            // Japanese has the adnominal alternatives 大きな and 小さな.
            Some(Upos::Det) if adjective.surface == "小さな" => "小".to_owned(),
            Some(Upos::Det) if adjective.surface == "大きな" => "大".to_owned(),
            Some(Upos::Adj) => translate_single_ja_term(
                &adjective.surface,
                adjective.lemma.as_deref(),
                adjective.upos,
            ),
            _ => return None,
        };
        if modifier.is_empty() || modifier.chars().any(is_kana) {
            return None;
        }
        let noun = lexical_np(&noun.surface)?;
        return Some((format!("{modifier}{noun}"), format!("{modifier}的{noun}")));
    }
    let noun = nominal(words)?.text;
    Some((noun.clone(), noun))
}

pub(super) fn is_voice_report(voice: &Atom, case: &Atom, verb: &Atom) -> bool {
    !voice.is_locked
        && voice.surface == "声"
        && voice.deprel == "obj"
        && voice.head == verb.doc_tok_idx
        && !case.is_locked
        && case.surface == "を"
        && case.deprel == "case"
        && case.head == voice.doc_tok_idx
        && !verb.is_locked
        && verb.upos == Some(Upos::Verb)
        && matches!(
            verb.lemma.as_deref().unwrap_or(&verb.surface),
            "かける" | "掛ける"
        )
}

// The compound かける / 掛ける directs a call, question or statement to
// someone. Keep that act distinct from its public-appeal reading.
fn addressed_verb(verb: &Atom) -> Option<&'static str> {
    let lemma = verb.lemma.as_deref().unwrap_or(&verb.surface);
    let stem = lemma
        .strip_suffix("かける")
        .or_else(|| lemma.strip_suffix("掛ける"));
    match (stem, lemma) {
        (Some("呼び"), _) => Some("招呼說"),
        (Some("話し" | "語り"), _) => Some("開口說"),
        (Some("問い"), _) => Some("問道"),
        (_, "呼ぶ" | "叫ぶ") => Some("喊道"),
        (_, "ささやく" | "囁く") => Some("低聲說"),
        _ => None,
    }
}

// A size determiner attached to 声 and marked by で describes how an
// action sounds. The same phrase can occur on either side of a quotation.
fn voice_manner(atoms: &[Atom]) -> Option<&'static str> {
    let [size, voice, case] = atoms else {
        return None;
    };
    if atoms.iter().any(|a| a.is_locked)
        || voice.doc_tok_idx.is_none()
        || size.head != voice.doc_tok_idx
        || size.deprel != "det"
        || voice.surface != "声"
        || !matches!(voice.deprel.as_str(), "obl" | "obl:arg")
        || case.surface != "で"
        || case.deprel != "case"
        || case.head != voice.doc_tok_idx
    {
        return None;
    }
    match size.surface.as_str() {
        "大きな" => Some("大聲"),
        "小さな" => Some("小聲"),
        _ => None,
    }
}

pub(super) fn bind_voice_manners(atoms: &mut Vec<Atom>) {
    for i in (0..atoms.len().saturating_sub(2)).rev() {
        let Some(text) = atoms.get(i..i + 3).and_then(voice_manner) else {
            continue;
        };
        let mut manner = atoms[i + 1].clone();
        manner.span.start_char = atoms[i].span.start_char;
        manner.span.end_char = atoms[i + 2].span.end_char;
        manner.surface = text.to_owned();
        manner.lemma = None;
        manner.upos = Some(Upos::Adv);
        manner.deprel = "advmod".into();
        manner.is_locked = true;
        atoms.splice(i..i + 3, [manner]);
    }
}

pub(super) fn named_call(inner: &[Atom], report: &[Atom]) -> Option<String> {
    let [name, rest @ ..] = inner else {
        return None;
    };
    if !has_name_suffix(name)
        || !rest.is_empty() && !matches!(rest[0].surface.as_str(), "、" | "，")
    {
        return None;
    }
    let [marker, words @ ..] = report else {
        return None;
    };
    if marker.is_locked || marker.surface != "と" || marker.head.is_none() {
        return None;
    }
    let index = words.iter().position(|a| a.doc_tok_idx == marker.head)?;
    let (manner, head) = words.split_at(index);
    let [verb, auxiliaries @ ..] = head else {
        return None;
    };
    if verb.is_locked
        || verb.upos != Some(Upos::Verb)
        || !auxiliaries.iter().all(|a| {
            !a.is_locked
                && a.upos == Some(Upos::Aux)
                && a.head == verb.doc_tok_idx
                && matches!(a.surface.as_str(), "た" | "まし" | "ます")
        })
    {
        return None;
    }
    let spoken = addressed_verb(verb)?;
    let how = match manner {
        [] => "",
        [_, voice, _] if voice.head == verb.doc_tok_idx => voice_manner(manner)?,
        _ => return None,
    };
    Some(format!("{how}{spoken}"))
}

// Quotation marks can enclose an address or the words being discussed.
// Follow the quotative particle to the predicate that owns this quotation.
fn reported_speech(atoms: &[Atom], quoted: Span) -> bool {
    let end = atoms.partition_point(|a| a.span.start_char < quoted.end_char);
    let Some((marker, rest)) = atoms[end..].split_first() else {
        return false;
    };
    if marker.is_locked || marker.surface != "と" || marker.head.is_none() {
        return false;
    }
    let Some(verb) = rest.iter().find(|a| a.doc_tok_idx == marker.head) else {
        return false;
    };
    if verb.is_locked || verb.upos != Some(Upos::Verb) || verb.deprel == "acl" {
        return false;
    }
    matches!(
        verb.lemma.as_deref().unwrap_or(&verb.surface),
        "言う" | "話す" | "伝える" | "聞く" | "尋ねる" | "答える"
    ) || addressed_verb(verb).is_some()
        || rest.windows(2).any(|p| is_voice_report(&p[0], &p[1], verb))
}

impl HonorificContext {
    pub(super) fn enter_sentence(&mut self, paragraph: usize, atoms: &[Atom]) {
        if self.paragraph != Some(paragraph) {
            self.bindings.clear();
            self.closing_quotes.clear();
            self.paragraph = Some(paragraph);
        }
        // A definition spoken by another character does not bind the narrator's
        // later words. Mark source quote ranges without inspecting any lock.
        self.quoted_ranges.clear();
        self.starts_inside_quote = !self.closing_quotes.is_empty();
        let mut start = atoms.first().map_or(0, |a| a.span.start_char);
        for atom in atoms.iter().filter(|a| !a.is_locked) {
            match atom.surface.as_str() {
                "「" | "『" => {
                    if self.closing_quotes.is_empty() {
                        start = atom.span.start_char;
                    }
                    self.closing_quotes
                        .push(if atom.surface == "「" { '」' } else { '』' });
                }
                "」" | "』"
                    if self.closing_quotes.last().copied() == atom.surface.chars().next() =>
                {
                    self.closing_quotes.pop();
                    if self.closing_quotes.is_empty() {
                        self.quoted_ranges.push(Span {
                            start_char: start,
                            end_char: atom.span.end_char,
                        });
                    }
                }
                _ => {}
            }
        }
        if !self.closing_quotes.is_empty()
            && let Some(last) = atoms.last()
        {
            self.quoted_ranges.push(Span {
                start_char: start,
                end_char: last.span.end_char,
            });
        }
    }

    fn record_definition(&mut self, label: Option<&str>, referent: Option<&str>) {
        // A protected label has a sentence-local placeholder. Never store it
        // as the source identity of a later mention.
        let Some(label) = label else { return };
        match self.bindings.entry(label.to_owned()) {
            Entry::Vacant(entry) => {
                entry.insert(referent.map(str::to_owned));
            }
            Entry::Occupied(mut entry) if entry.get().as_deref() != referent => {
                entry.insert(None);
            }
            _ => {}
        }
    }

    /// A locked actor may fill the quoted topic. Only an exposed source label
    /// and an exposed complement can establish a later reading.
    pub(super) fn definition(
        &mut self,
        atoms: &[Atom],
        locks: &SpanLockTable,
    ) -> Option<Definition> {
        if self.starts_inside_quote {
            return None;
        }
        let open = atoms.first()?;
        let (open_q, close_q) = match open.surface.as_str() {
            "「" => ('「', '」'),
            "『" => ('『', '』'),
            _ => return None,
        };
        let close = atoms
            .iter()
            .position(|a| a.surface == close_q.to_string())?;
        let inner = &atoms[1..close];
        let locked_actor = matches!((inner, locks.locks.as_slice()), ([actor], [lock])
            if actor.is_locked && lock.mandatory && actor.span == lock.source_span
                && actor.surface == lock.placeholder
                && matches!(lock.origin, LockOrigin::Glossary { kind: GlossaryKind::Actor, .. }));
        if inner.is_empty()
            || !locked_actor
                && inner
                    .iter()
                    .any(|a| a.is_locked || !a.surface.chars().all(is_kana))
        {
            return None;
        }
        let [topic, mut_rest @ ..] = &atoms[close + 1..] else {
            return None;
        };
        if topic.surface != "は" || topic.upos != Some(Upos::Adp) || topic.is_locked {
            return None;
        }
        let mut rest = mut_rest;
        if rest
            .first()
            .is_some_and(|a| a.surface == "、" || a.surface == "，")
        {
            rest = &rest[1..];
        }
        let label = joined(inner);
        let source_label = (!locked_actor).then_some(label.as_str());
        if !locked_actor && !locks.is_empty()
            || open.is_locked
            || atoms[close..].iter().any(|a| a.is_locked)
            || atoms
                .windows(2)
                .any(|pair| pair[0].span.end_char != pair[1].span.start_char)
        {
            // A second quoted topic may introduce a new sense. The protected
            // complement cannot be used to support the earlier definition.
            if let Some(binding) = self.bindings.get_mut(&label) {
                *binding = None;
            }
            return None;
        }
        let (end, words) = rest.split_last()?;
        let (ending, words) = if end.surface == "。" {
            ("。", words)
        } else {
            ("", rest)
        };
        let no = words
            .iter()
            .rposition(|a| a.surface == "の" && a.upos == Some(Upos::Adp))?;
        let tail = &words[no + 1..];
        // Morphology can split あだ名 into several tokens. Match an entire
        // definition noun and leave its outer finite ending separate.
        let (noun_end, nickname) = definition_word(tail)?;
        let outer = &tail[noun_end..];
        let negative = match outer {
            [copula]
                if copula.upos == Some(Upos::Aux)
                    && matches!(copula.surface.as_str(), "だ" | "です") =>
            {
                false
            }
            _ if matches!(
                joined(outer).as_str(),
                "ではありません" | "ではない" | "じゃありません" | "じゃない"
            ) =>
            {
                true
            }
            _ => {
                self.record_definition(source_label, None);
                return None;
            }
        };
        let complement = &words[..no];
        let meaning = if nickname {
            nominal(complement)
                .filter(|owner| owner.person)
                .map(|owner| (label.clone(), format!("是{}的綽號", owner.text)))
        } else {
            explained_noun(complement)
                .map(|(noun, description)| (noun, format!("指的是{description}")))
        };
        let Some((referent, predicate)) = meaning else {
            self.record_definition(source_label, None);
            return None;
        };
        self.record_definition(source_label, (!negative).then_some(referent.as_str()));
        let subject = if negative || locked_actor {
            label
        } else {
            referent
        };
        Some(Definition {
            source_pivot: complement.first()?.span.start_char,
            subject,
            open: open_q,
            close: close_q,
            predicate: if negative {
                format!("不{predicate}{ending}")
            } else {
                format!("{predicate}{ending}")
            },
        })
    }

    pub(super) fn resolve(
        &mut self,
        atoms: Vec<Atom>,
        chars: &[char],
        doc: &Document,
        sentence: usize,
        locks: &SpanLockTable,
    ) -> Vec<Atom> {
        if !atoms.iter().any(|a| !a.is_locked && a.surface == "さん") {
            return atoms;
        }
        let mut result = Vec::with_capacity(atoms.len());
        let mut cursor = 0;
        for (suffix_idx, pair) in atoms.windows(2).enumerate() {
            let (suffix, case) = (&pair[0], &pair[1]);
            if suffix.is_locked
                || suffix.surface != "さん"
                || case.is_locked
                || suffix.doc_tok_idx.is_none()
                || suffix.span.end_char != case.span.start_char
                || suffix_idx <= cursor
            {
                continue;
            }
            let previous = &atoms[suffix_idx - 1];
            if previous.span.end_char != suffix.span.start_char {
                continue;
            }
            let attached_case = case.deprel == "case"
                && case.head == suffix.doc_tok_idx
                && matches!(case.upos, Some(Upos::Adp | Upos::Part))
                && matches!(
                    case.surface.as_str(),
                    "は" | "が" | "を" | "に" | "へ" | "で" | "と" | "も" | "の"
                );
            // At the start of an utterance, a comma or closing quote can end a
            // direct address without a particle. Written quotations are already
            // protected as one source span before the honorific pass.
            let quotation = self.quoted_ranges.iter().find(|span| {
                span.start_char <= suffix.span.start_char && suffix.span.end_char <= span.end_char
            });
            let address = matches!(case.upos, Some(Upos::Punct))
                && matches!(case.surface.as_str(), "、" | "，" | "」" | "』")
                && quotation.is_none_or(|span| reported_speech(&atoms, *span))
                && (suffix_idx == 1
                    || matches!(
                        atoms[suffix_idx - 2].surface.as_str(),
                        "「" | "『" | "、" | "，"
                    ));
            if !(attached_case || address) {
                continue;
            }
            // In 名前さんのです the copula and genitive particle both depend
            // on さん. The possessive renderer consumes the original owner
            // phrase, including a glossary placeholder, and keeps question scope.
            let copular_owner = case.surface == "の"
                && atoms.get(suffix_idx + 2).is_some_and(|copula| {
                    !copula.is_locked
                        && copula.upos == Some(Upos::Aux)
                        && copula.deprel == "cop"
                        && copula.head == suffix.doc_tok_idx
                        && case.span.end_char == copula.span.start_char
                });
            if previous.is_locked
                && let Some(lock) = locks.locks.iter().find(|lock| {
                    lock.mandatory
                        && lock.source_span == previous.span
                        && lock.placeholder == previous.surface
                        && matches!(
                            lock.origin,
                            LockOrigin::Glossary {
                                kind: GlossaryKind::Actor,
                                ..
                            } | LockOrigin::InlineOpaque(_)
                        )
                })
            {
                // Case and copula depend on the suffix in the source parse.
                // Carry its grammatical head while preserving the exact caller
                // placeholder. The glossary hit retains its original span.
                let mut person = suffix.clone();
                person.span.start_char = previous.span.start_char;
                person.surface.clone_from(&previous.surface);
                person.lemma = Some(
                    match lock.origin {
                        LockOrigin::Glossary { .. } => PERSON_WITH_SUFFIX,
                        _ => HONORIFIC_NAME,
                    }
                    .to_owned(),
                );
                person.upos = Some(Upos::Propn);
                person.is_locked = true;
                result.extend_from_slice(&atoms[cursor..suffix_idx - 1]);
                result.push(person);
                cursor = suffix_idx + 1;
                continue;
            }
            let mut start = suffix_idx;
            while start > cursor {
                let prev = &atoms[start - 1];
                if prev.is_locked
                    || prev.span.end_char != atoms[start].span.start_char
                    || !prev.surface.chars().all(is_kana)
                    || is_case_or_sfp_to_strip(&prev.surface)
                        && !(prev.upos == Some(Upos::Part) && prev.deprel == "discourse")
                {
                    break;
                }
                start -= 1;
            }
            let stem = joined(&atoms[start..suffix_idx]);
            let binding = if quotation.is_some() {
                None
            } else {
                self.bindings.get(&stem)
            };
            if binding.is_none()
                && !previous.is_locked
                && let Some(reading) = person_reading(&atoms[..suffix_idx], suffix, doc)
            {
                if copular_owner && matches!(reading, PersonReading::Proven) {
                    continue;
                }
                result.extend_from_slice(&atoms[cursor..suffix_idx - 1]);
                let mut noun = suffix.clone();
                noun.span.start_char = previous.span.start_char;
                noun.surface = match reading {
                    PersonReading::Proven => translate_single_ja_term(
                        &previous.surface,
                        previous.lemma.as_deref(),
                        previous.upos,
                    ),
                    PersonReading::Ambiguous => {
                        self.unknown.push(unresolved(noun.span, sentence));
                        chars[noun.span.start_char..noun.span.end_char]
                            .iter()
                            .collect()
                    }
                };
                // The case particle attaches to the suffix. Keep that syntax
                // node as the head of the complete, already-rendered name.
                noun.lemma = Some(
                    match reading {
                        PersonReading::Proven => PERSON_WITH_SUFFIX,
                        PersonReading::Ambiguous => HONORIFIC_NAME,
                    }
                    .to_owned(),
                );
                noun.upos = Some(Upos::Propn);
                noun.is_locked = true;
                result.push(noun);
                cursor = suffix_idx + 1;
                continue;
            }
            if !attached_case || start == suffix_idx {
                continue;
            }
            let span = Span {
                start_char: atoms[start].span.start_char,
                end_char: suffix.span.end_char,
            };
            let at_start = if start == 0 {
                span.start_char == doc.segmentation.sentences[sentence].span.start_char
            } else {
                let left = &atoms[start - 1];
                !left.is_locked
                    && (matches!(left.surface.as_str(), "、" | "，" | "「" | "『")
                        || binding.is_some()
                            && left.upos == Some(Upos::Adp)
                            && left.deprel == "case"
                            && left.head.is_some_and(|head| {
                                atoms[..start - 1]
                                    .iter()
                                    .any(|a| a.doc_tok_idx == Some(head))
                            }))
            };
            if !at_start
                || !doc.segmentation.clauses.iter().any(|c| {
                    c.sentence == sentence
                        && c.span.start_char <= span.start_char
                        && case.span.end_char <= c.span.end_char
                })
                || !atoms[start..=suffix_idx + 1].iter().all(|a| {
                    a.doc_tok_idx
                        .and_then(|i| doc.tokens.get(i))
                        .is_some_and(|t| t.sentence == sentence && t.span == a.span)
                })
                || locks.locks.iter().any(|l| {
                    l.source_span.start_char < case.span.end_char
                        && span.start_char < l.source_span.end_char
                })
            {
                continue;
            }
            let text = if let Some(Some(bound)) = binding {
                bound.clone()
            } else {
                let source: String = chars[span.start_char..span.end_char].iter().collect();
                let pieces = &atoms[start..suffix_idx];
                let fragmented = pieces.len() >= 2
                    && pieces.iter().all(|a| {
                        matches!(a.upos, Some(Upos::Noun | Upos::Part))
                            || a.upos == Some(Upos::Verb)
                                && a.deprel == "acl"
                                && a.head == suffix.doc_tok_idx
                                && pieces.iter().any(|n| {
                                    n.upos == Some(Upos::Noun)
                                        && n.deprel == "compound"
                                        && n.head == suffix.doc_tok_idx
                                })
                            || a.upos == Some(Upos::Aux)
                                && a.deprel == "aux"
                                && pieces.iter().any(|n| {
                                    n.upos == Some(Upos::Noun)
                                        && n.doc_tok_idx.is_some()
                                        && n.doc_tok_idx == a.head
                                })
                    });
                if !matches!(case.surface.as_str(), "は" | "が")
                    || (binding.is_none()
                        && (!fragmented
                            || known_whole(&stem)
                            || stem
                                .char_indices()
                                .any(|(byte, _)| known_whole(&source[byte..]))))
                {
                    continue;
                }
                self.unknown.push(unresolved(span, sentence));
                source
            };
            result.extend_from_slice(&atoms[cursor..start]);
            let mut noun = suffix.clone();
            noun.span = span;
            noun.surface = text;
            noun.lemma = binding
                .and_then(Option::as_ref)
                .map(|_| DEFINED_ALIAS.to_owned());
            noun.upos = Some(Upos::Noun);
            noun.is_locked = true;
            result.push(noun);
            cursor = suffix_idx + 1;
        }
        result.extend_from_slice(&atoms[cursor..]);
        result
    }
}

fn known_whole(surface: &str) -> bool {
    faberline_nlp::lexicon::lookup_ja(surface).is_some()
        || lookup_curated_ja_word(surface).is_some()
        || lookup_curated_phrase(surface).is_some()
}

fn unresolved(span: Span, sentence: usize) -> TranslateUnknown {
    TranslateUnknown {
        kind: TranslateUnknownKind::LowConfidence,
        source_span: span,
        sentence: Some(sentence),
        detail: Some("retained an ambiguous source expression with its original suffix".to_owned()),
    }
}

enum PersonReading {
    Proven,
    Ambiguous,
}
fn person_reading(before: &[Atom], suffix: &Atom, doc: &Document) -> Option<PersonReading> {
    let (noun, prefix) = before.split_last()?;
    let token = doc.tokens.get(noun.doc_tok_idx?)?;
    if token.span != noun.span || token.upos != noun.upos {
        return None;
    }
    let entry = faberline_nlp::lexicon::lookup_ja(&noun.surface)?;
    if known_whole(&format!("{}{}", noun.surface, suffix.surface)) {
        return None;
    }
    if token.upos == Some(Upos::Noun)
        && entry.decided_upos == Some(Upos::Noun)
        && is_demonstrative_person_noun(noun)
    {
        return Some(PersonReading::Proven);
    }
    let overt_person = matches!(prefix, [.., descriptor, genitive]
        if descriptor.upos == Some(Upos::Noun) && !descriptor.is_locked
        && matches!(descriptor.surface.as_str(), "男性" | "女性")
        && descriptor.deprel == "nmod" && descriptor.head == suffix.doc_tok_idx
        && genitive.surface == "の" && !genitive.is_locked && genitive.deprel == "case"
        && genitive.head == descriptor.doc_tok_idx && genitive.span.start_char == descriptor.span.end_char
        && genitive.span.end_char == noun.span.start_char && noun.head == suffix.doc_tok_idx && noun.deprel == "compound");
    if token.upos != Some(Upos::Propn)
        || token.sem_class != Some(SemanticClass::Person)
        || !entry.upos_candidates.contains(&Upos::Propn)
        || entry.sem_class != Some(SemanticClass::Person)
    {
        return None;
    }
    let span = Span {
        start_char: noun.span.start_char,
        end_char: suffix.span.end_char,
    };
    // A name entity can end at the surname. The separate honorific still
    // owns its case when the proven proper noun is its compound dependent.
    let attached_person = entry.decided_upos == Some(Upos::Propn)
        && suffix.doc_tok_idx.is_some()
        && noun.head == suffix.doc_tok_idx
        && noun.deprel == "compound"
        && noun.span.end_char == suffix.span.start_char;
    let mut exact = false;
    for entity in doc
        .entities
        .iter()
        .filter(|e| e.span.start_char < span.end_char && span.start_char < e.span.end_char)
    {
        if entity.kind != EntityKind::Name
            || !(entity.span == span || attached_person && entity.span == noun.span)
        {
            return None;
        }
        exact = true;
    }
    exact.then_some(if entry.decided_upos == Some(Upos::Propn) || overt_person {
        PersonReading::Proven
    } else {
        PersonReading::Ambiguous
    })
}

pub(super) fn is_linked_person_suffix(before: &[Atom], suffix: &Atom, doc: &Document) -> bool {
    suffix.surface == "さん"
        && !suffix.is_locked
        && suffix.doc_tok_idx.is_some()
        && before.last().is_some_and(|name| {
            !name.is_locked
                && name.upos == Some(Upos::Propn)
                && name.span.end_char == suffix.span.start_char
                && name.deprel == "compound"
                && name.head == suffix.doc_tok_idx
        })
        && matches!(
            person_reading(before, suffix, doc),
            Some(PersonReading::Proven)
        )
}
