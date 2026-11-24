//! Apply verb glossary terms after the original syntax has supplied the grammar.

use crate::lock::{LockOrigin, LockedSpan, RestoreReport, SpanLockTable};
use crate::schema::{GlossaryKind, TranslateUnknownKind};

use super::{
    Atom, Document, Segment, Span, Token, Upos, build_sentence_atoms, classify_bunsetsu,
    dasu_case_link, household_entry, naide_adverbial, naide_ending, outward_look,
    physical_goal_head, polarity_contrast, split_ja_atoms, trip_after_meal,
};

pub(crate) fn kind(lock: &LockedSpan) -> Option<GlossaryKind> {
    match lock.origin {
        LockOrigin::Glossary { kind, .. } if lock.mandatory => Some(kind),
        _ => None,
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Coverage {
    Stem,
    Complete,
    BeforeRequest,
}

pub(super) struct PredicateTerm {
    verb: Span,
    pub(super) placeholder: String,
    pub(super) coverage: Coverage,
    pub(super) has_affirmation: bool,
    pub(super) has_request: bool,
    pub(super) has_negation: bool,
    pub(super) has_past: bool,
    pub(super) has_result_link: bool,
    pub(super) includes_native_object: bool,
}

struct SharedObject {
    verb: usize,
    object: usize,
    suffix_chars: Option<usize>,
}

#[derive(Default)]
pub(crate) struct PredicateTerms {
    terms: Vec<PredicateTerm>,
    question_verb: Option<Span>,
    site: Option<usize>,
    shared_object: Option<SharedObject>,
}

impl PredicateTerms {
    pub(super) fn add(&mut self, verb: Span, lock: &LockedSpan, coverage: Coverage) {
        let target = lock.replacement.trim();
        let request = ["請", "麻煩", "拜託"]
            .into_iter()
            .find_map(|prefix| target.strip_prefix(prefix));
        let action = request.unwrap_or(target).trim_start();
        self.terms.push(PredicateTerm {
            verb,
            placeholder: lock.placeholder.clone(),
            coverage,
            has_affirmation: target.starts_with('有'),
            has_request: request.is_some(),
            has_negation: action.starts_with(['沒', '未', '不', '無', '別', '勿'])
                || ["尚未", "還沒", "從未", "禁止"]
                    .iter()
                    .any(|prefix| action.starts_with(prefix)),
            has_past: target.ends_with(['了', '過'])
                || target.starts_with(['已', '曾'])
                || target.starts_with("未曾"),
            has_result_link: coverage == Coverage::Complete && target.ends_with('就'),
            includes_native_object: false,
        });
    }

    pub(super) fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    pub(super) fn verb(&self, atoms: &[Atom]) -> Option<&PredicateTerm> {
        self.terms.iter().find(|term| {
            atoms.iter().any(|atom| {
                !atom.is_locked && atom.upos == Some(Upos::Verb) && atom.span == term.verb
            })
        })
    }

    pub(super) fn is_question(&self, atoms: &[Atom]) -> bool {
        self.question_verb.is_some_and(|verb| {
            atoms
                .iter()
                .any(|atom| atom.upos == Some(Upos::Verb) && !atom.is_locked && atom.span == verb)
        })
    }

    pub(crate) fn restore_terms(
        &self,
        locks: &SpanLockTable,
        masked: &str,
        spans: &mut [Span],
    ) -> RestoreReport {
        let site = self.site.filter(|&index| {
            let term = &locks.locks[index];
            masked.matches(&term.placeholder).count() == 1
                && masked.matches(&format!("在{}", term.placeholder)).count() == 1
        });
        let shared = self.shared_object.as_ref().filter(|plan| {
            let verb = &locks.locks[plan.verb];
            let object = &locks.locks[plan.object];
            masked.matches(&verb.placeholder).count() == 1
                && masked.matches(&object.placeholder).count() == 1
                && masked
                    .split_once(&verb.placeholder)
                    .is_some_and(|(_, tail)| tail.trim_start().starts_with(&object.placeholder))
        });
        if site.is_none() && shared.is_none() {
            return locks.restore_with_spans(masked, spans);
        }
        let mut render = locks.clone();
        if let Some(index) = site {
            render.locks[index].replacement.remove(0);
        }
        if let Some(plan) = shared {
            render.locks[plan.object].replacement.clear();
        }
        let mut report = render.restore_with_spans(masked, spans);
        if let Some(index) = site
            && let Some(hit) = report.restored.iter_mut().find(|hit| hit.index == index)
        {
            hit.target_span.start_char -= 1;
        }
        if let Some(plan) = shared {
            let verb_end = report
                .restored
                .iter()
                .find(|hit| hit.index == plan.verb)
                .map(|hit| hit.target_span.end_char);
            if let Some(length) = plan.suffix_chars
                && let Some(end) = verb_end
                && let Some(object) = report
                    .restored
                    .iter_mut()
                    .find(|hit| hit.index == plan.object)
            {
                // Both source terms match the same already-emitted target object.
                object.target_span = Span {
                    start_char: end - length,
                    end_char: end,
                };
            } else {
                report.restored.retain(|hit| hit.index != plan.object);
                report.issues.push((
                    TranslateUnknownKind::PlaceholderDropped,
                    locks.locks[plan.object].source_span,
                    "The predicate glossary term includes the object with a different target wording."
                        .to_owned(),
                ));
            }
        }
        report
    }
}

pub(crate) fn may_restore(text: &str, locks: &SpanLockTable) -> bool {
    !text.contains(['「', '」', '『', '』', '"', '“', '”'])
        && (text.contains(['て', 'で']) || text.contains("ください"))
        && locks
            .locks
            .iter()
            .any(|lock| kind(lock) == Some(GlossaryKind::DomainTerm))
}

fn through(first: &Atom, last: &Atom) -> Span {
    Span {
        start_char: first.span.start_char,
        end_char: last.span.end_char,
    }
}

/// Each accepted pair uses a single contiguous verb and its own inflection.
fn separate_verbs(
    locks: &SpanLockTable,
    verbs: &[(Span, Span)],
) -> (SpanLockTable, PredicateTerms) {
    let mut render = SpanLockTable::new();
    let mut terms = PredicateTerms::default();
    for lock in &locks.locks {
        let matched = (kind(lock) == Some(GlossaryKind::DomainTerm))
            .then(|| {
                verbs
                    .iter()
                    .find(|(verb, whole)| lock.source_span == *verb || lock.source_span == *whole)
            })
            .flatten();
        if let Some((verb, whole)) = matched {
            let coverage = if lock.source_span == *whole {
                Coverage::Complete
            } else {
                Coverage::Stem
            };
            terms.add(*verb, lock, coverage);
        } else {
            render.locks.push(lock.clone());
        }
    }
    (render, terms)
}

fn connected_verb<'a>(before: &'a [Atom], following: &Atom) -> Option<(&'a Atom, &'a Atom)> {
    let [.., verb, connector] = before else {
        return None;
    };
    let id = verb.doc_tok_idx?;
    (verb.upos == Some(Upos::Verb)
        && matches!(verb.deprel.as_str(), "advcl" | "conj")
        && verb.head == following.doc_tok_idx
        && connector.upos == Some(Upos::Sconj)
        && matches!(connector.surface.as_str(), "て" | "で")
        && connector.deprel == "mark"
        && connector.head == Some(id)
        && verb.span.end_char == connector.span.start_char)
        .then_some((verb, connector))
}

fn simple_past(verb: &Atom, tail: &[Atom]) -> bool {
    let known = match tail {
        [past] => matches!(past.surface.as_str(), "た" | "だ" | "ました"),
        [polite, past] => polite.surface == "まし" && past.surface == "た",
        _ => false,
    };
    known
        && tail[0].span.start_char == verb.span.end_char
        && tail
            .windows(2)
            .all(|w| w[0].span.end_char == w[1].span.start_char)
        && tail.iter().all(|atom| {
            atom.upos == Some(Upos::Aux) && atom.deprel == "aux" && atom.head == verb.doc_tok_idx
        })
}

fn remove_question<'a>(atoms: &'a [Atom], verb: &Atom, punct: Option<&str>) -> (&'a [Atom], bool) {
    if let Some((last, rest)) = atoms.split_last()
        && last.surface == "か"
        && last.upos == Some(Upos::Part)
        && last.deprel == "discourse"
        && last.head == verb.doc_tok_idx
    {
        (rest, true)
    } else {
        (atoms, matches!(punct, Some("？" | "?")))
    }
}

fn object<'a>(atoms: &'a [Atom], verb: &Atom) -> Option<(&'a Atom, &'a Atom)> {
    let id = verb.doc_tok_idx?;
    let mut pairs = atoms.windows(2).filter_map(|pair| {
        let [noun, case] = pair else { return None };
        (noun.head == Some(id)
            && noun.deprel == "obj"
            && case.surface == "を"
            && dasu_case_link(noun, case))
        .then_some((noun, case))
    });
    let matched = pairs.next()?;
    pairs.next().is_none().then_some(matched)
}

enum Argument {
    Object,
    Outside,
    Destination,
    Site,
}

fn keep_arguments(locks: &mut SpanLockTable, arguments: &[(&Atom, &Atom, Argument)]) -> Option<()> {
    for lock in &mut locks.locks {
        let (noun, _, argument) = arguments.iter().find(|(noun, case, _)| {
            lock.source_span == noun.span || lock.source_span == through(noun, case)
        })?;
        let original = kind(lock)?;
        let renderer_kind = match (argument, original) {
            (
                Argument::Destination | Argument::Site,
                GlossaryKind::Place | GlossaryKind::DomainTerm,
            ) => GlossaryKind::Place,
            (Argument::Outside, GlossaryKind::Place) => GlossaryKind::Object,
            (
                Argument::Object | Argument::Outside,
                GlossaryKind::Object | GlossaryKind::DomainTerm,
            ) => GlossaryKind::Object,
            _ => return None,
        };
        lock.source_span = noun.span;
        if let LockOrigin::Glossary { kind, .. } = &mut lock.origin {
            *kind = renderer_kind;
        }
    }
    Some(())
}

fn gaze(
    before: &[Atom],
    after: &[Atom],
    punct: Option<&str>,
    doc: &Document,
    locks: &SpanLockTable,
) -> Option<(SpanLockTable, PredicateTerms)> {
    let index = after
        .iter()
        .position(|atom| atom.deprel == "root" && atom.upos == Some(Upos::Verb))?;
    let look = &after[index];
    let (open, connector) = connected_verb(before, look)?;
    let (after, question) = remove_question(after, look, punct);
    if open.lemma.as_deref() != Some("開ける")
        || !simple_past(look, &after[index + 1..])
        || after[..index].iter().any(|atom| atom.deprel == "nsubj")
    {
        return None;
    }
    let (outside, outside_case) = outward_look::outside_case(after, look, doc)?;
    let (window, window_case) = object(before, open)?;
    let (mut render, mut terms) = separate_verbs(
        locks,
        &[
            (open.span, through(open, connector)),
            (look.span, through(look, after.last()?)),
        ],
    );
    keep_arguments(
        &mut render,
        &[
            (window, window_case, Argument::Object),
            (outside, outside_case, Argument::Outside),
        ],
    )?;
    if question {
        terms.question_verb = Some(look.span);
    }
    (!terms.is_empty()).then_some((render, terms))
}

fn meal_contrast(
    sentence: usize,
    before: &[Atom],
    after: &[Atom],
    punct: Option<&str>,
    doc: &Document,
    chars: &[char],
    locks: &SpanLockTable,
) -> Option<(SpanLockTable, PredicateTerms)> {
    let index = after
        .iter()
        .position(|atom| atom.deprel == "root" && atom.upos == Some(Upos::Verb))?;
    let go = &after[index];
    let (meal, connector) = connected_verb(before, go)?;
    let (after, question) = remove_question(after, go, punct);
    let (render, mut terms) = separate_verbs(
        locks,
        &[
            (meal.span, through(meal, connector)),
            (go.span, through(go, after.last()?)),
        ],
    );
    if terms.is_empty() {
        return None;
    }
    let (render, _) =
        polarity_contrast::render_topics(sentence, before, after, doc, chars, &render)?;
    if question {
        terms.question_verb = Some(go.span);
    }
    Some((render, terms))
}

fn linked_destination(
    sentence: usize,
    before: &[Atom],
    after: &[Atom],
    punct: Option<&str>,
    doc: &Document,
    chars: &[char],
    locks: &SpanLockTable,
) -> Option<(SpanLockTable, PredicateTerms)> {
    let [place, case, go, tail @ ..] = after else {
        return None;
    };
    if go.deprel != "root"
        || !matches!(case.surface.as_str(), "に" | "へ")
        || !dasu_case_link(place, case)
        || !simple_past(go, tail)
    {
        return None;
    }
    let goal = go.doc_tok_idx?;
    let household = match go.lemma.as_deref() {
        Some("行く")
            if physical_goal_head(&classify_bunsetsu(
                vec![place.clone(), case.clone()],
                doc,
                sentence,
            )) == Some(goal) =>
        {
            false
        }
        Some("入る")
            if place.surface == "家"
                && case.surface == "に"
                && place.deprel == "obl"
                && place.head == Some(goal)
                && household_entry::allows_physical_entry(
                    doc,
                    chars,
                    goal,
                    household_entry::EntryUse::CallerDestination,
                ) =>
        {
            true
        }
        _ => return None,
    };
    let (eat, connector) = if let Some((eat, _, connector)) = naide_ending(before) {
        let omission = naide_adverbial(before, after, punct, chars)?;
        if !omission.past || !omission.join_unstated_subject || eat.head != go.doc_tok_idx {
            return None;
        }
        (eat, connector)
    } else {
        if household {
            return None;
        }
        trip_after_meal::prior_meal(before, after, doc, chars)?;
        connected_verb(before, go)?
    };
    let (meal, marker) = object(before, eat)?;
    if (household && eat.lemma.as_deref() != Some("脱ぐ"))
        || (!household
            && (eat.lemma.as_deref() != Some("食べる")
                || !trip_after_meal::is_meal_noun(
                    chars.get(meal.span.start_char..meal.span.end_char)?,
                )))
    {
        return None;
    }
    let (mut render, terms) = separate_verbs(
        locks,
        &[
            (eat.span, through(eat, connector)),
            (go.span, through(go, after.last()?)),
        ],
    );
    keep_arguments(
        &mut render,
        &[
            (meal, marker, Argument::Object),
            (place, case, Argument::Destination),
        ],
    )?;
    (!terms.is_empty()).then_some((render, terms))
}

fn polite_request(
    atoms: &[Atom],
    locks: &SpanLockTable,
) -> Option<(SpanLockTable, PredicateTerms)> {
    let (request, before) = atoms.split_last()?;
    let (verb, _, connector) = naide_ending(before)?;
    if request.surface != "ください"
        || request.deprel != "root"
        || verb.head != request.doc_tok_idx
        || !matches!(verb.lemma.as_deref(), Some("撮る" | "吸う"))
        || connector.span.end_char != request.span.start_char
    {
        return None;
    }
    let (noun, marker) = object(before, verb)?;
    let (mut render, mut terms) = separate_verbs(locks, &[(verb.span, through(verb, request))]);
    // ください lies outside the shorter match and still supplies the request.
    if let Some(index) = render.locks.iter().position(|lock| {
        kind(lock) == Some(GlossaryKind::DomainTerm) && lock.source_span == through(verb, connector)
    }) {
        let lock = render.locks.remove(index);
        terms.add(verb.span, &lock, Coverage::BeforeRequest);
    }
    let mut arguments = vec![(noun, marker, Argument::Object)];
    if let Some((site, marker)) = before.windows(2).find_map(|pair| {
        let [site, marker] = pair else { return None };
        (site.head == verb.doc_tok_idx
            && site.deprel == "obl"
            && marker.surface == "で"
            && dasu_case_link(site, marker))
        .then_some((site, marker))
    }) {
        arguments.push((site, marker, Argument::Site));
        terms.site = locks
            .locks
            .iter()
            .find(|lock| {
                (lock.source_span == site.span || lock.source_span == through(site, marker))
                    && lock.replacement.starts_with('在')
            })
            .map(|lock| lock.index);
    }
    keep_arguments(&mut render, &arguments)?;
    merge_request_object(&mut terms, locks, before, verb, noun, marker);
    (!terms.is_empty()).then_some((render, terms))
}

fn merge_request_object(
    terms: &mut PredicateTerms,
    locks: &SpanLockTable,
    clause: &[Atom],
    verb: &Atom,
    noun: &Atom,
    marker: &Atom,
) {
    let Some(term) = terms
        .terms
        .iter_mut()
        .find(|term| term.verb == verb.span && term.coverage != Coverage::Stem)
    else {
        return;
    };
    let Some(predicate_lock) = locks
        .locks
        .iter()
        .find(|lock| lock.placeholder == term.placeholder)
    else {
        return;
    };
    if clause
        .iter()
        .any(|atom| atom.head == noun.doc_tok_idx && atom.doc_tok_idx != marker.doc_tok_idx)
    {
        return;
    }
    let embedded = match (verb.lemma.as_deref(), noun.surface.as_str()) {
        (Some("撮る"), "写真") => &["拍照", "照片", "相片"][..],
        (Some("吸う"), "たばこ") => &["菸", "煙"][..],
        _ => return,
    };
    let alias = &predicate_lock.replacement;
    if !embedded.iter().any(|suffix| alias.ends_with(suffix)) {
        return;
    }
    let object = locks
        .locks
        .iter()
        .find(|lock| lock.source_span == noun.span || lock.source_span == through(noun, marker));
    if let Some(object) = object {
        let suffix_chars = alias
            .ends_with(&object.replacement)
            .then(|| object.replacement.chars().count());
        terms.shared_object = Some(SharedObject {
            verb: predicate_lock.index,
            object: object.index,
            suffix_chars,
        });
    } else {
        term.includes_native_object = true;
    }
}

pub(crate) fn restore(
    sentence: usize,
    doc: &Document,
    tokens: &[&Token],
    chars: &[char],
    locks: &SpanLockTable,
) -> Option<(SpanLockTable, PredicateTerms)> {
    let span = doc.segmentation.sentences.get(sentence)?.span;
    let mut atoms = build_sentence_atoms(sentence, chars, doc, tokens, &SpanLockTable::new());
    normalize_linked_comma(&mut atoms);
    let segments = split_ja_atoms(&atoms, doc, span, Some(chars));
    if let [
        Segment::Clause {
            atoms,
            trailing_punct,
            ..
        },
    ] = segments.as_slice()
        && matches!(trailing_punct.as_deref(), None | Some("。" | "."))
    {
        return polite_request(atoms, locks);
    }
    let [
        Segment::Clause {
            atoms: before,
            trailing_punct: Some(comma),
            ..
        },
        Segment::Clause {
            atoms: after,
            trailing_punct,
            ..
        },
    ] = segments.as_slice()
    else {
        return None;
    };
    if !matches!(comma.as_str(), "、" | "，" | ",") {
        return None;
    }
    let punct = trailing_punct.as_deref();
    if !matches!(punct, None | Some("。" | "." | "？" | "?")) {
        return None;
    }
    gaze(before, after, punct, doc, locks)
        .or_else(|| meal_contrast(sentence, before, after, punct, doc, chars, locks))
        .or_else(|| {
            if matches!(punct, Some("？" | "?")) {
                return None;
            }
            linked_destination(sentence, before, after, punct, doc, chars, locks)
        })
}

pub(super) fn normalize_linked_comma(atoms: &mut [Atom]) {
    for index in 1..atoms.len() {
        let [connector, comma] = &mut atoms[index - 1..=index] else {
            continue;
        };
        if !comma.is_locked
            && comma.upos == Some(Upos::Punct)
            && comma.surface == ","
            && connector.upos == Some(Upos::Sconj)
            && connector.deprel == "mark"
            && matches!(connector.surface.as_str(), "て" | "で")
        {
            comma.surface = "、".to_owned();
        }
    }
}

#[cfg(test)]
mod tests {
    use faberline_nlp::doc::{analyze_document, model::Backend};

    use crate::{
        GlossaryEntry, GlossaryKind, TargetVariant, TranslateOptions, TranslationGlossary,
        analyze_and_translate, doc_options_for_translate, translate_document,
    };

    use GlossaryKind::{DomainTerm, Object, Place};

    type Terms<'a> = &'a [(GlossaryKind, &'a str, &'a str)];

    fn replay(source: &str, terms: Terms<'_>, expected: &str) {
        let mut glossary = TranslationGlossary::new();
        for (kind, surface, replacement) in terms.iter().copied() {
            let entry = GlossaryEntry {
                canonical: surface.to_owned(),
                surfaces: vec![surface.to_owned()],
                targets: [(TargetVariant::ZhHant, replacement.to_owned())].into(),
                ..GlossaryEntry::default()
            };
            match kind {
                Object => glossary.objects.push(entry),
                Place => glossary.places.push(entry),
                _ => glossary.domain_terms.push(entry),
            }
        }
        let options = TranslateOptions::new(TargetVariant::ZhHant);
        let live = analyze_and_translate(source, &options, &glossary, None, Backend::Onnx).unwrap();
        let doc = analyze_document(
            source,
            &doc_options_for_translate(&options, &TranslationGlossary::new()),
        );
        let supplied = translate_document(
            source,
            &doc,
            #[cfg(feature = "narrative")]
            None,
            &options,
            &glossary,
            None,
        )
        .unwrap();
        assert_eq!(live.target_text, expected, "{source}");
        assert_eq!(supplied.target_text, live.target_text, "{source}");
        assert_eq!(supplied.glossary_hits, live.glossary_hits, "{source}");
        assert_eq!(terms.len(), live.glossary_hits.len());
        assert!(live.unknown.is_empty());
        let cut = |text: &str, span: faberline_nlp::Span| {
            text.chars()
                .skip(span.start_char)
                .take(span.end_char - span.start_char)
                .collect::<String>()
        };
        for hit in live.glossary_hits {
            let hit_source = cut(source, hit.source_span);
            let (kind, _, alias) = terms
                .iter()
                .find(|(_, surface, _)| *surface == hit_source)
                .unwrap();
            assert_eq!(&hit.kind, kind);
            assert_eq!(&cut(expected, hit.target_span), alias);
        }
    }

    #[test]
    fn inflection_and_argument_order_survive_predicate_terms() {
        let bank: serde_json::Value = serde_json::from_str(include_str!(
            "../../tests/data/ja_zh/naide_clause_dev_v1.json"
        ))
        .unwrap();
        let cases: &[(&str, Terms<'_>, &str)] = &[
            (
                "D01",
                &[(DomainTerm, "食べ", "享用")],
                "沒有享用早餐就去了學校。",
            ),
            (
                "D02",
                &[
                    (DomainTerm, "持たないで", "沒有攜帶"),
                    (DomainTerm, "出かけました", "出門"),
                ],
                "沒有攜帶傘就出門了。",
            ),
            (
                "D03",
                &[
                    (Object, "靴を", "鞋履"),
                    (DomainTerm, "脱がないで", "沒有脫下"),
                    (DomainTerm, "家に", "舊屋"),
                    (DomainTerm, "入りました", "進了"),
                ],
                "沒有脫下鞋履就進了舊屋。",
            ),
            (
                "D04",
                &[
                    (Place, "ここで", "此處"),
                    (Object, "写真を", "相片"),
                    (DomainTerm, "撮らないで", "不要拍"),
                ],
                "請在此處不要拍相片。",
            ),
            (
                "D05",
                &[
                    (DomainTerm, "たばこを", "香菸"),
                    (DomainTerm, "吸わないでください", "抽"),
                ],
                "請不要在這裡抽香菸。",
            ),
            (
                "D06",
                &[
                    (Object, "朝ご飯を", "早點"),
                    (DomainTerm, "食べて", "吃了"),
                    (DomainTerm, "学校へ", "新校舍"),
                    (DomainTerm, "行きました", "前往"),
                ],
                "吃了早點後，前往了新校舍。",
            ),
            (
                "D07",
                &[
                    (Object, "窓を", "百葉窗"),
                    (DomainTerm, "開けて", "揭開"),
                    (DomainTerm, "見ました", "察看"),
                ],
                "揭開百葉窗，往外察看了。",
            ),
            (
                "D07",
                &[(Place, "外を", "戶外"), (DomainTerm, "見", "察看")],
                "打開窗戶，察看了戶外。",
            ),
            (
                "D08",
                &[
                    (DomainTerm, "朝ご飯は", "早餐"),
                    (DomainTerm, "食べて", "享用"),
                    (DomainTerm, "学校には", "新校舍"),
                    (DomainTerm, "行きませんでした", "前往"),
                ],
                "早餐有享用，但沒有前往新校舍。",
            ),
        ];
        for (id, terms, expected) in cases {
            let row = bank["items"]
                .as_array()
                .unwrap()
                .iter()
                .find(|row| row["id"] == *id)
                .unwrap();
            assert_eq!(row["visibility"], "development-visible");
            assert_eq!(row["heldout"], false);
            replay(row["jp"].as_str().unwrap(), terms, expected);
        }
    }

    #[test]
    fn covered_negative_verb_keeps_direction_contrast_and_questions_distinct() {
        let terms = &[(DomainTerm, "行きませんでした", "未曾前往")];
        for comma in ["、", ",", ""] {
            let source = format!("朝ご飯は食べて{comma}学校には行きませんでした");
            replay(&format!("{source}。"), terms, "早餐有吃，但未曾前往學校。");
            replay(&format!("{source}か"), terms, "早餐吃，未曾前往學校嗎？");
            replay(&format!("{source}か。"), terms, "早餐吃，未曾前往學校嗎？");
        }
    }

    #[test]
    fn complete_site_and_object_terms_share_the_existing_target_text() {
        replay(
            "ここでたばこを吸わないでください。",
            &[
                (Place, "ここで", "在這裡"),
                (Object, "たばこを", "菸"),
                (DomainTerm, "吸わないでください", "請不要抽菸"),
            ],
            "在這裡請不要抽菸。",
        );
        replay(
            "ここで写真を撮らないでください。",
            &[
                (Place, "ここで", "在這裡"),
                (Object, "写真を", "照片"),
                (DomainTerm, "撮らないでください", "請不要拍照片"),
            ],
            "在這裡請不要拍照片。",
        );
        replay(
            "ここで写真を撮らないでください。",
            &[(DomainTerm, "撮らないでください", "請不要拍照")],
            "在這裡請不要拍照。",
        );
    }

    #[test]
    fn incompatible_object_alias_is_reported_without_a_false_hit() {
        let options = TranslateOptions::new(TargetVariant::ZhHant);
        let mut glossary = TranslationGlossary::new();
        for (group, source, target) in [
            (&mut glossary.objects, "写真を", "照片"),
            (
                &mut glossary.domain_terms,
                "撮らないでください",
                "請不要拍照",
            ),
        ] {
            group.push(GlossaryEntry {
                canonical: source.to_owned(),
                surfaces: vec![source.to_owned()],
                targets: [(TargetVariant::ZhHant, target.to_owned())].into(),
                ..GlossaryEntry::default()
            });
        }
        let source = "ここで写真を撮らないでください。";
        let output =
            analyze_and_translate(source, &options, &glossary, None, Backend::Onnx).unwrap();
        assert_eq!(output.target_text, "在這裡請不要拍照。");
        let [hit] = output.glossary_hits.as_slice() else {
            panic!("only the emitted phrase matches");
        };
        assert_eq!(hit.canonical, "撮らないでください");
        let [issue] = output.unknown.as_slice() else {
            panic!("report the incompatible object term");
        };
        assert_eq!(issue.kind, crate::TranslateUnknownKind::PlaceholderDropped);
        assert_eq!(
            issue.source_span,
            faberline_nlp::Span {
                start_char: 3,
                end_char: 6
            }
        );
    }
}
