//! Select the order record reading when a shop clerk confirms an order.

use super::{Atom, Upos, dasu_case_link};

pub(super) const CONFIRMED_ORDER: &str = "訂單";

/// Mark only the verified object in one rendered clause. Outside speakers
/// must not supply a role for an object inside a quotation.
pub(super) fn bind(atoms: &mut [Atom], source: &[Atom], chars: &[char]) {
    for (index, noun) in source.iter().enumerate() {
        if noun.is_locked
            || noun.surface != "注文"
            || noun.upos != Some(Upos::Noun)
            || noun.deprel != "obj"
            || !has_clerk_confirmation(source, index, chars)
        {
            continue;
        }
        if let Some(rendered) = atoms.iter_mut().find(|atom| {
            !atom.is_locked
                && atom.surface == noun.surface
                && atom.upos == noun.upos
                && atom.span == noun.span
                && atom.doc_tok_idx == noun.doc_tok_idx
        }) {
            rendered.lemma = Some(CONFIRMED_ORDER.to_owned());
        }
    }
}

fn has_clerk_confirmation(source: &[Atom], index: usize, chars: &[char]) -> bool {
    let noun = &source[index];
    let Some(governor) = noun.head else {
        return false;
    };
    // A topic modifier can describe a request rather than an order record.
    if source.windows(2).any(|pair| {
        let modifier = &pair[0];
        let case = &pair[1];
        modifier.head == noun.doc_tok_idx
            && modifier.deprel == "nmod"
            && matches!(case.surface.as_str(), "について" | "に関して")
            && dasu_case_link(modifier, case)
    }) {
        return false;
    }
    source
        .get(index + 1)
        .is_some_and(|case| case.surface == "を" && dasu_case_link(noun, case))
        && source.iter().any(|atom| {
            atom.doc_tok_idx == Some(governor)
                && !atom.is_locked
                && atom.upos == Some(Upos::Verb)
                && atom.lemma.as_deref() == Some("確認する")
                && outside_relative_clause(source, atom)
        })
        && source.windows(2).enumerate().any(|(subject_index, pair)| {
            let [subject, case] = pair else {
                return false;
            };
            subject.deprel == "nsubj"
                && subject.head == Some(governor)
                && matches!(case.surface.as_str(), "が" | "は")
                && dasu_case_link(subject, case)
                && is_clerk(source, subject_index, chars)
        })
}

fn outside_relative_clause<'a>(source: &'a [Atom], mut predicate: &'a Atom) -> bool {
    for _ in source {
        // With 確認している客, the analyzer may attach the apparent
        // subject to 確認し while いる carries the adnominal attachment.
        if matches!(predicate.deprel.as_str(), "acl" | "acl:relcl") {
            return false;
        }
        let Some(head) = predicate.head else {
            return true;
        };
        let Some(parent) = source.iter().find(|atom| atom.doc_tok_idx == Some(head)) else {
            return true;
        };
        predicate = parent;
    }
    false
}

fn is_clerk(source: &[Atom], index: usize, chars: &[char]) -> bool {
    let subject = &source[index];
    match chars.get(subject.span.start_char..subject.span.end_char) {
        // The honorific pass can combine both source tokens while retaining
        // the suffix's grammatical head and a localized or locked surface.
        Some(['店', '員'] | ['店', '員', 'さ', 'ん']) => true,
        Some(['さ', 'ん']) => index.checked_sub(1).is_some_and(|previous| {
            let clerk = &source[previous];
            clerk.head == subject.doc_tok_idx
                && clerk.deprel == "compound"
                && clerk.span.end_char == subject.span.start_char
                && chars.get(clerk.span.start_char..clerk.span.end_char) == Some(&['店', '員'])
        }),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::super::Span;
    use super::*;
    use crate::{TargetVariant, TranslateOptions, TranslationGlossary, analyze_and_translate};
    use faberline_nlp::doc::model::Backend;

    #[test]
    fn translation_keeps_the_order_as_a_noun_and_preserves_the_predicate() {
        for (source, expected) in [
            ("店員は注文を確認しましたか。", "店員確認了訂單嗎？"),
            ("店員は注文を確認しませんでした。", "店員沒有確認訂單。"),
            ("店員は注文を確認している。", "店員正在確認訂單。"),
            ("店員は本の注文を確認しました。", "店員確認了書的訂單。"),
            (
                "店員は注文した料理を確認しました。",
                "店員確認了點餐的菜餚。",
            ),
            (
                "店員は修理方法についての注文を確認しました。",
                "店員確認了修理方法的訂購。",
            ),
            (
                "店員は注文を確認している客を見ました。",
                "店員正在確認看訂購客。",
            ),
            (
                "店員は注文を確認した客を見ました。",
                "店員看了確認了訂購的客。",
            ),
        ] {
            let result = analyze_and_translate(
                source,
                &TranslateOptions::new(TargetVariant::ZhHant),
                &TranslationGlossary::new(),
                None,
                Backend::Onnx,
            )
            .unwrap();
            assert_eq!(result.target_text, expected, "{source}");
        }
    }

    fn fixture() -> (Vec<Atom>, Vec<char>) {
        use Upos::{Adp, Noun, Verb};

        let tokens = [
            ("店員", Noun, None, "compound", Some(1)),
            ("さん", Noun, None, "nsubj", Some(5)),
            ("が", Adp, None, "case", Some(1)),
            ("注文", Noun, None, "obj", Some(5)),
            ("を", Adp, None, "case", Some(3)),
            ("確認し", Verb, Some("確認する"), "root", None),
        ];
        let mut chars = Vec::new();
        let atoms = tokens
            .into_iter()
            .enumerate()
            .map(|(id, (surface, upos, lemma, deprel, head))| {
                let start_char = chars.len();
                chars.extend(surface.chars());
                Atom {
                    span: Span {
                        start_char,
                        end_char: chars.len(),
                    },
                    surface: surface.to_owned(),
                    lemma: lemma.map(str::to_owned),
                    upos: Some(upos),
                    doc_tok_idx: Some(id),
                    head,
                    deprel: deprel.to_owned(),
                    is_locked: false,
                }
            })
            .collect();
        (atoms, chars)
    }

    fn assert_order(source: &[Atom], chars: &[char], expected: Option<&str>) {
        let mut atoms = source.to_vec();
        bind(&mut atoms, source, chars);
        let before = source
            .iter()
            .find(|atom| atom.doc_tok_idx == Some(3))
            .unwrap();
        let after = atoms
            .iter()
            .find(|atom| atom.doc_tok_idx == Some(3))
            .unwrap();
        assert_eq!(after.lemma.as_deref(), expected);
        assert_eq!(after.span, before.span);
        assert_eq!(after.surface, before.surface);
        assert_eq!(after.upos, before.upos);
        assert_eq!(after.head, before.head);
        assert_eq!(after.deprel, before.deprel);
        assert_eq!(after.is_locked, before.is_locked);
    }

    #[test]
    fn uses_original_clerk_span_with_split_or_resolved_honorific() {
        let (mut source, chars) = fixture();
        assert_order(&source, &chars, Some(CONFIRMED_ORDER));

        let clerk = source.remove(0);
        let subject = &mut source[0];
        subject.span.start_char = clerk.span.start_char;
        subject.upos = Some(Upos::Propn);
        subject.surface = "[[PH0]]".to_owned();
        subject.is_locked = true;
        assert_order(&source, &chars, Some(CONFIRMED_ORDER));
    }

    #[test]
    fn keeps_request_process_and_compound_verb_readings() {
        let (mut request, chars) = fixture();
        request[5].lemma = Some("つける".to_owned());
        assert_order(&request, &chars, None);

        let (mut process, _) = fixture();
        process[3].deprel = "nmod".to_owned();
        process[4].surface = "の".to_owned();
        assert_order(&process, &chars, None);

        let (mut verb, _) = fixture();
        verb[3].surface = "注文し".to_owned();
        verb[3].upos = Some(Upos::Verb);
        verb[3].deprel = "acl".to_owned();
        verb[3].lemma = Some("注文する".to_owned());
        assert_order(&verb, &chars, Some("注文する"));
    }

    #[test]
    fn requires_local_subject_and_attached_object_case() {
        let (mut source, chars) = fixture();
        // A reporter outside a quotation is absent from its rendered clause
        // even if the source dependency graph attaches them to its verb.
        assert_order(&source[3..], &chars, None);

        source[4].head = Some(1);
        assert_order(&source, &chars, None);
        source[4].head = Some(3);
        source[4].surface = "と".to_owned();
        assert_order(&source, &chars, None);
    }

    #[test]
    fn leaves_source_and_rendered_locks_untouched() {
        let (mut source, chars) = fixture();
        let mut rendered = source.clone();
        rendered[3].is_locked = true;
        bind(&mut rendered, &source, &chars);
        assert_eq!(rendered[3].lemma, None);

        source[3].is_locked = true;
        assert_order(&source, &chars, None);
    }
}
