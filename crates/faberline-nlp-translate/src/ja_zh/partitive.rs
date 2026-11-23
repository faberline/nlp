//! Keep the noun role of a genitive half portion under glossary locks.

use super::{Atom, Upos, dasu_case_link};
use crate::lock::{LockOrigin, LockedSpan, SpanLockTable};
use crate::schema::GlossaryKind;

fn linked_half(whole: &Atom, genitive: &Atom, half: &Atom) -> bool {
    !half.is_locked
        && half.surface == "半分"
        && half.upos == Some(Upos::Noun)
        && half.doc_tok_idx.is_some()
        && whole.deprel == "nmod"
        && whole.head == half.doc_tok_idx
        && genitive.surface == "の"
        && dasu_case_link(whole, genitive)
        && genitive.span.end_char == half.span.start_char
}

pub(super) fn normalize(atoms: &mut [Atom]) {
    for index in 2..atoms.len() {
        if let [whole, genitive, half] = &mut atoms[index - 2..=index]
            && linked_half(whole, genitive, half)
        {
            // Keep the genitive and its existing noun-phrase rendering.
            half.surface = "一半".to_owned();
            half.is_locked = true;
        }
    }
}

fn is_caller_term(lock: &LockedSpan) -> bool {
    lock.mandatory
        && matches!(
            lock.origin,
            LockOrigin::Glossary {
                kind: GlossaryKind::DomainTerm | GlossaryKind::Object,
                ..
            }
        )
}

pub(super) fn may_repair(text: &str, chars: &[char], locks: &SpanLockTable) -> bool {
    text.contains("の半分")
        && locks.locks.iter().any(|lock| {
            is_caller_term(lock)
                && matches!(
                    chars.get(lock.source_span.start_char..lock.source_span.end_char),
                    Some(['半'] | ['分'] | ['の', '半'] | ['の', '半', '分'])
                )
        })
}

pub(super) struct Repair {
    genitive: Option<Atom>,
    portion: Atom,
    markers: Vec<String>,
}

fn repair(genitive: &Atom, half: &Atom, locks: &SpanLockTable) -> Option<Repair> {
    let boundary = genitive.span.start_char;
    let start = half.span.start_char;
    let end = half.span.end_char;
    let mut first = None;
    let mut last = None;
    for lock in &locks.locks {
        let span = lock.source_span;
        if span.start_char >= end || span.end_char <= start {
            continue;
        }
        if !is_caller_term(lock) {
            return None;
        }
        match (span.start_char, span.end_char) {
            (a, b) if a == boundary && (b == start + 1 || b == end) && first.is_none() => {
                first = Some(lock);
            }
            (a, b) if a == start && b == start + 1 && first.is_none() => first = Some(lock),
            (a, b) if a == start + 1 && b == end && last.is_none() => last = Some(lock),
            _ => return None,
        }
    }
    let surface = match (first, last) {
        // 分 is the portion suffix. A protected 半 supplies the portion;
        // a protected 分 still needs the untranslated half quantity.
        (Some(a), None) => a.placeholder.clone(),
        (None, Some(b)) => format!("半{}", b.placeholder),
        (Some(a), Some(b)) => format!("{}{}", a.placeholder, b.placeholder),
        (None, None) => return None,
    };
    let mut portion = half.clone();
    portion.surface = surface;
    portion.is_locked = true;
    let genitive = first
        .is_some_and(|lock| lock.source_span.start_char == boundary)
        .then(|| genitive.clone());
    let markers = first
        .into_iter()
        .chain(last)
        .map(|lock| lock.placeholder.clone())
        .collect();
    Some(Repair {
        genitive,
        portion,
        markers,
    })
}

pub(super) fn prove(source: &[Atom], locks: &SpanLockTable) -> Vec<Repair> {
    source
        .windows(3)
        .filter_map(|trio| {
            let [whole, genitive, half] = trio else {
                return None;
            };
            if !linked_half(whole, genitive, half) {
                return None;
            }
            repair(genitive, half, locks)
        })
        .collect()
}

/// A recovered の still carries the noun relation. Its marker can supply 的.
pub(super) fn supplies_genitive(half: &Atom, modifier: &[Atom], locks: &[LockedSpan]) -> bool {
    let Some(genitive) = modifier.last() else {
        return false;
    };
    let start = half.span.start_char;
    half.is_locked
        && half.upos == Some(Upos::Noun)
        && genitive.surface == "の"
        && genitive.span.end_char == start
        && locks.iter().any(|lock| {
            is_caller_term(lock)
                && lock.source_span.start_char == genitive.span.start_char
                && [start + 1, half.span.end_char].contains(&lock.source_span.end_char)
                && half.surface.starts_with(&lock.placeholder)
                && lock.replacement.trim_start().starts_with('的')
        })
}

impl Repair {
    pub(super) fn apply(&self, atoms: &mut Vec<Atom>) {
        let source_start = self
            .genitive
            .as_ref()
            .map_or(self.portion.span.start_char, |atom| atom.span.start_char);
        let source_end = self.portion.span.end_char;
        let start = atoms.partition_point(|atom| atom.span.end_char <= source_start);
        let end = start + atoms[start..].partition_point(|atom| atom.span.start_char < source_end);
        let parts = &atoms[start..end];
        if parts.iter().any(|a| {
            a.span.start_char < source_start
                || a.span.end_char > source_end
                || a.is_locked && !self.markers.contains(&a.surface)
        }) || self.markers.iter().any(|marker| {
            parts
                .iter()
                .filter(|a| a.is_locked && a.surface == *marker)
                .count()
                != 1
        }) {
            return;
        }
        // The original lock table still supplies each hit's exact source span.
        atoms.splice(
            start..end,
            self.genitive.iter().cloned().chain([self.portion.clone()]),
        );
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        Backend, GlossaryEntry, TargetVariant, TranslateOptions, TranslationGlossary,
        analyze_and_translate,
    };

    #[test]
    fn glossary_keeps_the_portion_and_exact_hits() {
        let options = TranslateOptions::new(TargetVariant::ZhHant);
        for (source, entries, expected) in [
            (
                "ケーキの半分を食べました。",
                &[("半", "半量")][..],
                "吃了蛋糕的半量。",
            ),
            (
                "ケーキの半分を食べました。",
                &[("分", "份")][..],
                "吃了蛋糕的半份。",
            ),
            (
                "ケーキの半分を食べました。",
                &[("半", "半"), ("分", "份")][..],
                "吃了蛋糕的半份。",
            ),
            (
                "ケーキの半分を食べました。",
                &[("の半", "的半"), ("分", "份")][..],
                "吃了蛋糕的半份。",
            ),
            (
                "ケーキの半分を食べました。",
                &[("ケーキ", "點心"), ("の半", "50%")][..],
                "吃了點心的50%。",
            ),
            (
                "この本の半分を読みました。",
                &[("の半分", "前半")][..],
                "讀了這本書的前半。",
            ),
            (
                "この本の半分を読みました。",
                &[("の半分", "的一半")][..],
                "讀了這本書的一半。",
            ),
            (
                "ケーキの半分とパンの半分を食べました。",
                &[("半", "50%")][..],
                "吃了蛋糕的50%和麵包的50%。",
            ),
            (
                "午後二時半にケーキの半分を食べました。",
                &[("半", "半量")][..],
                "在下午兩點半量吃了蛋糕的半量。",
            ),
            (
                "会議は午後二時半に始まります。",
                &[("半", "半量")][..],
                "會議在下午兩點半量開始。",
            ),
        ] {
            let glossary = TranslationGlossary {
                domain_terms: entries
                    .iter()
                    .map(|&(jp, zh)| GlossaryEntry {
                        canonical: jp.to_owned(),
                        surfaces: vec![jp.to_owned()],
                        targets: [(TargetVariant::ZhHant, zh.to_owned())]
                            .into_iter()
                            .collect(),
                        ..GlossaryEntry::default()
                    })
                    .collect(),
                ..TranslationGlossary::new()
            };
            let doc =
                analyze_and_translate(source, &options, &glossary, None, Backend::Onnx).unwrap();
            assert_eq!(doc.target_text, expected, "{source}");
            let expected_hits = entries
                .iter()
                .map(|&(jp, _)| source.matches(jp).count())
                .sum::<usize>();
            assert_eq!(doc.glossary_hits.len(), expected_hits);
            for hit in doc.glossary_hits {
                let expected = entries
                    .iter()
                    .find(|&&(jp, _)| jp == hit.canonical)
                    .unwrap();
                let slice = |text: &str, span: crate::Span| {
                    text.chars()
                        .skip(span.start_char)
                        .take(span.end_char - span.start_char)
                        .collect::<String>()
                };
                assert_eq!(slice(source, hit.source_span), expected.0);
                assert_eq!(slice(&doc.target_text, hit.target_span), expected.1);
            }
        }
    }
}
