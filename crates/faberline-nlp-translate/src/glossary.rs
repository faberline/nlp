//! Glossary matching (leftmost-longest with ASCII word boundaries) and deterministic entity
//! locale formatting over `nlp.doc.v1`.

use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};
use faberline_nlp::Span;
use faberline_nlp::doc::{Entity, EntityKind};

use crate::schema::{
    GlossaryEntry, GlossaryKind, PronounClass, SpeechRegister, TargetVariant, TranslationGlossary,
};

#[derive(Clone, Debug)]
struct CompiledPattern {
    surface_chars: usize,
    ascii_edges: (bool, bool),
    canonical: String,
    kind: GlossaryKind,
    target_text: String,
    pronoun_class: PronounClass,
    speech_register: SpeechRegister,
}

/// Compiled leftmost-longest glossary matcher for a specific `TargetVariant`.
pub struct CompiledGlossary {
    ac: Option<AhoCorasick>,
    patterns: Vec<CompiledPattern>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GlossaryMatch {
    pub span: Span,
    pub canonical: String,
    pub kind: GlossaryKind,
    pub target_text: String,
    pub pronoun_class: PronounClass,
    pub speech_register: SpeechRegister,
}

impl CompiledGlossary {
    pub fn compile(
        glossary: &TranslationGlossary,
        target: TargetVariant,
        fallback_convert: impl Fn(&str, TargetVariant) -> String,
    ) -> Self {
        let mut raw: Vec<(String, CompiledPattern)> = Vec::new();
        let groups = [
            (&glossary.actors, GlossaryKind::Actor, 0u8),
            (&glossary.places, GlossaryKind::Place, 1u8),
            (&glossary.objects, GlossaryKind::Object, 2u8),
            (&glossary.domain_terms, GlossaryKind::DomainTerm, 3u8),
        ];

        for (entries, kind, _prio) in groups {
            for entry in entries {
                let rendered = resolve_entry_target(entry, target, &fallback_convert);
                for surf in &entry.surfaces {
                    if surf.is_empty() || surf.contains('\n') || surf.contains('\0') {
                        continue;
                    }
                    let first_ascii = surf
                        .chars()
                        .next()
                        .is_some_and(|c| c.is_ascii_alphanumeric());
                    let last_ascii = surf
                        .chars()
                        .next_back()
                        .is_some_and(|c| c.is_ascii_alphanumeric());
                    raw.push((
                        surf.clone(),
                        CompiledPattern {
                            surface_chars: surf.chars().count(),
                            ascii_edges: (first_ascii, last_ascii),
                            canonical: entry.canonical.clone(),
                            kind,
                            target_text: rendered.clone(),
                            pronoun_class: entry.pronoun_class,
                            speech_register: entry.speech_register,
                        },
                    ));
                }
            }
        }

        // Sort longer surfaces first so Aho-Corasick LeftmostLongest breaks exact-start ties cleanly.
        raw.sort_by(|a, b| {
            b.1.surface_chars
                .cmp(&a.1.surface_chars)
                .then_with(|| a.0.cmp(&b.0))
        });
        raw.dedup_by(|a, b| a.0 == b.0);

        if raw.is_empty() {
            return Self {
                ac: None,
                patterns: Vec::new(),
            };
        }

        let needles: Vec<&str> = raw.iter().map(|(s, _)| s.as_str()).collect();
        let ac = AhoCorasickBuilder::new()
            .match_kind(MatchKind::LeftmostLongest)
            .build(&needles)
            .ok();
        let patterns = raw.into_iter().map(|(_, p)| p).collect();

        Self { ac, patterns }
    }

    pub fn is_empty(&self) -> bool {
        self.ac.is_none()
    }

    /// Scan `slice_text` (whose first character is at document scalar offset `base_char`) and
    /// return non-overlapping glossary matches that respect ASCII word boundaries and never cut
    /// inside `blocked_spans`.
    pub fn find_matches(
        &self,
        slice_text: &str,
        chars: &[char],
        base_char: usize,
        blocked_spans: &[Span],
    ) -> Vec<GlossaryMatch> {
        let Some(ac) = &self.ac else {
            return Vec::new();
        };
        if slice_text.is_empty() {
            return Vec::new();
        }

        // Byte offset -> scalar offset within `slice_text`.
        let mut byte_to_char = vec![0usize; slice_text.len() + 1];
        for (ci, (bi, ch)) in slice_text.char_indices().enumerate() {
            byte_to_char[bi] = ci;
            byte_to_char[bi + ch.len_utf8()] = ci + 1;
        }

        let mut out = Vec::new();
        for m in ac.find_iter(slice_text) {
            let pat = &self.patterns[m.pattern().as_usize()];
            let local_start = byte_to_char[m.start()];
            let local_end = byte_to_char[m.end()];
            let start_char = base_char + local_start;
            let end_char = base_char + local_end;

            // ASCII word boundary guard (matches `nlp.doc.v1` EntityKind::Name rule).
            if pat.ascii_edges.0
                && start_char > 0
                && chars
                    .get(start_char - 1)
                    .is_some_and(|c| c.is_ascii_alphanumeric())
            {
                continue;
            }
            if pat.ascii_edges.1
                && chars
                    .get(end_char)
                    .is_some_and(|c| c.is_ascii_alphanumeric())
            {
                continue;
            }

            let span = Span {
                start_char,
                end_char,
            };
            if blocked_spans
                .iter()
                .any(|b| span.start_char < b.end_char && b.start_char < span.end_char)
            {
                continue;
            }

            out.push(GlossaryMatch {
                span,
                canonical: pat.canonical.clone(),
                kind: pat.kind,
                target_text: pat.target_text.clone(),
                pronoun_class: pat.pronoun_class,
                speech_register: pat.speech_register,
            });
        }
        out
    }
}

fn resolve_entry_target(
    entry: &GlossaryEntry,
    target: TargetVariant,
    fallback_convert: &impl Fn(&str, TargetVariant) -> String,
) -> String {
    if let Some(exact) = entry.targets.get(&target) {
        return exact.clone();
    }
    // Same-language sibling fallback (e.g. `en-US` <-> `en-GB` or `zh-Hant` <-> `zh-Hans`).
    match target {
        TargetVariant::EnUs => {
            if let Some(gb) = entry.targets.get(&TargetVariant::EnGb) {
                return fallback_convert(gb, TargetVariant::EnUs);
            }
        }
        TargetVariant::EnGb => {
            if let Some(us) = entry.targets.get(&TargetVariant::EnUs) {
                return fallback_convert(us, TargetVariant::EnGb);
            }
        }
        TargetVariant::ZhHant => {
            if let Some(hans) = entry.targets.get(&TargetVariant::ZhHans) {
                return fallback_convert(hans, TargetVariant::ZhHant);
            }
        }
        TargetVariant::ZhHans => {
            if let Some(hant) = entry.targets.get(&TargetVariant::ZhHant) {
                return fallback_convert(hant, TargetVariant::ZhHans);
            }
        }
        _ => {}
    }
    // If any source surface belongs to a Chinese/English script and target is a sibling variant,
    // convert the primary surface; otherwise fall back to humanized canonical or primary surface.
    if let Some(primary) = entry.surfaces.first() {
        if target.is_zh() && primary.chars().any(is_han_char) {
            return fallback_convert(primary, target);
        }
        if target.is_en() && primary.is_ascii() {
            return fallback_convert(primary, target);
        }
        if target.is_en() && !entry.canonical.is_empty() {
            return humanize_canonical(&entry.canonical, entry_is_proper(entry));
        }
        return primary.clone();
    }
    entry.canonical.clone()
}

fn entry_is_proper(entry: &GlossaryEntry) -> bool {
    !entry.canonical.is_empty()
}

fn humanize_canonical(canonical: &str, title_case: bool) -> String {
    let words: Vec<String> = canonical
        .split(['_', '-'])
        .filter(|w| !w.is_empty())
        .map(|w| {
            if title_case {
                let mut cs = w.chars();
                match cs.next() {
                    Some(first) => {
                        let mut s = String::new();
                        for u in first.to_uppercase() {
                            s.push(u);
                        }
                        s.push_str(cs.as_str());
                        s
                    }
                    None => String::new(),
                }
            } else {
                w.to_owned()
            }
        })
        .collect();
    words.join(" ")
}

fn is_han_char(c: char) -> bool {
    matches!(
        c as u32,
        0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0x20000..=0x2CEAF | 0xF900..=0xFAFF
    )
}

/// Deterministic target-locale formatter for Layer-3 `Entity` (`Date`, `Time`, `Number`, `Quantity`).
/// Returns `Some(formatted)` when `entity` has a normalized `value` that localizes cleanly to `target`.
pub fn format_entity_for_target(entity: &Entity, target: TargetVariant) -> Option<String> {
    match entity.kind {
        EntityKind::Date => {
            let val = entity.value.as_deref()?;
            format_iso_date(val, target)
        }
        EntityKind::Time => {
            let val = entity.value.as_deref()?;
            Some(val.to_owned())
        }
        EntityKind::Number => {
            let val = entity.value.as_deref()?;
            if target.is_en() {
                Some(group_thousands_ascii(val))
            } else {
                Some(val.to_owned())
            }
        }
        EntityKind::Quantity => {
            let val = entity.value.as_deref()?;
            let unit = entity.unit.as_deref()?;
            if unit == "%" {
                return Some(format!("{val}%"));
            }
            let norm_val = if target.is_en() {
                group_thousands_ascii(val)
            } else {
                val.to_owned()
            };
            match map_common_unit(unit, target) {
                Some((u_str, space)) => {
                    if space {
                        Some(format!("{norm_val} {u_str}"))
                    } else {
                        Some(format!("{norm_val}{u_str}"))
                    }
                }
                None => None,
            }
        }
        _ => None,
    }
}

fn format_iso_date(iso: &str, target: TargetVariant) -> Option<String> {
    const EN_MONTHS: [&str; 12] = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ];
    if let Some(mm_dd) = iso.strip_prefix("--") {
        let (m_str, d_str) = mm_dd.split_once('-')?;
        let m: usize = m_str.parse().ok()?;
        let d: u32 = d_str.parse().ok()?;
        if !(1..=12).contains(&m) {
            return None;
        }
        return Some(match target {
            TargetVariant::ZhHant | TargetVariant::ZhHans | TargetVariant::Ja => {
                format!("{m}月{d}日")
            }
            TargetVariant::Ko => format!("{m}월 {d}일"),
            TargetVariant::EnUs => format!("{} {d}", EN_MONTHS[m - 1]),
            TargetVariant::EnGb => format!("{d} {}", EN_MONTHS[m - 1]),
        });
    }

    let parts: Vec<&str> = iso.split('-').collect();
    match parts.as_slice() {
        [y] if y.len() == 4 => {
            let yr: u32 = y.parse().ok()?;
            Some(match target {
                TargetVariant::ZhHant | TargetVariant::ZhHans | TargetVariant::Ja => {
                    format!("{yr}年")
                }
                TargetVariant::Ko => format!("{yr}년"),
                TargetVariant::EnUs | TargetVariant::EnGb => format!("{yr}"),
            })
        }
        [y, m_s] => {
            let yr: u32 = y.parse().ok()?;
            let m: usize = m_s.parse().ok()?;
            if !(1..=12).contains(&m) {
                return None;
            }
            Some(match target {
                TargetVariant::ZhHant | TargetVariant::ZhHans | TargetVariant::Ja => {
                    format!("{yr}年{m}月")
                }
                TargetVariant::Ko => format!("{yr}년 {m}월"),
                TargetVariant::EnUs | TargetVariant::EnGb => format!("{} {yr}", EN_MONTHS[m - 1]),
            })
        }
        [y, m_s, d_s] => {
            let yr: u32 = y.parse().ok()?;
            let m: usize = m_s.parse().ok()?;
            let d: u32 = d_s.parse().ok()?;
            if !(1..=12).contains(&m) {
                return None;
            }
            Some(match target {
                TargetVariant::ZhHant | TargetVariant::ZhHans | TargetVariant::Ja => {
                    format!("{yr}年{m}月{d}日")
                }
                TargetVariant::Ko => format!("{yr}년 {m}월 {d}일"),
                TargetVariant::EnUs => format!("{} {d}, {yr}", EN_MONTHS[m - 1]),
                TargetVariant::EnGb => format!("{d} {} {yr}", EN_MONTHS[m - 1]),
            })
        }
        _ => None,
    }
}

fn group_thousands_ascii(num_str: &str) -> String {
    let (int_part, frac_part) = match num_str.split_once('.') {
        Some((i, f)) => (i, Some(f)),
        None => (num_str, None),
    };
    if !int_part.bytes().all(|b| b.is_ascii_digit()) || int_part.len() <= 3 {
        return num_str.to_owned();
    }
    let mut grouped = String::with_capacity(num_str.len() + int_part.len() / 3);
    let len = int_part.len();
    for (idx, ch) in int_part.chars().enumerate() {
        if idx > 0 && (len - idx) % 3 == 0 {
            grouped.push(',');
        }
        grouped.push(ch);
    }
    if let Some(f) = frac_part {
        grouped.push('.');
        grouped.push_str(f);
    }
    grouped
}

fn map_common_unit(unit: &str, target: TargetVariant) -> Option<(&'static str, bool)> {
    match (unit, target) {
        ("kg" | "公斤", TargetVariant::EnUs | TargetVariant::EnGb) => Some(("kg", true)),
        ("km" | "公里", TargetVariant::EnUs | TargetVariant::EnGb) => Some(("km", true)),
        ("公斤" | "kg", TargetVariant::ZhHant | TargetVariant::ZhHans) => Some(("公斤", false)),
        ("公里" | "km", TargetVariant::ZhHant | TargetVariant::ZhHans) => Some(("公里", false)),
        ("分鐘" | "分钟", TargetVariant::EnUs | TargetVariant::EnGb) => Some(("minutes", true)),
        ("小時" | "小时", TargetVariant::EnUs | TargetVariant::EnGb) => Some(("hours", true)),
        ("天" | "日" | "days", TargetVariant::EnUs | TargetVariant::EnGb) => Some(("days", true)),
        ("年", TargetVariant::EnUs | TargetVariant::EnGb) => Some(("years", true)),
        ("分鐘" | "分钟", TargetVariant::ZhHant) => Some(("分鐘", false)),
        ("分鐘" | "分钟", TargetVariant::ZhHans) => Some(("分钟", false)),
        ("小時" | "小时", TargetVariant::ZhHant) => Some(("小時", false)),
        ("小時" | "小时", TargetVariant::ZhHans) => Some(("小时", false)),
        ("分鐘" | "分钟", TargetVariant::Ja) => Some(("分", false)),
        ("小時" | "小时", TargetVariant::Ja) => Some(("時間", false)),
        ("分鐘" | "分钟", TargetVariant::Ko) => Some(("분", false)),
        ("小時" | "小时", TargetVariant::Ko) => Some(("시간", false)),
        _ => None,
    }
}
