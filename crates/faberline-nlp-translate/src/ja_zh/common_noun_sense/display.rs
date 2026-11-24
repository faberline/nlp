//! Render a protected quotation displayed on a counting device's screen.

use super::{
    Atom, ClauseAlignment, Document, Upos, align_boundary, attached_to, dasu_case_link, like,
    noun_phrase, render as noun_reading, sentence_body, translate_single_ja_term,
};

pub(super) fn render(
    atoms: &[Atom],
    doc: &Document,
    sentence: usize,
    chars: &[char],
) -> Option<(String, Vec<ClauseAlignment>)> {
    let (body, punctuation) = sentence_body(atoms)?;
    let [
        amount @ ..,
        object_case,
        count,
        device,
        genitive,
        screen,
        location,
        quoted,
        quote_case,
        display,
        make,
        passive,
        connective,
        ongoing,
        polite,
    ] = body
    else {
        return None;
    };
    let counted = amount.last()?;
    let quoted_source = chars.get(quoted.span.start_char..quoted.span.end_char)?;
    // Only the quoted material may be opaque. All surrounding roles must be
    // supported by the original dependency links.
    if amount.is_empty()
        || body.iter().filter(|atom| atom.is_locked).count() != 1
        || !quoted.is_locked
        || quoted.lemma.is_some()
        || !quoted.surface.starts_with("[[PH")
        || !quoted.surface.ends_with("]]")
        || !matches!(quoted_source, ['`', _, .., '`'])
        || !like(count, &["数える", "計測する", "カウントする"])
        || count.upos != Some(Upos::Verb)
        || count.deprel != "acl"
        || count.head != device.doc_tok_idx
        || !attached_to(counted, object_case, count, "obj", "を")
        || noun_reading(&device.surface, device.lemma.as_deref()) != Some("計數器")
        || device.deprel != "nmod"
        || device.head != screen.doc_tok_idx
        || genitive.surface != "の"
        || !dasu_case_link(device, genitive)
        || !like(screen, &["画面"])
        || !attached_to(screen, location, make, "obl", "に")
        || quoted.head != make.doc_tok_idx
        || quoted.deprel != "obl"
        || quoted.doc_tok_idx.is_none()
        || quote_case.is_locked
        || quote_case.upos != Some(Upos::Adp)
        || quote_case.deprel != "case"
        || quote_case.head != quoted.doc_tok_idx
        || quote_case.surface != "と"
        || !chars
            .get(quoted.span.end_char..quote_case.span.start_char)?
            .iter()
            .all(|ch| ch.is_whitespace())
        || !like(display, &["表示"])
        || display.head != make.doc_tok_idx
        || !matches!(display.deprel.as_str(), "obl" | "compound")
        || !like(make, &["する"])
        || make.upos != Some(Upos::Verb)
        || make.deprel != "advcl"
        || make.head != ongoing.doc_tok_idx
        || passive.head != make.doc_tok_idx
        || passive.deprel != "aux"
        || passive.lemma.as_deref() != Some("れる")
        || connective.head != make.doc_tok_idx
        || connective.deprel != "mark"
        || connective.surface != "て"
        || !like(ongoing, &["いる"])
        || ongoing.deprel != "root"
        || polite.head != ongoing.doc_tok_idx
        || polite.deprel != "aux"
        || polite.surface != "ます"
    {
        return None;
    }
    let amount = noun_phrase(amount, doc, sentence)?;
    let screen = translate_single_ja_term(&screen.surface, screen.lemma.as_deref(), screen.upos);
    let prefix = format!("計算{amount}的計數器{screen}");
    let cut = prefix.chars().count();
    let text = format!("{prefix}顯示著 {}{punctuation}", quoted.surface);
    let alignments = align_boundary(
        doc,
        sentence,
        location.span.end_char,
        cut,
        text.chars().count(),
    );
    Some((text, alignments))
}
