//! Distinguish a household destination from a linked family role.

use faberline_nlp::doc::{Document, SyntaxToken, Upos};

const FAMILY_ROLES: &[&str] = &["婿", "嫁", "養子", "養女", "婿養子", "家族"];
const HOME_ROLES: &[&str] = &["主婦", "主夫", "専業主婦", "専業主夫"];

pub(super) fn allows_physical_entry(doc: &Document, source: &[char], entry: usize) -> bool {
    let Some(token) = doc.tokens.get(entry) else {
        return false;
    };
    let Some(sentence) = doc.segmentation.sentences.get(token.sentence) else {
        return false;
    };
    if source
        .get(sentence.span.start_char..sentence.span.end_char)
        .is_none()
    {
        return false;
    }
    let start = doc
        .tokens
        .partition_point(|token| token.span.start_char < sentence.span.start_char);
    let end = doc
        .tokens
        .partition_point(|token| token.span.start_char < sentence.span.end_char);
    let Some(links) = doc.syntax.get(start..end) else {
        return false;
    };
    let Some(entry_link) = doc.syntax.get(entry).filter(|link| {
        link.token == entry && link.span == token.span && link.sentence == token.sentence
    }) else {
        return false;
    };
    let children = |head, relation: &'static str| {
        links
            .iter()
            .filter(move |link| link.head == Some(head) && link.deprel == relation)
    };
    let surface_is = |link: &SyntaxToken, spelling: &str| {
        source
            .get(link.span.start_char..link.span.end_char)
            .is_some_and(|chars| chars.iter().copied().eq(spelling.chars()))
    };
    let form_is = |link: &SyntaxToken, lemma: &str| {
        doc.tokens[link.token].lemma.as_deref() == Some(lemma) || surface_is(link, lemma)
    };
    let has_argument = |verb, relation, spellings: &[&str], particle| {
        children(verb, relation).any(|noun| {
            doc.tokens[noun.token].upos == Some(Upos::Noun)
                && spellings.iter().any(|word| surface_is(noun, word))
                && children(noun.token, "case").any(|case| {
                    noun.span.end_char == case.span.start_char && surface_is(case, particle)
                })
        })
    };
    if links.iter().enumerate().any(|(offset, link)| {
        link.token != start + offset
            || link.sentence != token.sentence
            || link.span != doc.tokens[link.token].span
    }) {
        return false;
    }

    if has_argument(entry, "obl", FAMILY_ROLES, "として") {
        return false;
    }
    // Adoption goes with the preceding reception predicate. An active
    // reception can describe a different actor's action before going inside.
    if children(entry, "advcl").any(|prior| {
        prior.token < entry
            && form_is(prior, "迎える")
            && has_argument(prior.token, "obl", FAMILY_ROLES, "に")
            && children(prior.token, "aux").any(|suffix| form_is(suffix, "られる"))
    }) {
        return false;
    }

    let Some(result) = entry_link
        .head
        .filter(|head| entry_link.deprel == "advcl" && *head > entry && *head < end)
        .and_then(|head| doc.syntax.get(head))
    else {
        return true;
    };
    // A linked change in role or a move from employment to household
    // care leaves the physical interpretation unproven.
    let homemaking = form_is(result, "なる") && has_argument(result.token, "obl", HOME_ROLES, "に");
    let caregiving = form_is(result, "専念")
        && has_argument(result.token, "obl", &["子育て", "育児", "家事"], "に")
        && children(entry, "advcl").any(|prior| {
            prior.token < entry
                && (form_is(prior, "退職する")
                    || (form_is(prior, "辞める")
                        && has_argument(prior.token, "obj", &["仕事", "職"], "を")))
        });
    !homemaking && !caregiving
}
