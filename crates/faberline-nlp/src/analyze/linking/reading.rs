use super::{
    ClauseCandidate, ClauseScope, HYPOTHETICAL, NarrativeAnalyzer, REPORTED, Source, Span, Unit,
    UnitScope, candidate_scope, contains_any, continuation, ends_with_question_particle,
    inside_lexical_term, nominal_measure, scopes_for_units, split_units, trim_clause_end,
};
use crate::Lexicon;

/// A performed reading or checking action with an explicit, known local actor.
/// Ranges count Unicode scalar values in the original chapter.
#[derive(Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ReadingAction {
    /// The original retained clause, including any terminal punctuation.
    pub clause: Span,
    /// The written verb, including any completion built into it, such as 完 in 讀完.
    pub predicate: Span,
    pub actor: String,
    pub actor_surface: Span,
    /// The complete written object, including its classifier when present.
    pub object: Span,
    /// A separate written 了, when used as completion evidence.
    pub completion: Option<Span>,
}

const COMPLETED: &[&str] = &[
    "閱讀完",
    "阅读完",
    "查閱完",
    "查阅完",
    "翻閱完",
    "翻阅完",
    "檢查完",
    "检查完",
    "查看完",
    "核對完",
    "核对完",
    "讀完",
    "读完",
    "讀畢",
    "读毕",
    "閱畢",
    "阅毕",
];
const READING: &[&str] = &[
    "閱讀", "阅读", "查閱", "查阅", "翻閱", "翻阅", "翻看", "查看", "細讀", "细读", "檢查", "检查",
    "核對", "核对", "審閱", "审阅", "讀", "读",
];

fn known_object(text: &str, lexicon: &Lexicon<'_>) -> bool {
    let noun = nominal_measure(text).or_else(|| {
        [
            "一封", "一份", "一冊", "一册", "這封", "这封", "那封", "這份", "这份", "那份",
        ]
        .iter()
        .find_map(|classifier| text.strip_prefix(classifier))
        .map(str::trim_start)
    });
    lexicon
        .objects
        .iter()
        .any(|&(surface, _)| !surface.is_empty() && (surface == text || Some(surface) == noun))
}

fn completed_object(
    source: &Source<'_>,
    lexicon: &Lexicon<'_>,
    start: usize,
    end: usize,
    lexical_completion: bool,
) -> Option<(Span, Option<Span>)> {
    let whole = source.trim(start, end)?;
    let text = source.slice(&whole);
    if known_object(text, lexicon) {
        return lexical_completion.then_some((whole, None));
    }
    let start = source.bytes[whole.start_char];
    let end = source.bytes[whole.end_char];
    let width = '了'.len_utf8();
    for (marker, argument) in [
        text.starts_with('了')
            .then(|| (start, source.trim(start + width, end))),
        (lexical_completion && text.ends_with('了'))
            .then(|| (end - width, source.trim(start, end - width))),
    ]
    .into_iter()
    .flatten()
    {
        if let Some(argument) = argument
            && known_object(source.slice(&argument), lexicon)
        {
            return Some((argument, Some(source.span(marker, marker + width))));
        }
    }
    None
}

fn follows_report(source: &Source<'_>, unit: &Unit) -> bool {
    let before = &source.text[..source.bytes[unit.evidence.start_char]];
    unit.first_in_line
        && before
            .trim_end()
            .rsplit('\n')
            .next()
            .and_then(|line| line.strip_suffix(['：', ':']))
            .is_some_and(|lead| {
                source.has_speech(lead)
                    || lead.trim_end().ends_with(['說', '说', '稱', '称'])
                    || contains_any(lead, &["寫道", "写道", "記載", "记载"])
            })
}

pub(super) fn asserted_unit(source: &Source<'_>, unit: &Unit, scope: &UnitScope) -> bool {
    let text = source.slice(&unit.evidence);
    let sentence = source.slice(&unit.sentence);
    !(unit.quote
        || !scope.quotes.is_empty()
        || sentence.contains(['？', '?'])
        || ends_with_question_particle(sentence)
        || trim_clause_end(sentence).ends_with('吧')
        || source.has_speech(sentence)
        || contains_any(sentence, HYPOTHETICAL)
        || contains_any(sentence, REPORTED)
        || contains_any(sentence, &["就能", "便能", "才會", "才会"])
        || contains_any(
            sentence,
            &[
                "說，", "说，", "說,", "说,", "問，", "问，", "寫道", "写道", "聲稱", "声称",
            ],
        )
        || text.contains(|ch| matches!(ch, '\n'..='\r' | '\u{85}' | '\u{2028}' | '\u{2029}'))
        || follows_report(source, unit))
}

fn reading_action(
    source: &Source<'_>,
    lexicon: &Lexicon<'_>,
    unit: &Unit,
    scope: &UnitScope,
) -> Option<ReadingAction> {
    if !asserted_unit(source, unit, scope) {
        return None;
    }
    let text = source.slice(&unit.evidence);

    let lead = continuation(text).unwrap_or(text).trim_start();
    let start = source.bytes[unit.evidence.end_char] - lead.len();
    let end = start + trim_clause_end(lead).len();
    let (0, actor_word, actor) = source.first_name(start, end, lexicon)? else {
        return None;
    };
    let actor_end = start + actor_word.len();
    let action = source.text[actor_end..end].trim_start();
    let action = ["仔細", "仔细", "認真", "认真", "已經", "已经", "早已", "又"]
        .iter()
        .find_map(|modifier| action.strip_prefix(modifier))
        .unwrap_or(action)
        .trim_start();
    let (verb, lexical_completion) = COMPLETED
        .iter()
        .find(|verb| action.starts_with(**verb))
        .map(|&verb| (verb, true))
        .or_else(|| {
            READING
                .iter()
                .find(|verb| action.starts_with(**verb))
                .map(|&verb| (verb, false))
        })?;
    let verb_start = end - action.len();
    let verb_end = verb_start + verb.len();
    let predicate = source.span(verb_start, verb_end);
    let actor_surface = source.span(start, actor_end);
    let (object, completion) =
        completed_object(source, lexicon, verb_end, end, lexical_completion)?;
    if completion.is_none()
        && (unit.evidence.end_char != unit.sentence.end_char || !text.ends_with(['。', '.']))
    {
        return None;
    }
    let candidate = ClauseCandidate {
        actor: Some(actor_surface),
        theme: Some(object),
        ..ClauseCandidate::new(predicate)
    };
    if candidate_scope(source, scope, &candidate).0 != ClauseScope::Narrated
        || inside_lexical_term(source, lexicon, unit.evidence, predicate)
    {
        return None;
    }
    Some(ReadingAction {
        clause: unit.evidence,
        predicate,
        actor: actor.to_owned(),
        actor_surface,
        object,
        completion,
    })
}

pub(in crate::analyze) fn discover(
    analyzer: &NarrativeAnalyzer<'_>,
    chapter_text: &str,
    title: &str,
) -> Vec<ReadingAction> {
    let mut source = Source::new(chapter_text);
    let lexicon = &analyzer.lexicon;
    let units = split_units(&mut source, title, lexicon, Some(analyzer));
    let scopes = scopes_for_units(&source, &units, true);
    units
        .iter()
        .zip(&scopes)
        .filter_map(|(unit, scope)| reading_action(&source, lexicon, unit, scope))
        .collect()
}
