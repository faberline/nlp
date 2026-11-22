//! Recover grammar that a caller glossary collapsed during source tokenization.

use super::{Atom, Document, SpanLockTable, Token, calendar::glossary, handover};

pub(crate) enum GlossaryRenderLocks {
    Handover(handover::HandoverRenderLocks),
    Time(glossary::TimeRenderLocks),
}

impl GlossaryRenderLocks {
    pub(crate) fn table(&self) -> &SpanLockTable {
        match self {
            Self::Handover(plan) => &plan.table,
            Self::Time(plan) => &plan.table,
        }
    }

    pub(super) fn apply(&self, atoms: &mut Vec<Atom>) {
        match self {
            Self::Handover(plan) => plan.apply(atoms),
            Self::Time(plan) => plan.apply(atoms),
        }
    }
}

pub(crate) fn may_restore_glossary_syntax(
    text: &str,
    chars: &[char],
    locks: &SpanLockTable,
) -> bool {
    !text.contains(['「', '」', '『', '』'])
        && (handover::has_handover_glossary(text, locks) || glossary::may_cover_time(chars, locks))
}

pub(crate) fn glossary_render_locks(
    sentence: usize,
    source: &Document,
    tokens: &[&Token],
    chars: &[char],
    locks: &SpanLockTable,
) -> Option<GlossaryRenderLocks> {
    handover::handover_render_locks(sentence, source, tokens, chars, locks)
        .map(GlossaryRenderLocks::Handover)
        .or_else(|| {
            glossary::time_render_locks(sentence, source, tokens, chars, locks)
                .map(GlossaryRenderLocks::Time)
        })
}
