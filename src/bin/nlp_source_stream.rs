//! Experimental source-order extraction. No global story order or identity is established.
#[path = "nlp_source_stream/lexicon.rs"]
mod lexicon;
#[path = "nlp_source_stream/sparse.rs"]
mod sparse;

use aho_corasick::AhoCorasick;
use anyhow::{Context, Result, ensure};
use clap::Parser;
use faberline_nlp::automatic_spelling::{self, AlternativeStatus, AutoPolicy, ScanStatus};
use faberline_nlp::safe_written_observations::{
    self, QuotationForm, QuoteBoundaryAlternative, QuoteScanDiagnostic, WrittenQuote,
};
use faberline_nlp::source_cross_chapter::{
    CrossChapterCandidates, CrossChapterPolicy, SourcePosition,
};
use faberline_nlp::written_cessation_evidence::SourceRange as CessationRange;
use faberline_nlp::written_speech_alias_evidence::SourceRange as WrittenRange;
use faberline_nlp::{
    ClauseOrderCueKind, Lexicon, MentionLabelBasis, MentionPredicateBasis, NarrativeAnalyzer, Span,
    source_timeline_stream::{Chapter, MentionRef, SourceTimelineStream},
    text::Locale,
    timeline::LocatedSpan,
};
use faberline_nlp::{scoped_transfer, scoped_written_observations, source_scope::SourceScope};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{self, BufWriter, Read, Write},
    path::{Path, PathBuf},
};

const MAX_MANIFEST_BYTES: u64 = 16 * 1024 * 1024;
const MAX_CHAPTER_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Parser)]
#[command(
    about = "Experimental Chinese (Taiwan) source observations; use trusted, unchanged input directories"
)]
struct Args {
    #[arg(long)]
    manifest: PathBuf,
    #[arg(
        long,
        required_unless_present = "all",
        conflicts_with = "all",
        value_delimiter = ','
    )]
    chapters: Vec<u32>,
    #[arg(long)]
    all: bool,
    #[arg(long)]
    work_lexicon: Option<PathBuf>,
    /// Include unverified written quotations and related cues, chapter by chapter.
    #[arg(long)]
    written_quotes: bool,
    /// Include unverified written time references, with no event association.
    #[arg(long)]
    written_time_cues: bool,
    /// Include unverified written give/receive cues.
    #[arg(long)]
    written_transfer_cues: bool,
    /// Include unverified written transfer-role fragments.
    #[arg(long)]
    written_transfer_role_candidates: bool,
    /// Include alternative, unverified written role boundaries.
    #[arg(long)]
    written_transfer_role_alternatives: bool,
    /// Include parser-selected, unverified written predicate ranges.
    #[arg(long)]
    written_predicate_observations: bool,
    /// Emit bounded, unverified cross-chapter written-form candidates.
    #[arg(long)]
    cross_chapter_candidates: bool,
    /// Restrict written quotation and transfer observations to admitted source segments.
    #[arg(long)]
    scoped_written_observations: bool,
    /// Emit bounded spelling candidates, not corrections.
    #[arg(long)]
    spelling_review: bool,
    #[arg(long, requires_all=["spelling_review","spelling_end_char"])]
    spelling_start_char: Option<usize>,
    #[arg(long, requires_all=["spelling_review","spelling_start_char"])]
    spelling_end_char: Option<usize>,
    #[arg(long, requires = "spelling_review")]
    spelling_work_budget: Option<usize>,
    #[arg(long, requires = "spelling_review")]
    spelling_max_alternatives: Option<usize>,
}

#[derive(Deserialize)]
struct Manifest {
    count: usize,
    chapters: Vec<Entry>,
}
#[derive(Deserialize)]
struct Entry {
    order: u32,
    title: String,
    file: String,
    status: String,
}

struct Meta {
    chapter: u32,
    sha256: String,
    chars: usize,
    newlines: Vec<usize>,
}

// Only source identifiers and the bounded set of written-name positions are
// retained. Previous chapter content remains the caller's responsibility.
struct CrossMeta {
    sha256: String,
    names: Vec<SourcePosition>,
}

fn validate_cross_position(
    position: &SourcePosition,
    chapter: u32,
    source: &str,
    meta: &Meta,
    offsets: &sparse::SparseOffsets,
) -> Result<()> {
    ensure!(
        position.chapter == chapter,
        "cross-chapter source ordinal differs"
    );
    let span = position.scalars;
    meta.evidence(span)?;
    ensure!(
        position.bytes.start < position.bytes.end,
        "cross-chapter byte range is empty"
    );
    let written = source
        .get(position.bytes.clone())
        .context("cross-chapter byte range is invalid")?;
    let start = offsets.byte_offset(source, span.start_char);
    let end = offsets.byte_offset(source, span.end_char);
    ensure!(
        position.bytes.start == start
            && position.bytes.end == end
            && written == &source[start..end],
        "cross-chapter byte and scalar positions differ"
    );
    Ok(())
}

fn cross_source_value(position: &SourcePosition, sha256: &str) -> Result<Value> {
    let mut value = serde_json::to_value(position)?;
    value
        .as_object_mut()
        .context("cross-chapter source is not an object")?
        .insert("sha256".into(), json!(sha256));
    Ok(value)
}
impl Meta {
    fn new(chapter: u32, bytes: &[u8], source: &str) -> Self {
        Self {
            chapter,
            sha256: format!("{:x}", Sha256::digest(bytes)),
            chars: source.chars().count(),
            newlines: source
                .chars()
                .enumerate()
                .filter_map(|(i, c)| (c == '\n').then_some(i))
                .collect(),
        }
    }
    fn evidence(&self, span: Span) -> Result<Value> {
        ensure!(
            span.start_char < span.end_char && span.end_char <= self.chars,
            "invalid source evidence"
        );
        Ok(
            json!({"chapter":self.chapter,"sha256":self.sha256,"start_char":span.start_char,
            "end_char":span.end_char,"line":self.newlines.partition_point(|&i| i < span.start_char)+1}),
        )
    }
}

fn read_capped(path: &Path, limit: u64) -> Result<Vec<u8>> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    ensure!(file.metadata()?.is_file(), "input must be a regular file");
    let mut bytes = Vec::new();
    file.take(limit + 1).read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= limit,
        "input exceeds configured byte limit"
    );
    Ok(bytes)
}

fn record(out: &mut impl Write, value: Value) -> Result<()> {
    serde_json::to_writer(&mut *out, &value)?;
    writeln!(out)?;
    Ok(())
}

struct DigestWriter<'a, W: Write> {
    inner: &'a mut W,
    digest: Sha256,
    lines: usize,
}
impl<W: Write> Write for DigestWriter<'_, W> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let written = self.inner.write(bytes)?;
        self.digest.update(&bytes[..written]);
        let new_lines = bytes[..written]
            .iter()
            .filter(|&&byte| byte == b'\n')
            .count();
        self.lines = self
            .lines
            .checked_add(new_lines)
            .ok_or_else(|| io::Error::other("stream record count overflow"))?;
        Ok(written)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

fn meta_for<'a>(chapter: u32, current: &'a Meta, previous: Option<&'a Meta>) -> Result<&'a Meta> {
    if current.chapter == chapter {
        return Ok(current);
    }
    previous
        .filter(|meta| meta.chapter == chapter)
        .context("relation references unavailable chapter")
}

fn identifier(
    reference: MentionRef,
    anchor: LocatedSpan,
    current: &Meta,
    previous: Option<&Meta>,
) -> Result<String> {
    ensure!(
        reference.chapter == anchor.chapter,
        "mention anchor chapter mismatch"
    );
    let meta = meta_for(reference.chapter, current, previous)?;
    meta.evidence(anchor.span)?;
    Ok(format!(
        "stream-v1-{}-{}-{}-{}-{}",
        meta.chapter,
        meta.sha256,
        anchor.span.start_char,
        anchor.span.end_char,
        reference.event_index
    ))
}

fn evidence(location: LocatedSpan, current: &Meta, previous: Option<&Meta>) -> Result<Value> {
    meta_for(location.chapter, current, previous)?.evidence(location.span)
}

fn written_evidence(
    source: &str,
    meta: &Meta,
    offsets: &sparse::SparseOffsets,
    range: WrittenRange,
) -> Result<Value> {
    let proof = meta.evidence(range.scalars)?;
    let bytes = source
        .get(range.byte_start..range.byte_end)
        .context("invalid written byte range")?;
    let start = offsets.byte_offset(source, range.scalars.start_char);
    let end = offsets.byte_offset(source, range.scalars.end_char);
    ensure!(
        start == range.byte_start && end == range.byte_end && bytes == &source[start..end],
        "written byte and character ranges differ"
    );
    Ok(proof)
}

fn cessation_evidence(
    source: &str,
    meta: &Meta,
    offsets: &sparse::SparseOffsets,
    range: CessationRange,
) -> Result<Value> {
    written_evidence(
        source,
        meta,
        offsets,
        WrittenRange {
            scalars: range.scalars,
            byte_start: range.byte_start,
            byte_end: range.byte_end,
        },
    )
}

fn quote_value(
    source: &str,
    meta: &Meta,
    offsets: &sparse::SparseOffsets,
    quote: &WrittenQuote,
) -> Result<Value> {
    let nested: Vec<_> = quote
        .nested_quotations
        .iter()
        .map(|q| quote_value(source, meta, offsets, q))
        .collect::<Result<_>>()?;
    Ok(
        json!({"evidence":written_evidence(source,meta,offsets,quote.source)?,
        "inner":quote.inner.map(|r| written_evidence(source,meta,offsets,r)).transpose()?,
        "form":quote.form,"delimiter_status":quote.delimiter_status,"delimiter_basis":quote.delimiter_basis,
        "nested_quotations":nested,"speaker_identity":"unknown","speaker_role":"unknown",
        "occurrence":"unknown","narrative_scope":"unknown"}),
    )
}

fn quote_forms(
    out: &mut impl Write,
    source: &str,
    meta: &Meta,
    offsets: &sparse::SparseOffsets,
    quotes: &[WrittenQuote],
    alternatives: &[QuoteBoundaryAlternative],
    diagnostics: &[QuoteScanDiagnostic],
) -> Result<()> {
    for (index, quote) in quotes.iter().enumerate() {
        record(
            out,
            json!({"type":"written_quote","chapter":meta.chapter,"quote_index":index,
            "quote":quote_value(source,meta,offsets,quote)?}),
        )?;
    }
    for alternative in alternatives {
        record(
            out,
            json!({"type":"written_quote_boundary_alternative",
            "evidence":written_evidence(source,meta,offsets,alternative.source)?,
            "delimiter_status":alternative.delimiter_status,"reason":alternative.reason}),
        )?;
    }
    for diagnostic in diagnostics {
        record(
            out,
            json!({"type":"written_quote_diagnostic",
            "evidence":written_evidence(source,meta,offsets,diagnostic.source)?,
            "status":diagnostic.status,"reason":diagnostic.reason,"delimiter_status":diagnostic.delimiter_status}),
        )?;
    }
    Ok(())
}

fn write_time_cues(
    out: &mut impl Write,
    source: &str,
    title: &str,
    meta: &Meta,
    analyzer: &NarrativeAnalyzer<'_>,
) -> Result<()> {
    let (clauses, cues) = analyzer.link_clause_candidates_with_cues(source, title, &[]);
    ensure!(clauses.clauses.is_empty(), "unexpected time-cue endpoints");
    let mut spans = Vec::new();
    for cue in cues
        .into_iter()
        .filter(|cue| cue.kind == ClauseOrderCueKind::Reference)
    {
        ensure!(
            cue.before.is_none() && cue.after.is_none(),
            "unexpected time-cue association"
        );
        meta.evidence(cue.cue)?;
        spans.push(cue.cue);
    }
    spans.sort_unstable_by_key(|span| (span.start_char, span.end_char));
    spans.dedup();
    let offsets = sparse::SparseOffsets::new(source);
    for span in spans {
        let mut proof = meta.evidence(span)?;
        let start = offsets.byte_offset(source, span.start_char);
        let end = offsets.byte_offset(source, span.end_char);
        let text = source
            .get(start..end)
            .context("invalid time-cue byte range")?;
        proof["start_byte"] = json!(start);
        proof["end_byte"] = json!(end);
        record(
            out,
            json!({"type":"written_time_cue","chapter":meta.chapter,"source":proof,
            "written_text":text,"association_status":"unknown","identity_status":"unknown",
            "occurrence_status":"unknown","narrative_scope_status":"unknown","coverage":"partial_heuristic"}),
        )?;
    }
    Ok(())
}

fn write_transfer_cues(
    out: &mut impl Write,
    source: &str,
    meta: &Meta,
    analyzer: &NarrativeAnalyzer<'_>,
) -> Result<()> {
    let offsets = sparse::SparseOffsets::new(source);
    let mut previous_end = 0;
    for cue in analyzer.discover_written_transfer_cues(source) {
        ensure!(
            previous_end <= cue.cue.start_char,
            "transfer cues overlap or are out of order"
        );
        let mut proof = meta.evidence(cue.cue)?;
        let start = offsets.byte_offset(source, cue.cue.start_char);
        let end = offsets.byte_offset(source, cue.cue.end_char);
        let text = source
            .get(start..end)
            .context("invalid transfer-cue byte range")?;
        proof["start_byte"] = json!(start);
        proof["end_byte"] = json!(end);
        record(
            out,
            json!({"type":"written_transfer_cue","chapter":meta.chapter,"source":proof,
            "written_text":text,"written_direction":cue.direction,"direction_status":"written_form_only",
            "association_status":"unknown","identity_status":"unknown","role_status":"unknown",
            "occurrence_status":"unknown","narrative_scope_status":"unknown","coverage":"partial_heuristic"}),
        )?;
        previous_end = cue.cue.end_char;
    }
    Ok(())
}

fn write_transfer_role_candidates(
    out: &mut impl Write,
    source: &str,
    meta: &Meta,
    analyzer: &NarrativeAnalyzer<'_>,
) -> Result<()> {
    let offsets = sparse::SparseOffsets::new(source);
    let evidence = |span: Span| -> Result<Value> {
        let mut proof = meta.evidence(span)?;
        let start = offsets.byte_offset(source, span.start_char);
        let end = offsets.byte_offset(source, span.end_char);
        let text = source
            .get(start..end)
            .context("invalid transfer-role byte range")?;
        proof["start_byte"] = json!(start);
        proof["end_byte"] = json!(end);
        proof["written_text"] = json!(text);
        Ok(proof)
    };
    for candidate in analyzer.discover_written_transfer_role_candidates(source) {
        record(
            out,
            json!({"type":"written_transfer_role_candidate","chapter":meta.chapter,
            "cue":evidence(candidate.cue)?,"construction":evidence(candidate.construction)?,
            "giver_candidate":candidate.giver.map(&evidence).transpose()?,
            "item_candidate":candidate.item.map(&evidence).transpose()?,
            "recipient_candidate":candidate.recipient.map(&evidence).transpose()?,
            "written_direction":candidate.direction,"direction_status":"written_form_only",
            "association_status":"unknown","identity_status":"unknown","role_status":"unknown",
            "occurrence_status":"unknown","narrative_scope_status":"unknown","coverage":"partial_heuristic"}),
        )?;
    }
    Ok(())
}

fn write_transfer_role_alternatives(
    out: &mut impl Write,
    source: &str,
    meta: &Meta,
    analyzer: &NarrativeAnalyzer<'_>,
) -> Result<()> {
    let offsets = sparse::SparseOffsets::new(source);
    let evidence = |span: Span| -> Result<Value> {
        let mut proof = meta.evidence(span)?;
        let start = offsets.byte_offset(source, span.start_char);
        let end = offsets.byte_offset(source, span.end_char);
        let text = source
            .get(start..end)
            .context("invalid alternative byte range")?;
        proof["start_byte"] = json!(start);
        proof["end_byte"] = json!(end);
        proof["written_text"] = json!(text);
        Ok(proof)
    };
    for observation in analyzer.discover_written_transfer_role_alternatives(source) {
        let giver = observation
            .giver_alternatives
            .into_iter()
            .map(&evidence)
            .collect::<Result<Vec<_>>>()?;
        let recipient = observation
            .recipient_alternatives
            .into_iter()
            .map(&evidence)
            .collect::<Result<Vec<_>>>()?;
        record(
            out,
            json!({"type":"written_transfer_role_alternatives","chapter":meta.chapter,
            "cue":evidence(observation.cue)?,"construction":evidence(observation.construction)?,
            "giver_alternatives":giver,"recipient_alternatives":recipient,
            "boundary_status":"unknown","interpretation_status":"unknown","association_status":"unknown",
            "identity_status":"unknown","role_status":"unknown","occurrence_status":"unknown",
            "narrative_scope_status":"unknown","coverage":"partial_heuristic"}),
        )?;
    }
    Ok(())
}

fn write_quotes(
    out: &mut impl Write,
    source: &str,
    title: &str,
    meta: &Meta,
    analyzer: &NarrativeAnalyzer<'_>,
    vocabulary: &Lexicon<'_>,
) -> Result<bool> {
    let offsets = sparse::SparseOffsets::new(source);
    let degraded = match safe_written_observations::inspect_with_analyzer(
        source, title, analyzer, vocabulary,
    ) {
        Ok(report) => {
            record(
                out,
                json!({"type":"written_quote_summary","chapter":meta.chapter,"sha256":meta.sha256,
                "status":"inspected","coverage":report.quote_coverage,"speech_coverage":report.speech_coverage,
                "cessation_coverage":report.cessation_coverage,"written_claim_coverage":report.written_claim_coverage,
                "identity_status":"unknown","occurrence_status":"unknown"}),
            )?;
            quote_forms(
                out,
                source,
                meta,
                &offsets,
                &report.quotations,
                &report.alternative_quote_boundaries,
                &report.quote_scan_diagnostics,
            )?;
            for speech in &report.speech {
                record(
                    out,
                    json!({"type":"written_speech_observation","chapter":meta.chapter,
                    "kind":speech.written.kind,"cue":written_evidence(source,meta,&offsets,speech.written.cue)?,
                    "payload":written_evidence(source,meta,&offsets,speech.written.payload)?,
                    "written_speaker":speech.written.written_speaker.map(|r| written_evidence(source,meta,&offsets,r)).transpose()?,
                    "caller_labels":speech.written.caller_labels,"attribution_basis":speech.written.attribution_basis,
                    "potential_quote_index":speech.potential_quote_index,"identity":"unknown","role":"unknown",
                    "occurrence":"unknown","narrative_scope":"unknown","quote_attribution":"unknown"}),
                )?;
            }
            for cue in &report.written_negated_cues {
                record(
                    out,
                    json!({"type":"written_negated_speech_cue","chapter":meta.chapter,
                    "cue":written_evidence(source,meta,&offsets,cue.cue)?,"negation":written_evidence(source,meta,&offsets,cue.negation)?,
                    "written_subject":cue.written_subject.map(|r| written_evidence(source,meta,&offsets,r)).transpose()?,
                    "caller_labels":cue.caller_labels,"potential_quote_index":cue.potential_quote_index,"basis":cue.basis,
                    "identity":"unknown","role":"unknown","occurrence":"unknown","narrative_scope":"unknown","quote_attribution":"unknown"}),
                )?;
            }
            for (index, cessation) in report.cessation.iter().enumerate() {
                record(
                    out,
                    json!({"type":"written_cessation","chapter":meta.chapter,"cessation_index":index,
                    "preceding_quote_index":cessation.preceding_quote_index,
                    "tag":cessation_evidence(source,meta,&offsets,cessation.written.tag)?,
                    "written_subject":cessation_evidence(source,meta,&offsets,cessation.written.subject_surface)?,
                    "cue":cessation_evidence(source,meta,&offsets,cessation.written.cue)?,
                    "caller_labels":cessation.written.caller_labels,"identity":"unknown","role":"unknown",
                    "occurrence":"unknown","narrative_scope":"unknown","quote_attribution":"unknown"}),
                )?;
            }
            for claim in &report.written_claims {
                record(
                    out,
                    json!({"type":"written_claim","chapter":meta.chapter,
                    "evidence":written_evidence(source,meta,&offsets,claim.source)?,"potential_quote_index":claim.potential_quote_index,
                    "potential_cessation_index":claim.potential_cessation_index,"selection_basis":claim.selection_basis,
                    "identity":"unknown","role":"unknown","occurrence":"unknown","narrative_scope":"unknown","quote_attribution":"unknown"}),
                )?;
            }
            for diagnostic in &report.cessation_diagnostics {
                record(
                    out,
                    json!({"type":"written_cessation_diagnostic","chapter":meta.chapter,
                    "preceding_quote_index":diagnostic.preceding_quote_index,
                    "evidence":written_evidence(source,meta,&offsets,diagnostic.candidate_source)?,
                    "status":diagnostic.status,"reason":diagnostic.reason}),
                )?;
            }
            !report.quote_scan_diagnostics.is_empty() || !report.cessation_diagnostics.is_empty()
        }
        Err(error) => {
            let forms = safe_written_observations::inspect_quote_forms(source);
            record(
                out,
                json!({"type":"written_quote_summary","chapter":meta.chapter,"sha256":meta.sha256,
                "status":"partial","coverage":forms.quote_coverage,"speech_coverage":"unexamined",
                "cessation_coverage":"unexamined","written_claim_coverage":"unexamined","reason":format!("{error:?}"),
                "identity_status":"unknown","occurrence_status":"unknown"}),
            )?;
            quote_forms(
                out,
                source,
                meta,
                &offsets,
                &forms.quotations,
                &forms.alternative_quote_boundaries,
                &forms.quote_scan_diagnostics,
            )?;
            true
        }
    };
    Ok(degraded)
}

fn quote_is_degraded(quote: &WrittenQuote) -> bool {
    quote.form != QuotationForm::Complete || quote.nested_quotations.iter().any(quote_is_degraded)
}

// The scope owns no source text. Segment ranges and every nested observation
// refer to the original chapter, never to a concatenation of visible pieces.
fn write_scoped_observations(
    out: &mut impl Write,
    source: &str,
    title: &str,
    meta: &Meta,
    analyzer: &NarrativeAnalyzer<'_>,
    vocabulary: &Lexicon<'_>,
    args: &Args,
) -> Result<bool> {
    let scope = SourceScope::new(source, title, vocabulary);
    let offsets = sparse::SparseOffsets::new(source);
    let mut segments = Vec::with_capacity(scope.segments().len());
    for (segment_id, segment) in scope.segments().iter().enumerate() {
        // Validate both offset systems against the exact full chapter.
        written_evidence(source, meta, &offsets, segment.source)?;
        segments.push(json!({"segment_id":segment_id,"source":segment.source}));
    }
    // The analyzer and scope originate from the same vocabulary in run().
    // The quote adapter also verifies this, including when there are no segments.
    let quotes = if args.written_quotes {
        Some(
            scoped_written_observations::inspect_with_analyzer(source, title, analyzer, vocabulary)
                .map_err(|e| anyhow::anyhow!("scoped written observations: {e:?}"))?,
        )
    } else {
        None
    };
    let transfer = if args.written_transfer_cues
        || args.written_transfer_role_candidates
        || args.written_transfer_role_alternatives
    {
        Some(scoped_transfer::inspect(&scope, analyzer)?)
    } else {
        None
    };
    if let Some(report) = &quotes {
        ensure!(
            report.segments.len() == scope.segments().len(),
            "scoped quote segment count differs"
        );
        for (id, segment) in report.segments.iter().enumerate() {
            ensure!(
                segment.segment_id == id && segment.source == scope.segments()[id].source,
                "scoped quote segment differs from admitted source"
            );
        }
    }
    if let Some(report) = &transfer {
        ensure!(
            report.segments.len() == scope.segments().len(),
            "scoped transfer segment count differs"
        );
        for (id, segment) in report.segments.iter().enumerate() {
            ensure!(
                *segment == scope.segments()[id].source,
                "scoped transfer segment differs from admitted source"
            );
        }
    }
    let cue_count = if args.written_transfer_cues {
        transfer.as_ref().map_or(0, |r| r.cues.len())
    } else {
        0
    };
    let role_count = if args.written_transfer_role_candidates {
        transfer.as_ref().map_or(0, |r| r.roles.len())
    } else {
        0
    };
    let alternative_count = if args.written_transfer_role_alternatives {
        transfer.as_ref().map_or(0, |r| r.alternatives.len())
    } else {
        0
    };
    record(
        out,
        json!({"type":"written_scoped_summary","chapter":meta.chapter,"sha256":meta.sha256,
        "segments":segments,"scope_coverage":"partial_or_unknown","cross_segment_associations":"not_examined",
        "source_binding_status":"caller_must_bind_exact_input","identity_status":"unknown",
        "quote_segments":quotes.as_ref().map_or(0,|r| r.segments.len()),"transfer_cues":cue_count,
        "transfer_role_candidates":role_count,"transfer_role_alternatives":alternative_count}),
    )?;
    let mut degraded = false;
    if let Some(report) = quotes {
        for segment in report.segments {
            degraded |= segment.report.quotations.iter().any(quote_is_degraded)
                || !segment.report.quote_scan_diagnostics.is_empty()
                || !segment.report.cessation_diagnostics.is_empty();
            record(
                out,
                json!({"type":"written_scoped_quote_segment","chapter":meta.chapter,"sha256":meta.sha256,
                "segment_id":segment.segment_id,"source":segment.source,"index_scope":segment.index_scope,
                "boundary_scope":segment.boundary_scope,"report":segment.report}),
            )?;
        }
    }
    if let Some(report) = transfer {
        if args.written_transfer_cues {
            for item in report.cues {
                record(
                    out,
                    json!({"type":"written_scoped_transfer_cue",
                "chapter":meta.chapter,"sha256":meta.sha256,"observation":item}),
                )?;
            }
        }
        if args.written_transfer_role_candidates {
            for item in report.roles {
                record(
                    out,
                    json!({"type":"written_scoped_transfer_role_candidate",
                "chapter":meta.chapter,"sha256":meta.sha256,"direction_status":"written_form_only",
                "observation":item}),
                )?;
            }
        }
        if args.written_transfer_role_alternatives {
            for item in report.alternatives {
                record(
                    out,
                    json!({"type":"written_scoped_transfer_role_alternatives",
                "chapter":meta.chapter,"sha256":meta.sha256,"observation":item}),
                )?;
            }
        }
    }
    Ok(degraded)
}

fn name_matcher(vocabulary: &Lexicon<'_>) -> Result<Option<AhoCorasick>> {
    let mut names = Vec::new();
    let mut bytes = 0usize;
    for (surface, _) in vocabulary.actors {
        if surface.is_empty() {
            continue;
        }
        bytes = bytes.saturating_add(surface.len());
        names.push(*surface);
        if names.len() > 65_536 || bytes > 1_048_576 {
            return Ok(None);
        }
    }
    Ok(Some(
        AhoCorasick::new(names).context("build spelling name protection")?,
    ))
}

fn write_spelling(
    out: &mut impl Write,
    source: &str,
    meta: &Meta,
    matcher: Option<&AhoCorasick>,
    args: &Args,
) -> Result<bool> {
    let start_char = args.spelling_start_char.unwrap_or(0);
    let end_char = args.spelling_end_char.unwrap_or(meta.chars);
    ensure!(
        start_char <= end_char && end_char <= meta.chars,
        "invalid spelling range"
    );
    if args.spelling_start_char.is_some() {
        ensure!(start_char < end_char, "empty spelling range");
    }
    let offsets = sparse::SparseOffsets::new(source);
    let start_byte = offsets.byte_offset(source, start_char);
    let end_byte = offsets.byte_offset(source, end_char);
    let segment = &source[start_byte..end_byte];
    let scan_range = if start_char == end_char {
        Value::Null
    } else {
        meta.evidence(Span {
            start_char,
            end_char,
        })?
    };
    let Some(matcher) = matcher else {
        record(
            out,
            json!({"type":"spelling_review_summary","chapter":meta.chapter,"sha256":meta.sha256,
            "scan_status":"partial_name_protection_limit","coverage":"unknown","grammar":"unknown","meaning":"unknown",
            "names_protected":false,"scan_range":scan_range,"recheck_from_char":start_char,
            "resume_status":"unavailable_caller_segments_input"}),
        )?;
        return Ok(true);
    };
    let protected: Vec<_> = matcher
        .find_overlapping_iter(source)
        .filter(|hit| hit.start() < end_byte && hit.end() > start_byte)
        .take(100_001)
        .map(|hit| hit.start().max(start_byte) - start_byte..hit.end().min(end_byte) - start_byte)
        .collect();
    if protected.len() > 100_000 {
        record(
            out,
            json!({"type":"spelling_review_summary","chapter":meta.chapter,"sha256":meta.sha256,
            "scan_status":"partial_name_protection_limit","coverage":"unknown","grammar":"unknown","meaning":"unknown",
            "names_protected":false,"scan_range":scan_range,"recheck_from_char":start_char,
            "resume_status":"unavailable_caller_segments_input"}),
        )?;
        return Ok(true);
    }
    let mut policy = AutoPolicy {
        protected_bytes: protected,
        ..AutoPolicy::default()
    };
    if let Some(v) = args.spelling_work_budget {
        policy.work_budget = v;
    }
    if let Some(v) = args.spelling_max_alternatives {
        policy.max_alternatives_per_window = v;
    }
    let report = automatic_spelling::scan(segment, &policy)
        .map_err(|e| anyhow::anyhow!("spelling scan failed: {e:?}"))?;
    let status = match report.status {
        ScanStatus::FinishedConfiguredWindows => "finished_configured_windows",
        ScanStatus::PartialBudget => "partial_budget",
        ScanStatus::PartialReportedWindowLimit => "partial_report_limit",
    };
    let degraded = report.status != ScanStatus::FinishedConfiguredWindows
        || report.recheck_from_scalar.is_some();
    record(
        out,
        json!({"type":"spelling_review_summary","chapter":meta.chapter,"sha256":meta.sha256,
        "scan_status":status,"scan_range":scan_range,"coverage":"unknown","grammar":"unknown","meaning":"unknown",
        "work_used":report.work_used,"dictionary_probes":report.dictionary_probes,"inspected_windows":report.inspected_windows,
        "partial_windows":report.partial_windows,"protected_windows":report.protected_windows,
        "dictionary_scope":report.dictionary_scope,"operation_scope":report.operation_scope,
        "work_budget":policy.work_budget,"max_alternatives_per_window":policy.max_alternatives_per_window,
        "max_reported_windows":policy.max_reported_windows,"resume_status":report.resume_status,
        "recheck_from_char":report.recheck_from_scalar.map(|offset| start_char+offset),
        "names_protected":true,"name_protection_scope":"supplied_actor_surfaces_only"}),
    )?;
    for candidate in report.candidates {
        let span = Span {
            start_char: start_char + candidate.source.scalars.start,
            end_char: start_char + candidate.source.scalars.end,
        };
        ensure!(
            segment.get(candidate.source.bytes.clone()) == Some(candidate.observed.as_str()),
            "spelling bytes differ from source"
        );
        ensure!(
            segment[..candidate.source.bytes.start].chars().count()
                == candidate.source.scalars.start
                && span.start_char + candidate.observed.chars().count() == span.end_char,
            "spelling character range differs from source"
        );
        let alternatives = match candidate.alternatives {
            AlternativeStatus::FinishedBasicCjkAlphabet => "finished_basic_cjk_alphabet",
            AlternativeStatus::PartialAlternativeLimit => "partial_alternative_limit",
            AlternativeStatus::PartialBudget => "partial_budget",
        };
        let suggestions: Vec<_> = candidate
            .suggestions
            .into_iter()
            .map(|s| json!({"text":s.spelling,"basis":s.basis,"correctness":"unknown"}))
            .collect();
        record(
            out,
            json!({"type":"spelling_candidate","observed":candidate.observed,"evidence":meta.evidence(span)?,
            "alternatives_status":alternatives,"correctness":"unknown","identity":"unknown","suggestions":suggestions}),
        )?;
    }
    Ok(degraded)
}

fn run(args: Args, out: &mut impl Write) -> Result<()> {
    if let Some(v) = args.spelling_work_budget {
        ensure!(
            (1..=3_000_000).contains(&v),
            "spelling work budget must be between 1 and 3000000"
        );
    }
    if let Some(v) = args.spelling_max_alternatives {
        ensure!(
            (1..=256).contains(&v),
            "spelling alternatives limit must be between 1 and 256"
        );
    }
    let manifest_path = args.manifest.canonicalize().context("open manifest")?;
    let root = manifest_path.parent().context("manifest has no parent")?;
    let manifest_bytes = read_capped(&manifest_path, MAX_MANIFEST_BYTES)?;
    let manifest_hash = format!("{:x}", Sha256::digest(&manifest_bytes));
    let manifest: Manifest = serde_json::from_slice(&manifest_bytes)?;
    ensure!(
        manifest.count == manifest.chapters.len(),
        "manifest count mismatch"
    );
    let total = manifest.count;
    let mut entries = BTreeMap::new();
    for entry in manifest.chapters {
        ensure!(
            entry.order > 0 && entries.insert(entry.order, entry).is_none(),
            "invalid or duplicate ordinal"
        );
    }
    let selected: Vec<Entry> = if args.all {
        entries.into_values().collect()
    } else {
        let mut picked = BTreeMap::new();
        for ordinal in args.chapters.iter().copied() {
            let item = entries
                .remove(&ordinal)
                .context("unknown or repeated chapter")?;
            picked.insert(ordinal, item);
        }
        picked.into_values().collect()
    };
    let selected_count = selected.len();
    if args.spelling_start_char.is_some() {
        ensure!(
            selected_count == 1,
            "a spelling range requires exactly one selected chapter"
        );
    }
    let work_bytes = args
        .work_lexicon
        .as_deref()
        .map(|path| read_capped(path, MAX_MANIFEST_BYTES))
        .transpose()?;
    let vocabulary_hash = work_bytes
        .as_deref()
        .map(|b| format!("{:x}", Sha256::digest(b)));
    let vocabulary_origin = if work_bytes.is_some() {
        "work_lexicon_file"
    } else {
        "empty_default"
    };
    let work = work_bytes
        .as_deref()
        .map(lexicon::WorkFile::parse)
        .transpose()?;
    let work_refs = work.as_ref().map(lexicon::WorkFile::refs);
    let vocabulary = work_refs
        .as_ref()
        .map_or_else(Lexicon::default, |refs| refs.lexicon());
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &vocabulary)?;
    let spelling_names = if args.spelling_review {
        name_matcher(&vocabulary)?
    } else {
        None
    };
    let cross_policy = CrossChapterPolicy {
        history_distance: 2,
        max_chapter_bytes: MAX_CHAPTER_BYTES as usize,
        max_terms: 65_536,
        max_term_bytes: 1024 * 1024,
        max_names_per_chapter: 256,
        max_pronouns_per_chapter: 256,
        max_pairs_per_pronoun: 32,
        max_pairs_per_chapter: 4096,
    };
    let mut cross = if args.cross_chapter_candidates {
        Some(CrossChapterCandidates::new(
            Locale::ChineseTaiwan,
            vocabulary,
            cross_policy,
        )?)
    } else {
        None
    };
    let mut cross_history = BTreeMap::<u32, CrossMeta>::new();
    let (mut cross_summaries, mut cross_candidates) = (0usize, 0usize);
    let mut body = DigestWriter {
        inner: &mut *out,
        digest: Sha256::new(),
        lines: 0,
    };
    let mut header = json!({"type":"stream_start","schema":if args.scoped_written_observations {
            "avl-source-stream-v3-scoped-written" } else if args.cross_chapter_candidates {
            "avl-source-stream-v2-cross-candidates" } else { "avl-source-stream-v1" },"manifest_sha256":manifest_hash,
        "total_chapters":total,"selected_chapters":selected_count,"completion":"pending",
        "extraction_coverage":"unknown","identity_status":"unknown","occurrence_status":"unknown",
        "story_order":"not_computed","global_cycle_check":"not_computed",
        "quote_coverage":if args.written_quotes {"partial_unverified_delimiter_candidates"} else {"not_computed"},
        "time_cue_coverage":if args.written_time_cues {"partial_heuristic"} else {"not_computed"},
        "transfer_cue_coverage":if args.written_transfer_cues {"partial_heuristic"} else {"not_computed"},
        "transfer_role_candidate_coverage":if args.written_transfer_role_candidates {"partial_heuristic"} else {"not_computed"},
        "transfer_role_alternative_coverage":if args.written_transfer_role_alternatives {"partial_heuristic"} else {"not_computed"},
        "predicate_observation_coverage":if args.written_predicate_observations {"partial_heuristic"} else {"not_computed"},
        "producer_binding":"not_recorded",
        "spelling_coverage":if args.spelling_review {"unknown"} else {"not_computed"},
        "work_lexicon_sha256":vocabulary_hash});
    if args.cross_chapter_candidates {
        header["cross_chapter_coverage"] = json!("unknown");
    }
    if args.scoped_written_observations {
        header["quote_transfer_scope"] = json!("admitted_segments");
        header["cross_segment_associations"] = json!("not_examined");
        header["cross_chapter_candidates_enabled"] = json!(args.cross_chapter_candidates);
    }
    record(&mut body, header)?;
    let mut stream = SourceTimelineStream::new();
    let mut previous: Option<Meta> = None;
    let (mut completed, mut mentions, mut relations, mut quote_degraded, mut spelling_degraded) =
        (0usize, 0usize, 0usize, 0usize, 0usize);
    for entry in selected {
        ensure!(entry.status == "ok", "selected chapter is not ready");
        let chapter_path = root
            .join(&entry.file)
            .canonicalize()
            .context("open chapter")?;
        ensure!(
            chapter_path.starts_with(root),
            "chapter path escapes manifest directory"
        );
        let bytes = read_capped(&chapter_path, MAX_CHAPTER_BYTES)?;
        let source = std::str::from_utf8(&bytes).context("chapter is not UTF-8")?;
        let meta = Meta::new(entry.order, &bytes, source);
        if let (Some(start_char), Some(end_char)) =
            (args.spelling_start_char, args.spelling_end_char)
        {
            ensure!(
                start_char < end_char && end_char <= meta.chars,
                "invalid spelling range"
            );
        }
        let (analysis, frames, observations, predicates) = if args.written_predicate_observations {
            let (analysis, frames, actors, predicates) =
                analyzer.analyze_source_mentions_with_observations(source, &entry.title);
            (analysis, frames, actors, Some(predicates))
        } else {
            let (analysis, frames, actors) =
                analyzer.analyze_source_mentions_with_actor_observations(source, &entry.title);
            (analysis, frames, actors, None)
        };
        drop(frames);
        ensure!(
            observations.len() == analysis.events.len(),
            "actor observation count differs from events"
        );
        if let Some(items) = &predicates {
            ensure!(
                items.len() == analysis.events.len(),
                "predicate observation count differs from events"
            );
        }
        let reviews = analysis.reviews;
        let chunk = stream.push(Chapter {
            order: entry.order,
            text: source.to_owned(),
            events: analysis.events,
        })?;
        let current = stream.current().context("missing current chapter")?;
        let actor_offsets = (observations
            .iter()
            .any(|item| item.written_surface.is_some())
            || predicates
                .as_ref()
                .is_some_and(|items| items.iter().any(|item| item.written_predicate.is_some())))
        .then(|| sparse::SparseOffsets::new(&current.text));
        for mention in &chunk.mentions {
            let event = &current.events[mention.reference.event_index];
            let observation = observations
                .get(mention.reference.event_index)
                .context("missing actor observation")?;
            ensure!(
                observation.event_index == mention.reference.event_index,
                "actor observation index mismatch"
            );
            let (surface, written_text) = if let Some(span) = observation.written_surface {
                let mut proof = meta.evidence(span)?;
                let offsets = actor_offsets.as_ref().context("missing actor offsets")?;
                let start = offsets.byte_offset(&current.text, span.start_char);
                let end = offsets.byte_offset(&current.text, span.end_char);
                let text = current
                    .text
                    .get(start..end)
                    .context("invalid actor surface bytes")?;
                proof["source_sha256"] = json!(meta.sha256);
                proof["start_byte"] = json!(start);
                proof["end_byte"] = json!(end);
                (proof, Some(text))
            } else {
                (Value::Null, None)
            };
            let label_origin = match observation.label_basis {
                MentionLabelBasis::CallerVocabularyExactSurfaceAndLabel => {
                    Some(if work_bytes.is_some() {
                        "caller_vocabulary"
                    } else {
                        "empty_vocabulary"
                    })
                }
                MentionLabelBasis::LegacyParserLabelUnverified => Some("parser_unverified"),
                MentionLabelBasis::Unavailable => None,
            };
            let evidence: Vec<_> = mention
                .evidence
                .iter()
                .map(|span| evidence(*span, &meta, previous.as_ref()))
                .collect::<Result<_>>()?;
            let anchor = *mention
                .evidence
                .first()
                .context("mention lacks primary source")?;
            let mut record_value = json!({"type":"source_mention","id":identifier(mention.reference,anchor,&meta,previous.as_ref())?,
                "chapter":entry.order,"action":event.action,"evidence":evidence,"identity_status":"unknown",
                "occurrence_status":"unknown","cross_mention_status":"unknown","actor":null,"object":null,
                "actor_observation":{"written_surface":surface,"written_text":written_text,
                    "label":observation.legacy_label_hint,"label_origin":label_origin,"label_basis":observation.label_basis,
                    "caller_labels":observation.caller_labels,"label_mapping_status":observation.label_mapping_status,
                    "association_basis":observation.association_basis,"vocabulary_origin_detail":vocabulary_origin,
                    "vocabulary_sha256":vocabulary_hash,"identity_status":observation.identity_status,
                    "role_status":observation.role_status,"occurrence_status":observation.occurrence_status,
                    "narrative_scope_status":observation.narrative_scope_status},
                "parser_hints":{"kind":event.kind,"status":event.status,"actor":event.actor,
                    "object":event.object,"label_status":if event.actor.is_some() {"unverified_parser_hint"} else {"absent"}}});
            if let Some(items) = &predicates {
                let predicate = items
                    .get(mention.reference.event_index)
                    .context("missing predicate observation")?;
                ensure!(
                    predicate.event_index == mention.reference.event_index,
                    "predicate observation index mismatch"
                );
                let source = if let Some(span) = predicate.written_predicate {
                    ensure!(
                        predicate.basis != MentionPredicateBasis::Unavailable,
                        "predicate has no selection basis"
                    );
                    ensure!(
                        mention
                            .evidence
                            .iter()
                            .any(|proof| proof.chapter == entry.order
                                && proof.span.start_char <= span.start_char
                                && span.end_char <= proof.span.end_char),
                        "predicate outside mention evidence"
                    );
                    let mut proof = meta.evidence(span)?;
                    let offsets = actor_offsets
                        .as_ref()
                        .context("missing predicate offsets")?;
                    let start = offsets.byte_offset(&current.text, span.start_char);
                    let end = offsets.byte_offset(&current.text, span.end_char);
                    proof["start_byte"] = json!(start);
                    proof["end_byte"] = json!(end);
                    proof["written_text"] = json!(
                        current
                            .text
                            .get(start..end)
                            .context("invalid predicate bytes")?
                    );
                    proof
                } else {
                    ensure!(
                        predicate.basis == MentionPredicateBasis::Unavailable,
                        "missing predicate has selection basis"
                    );
                    Value::Null
                };
                record_value["written_predicate_observation"] = json!({"source":source,"basis":predicate.basis,
                    "identity_status":"unknown","role_status":"unknown","occurrence_status":"unknown",
                    "association_status":"unknown","narrative_scope_status":"unknown"});
            }
            record(&mut body, record_value)?;
        }
        for relation in &chunk.relations {
            let refs: Vec<_> = relation
                .evidence
                .iter()
                .map(|span| evidence(*span, &meta, previous.as_ref()))
                .collect::<Result<_>>()?;
            record(
                &mut body,
                json!({"type":"source_relation_hint","before":identifier(relation.before,relation.evidence[0],&meta,previous.as_ref())?,
                "after":identifier(relation.after,relation.evidence[2],&meta,previous.as_ref())?,"evidence":refs,
                "status":"unknown","identity_status":"unknown","occurrence_status":"unknown",
                "global_cycle_status":"not_computed"}),
            )?;
        }
        for review in reviews {
            record(
                &mut body,
                json!({"type":"review","kind":review.kind,"detail":review.detail,"evidence":meta.evidence(review.evidence)?}),
            )?;
        }
        if args.written_time_cues {
            write_time_cues(&mut body, &current.text, &entry.title, &meta, &analyzer)?;
        }
        if args.scoped_written_observations {
            if write_scoped_observations(
                &mut body,
                &current.text,
                &entry.title,
                &meta,
                &analyzer,
                &vocabulary,
                &args,
            )? {
                quote_degraded = quote_degraded
                    .checked_add(1)
                    .context("quote count overflow")?;
            }
        } else {
            if args.written_transfer_cues {
                write_transfer_cues(&mut body, &current.text, &meta, &analyzer)?;
            }
            if args.written_transfer_role_candidates {
                write_transfer_role_candidates(&mut body, &current.text, &meta, &analyzer)?;
            }
            if args.written_transfer_role_alternatives {
                write_transfer_role_alternatives(&mut body, &current.text, &meta, &analyzer)?;
            }
            if args.written_quotes
                && write_quotes(
                    &mut body,
                    &current.text,
                    &entry.title,
                    &meta,
                    &analyzer,
                    &vocabulary,
                )?
            {
                quote_degraded = quote_degraded
                    .checked_add(1)
                    .context("quote count overflow")?;
            }
        }
        if args.spelling_review
            && write_spelling(
                &mut body,
                &current.text,
                &meta,
                spelling_names.as_ref(),
                &args,
            )?
        {
            spelling_degraded = spelling_degraded
                .checked_add(1)
                .context("spelling count overflow")?;
        }
        if let Some(cross) = cross.as_mut() {
            // Even an empty selected chapter is pushed. A skipped ordinal is
            // reported by the library as an unexamined history gap.
            let observation = cross.push(entry.order, &current.text, &entry.title)?;
            ensure!(
                observation.chapter == entry.order,
                "cross-chapter summary ordinal differs"
            );
            let offsets = sparse::SparseOffsets::new(&current.text);
            for name in &observation.names {
                validate_cross_position(name, entry.order, &current.text, &meta, &offsets)?;
                let text = current
                    .text
                    .get(name.bytes.clone())
                    .context("invalid name source")?;
                ensure!(
                    vocabulary
                        .actors
                        .iter()
                        .any(|(surface, _)| *surface == text),
                    "cross-chapter name is not a caller vocabulary surface"
                );
            }
            for pronoun in &observation.pronouns {
                validate_cross_position(
                    &pronoun.source,
                    entry.order,
                    &current.text,
                    &meta,
                    &offsets,
                )?;
                ensure!(
                    current.text.get(pronoun.source.bytes.clone())
                        == Some(pronoun.written_text.as_str()),
                    "cross-chapter written token differs from source"
                );
            }
            let candidate_count = observation.candidates.len();
            for candidate in &observation.candidates {
                ensure!(
                    candidate.current_pronoun.chapter == entry.order
                        && observation
                            .pronouns
                            .iter()
                            .any(|token| token.source == candidate.current_pronoun),
                    "cross-chapter current endpoint is not an observed token"
                );
                ensure!(
                    candidate.prior_name.chapter >= observation.history_start_chapter
                        && candidate.prior_name.chapter < entry.order,
                    "cross-chapter prior endpoint is outside the configured window"
                );
                let prior = cross_history
                    .get(&candidate.prior_name.chapter)
                    .context("cross-chapter prior source metadata unavailable")?;
                ensure!(
                    prior.names.contains(&candidate.prior_name),
                    "cross-chapter prior endpoint is not an observed name"
                );
            }
            let mut summary = serde_json::to_value(&observation)?;
            summary
                .as_object_mut()
                .context("cross-chapter summary is not an object")?
                .remove("candidates");
            record(
                &mut body,
                json!({"type":"cross_chapter_summary","chapter":entry.order,
                "chapter_sha256":meta.sha256,"observation":summary,"candidate_records":candidate_count}),
            )?;
            for candidate in &observation.candidates {
                let prior = cross_history
                    .get(&candidate.prior_name.chapter)
                    .context("cross-chapter prior source metadata unavailable")?;
                record(
                    &mut body,
                    json!({"type":"cross_chapter_candidate","chapter":entry.order,
                    "prior_name":cross_source_value(&candidate.prior_name,&prior.sha256)?,
                    "current_pronoun":cross_source_value(&candidate.current_pronoun,&meta.sha256)?,
                    "basis":candidate.basis,"identity_status":candidate.identity_status,
                    "role_status":candidate.role_status,"association_status":candidate.association_status,
                    "occurrence_status":candidate.occurrence_status,"narrative_scope_status":candidate.narrative_scope_status}),
                )?;
            }
            cross_summaries = cross_summaries
                .checked_add(1)
                .context("cross summary count overflow")?;
            cross_candidates = cross_candidates
                .checked_add(candidate_count)
                .context("cross candidate count overflow")?;
            let floor = observation.history_start_chapter;
            cross_history.retain(|&chapter, _| chapter >= floor);
            cross_history.insert(
                entry.order,
                CrossMeta {
                    sha256: meta.sha256.clone(),
                    names: observation.names,
                },
            );
        }
        mentions = mentions
            .checked_add(chunk.mentions.len())
            .context("mention count overflow")?;
        relations = relations
            .checked_add(chunk.relations.len())
            .context("relation count overflow")?;
        completed = completed
            .checked_add(1)
            .context("completed chapter count overflow")?;
        record(
            &mut body,
            json!({"type":"chapter_complete","chapter":entry.order,"sha256":meta.sha256,
            "mentions":chunk.mentions.len(),"relations":chunk.relations.len(),"extraction_coverage":"unknown"}),
        )?;
        previous = Some(meta);
    }
    let body_hash = format!("{:x}", body.digest.clone().finalize());
    let body_lines = body.lines;
    let mut completion = json!({"type":"stream_complete","manifest_sha256":manifest_hash,
        "prior_records_sha256":body_hash,"prior_records":body_lines,
        "selected_chapters":selected_count,"completed_chapters":completed,"mentions":mentions,
        "relations":relations,"extraction_coverage":"unknown","global_story_order":"not_computed",
        "time_cue_coverage":if args.written_time_cues {"partial_heuristic"} else {"not_computed"},
        "transfer_cue_coverage":if args.written_transfer_cues {"partial_heuristic"} else {"not_computed"},
        "transfer_role_candidate_coverage":if args.written_transfer_role_candidates {"partial_heuristic"} else {"not_computed"},
        "transfer_role_alternative_coverage":if args.written_transfer_role_alternatives {"partial_heuristic"} else {"not_computed"},
        "predicate_observation_coverage":if args.written_predicate_observations {"partial_heuristic"} else {"not_computed"},
        "quote_observation":if args.written_quotes {"partial_unverified"} else {"not_computed"},
        "quote_degraded_chapters":if args.written_quotes {Some(quote_degraded)} else {None},
        "spelling_coverage":if args.spelling_review {"unknown"} else {"not_computed"},
        "spelling_degraded_chapters":if args.spelling_review {Some(spelling_degraded)} else {None}});
    if args.cross_chapter_candidates {
        completion["cross_chapter_coverage"] = json!("unknown");
        completion["cross_chapter_summaries"] = json!(cross_summaries);
        completion["cross_chapter_candidates"] = json!(cross_candidates);
    }
    if args.scoped_written_observations {
        completion["quote_transfer_scope"] = json!("admitted_segments");
        completion["cross_segment_associations"] = json!("not_examined");
        completion["scoped_written_summaries"] = json!(completed);
    }
    record(out, completion)?;
    Ok(())
}

fn main() -> Result<()> {
    let mut out = BufWriter::new(io::stdout().lock());
    run(Args::parse(), &mut out)?;
    out.flush()?;
    Ok(())
}

#[cfg(test)]
mod receipt_tests {
    use super::*;

    #[test]
    fn record_count_overflow_stops_the_writer() {
        let mut output = Vec::new();
        let mut writer = DigestWriter {
            inner: &mut output,
            digest: Sha256::new(),
            lines: usize::MAX,
        };
        assert!(writer.write_all(b"{}\n").is_err());
    }
}
