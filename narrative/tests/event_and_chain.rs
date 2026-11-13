use faberline_nlp::Span;
use faberline_nlp::doc::{
    Document, Entity, EntityKind, Lang, Reason, Score, SyntaxToken, Unknown, Upos, analyze_document,
};
use faberline_nlp_narrative::extract_core;
use faberline_nlp_narrative::schema::{
    AliasEntry, ArgumentBasis, CHAPTER_NARRATIVE_SCHEMA, ChapterNarrative, ContextZone, EdgeKind,
    EventRole, Factuality, LocatedSpan, MentionKind, NarrativeUnknownKind, SpeakerBasis,
    WorkVocabulary, text_sha256_hex,
};
use std::collections::BTreeSet;

fn char_slice(text: &str, span: Span) -> String {
    let chars: Vec<char> = text.chars().collect();
    assert!(
        span.start_char <= span.end_char && span.end_char <= chars.len(),
        "span {:?} out of bounds for len {}",
        span,
        chars.len()
    );
    chars[span.start_char..span.end_char].iter().collect()
}

fn assert_all_spans_in_bounds(text: &str, core: &faberline_nlp_narrative::CoreExtraction) {
    let n = text.chars().count();
    let check = |s: Span, label: &str| {
        assert!(
            s.start_char <= s.end_char && s.end_char <= n,
            "{label} span {s:?} out of bounds (text_chars={n})"
        );
    };
    for d in &core.dialogues {
        check(d.span, "dialogue.span");
        check(d.content_span, "dialogue.content_span");
        if let Some(s) = d.speech_verb {
            check(s, "dialogue.speech_verb");
        }
        if let Some(s) = d.speaker_span {
            check(s, "dialogue.speaker_span");
        }
    }
    for e in &core.events {
        check(e.clause_span, "event.clause_span");
        check(e.predicate_span, "event.predicate_span");
        if let Some(s) = e.factuality_cue {
            check(s, "event.factuality_cue");
        }
        for a in &e.arguments {
            check(a.span, "event.arg.span");
        }
        for t in &e.temporal_anchors {
            check(t.span, "event.temporal_anchor.span");
        }
    }
    for m in &core.mentions {
        check(m.span, "mention.span");
        if let Some(s) = m.resolved_antecedent {
            check(s, "mention.resolved_antecedent");
        }
    }
    for u in &core.unknown {
        check(u.span, "unknown.span");
    }
}

fn sample_vocab() -> WorkVocabulary {
    WorkVocabulary {
        actors: vec![
            AliasEntry {
                surface: "林遠".to_owned(),
                canonical: "lin_yuan".to_owned(),
            },
            AliasEntry {
                surface: "蘇晴".to_owned(),
                canonical: "su_qing".to_owned(),
            },
            AliasEntry {
                surface: "陳伯".to_owned(),
                canonical: "chen_bo".to_owned(),
            },
            AliasEntry {
                surface: "周平".to_owned(),
                canonical: "zhou_ping".to_owned(),
            },
            AliasEntry {
                surface: "Alice".to_owned(),
                canonical: "alice".to_owned(),
            },
        ],
        places: vec![
            AliasEntry {
                surface: "青石鎮".to_owned(),
                canonical: "qingshi_town".to_owned(),
            },
            AliasEntry {
                surface: "石室".to_owned(),
                canonical: "stone_room".to_owned(),
            },
        ],
        objects: vec![
            AliasEntry {
                surface: "古銅鏡".to_owned(),
                canonical: "bronze_mirror".to_owned(),
            },
            AliasEntry {
                surface: "玉佩".to_owned(),
                canonical: "jade_pendant".to_owned(),
            },
        ],
        domain_terms: vec!["油燈".to_owned(), "木門".to_owned(), "木匣".to_owned()],
    }
}

fn analyze_with_vocab(text: &str, vocab: &WorkVocabulary) -> Document {
    analyze_document(text, &vocab.to_doc_options())
}

#[test]
fn topic_chain_inheritance_multi_clause_zh_tw() {
    let text = "林遠推開木門，走入石室，點燃油燈。";
    let vocab = sample_vocab();
    let doc = analyze_with_vocab(text, &vocab);
    let core = extract_core(1, text, &doc, &vocab);
    assert_all_spans_in_bounds(text, &core);

    assert_eq!(core.events.len(), 3, "expected 3 clause events");
    for ev in &core.events {
        assert_eq!(ev.zone, ContextZone::Narrative);
        assert_eq!(ev.factuality, Factuality::Actual);
        let agent = ev
            .arguments
            .iter()
            .find(|a| a.role == EventRole::Agent)
            .expect("each clause should have Agent");
        assert_eq!(char_slice(text, agent.span), "林遠");
        assert_eq!(agent.canonical.as_deref(), Some("lin_yuan"));
    }

    assert_eq!(
        core.events[0]
            .arguments
            .iter()
            .find(|a| a.role == EventRole::Agent)
            .unwrap()
            .basis,
        ArgumentBasis::DirectSyntax
    );
    assert_eq!(
        core.events[1]
            .arguments
            .iter()
            .find(|a| a.role == EventRole::Agent)
            .unwrap()
            .basis,
        ArgumentBasis::IntraSentenceTopicChain
    );
    assert_eq!(
        core.events[2]
            .arguments
            .iter()
            .find(|a| a.role == EventRole::Agent)
            .unwrap()
            .basis,
        ArgumentBasis::IntraSentenceTopicChain
    );

    let pro_drops: Vec<_> = core
        .mentions
        .iter()
        .filter(|m| m.kind == MentionKind::ZeroPronounProDrop)
        .collect();
    assert_eq!(pro_drops.len(), 2);
    for m in pro_drops {
        assert_eq!(m.canonical.as_deref(), Some("lin_yuan"));
        assert_eq!(char_slice(text, m.resolved_antecedent.unwrap()), "林遠");
    }
    assert!(
        core.unknown.is_empty(),
        "unexpected unknowns: {:?}",
        core.unknown
    );
}

#[test]
fn disposal_jiang_is_patient_and_actual_not_modal() {
    let text = "蘇晴從袖中取出玉佩，將玉佩交給守衛。";
    let vocab = sample_vocab();
    let doc = analyze_with_vocab(text, &vocab);
    let core = extract_core(1, text, &doc, &vocab);
    assert_all_spans_in_bounds(text, &core);

    assert_eq!(core.events.len(), 2);

    let ev0 = &core.events[0];
    assert_eq!(ev0.factuality, Factuality::Actual);
    let agent0 = ev0
        .arguments
        .iter()
        .find(|a| a.role == EventRole::Agent)
        .expect("ev0 agent");
    assert_eq!(char_slice(text, agent0.span), "蘇晴");
    let patient0 = ev0
        .arguments
        .iter()
        .find(|a| a.role == EventRole::Patient)
        .expect("ev0 patient");
    assert_eq!(char_slice(text, patient0.span), "玉佩");

    let ev1 = &core.events[1];
    assert_eq!(
        ev1.factuality,
        Factuality::Actual,
        "disposal 將 must stay Actual"
    );
    assert_eq!(ev1.factuality_cue, None);
    let patient1 = ev1
        .arguments
        .iter()
        .find(|a| a.role == EventRole::Patient)
        .expect("ev1 disposal patient");
    assert_eq!(char_slice(text, patient1.span), "玉佩");
    assert_eq!(patient1.basis, ArgumentBasis::DisposalJiang);
    assert_eq!(patient1.canonical.as_deref(), Some("jade_pendant"));

    let agent1 = ev1
        .arguments
        .iter()
        .find(|a| a.role == EventRole::Agent)
        .expect("ev1 inherited agent");
    assert_eq!(char_slice(text, agent1.span), "蘇晴");
    assert_eq!(agent1.basis, ArgumentBasis::IntraSentenceTopicChain);
}

#[test]
fn passive_bei_realigns_patient_and_agent() {
    let text = "木門被狂風吹開。";
    let vocab = sample_vocab();
    let doc = analyze_with_vocab(text, &vocab);
    let core = extract_core(1, text, &doc, &vocab);
    assert_all_spans_in_bounds(text, &core);

    assert_eq!(core.events.len(), 1);
    let ev = &core.events[0];
    assert_eq!(ev.factuality, Factuality::Actual);

    let patient = ev
        .arguments
        .iter()
        .find(|a| a.role == EventRole::Patient)
        .expect("passive patient");
    assert_eq!(char_slice(text, patient.span), "木門");
    assert_eq!(patient.basis, ArgumentBasis::PassiveBei);

    let agent = ev
        .arguments
        .iter()
        .find(|a| a.role == EventRole::Agent)
        .expect("passive agent");
    assert_eq!(char_slice(text, agent.span), "狂風");
    assert_eq!(agent.basis, ArgumentBasis::PassiveBei);
}

#[test]
fn quote_isolation_and_post_quote_speaker_attribution() {
    let text = "「明日我們不走水路，」陳伯低聲說道。";
    let vocab = sample_vocab();
    let doc = analyze_with_vocab(text, &vocab);
    let core = extract_core(1, text, &doc, &vocab);
    assert_all_spans_in_bounds(text, &core);

    assert_eq!(core.dialogues.len(), 1);
    let d = &core.dialogues[0];
    assert_eq!(char_slice(text, d.span), "「明日我們不走水路，」");
    assert_eq!(char_slice(text, d.content_span), "明日我們不走水路，");
    assert_eq!(d.basis, SpeakerBasis::PostQuoteSpeechVerb);
    assert_eq!(char_slice(text, d.speech_verb.unwrap()), "說道");
    assert_eq!(char_slice(text, d.speaker_span.unwrap()), "陳伯");
    assert_eq!(d.speaker_canonical.as_deref(), Some("chen_bo"));

    assert_eq!(core.events.len(), 2);
    let quoted_ev = &core.events[0];
    assert_eq!(quoted_ev.zone, ContextZone::DirectQuote { quote_index: 0 });
    assert_eq!(quoted_ev.factuality, Factuality::Quoted);

    let frame_ev = &core.events[1];
    assert_eq!(frame_ev.zone, ContextZone::Narrative);
    assert_eq!(frame_ev.factuality, Factuality::Actual);
    assert_eq!(char_slice(text, frame_ev.predicate_span), "說道");
    let speaker_arg = frame_ev
        .arguments
        .iter()
        .find(|a| a.role == EventRole::Agent)
        .expect("framing speaker agent");
    assert_eq!(char_slice(text, speaker_arg.span), "陳伯");
    assert!(
        frame_ev.temporal_anchors.is_empty(),
        "DirectQuote temporal entities must not leak onto Narrative framing event"
    );
}

#[test]
fn factuality_classification_and_idiom_guards() {
    let vocab = sample_vocab();

    let text_idioms = "林遠不久抵達青石鎮。蘇晴不禁皺起眉頭。";
    let doc_idioms = analyze_with_vocab(text_idioms, &vocab);
    let core_idioms = extract_core(1, text_idioms, &doc_idioms, &vocab);
    assert_all_spans_in_bounds(text_idioms, &core_idioms);
    assert_eq!(core_idioms.events.len(), 2);
    assert_eq!(core_idioms.events[0].factuality, Factuality::Actual);
    assert_eq!(core_idioms.events[1].factuality, Factuality::Actual);

    let text_neg = "蘇晴沒有打開木匣。";
    let doc_neg = analyze_with_vocab(text_neg, &vocab);
    let core_neg = extract_core(1, text_neg, &doc_neg, &vocab);
    assert_all_spans_in_bounds(text_neg, &core_neg);
    assert_eq!(core_neg.events.len(), 1);
    assert_eq!(core_neg.events[0].factuality, Factuality::Negated);
    assert_eq!(
        char_slice(text_neg, core_neg.events[0].factuality_cue.unwrap()),
        "沒有"
    );

    let text_hyp = "若是明日下雨，眾人便暫留客棧。";
    let doc_hyp = analyze_with_vocab(text_hyp, &vocab);
    let core_hyp = extract_core(1, text_hyp, &doc_hyp, &vocab);
    assert_all_spans_in_bounds(text_hyp, &core_hyp);
    assert_eq!(core_hyp.events.len(), 2);
    assert_eq!(core_hyp.events[0].factuality, Factuality::Hypothetical);
    assert_eq!(core_hyp.events[1].factuality, Factuality::Hypothetical);

    let text_mod = "他們即將動身前往港口。";
    let doc_mod = analyze_with_vocab(text_mod, &vocab);
    let core_mod = extract_core(1, text_mod, &doc_mod, &vocab);
    assert_all_spans_in_bounds(text_mod, &core_mod);
    assert_eq!(core_mod.events.len(), 1);
    assert_eq!(core_mod.events[0].factuality, Factuality::ModalOrFuture);
    assert_eq!(
        char_slice(text_mod, core_mod.events[0].factuality_cue.unwrap()),
        "即將"
    );
}

#[test]
fn never_guess_ambiguity_for_pro_drop_and_pronoun() {
    let text = "林遠與周平走入大廳，坐了下來，他端起茶杯。";
    let vocab = sample_vocab();
    let doc = analyze_with_vocab(text, &vocab);
    let core = extract_core(1, text, &doc, &vocab);
    assert_all_spans_in_bounds(text, &core);

    assert_eq!(core.events.len(), 3);
    // Clause 1 (`坐了下來`) must NOT inherit a guessed Agent.
    assert!(
        core.events[1]
            .arguments
            .iter()
            .all(|a| a.role != EventRole::Agent),
        "ambiguous pro-drop clause must not receive a guessed Agent"
    );

    // Pronoun `他` must NOT receive a guessed resolved_antecedent.
    let he_mention = core
        .mentions
        .iter()
        .find(|m| m.kind == MentionKind::Pronoun && char_slice(text, m.span) == "他")
        .expect("pronoun 他 mention");
    assert_eq!(he_mention.resolved_antecedent, None);
    assert_eq!(he_mention.canonical, None);

    assert!(
        core.unknown
            .iter()
            .any(|u| u.kind == NarrativeUnknownKind::ProDropAmbiguous),
        "expected ProDropAmbiguous, got {:?}",
        core.unknown
    );
    assert!(
        core.unknown
            .iter()
            .any(|u| u.kind == NarrativeUnknownKind::PronounAmbiguous),
        "expected PronounAmbiguous, got {:?}",
        core.unknown
    );
}

#[test]
fn english_and_mixed_prose_plus_nested_quotes_and_speaker_ambiguous() {
    let vocab = sample_vocab();

    let text_en = "Alice opened the wooden door, walked into the room, and lit a lantern.";
    let doc_en = analyze_with_vocab(text_en, &vocab);
    let core_en = extract_core(2, text_en, &doc_en, &vocab);
    assert_all_spans_in_bounds(text_en, &core_en);
    assert_eq!(core_en.events.len(), 3);
    for ev in &core_en.events {
        let ag = ev
            .arguments
            .iter()
            .find(|a| a.role == EventRole::Agent)
            .expect("Alice should propagate across English clauses");
        assert_eq!(ag.canonical.as_deref(), Some("alice"));
    }

    let text_mixed = "林遠 walked into the stone hall, and he picked up the 古銅鏡.";
    let doc_mixed = analyze_with_vocab(text_mixed, &vocab);
    let core_mixed = extract_core(3, text_mixed, &doc_mixed, &vocab);
    assert_all_spans_in_bounds(text_mixed, &core_mixed);
    assert_eq!(core_mixed.events.len(), 2);
    let he_mention = core_mixed
        .mentions
        .iter()
        .find(|m| m.kind == MentionKind::Pronoun && char_slice(text_mixed, m.span) == "he")
        .expect("English pronoun `he`");
    assert_eq!(
        char_slice(text_mixed, he_mention.resolved_antecedent.unwrap()),
        "林遠"
    );

    let text_ambig_spk = "林遠與周平說道：「明日動身。」";
    let doc_ambig_spk = analyze_with_vocab(text_ambig_spk, &vocab);
    let core_ambig_spk = extract_core(4, text_ambig_spk, &doc_ambig_spk, &vocab);
    assert_all_spans_in_bounds(text_ambig_spk, &core_ambig_spk);
    assert_eq!(core_ambig_spk.dialogues.len(), 1);
    assert_eq!(core_ambig_spk.dialogues[0].basis, SpeakerBasis::Unknown);
    assert_eq!(core_ambig_spk.dialogues[0].speaker_span, None);
    assert!(
        core_ambig_spk
            .unknown
            .iter()
            .any(|u| u.kind == NarrativeUnknownKind::SpeakerAmbiguous)
    );

    let text_nested = "陳伯道：「方才林遠說『夜路難行』，眾人小心。」";
    let doc_nested = analyze_with_vocab(text_nested, &vocab);
    let core_nested = extract_core(5, text_nested, &doc_nested, &vocab);
    assert_all_spans_in_bounds(text_nested, &core_nested);
    assert_eq!(core_nested.dialogues.len(), 1, "only outer quote emitted");
    assert_eq!(
        core_nested.dialogues[0].basis,
        SpeakerBasis::PreQuoteSpeechVerb
    );
    assert_eq!(
        char_slice(text_nested, core_nested.dialogues[0].speaker_span.unwrap()),
        "陳伯"
    );
}

#[test]
fn layer2_syntax_gate_and_located_span_traits() {
    // Test manual Ord / PartialOrd / Hash on LocatedSpan.
    let a = LocatedSpan {
        chapter: 1,
        span: Span {
            start_char: 0,
            end_char: 2,
        },
    };
    let b = LocatedSpan {
        chapter: 1,
        span: Span {
            start_char: 4,
            end_char: 6,
        },
    };
    let c = LocatedSpan {
        chapter: 2,
        span: Span {
            start_char: 0,
            end_char: 2,
        },
    };
    let mut set = BTreeSet::new();
    set.insert(c);
    set.insert(b);
    set.insert(a);
    assert_eq!(set.into_iter().collect::<Vec<_>>(), vec![a, b, c]);

    // Test Layer-2 syntax gate with synthetic UPOS + SyntaxToken annotations:
    // Clause 0: `林遠打開木門` (`打開` is root, `林遠` nsubj 0.96, `木門` obj 0.94,
    // plus an illegal cross-clause arc and low-score < 0.35 arc that MUST be rejected).
    let text = "林遠推開木門，取出古銅鏡。";
    let vocab = sample_vocab();
    let mut doc = analyze_with_vocab(text, &vocab);
    let idx_of = |w: &str| -> usize {
        doc.tokens
            .iter()
            .position(|t| char_slice(text, t.span) == w)
            .unwrap_or_else(|| panic!("missing token {w:?}"))
    };
    let t_lin = idx_of("林遠");
    let t_open = idx_of("推開");
    let t_door = idx_of("木門");
    let t_take = idx_of("取出");
    let t_mirror = idx_of("古銅鏡");
    doc.tokens[t_lin].upos = Some(Upos::Propn);
    doc.tokens[t_open].upos = Some(Upos::Verb);
    doc.tokens[t_door].upos = Some(Upos::Noun);
    doc.tokens[t_take].upos = Some(Upos::Verb);
    doc.tokens[t_mirror].upos = Some(Upos::Noun);
    doc.syntax = vec![
        SyntaxToken {
            token: t_lin,
            sentence: 0,
            span: doc.tokens[t_lin].span,
            head: Some(t_open),
            deprel: "nsubj".to_owned(),
            score: Score(960),
        },
        SyntaxToken {
            token: t_open,
            sentence: 0,
            span: doc.tokens[t_open].span,
            head: None,
            deprel: "root".to_owned(),
            score: Score(980),
        },
        SyntaxToken {
            token: t_door,
            sentence: 0,
            span: doc.tokens[t_door].span,
            head: Some(t_open),
            deprel: "obj".to_owned(),
            score: Score(940),
        },
        // Cross-clause arc (`古銅鏡` -> `推開` in clause 0, plus low score 300): MUST be gated out!
        SyntaxToken {
            token: t_take,
            sentence: 0,
            span: doc.tokens[t_take].span,
            head: Some(t_open),
            deprel: "conj".to_owned(),
            score: Score(900),
        },
        SyntaxToken {
            token: t_mirror,
            sentence: 0,
            span: doc.tokens[t_mirror].span,
            head: Some(t_open),
            deprel: "obj".to_owned(),
            score: Score(300),
        },
    ];
    let core = extract_core(1, text, &doc, &vocab);
    assert_eq!(core.events.len(), 2);
    // Cross-clause arc on `古銅鏡` must NOT attach `古銅鏡` to `推開` (event 0).
    assert_eq!(core.events[0].arguments.len(), 2);
    // Low-confidence (`300 < 350`) arc on `古銅鏡` must NOT attach to event 1 either.
    assert!(
        core.events[1]
            .arguments
            .iter()
            .all(|a| a.role != EventRole::Patient)
    );

    // Test PredicateLowConfidence abstention when `doc.unknown` suppresses the clause head.
    let text_lc = "林遠察看木匣。";
    let mut doc_lc = analyze_with_vocab(text_lc, &vocab);
    // Tokens: 0=林遠, 1=察看, 2=木匣, 3=。
    doc_lc.tokens[0].upos = Some(Upos::Propn);
    doc_lc.tokens[1].upos = None;
    doc_lc.tokens[2].upos = Some(Upos::Noun);
    doc_lc.unknown.push(Unknown {
        layer: "pos".to_owned(),
        span: doc_lc.tokens[1].span,
        reason: Reason::LowConfidence,
    });
    // Ensure entities match cleanly.
    doc_lc.entities.push(Entity {
        kind: EntityKind::Name,
        span: doc_lc.tokens[0].span,
        sentence: 0,
        value: None,
        unit: None,
    });
    assert_eq!(doc_lc.tokens[0].lang, Lang::Zh);
    let core_lc = extract_core(1, text_lc, &doc_lc, &vocab);
    assert!(
        core_lc.events.is_empty(),
        "low confidence predicate must abstain"
    );
    assert!(
        core_lc
            .unknown
            .iter()
            .any(|u| u.kind == NarrativeUnknownKind::PredicateLowConfidence)
    );
}

#[test]
fn acl_modifier_skip_compound_merge_and_schema_serde_contract() {
    let vocab = sample_vocab();

    // Pre-nominal `acl` relative-clause modifier `打開的` must be skipped in favor of matrix `拿出`.
    let text = "林遠從桌上的木匣中取出玉佩。";
    let mut doc = analyze_with_vocab(text, &vocab);
    for tok in &mut doc.tokens {
        let s = char_slice(text, tok.span);
        tok.upos = Some(match s.as_str() {
            "林遠" => Upos::Propn,
            "從" => Upos::Adp,
            "的" => Upos::Part,
            "桌上" | "木匣" | "中" | "玉佩" => Upos::Noun,
            "取出" => Upos::Verb,
            _ => Upos::Punct,
        });
    }
    let idx_of = |w: &str| -> usize {
        doc.tokens
            .iter()
            .position(|t| char_slice(text, t.span) == w)
            .unwrap()
    };
    let t_lin = idx_of("林遠");
    let t_table = idx_of("桌上");
    let t_box = idx_of("木匣");
    let t_take = idx_of("取出");
    let t_jade = idx_of("玉佩");
    doc.tokens[t_table].upos = Some(Upos::Verb);
    doc.syntax = vec![
        SyntaxToken {
            token: t_lin,
            sentence: 0,
            span: doc.tokens[t_lin].span,
            head: Some(t_take),
            deprel: "nsubj".to_owned(),
            score: Score(950),
        },
        SyntaxToken {
            token: t_table,
            sentence: 0,
            span: doc.tokens[t_table].span,
            head: Some(t_box),
            deprel: "acl".to_owned(),
            score: Score(800),
        },
        SyntaxToken {
            token: t_box,
            sentence: 0,
            span: doc.tokens[t_box].span,
            head: Some(t_take),
            deprel: "obl".to_owned(),
            score: Score(850),
        },
        SyntaxToken {
            token: t_take,
            sentence: 0,
            span: doc.tokens[t_take].span,
            head: None,
            deprel: "root".to_owned(),
            score: Score(970),
        },
        SyntaxToken {
            token: t_jade,
            sentence: 0,
            span: doc.tokens[t_jade].span,
            head: Some(t_take),
            deprel: "obj".to_owned(),
            score: Score(930),
        },
    ];
    let core = extract_core(1, text, &doc, &vocab);
    assert_eq!(core.events.len(), 1);
    assert_eq!(char_slice(text, core.events[0].predicate_span), "取出");

    // Verify temporal anchor attachment and `ChapterNarrative` JSON serialization round-trip.
    let text_t = "三日後，林遠回到青石鎮。";
    let doc_t = analyze_with_vocab(text_t, &vocab);
    let core_t = extract_core(1, text_t, &doc_t, &vocab);
    assert_all_spans_in_bounds(text_t, &core_t);
    assert_eq!(core_t.events.len(), 1);
    assert!(
        !core_t.events[0].temporal_anchors.is_empty(),
        "leading temporal clause anchor should attach to sentence event"
    );

    let chap = ChapterNarrative {
        schema: CHAPTER_NARRATIVE_SCHEMA.to_owned(),
        chapter: 1,
        title: Some("Chapter 1".to_owned()),
        text_chars: text_t.chars().count(),
        text_sha256: text_sha256_hex(text_t),
        dialogues: core_t.dialogues,
        events: core_t.events,
        mentions: core_t.mentions,
        edges: Vec::new(),
        event_dag_layers: vec![Some(0)],
        conflict_cycles_broken: 0,
        unknown: core_t.unknown,
    };
    let json = serde_json::to_string(&chap).expect("serialize ChapterNarrative");
    assert!(json.contains("\"zone\":\"narrative\""));
    assert!(json.contains("\"factuality\":\"actual\""));
    let roundtrip: ChapterNarrative =
        serde_json::from_str(&json).expect("deserialize ChapterNarrative");
    assert_eq!(roundtrip, chap);
    let _edge_kind_check = EdgeKind::Before;
}

#[test]
fn model_syntax_low_conf_abstains_without_zero_score_fallback_and_negation_compounds() {
    let vocab = sample_vocab();

    // 1. Model-analyzed document where sentence 1 has EVERY token suppressed by `syntax: low_confidence`
    // (while sentence 0 has valid syntax). Sentence 1 must abstain with `PredicateLowConfidence` and
    // must NEVER fall back to `extract_args_no_syntax` emitting `DirectSyntax` with `score_milli == 0`.
    let text = "林遠推開木門。蘇晴察看古銅鏡。";
    let mut doc = analyze_with_vocab(text, &vocab);
    let sent0_toks: Vec<usize> = doc
        .tokens
        .iter()
        .enumerate()
        .filter(|(_, t)| t.sentence == 0)
        .map(|(i, _)| i)
        .collect();
    let sent1_toks: Vec<usize> = doc
        .tokens
        .iter()
        .enumerate()
        .filter(|(_, t)| t.sentence == 1)
        .map(|(i, _)| i)
        .collect();
    for &ti in &sent0_toks {
        let s = char_slice(text, doc.tokens[ti].span);
        doc.tokens[ti].upos = Some(match s.as_str() {
            "林遠" => Upos::Propn,
            "推開" => Upos::Verb,
            "木門" => Upos::Noun,
            _ => Upos::Punct,
        });
    }
    let s0_lin = sent0_toks
        .iter()
        .copied()
        .find(|&ti| char_slice(text, doc.tokens[ti].span) == "林遠")
        .unwrap();
    let s0_open = sent0_toks
        .iter()
        .copied()
        .find(|&ti| char_slice(text, doc.tokens[ti].span) == "推開")
        .unwrap();
    let s0_door = sent0_toks
        .iter()
        .copied()
        .find(|&ti| char_slice(text, doc.tokens[ti].span) == "木門")
        .unwrap();
    doc.syntax = vec![
        SyntaxToken {
            token: s0_lin,
            sentence: 0,
            span: doc.tokens[s0_lin].span,
            head: Some(s0_open),
            deprel: "nsubj".to_owned(),
            score: Score(960),
        },
        SyntaxToken {
            token: s0_open,
            sentence: 0,
            span: doc.tokens[s0_open].span,
            head: None,
            deprel: "root".to_owned(),
            score: Score(980),
        },
        SyntaxToken {
            token: s0_door,
            sentence: 0,
            span: doc.tokens[s0_door].span,
            head: Some(s0_open),
            deprel: "obj".to_owned(),
            score: Score(940),
        },
    ];
    for &ti in &sent1_toks {
        let s = char_slice(text, doc.tokens[ti].span);
        if s == "。" {
            doc.tokens[ti].upos = Some(Upos::Punct);
        } else {
            // Leave `upos` as `Some(Verb)` or `None` while suppressing syntax completely.
            if s == "察看" {
                doc.tokens[ti].upos = Some(Upos::Verb);
            }
            doc.unknown.push(Unknown {
                layer: "syntax".to_owned(),
                span: doc.tokens[ti].span,
                reason: Reason::LowConfidence,
            });
        }
    }

    let core = extract_core(1, text, &doc, &vocab);
    assert_eq!(core.events.len(), 1, "sentence 1 must abstain");
    assert!(
        core.unknown
            .iter()
            .any(|u| u.kind == NarrativeUnknownKind::PredicateLowConfidence),
        "expected PredicateLowConfidence for suppressed sentence 1"
    );
    for ev in &core.events {
        for arg in &ev.arguments {
            if arg.basis == ArgumentBasis::DirectSyntax {
                assert!(arg.score_milli >= 350);
            }
        }
    }

    // 2. Multi-character negation tokens (`不會`, `不能`, `不敢`, `無法`, `未能`) vs `不得不` idiom guard.
    let neg_text = "林遠不會輕易動手。蘇晴不能離開青石鎮。陳伯無法打開木匣。林遠不得不離開客棧。";
    let neg_doc = analyze_with_vocab(neg_text, &vocab);
    let neg_core = extract_core(1, neg_text, &neg_doc, &vocab);
    assert_eq!(neg_core.events.len(), 4);
    assert_eq!(neg_core.events[0].factuality, Factuality::Negated);
    assert_eq!(neg_core.events[1].factuality, Factuality::Negated);
    assert_eq!(neg_core.events[2].factuality, Factuality::Negated);
    assert_eq!(neg_core.events[3].factuality, Factuality::Actual);

    // 3. Passive `由` frame stops at embedded verb (`帶`) and never includes `由` in agent span.
    let pass_text = "此行由陳伯帶眾人立即出發。";
    let mut pass_doc = analyze_with_vocab(pass_text, &vocab);
    for tok in &mut pass_doc.tokens {
        let s = char_slice(pass_text, tok.span);
        if s == "帶" || s == "出發" {
            tok.upos = Some(Upos::Verb);
        } else if s == "立即" {
            tok.upos = Some(Upos::Adv);
        }
    }
    let pass_core = extract_core(1, pass_text, &pass_doc, &vocab);
    assert!(!pass_core.events.is_empty());
    for ev in &pass_core.events {
        for arg in &ev.arguments {
            if arg.basis == ArgumentBasis::PassiveBei && arg.role == EventRole::Agent {
                let s = char_slice(pass_text, arg.span);
                assert!(
                    !s.starts_with('由'),
                    "agent span must not start with 由: {s}"
                );
                assert!(
                    !s.contains('帶'),
                    "agent span must stop before verb 帶: {s}"
                );
            }
        }
    }
}
