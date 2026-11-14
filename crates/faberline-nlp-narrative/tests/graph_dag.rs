use faberline_nlp::Span;
use faberline_nlp::doc::{
    Document, EntityKind, Options, Relation, RelationKind, RelationLevel, analyze_document,
};
use faberline_nlp_narrative::graph::{
    BookEventRef, BookTimelineGraph, build_chapter_dag_from_narrative, build_chapter_graph,
};
use faberline_nlp_narrative::schema::{
    CHAPTER_NARRATIVE_SCHEMA, CandidateEvent, ChapterNarrative, ChronologyOrder, ContextZone,
    EdgeEvidenceKind, EdgeKind, EdgeStatus, Factuality, NarrativeUnknownKind, TemporalAnchor,
};

fn find_char_span(text: &str, needle: &str) -> Span {
    let byte_start = text
        .find(needle)
        .unwrap_or_else(|| panic!("needle {needle:?} not found in {text:?}"));
    let start_char = text[..byte_start].chars().count();
    let end_char = start_char + needle.chars().count();
    Span {
        start_char,
        end_char,
    }
}

fn find_clause_for_span(doc: &Document, span: Span) -> (usize, usize, usize, Span) {
    for (c_idx, clause) in doc.segmentation.clauses.iter().enumerate() {
        if clause.span.start_char <= span.start_char && span.end_char <= clause.span.end_char {
            let sent = &doc.segmentation.sentences[clause.sentence];
            return (sent.paragraph, clause.sentence, c_idx, clause.span);
        }
    }
    panic!("no clause found containing span {span:?}");
}

fn make_event(
    chapter: u32,
    idx: usize,
    text: &str,
    doc: &Document,
    predicate: &str,
    zone: ContextZone,
    factuality: Factuality,
) -> CandidateEvent {
    let pred_span = find_char_span(text, predicate);
    let (paragraph, sentence, clause, clause_span) = find_clause_for_span(doc, pred_span);
    let pred_token = doc
        .tokens
        .iter()
        .position(|t| {
            t.span.start_char <= pred_span.start_char && pred_span.end_char <= t.span.end_char
        })
        .unwrap_or(0);
    CandidateEvent {
        event_id: format!("c{chapter:04}-e{idx:04}"),
        chapter,
        paragraph,
        sentence,
        clause,
        clause_span,
        predicate_token: pred_token,
        predicate_span: pred_span,
        zone,
        factuality,
        factuality_cue: None,
        arguments: Vec::new(),
        temporal_anchors: Vec::new(),
    }
}

#[test]
fn sequence_and_intra_sentence_flow_transitive_before() {
    let text = "林遠推開木門之後，走入石室，點燃油燈。";
    let doc = analyze_document(text, &Options::default());
    let events = vec![
        make_event(
            1,
            0,
            text,
            &doc,
            "推開",
            ContextZone::Narrative,
            Factuality::Actual,
        ),
        make_event(
            1,
            1,
            text,
            &doc,
            "走入",
            ContextZone::Narrative,
            Factuality::Actual,
        ),
        make_event(
            1,
            2,
            text,
            &doc,
            "點燃",
            ContextZone::Narrative,
            Factuality::Actual,
        ),
    ];

    let build = build_chapter_graph(1, text, &doc, &events);
    assert!(!build.dag.has_cycle());
    assert_eq!(build.conflict_cycles_broken, 0);
    assert_eq!(build.event_dag_layers, vec![Some(0), Some(1), Some(2)]);

    assert_eq!(build.dag.query_order(0, 1), ChronologyOrder::Before);
    assert_eq!(build.dag.query_order(1, 2), ChronologyOrder::Before);
    assert_eq!(build.dag.query_order(0, 2), ChronologyOrder::Before);
    assert_eq!(build.dag.query_order(2, 0), ChronologyOrder::After);

    assert!(build.edges.iter().any(|e| {
        e.from_event == 0
            && e.to_event == 1
            && e.kind == EdgeKind::Before
            && e.evidence_kind == EdgeEvidenceKind::ExplicitDiscourseRelation
            && e.confidence_tier == 1
            && e.status == EdgeStatus::Supported
    }));
    assert!(build.edges.iter().any(|e| {
        e.from_event == 1
            && e.to_event == 2
            && e.kind == EdgeKind::Before
            && e.evidence_kind == EdgeEvidenceKind::IntraSentenceClauseFlow
            && e.confidence_tier == 4
            && e.status == EdgeStatus::Supported
    }));
}

#[test]
fn independent_paragraphs_without_shift_are_incomparable() {
    let text = "林遠在書房整理古籍。\n\n蘇晴在庭院中練劍。";
    let doc = analyze_document(text, &Options::default());
    let events = vec![
        make_event(
            1,
            0,
            text,
            &doc,
            "整理",
            ContextZone::Narrative,
            Factuality::Actual,
        ),
        make_event(
            1,
            1,
            text,
            &doc,
            "練劍",
            ContextZone::Narrative,
            Factuality::Actual,
        ),
    ];

    let build = build_chapter_graph(1, text, &doc, &events);
    assert!(!build.dag.has_cycle());
    assert!(build.edges.is_empty());
    assert_eq!(build.event_dag_layers, vec![Some(0), Some(0)]);
    assert_eq!(build.dag.query_order(0, 1), ChronologyOrder::Incomparable);
    assert_eq!(build.dag.query_order(1, 0), ChronologyOrder::Incomparable);
}

#[test]
fn simultaneous_equivalence_and_before_transitivity() {
    let text = "林遠仔細檢查門鎖。與此同時，蘇晴守在石梯旁。隨後兩人一同走進地下密室。";
    let doc = analyze_document(text, &Options::default());
    let events = vec![
        make_event(
            1,
            0,
            text,
            &doc,
            "檢查",
            ContextZone::Narrative,
            Factuality::Actual,
        ),
        make_event(
            1,
            1,
            text,
            &doc,
            "守在",
            ContextZone::Narrative,
            Factuality::Actual,
        ),
        make_event(
            1,
            2,
            text,
            &doc,
            "走進",
            ContextZone::Narrative,
            Factuality::Actual,
        ),
    ];

    let build = build_chapter_graph(1, text, &doc, &events);
    assert!(!build.dag.has_cycle());
    assert_eq!(build.dag.query_order(0, 1), ChronologyOrder::Simultaneous);
    assert_eq!(build.dag.query_order(1, 2), ChronologyOrder::Before);
    assert_eq!(build.dag.query_order(0, 2), ChronologyOrder::Before);
    assert_eq!(build.dag.query_order(2, 0), ChronologyOrder::After);

    let l0 = build.event_dag_layers[0].expect("event 0 layer");
    let l1 = build.event_dag_layers[1].expect("event 1 layer");
    let l2 = build.event_dag_layers[2].expect("event 2 layer");
    assert_eq!(l0, l1);
    assert!(l1 < l2);
}

#[test]
fn backward_causal_orients_cause_before_effect() {
    let text = "之所以失敗，是因為機關生鏽。";
    let doc = analyze_document(text, &Options::default());
    let events = vec![
        make_event(
            1,
            0,
            text,
            &doc,
            "失敗",
            ContextZone::Narrative,
            Factuality::Actual,
        ),
        make_event(
            1,
            1,
            text,
            &doc,
            "生鏽",
            ContextZone::Narrative,
            Factuality::Actual,
        ),
    ];

    let build = build_chapter_graph(1, text, &doc, &events);
    assert!(!build.dag.has_cycle());
    assert_eq!(build.dag.query_order(1, 0), ChronologyOrder::Before);
    assert_eq!(build.dag.query_order(0, 1), ChronologyOrder::After);
    assert_eq!(build.event_dag_layers, vec![Some(1), Some(0)]);
    assert!(build.edges.iter().any(|e| {
        e.kind == EdgeKind::Cause
            && e.from_event == 1
            && e.to_event == 0
            && e.status == EdgeStatus::Supported
    }));
}

#[test]
fn carbon_false_positive_filters_reject_monoclausal_tongyang_and_determiner_lingwai() {
    // 1. Monoclausal `S，同樣 V`
    let text1 = "林遠，同樣點了點頭。";
    let doc1 = analyze_document(text1, &Options::default());
    let events1 = vec![make_event(
        1,
        0,
        text1,
        &doc1,
        "點了點頭",
        ContextZone::Narrative,
        Factuality::Actual,
    )];
    let build1 = build_chapter_graph(1, text1, &doc1, &events1);
    assert!(
        !build1
            .edges
            .iter()
            .any(|e| e.kind == EdgeKind::Simultaneous),
        "monoclausal 同樣 must not emit Simultaneous edge: {:?}",
        build1.edges
    );

    // 2. Determiner `另外三名守衛守在門外` following another sentence in the same paragraph
    let text2 = "陳伯坐在大廳內喝茶。另外三名守衛守在門外。";
    let doc2 = analyze_document(text2, &Options::default());
    let events2 = vec![
        make_event(
            1,
            0,
            text2,
            &doc2,
            "喝茶",
            ContextZone::Narrative,
            Factuality::Actual,
        ),
        make_event(
            1,
            1,
            text2,
            &doc2,
            "守在",
            ContextZone::Narrative,
            Factuality::Actual,
        ),
    ];
    let build2 = build_chapter_graph(1, text2, &doc2, &events2);
    assert!(
        !build2
            .edges
            .iter()
            .any(|e| e.kind == EdgeKind::Simultaneous),
        "determiner 另外三名 must not emit Simultaneous edge: {:?}",
        build2.edges
    );
    assert_eq!(build2.dag.query_order(0, 1), ChronologyOrder::Incomparable);
}

#[test]
fn flashback_guard_prevents_forward_edge_after_present_event() {
    let text = "林遠打開木匣。\n\n三年前，師父曾來過此地，並留下這面古銅鏡。";
    let doc = analyze_document(text, &Options::default());
    let events = vec![
        make_event(
            1,
            0,
            text,
            &doc,
            "打開",
            ContextZone::Narrative,
            Factuality::Actual,
        ),
        make_event(
            1,
            1,
            text,
            &doc,
            "來過",
            ContextZone::Narrative,
            Factuality::Actual,
        ),
        make_event(
            1,
            2,
            text,
            &doc,
            "留下",
            ContextZone::Narrative,
            Factuality::Actual,
        ),
    ];

    let build = build_chapter_graph(1, text, &doc, &events);
    assert!(!build.dag.has_cycle());
    assert_eq!(build.dag.query_order(0, 1), ChronologyOrder::Incomparable);
    assert_eq!(build.dag.query_order(0, 2), ChronologyOrder::Incomparable);
    assert!(!build.edges.iter().any(|e| {
        e.evidence_kind == EdgeEvidenceKind::TemporalAnchorShift
            || (e.from_event == 0 && e.to_event == 1)
    }));
}

#[test]
fn temporal_anchor_shift_forward_advances_chronology() {
    let text = "林遠離開青石鎮。\n\n三日後，他抵達山腰客棧。";
    let doc = analyze_document(text, &Options::default());
    let events = vec![
        make_event(
            1,
            0,
            text,
            &doc,
            "離開",
            ContextZone::Narrative,
            Factuality::Actual,
        ),
        make_event(
            1,
            1,
            text,
            &doc,
            "抵達",
            ContextZone::Narrative,
            Factuality::Actual,
        ),
    ];

    let build = build_chapter_graph(1, text, &doc, &events);
    assert!(!build.dag.has_cycle());
    assert_eq!(build.dag.query_order(0, 1), ChronologyOrder::Before);
    assert_eq!(build.event_dag_layers, vec![Some(0), Some(1)]);
    assert!(build.edges.iter().any(|e| {
        e.from_event == 0
            && e.to_event == 1
            && e.kind == EdgeKind::Before
            && e.evidence_kind == EdgeEvidenceKind::TemporalAnchorShift
            && e.confidence_tier == 3
            && e.status == EdgeStatus::Supported
    }));
}

#[test]
fn factuality_and_quote_gate_isolates_non_actual_events() {
    let text = "「我們明日出發，」陳伯說道。因為他沒有找到鑰匙，所以無法開門。如果天候轉晴，他將動身前往北門。";
    let doc = analyze_document(text, &Options::default());
    let events = vec![
        make_event(
            1,
            0,
            text,
            &doc,
            "出發",
            ContextZone::DirectQuote { quote_index: 0 },
            Factuality::Quoted,
        ),
        make_event(
            1,
            1,
            text,
            &doc,
            "說道",
            ContextZone::Narrative,
            Factuality::Actual,
        ),
        make_event(
            1,
            2,
            text,
            &doc,
            "找到",
            ContextZone::Narrative,
            Factuality::Negated,
        ),
        make_event(
            1,
            3,
            text,
            &doc,
            "開門",
            ContextZone::Narrative,
            Factuality::Negated,
        ),
        make_event(
            1,
            4,
            text,
            &doc,
            "轉晴",
            ContextZone::Narrative,
            Factuality::Hypothetical,
        ),
        make_event(
            1,
            5,
            text,
            &doc,
            "動身",
            ContextZone::Narrative,
            Factuality::ModalOrFuture,
        ),
    ];

    let build = build_chapter_graph(1, text, &doc, &events);
    assert!(!build.dag.has_cycle());

    // Only event 1 (`說道`) is Narrative + Actual.
    assert_eq!(build.event_dag_layers[0], None);
    assert_eq!(build.event_dag_layers[1], Some(0));
    assert_eq!(build.event_dag_layers[2], None);
    assert_eq!(build.event_dag_layers[3], None);
    assert_eq!(build.event_dag_layers[4], None);
    assert_eq!(build.event_dag_layers[5], None);

    // Explicit Cause / Condition edges touching non-Actual events remain inspectable with NonChronologicalFactuality.
    assert!(
        build.edges.iter().any(
            |e| e.status == EdgeStatus::NonChronologicalFactuality && e.kind == EdgeKind::Cause
        )
    );
    assert!(build.edges.iter().any(
        |e| e.status == EdgeStatus::NonChronologicalFactuality && e.kind == EdgeKind::Condition
    ));
    assert!(
        build
            .edges
            .iter()
            .all(|e| e.status != EdgeStatus::Supported)
    );

    for i in [0usize, 2, 3, 4, 5] {
        assert_eq!(build.dag.query_order(1, i), ChronologyOrder::Incomparable);
        assert_eq!(build.dag.query_order(i, 1), ChronologyOrder::Incomparable);
    }
    assert_eq!(build.dag.query_order(2, 3), ChronologyOrder::Incomparable);
    assert_eq!(build.dag.query_order(4, 5), ChronologyOrder::Incomparable);
}

#[test]
fn adversarial_contradictory_cycles_are_deterministically_demoted() {
    // 1. Tier 1 vs Tier 4 cycle: 3 clauses `A, B, C` where Tier 1 says `C -> A` (e.g. `之前`),
    // while Tier 4 says `A -> B` and `B -> C`. Tier 4 edge `B -> C` must be demoted.
    let text = "甲先入屋，乙緊隨其後，丙進入院子。";
    let mut doc = analyze_document(text, &Options::default());
    let events = vec![
        make_event(
            1,
            0,
            text,
            &doc,
            "入屋",
            ContextZone::Narrative,
            Factuality::Actual,
        ),
        make_event(
            1,
            1,
            text,
            &doc,
            "緊隨",
            ContextZone::Narrative,
            Factuality::Actual,
        ),
        make_event(
            1,
            2,
            text,
            &doc,
            "進入",
            ContextZone::Narrative,
            Factuality::Actual,
        ),
    ];

    // Inject an explicit Tier 1 `Sequence` (`C 之後，A` -> `C Before A`) spanning clause 2 -> clause 0.
    doc.relations.push(Relation {
        kind: RelationKind::Sequence,
        level: RelationLevel::Clause,
        connector: find_char_span(text, "進入"),
        correlate: None,
        arg1: events[2].clause_span,
        arg2: events[0].clause_span,
    });

    let build = build_chapter_graph(1, text, &doc, &events);
    assert!(!build.dag.has_cycle());
    assert_eq!(build.conflict_cycles_broken, 1);
    assert_eq!(build.cycle_unknowns.len(), 1);
    assert_eq!(
        build.cycle_unknowns[0].kind,
        NarrativeUnknownKind::CycleConflictDemoted
    );

    // Tier 1 edge (2 -> 0) and first Tier 4 edge (0 -> 1) survive; second Tier 4 edge (1 -> 2) is demoted.
    assert!(build.edges.iter().any(|e| {
        e.from_event == 2
            && e.to_event == 0
            && e.confidence_tier == 1
            && e.status == EdgeStatus::Supported
    }));
    assert!(build.edges.iter().any(|e| {
        e.from_event == 1
            && e.to_event == 2
            && e.confidence_tier == 4
            && e.status == EdgeStatus::ConflictCycle
    }));
    assert_eq!(build.dag.query_order(2, 0), ChronologyOrder::Before);
    assert_eq!(build.dag.query_order(0, 1), ChronologyOrder::Before);
    assert_eq!(build.dag.query_order(2, 1), ChronologyOrder::Before);

    // 2. Simultaneous vs strict precedence conflict in deserialized `ChapterNarrative`.
    let chapter_nar = ChapterNarrative {
        schema: CHAPTER_NARRATIVE_SCHEMA.to_owned(),
        chapter: 1,
        title: None,
        text_chars: text.chars().count(),
        text_sha256: doc.text_sha256.clone(),
        dialogues: Vec::new(),
        events: events.clone(),
        mentions: Vec::new(),
        edges: vec![
            faberline_nlp_narrative::schema::TimelineEdge {
                edge_id: "c0001-g0000".to_owned(),
                kind: EdgeKind::Before,
                evidence_kind: EdgeEvidenceKind::ExplicitDiscourseRelation,
                status: EdgeStatus::Supported,
                from_event: 0,
                to_event: 1,
                connector_span: None,
                evidence_spans: Vec::new(),
                confidence_tier: 1,
            },
            faberline_nlp_narrative::schema::TimelineEdge {
                edge_id: "c0001-g0001".to_owned(),
                kind: EdgeKind::Simultaneous,
                evidence_kind: EdgeEvidenceKind::ExplicitDiscourseRelation,
                status: EdgeStatus::Supported,
                from_event: 0,
                to_event: 1,
                connector_span: None,
                evidence_spans: Vec::new(),
                confidence_tier: 2,
            },
        ],
        event_dag_layers: vec![Some(0), Some(1), Some(0)],
        conflict_cycles_broken: 0,
        unknown: Vec::new(),
    };

    let dag = build_chapter_dag_from_narrative(&chapter_nar);
    assert!(!dag.has_cycle());
    assert_eq!(dag.query_order(0, 1), ChronologyOrder::Before);
}

#[test]
fn pre_event_before_frame_reverses_sequence() {
    let text = "在離開客棧之前，林遠檢查了行囊。";
    let doc = analyze_document(text, &Options::default());
    let events = vec![
        make_event(
            1,
            0,
            text,
            &doc,
            "離開",
            ContextZone::Narrative,
            Factuality::Actual,
        ),
        make_event(
            1,
            1,
            text,
            &doc,
            "檢查",
            ContextZone::Narrative,
            Factuality::Actual,
        ),
    ];

    let build = build_chapter_graph(1, text, &doc, &events);
    assert!(!build.dag.has_cycle());
    assert_eq!(build.dag.query_order(1, 0), ChronologyOrder::Before);
    assert_eq!(build.event_dag_layers, vec![Some(1), Some(0)]);
}

#[test]
fn book_timeline_graph_cross_chapter_and_flashback_boundary() {
    let text1 = "林遠走出青石鎮。";
    let doc1 = analyze_document(text1, &Options::default());
    let events1 = vec![make_event(
        1,
        0,
        text1,
        &doc1,
        "走出",
        ContextZone::Narrative,
        Factuality::Actual,
    )];
    let g1 = build_chapter_graph(1, text1, &doc1, &events1);
    let ch1 = ChapterNarrative {
        schema: CHAPTER_NARRATIVE_SCHEMA.to_owned(),
        chapter: 1,
        title: Some("Chapter 1".to_owned()),
        text_chars: doc1.text_chars,
        text_sha256: doc1.text_sha256,
        dialogues: Vec::new(),
        events: events1,
        mentions: Vec::new(),
        edges: g1.edges,
        event_dag_layers: g1.event_dag_layers,
        conflict_cycles_broken: g1.conflict_cycles_broken,
        unknown: g1.cycle_unknowns,
    };

    let text2 = "他抵達山腳茶館。";
    let doc2 = analyze_document(text2, &Options::default());
    let events2 = vec![make_event(
        2,
        0,
        text2,
        &doc2,
        "抵達",
        ContextZone::Narrative,
        Factuality::Actual,
    )];
    let g2 = build_chapter_graph(2, text2, &doc2, &events2);
    let ch2 = ChapterNarrative {
        schema: CHAPTER_NARRATIVE_SCHEMA.to_owned(),
        chapter: 2,
        title: Some("Chapter 2".to_owned()),
        text_chars: doc2.text_chars,
        text_sha256: doc2.text_sha256,
        dialogues: Vec::new(),
        events: events2,
        mentions: Vec::new(),
        edges: g2.edges,
        event_dag_layers: g2.event_dag_layers,
        conflict_cycles_broken: g2.conflict_cycles_broken,
        unknown: g2.cycle_unknowns,
    };

    // Chapter 3 opens with a flashback marker (`往事回憶` + retrospective temporal anchor).
    let text3 = "數年前，陳伯曾在此地開過一間茶館。";
    let doc3 = analyze_document(text3, &Options::default());
    let mut ev3 = make_event(
        3,
        0,
        text3,
        &doc3,
        "開過",
        ContextZone::Narrative,
        Factuality::Actual,
    );
    ev3.temporal_anchors.push(TemporalAnchor {
        entity_index: 0,
        kind: EntityKind::RelativeDate,
        span: find_char_span(text3, "數年前"),
        value: Some("-3 years".to_owned()),
        unit: Some("年前".to_owned()),
    });
    let events3 = vec![ev3];
    let g3 = build_chapter_graph(3, text3, &doc3, &events3);
    let ch3 = ChapterNarrative {
        schema: CHAPTER_NARRATIVE_SCHEMA.to_owned(),
        chapter: 3,
        title: Some("往事回憶".to_owned()),
        text_chars: doc3.text_chars,
        text_sha256: doc3.text_sha256,
        dialogues: Vec::new(),
        events: events3,
        mentions: Vec::new(),
        edges: g3.edges,
        event_dag_layers: g3.event_dag_layers,
        conflict_cycles_broken: g3.conflict_cycles_broken,
        unknown: g3.cycle_unknowns,
    };

    let book = BookTimelineGraph::from_chapters(&[ch3, ch1, ch2]);
    let r1 = BookEventRef {
        chapter: 1,
        event_index: 0,
    };
    let r2 = BookEventRef {
        chapter: 2,
        event_index: 0,
    };
    let r3 = BookEventRef {
        chapter: 3,
        event_index: 0,
    };

    assert_eq!(book.query_order(r1, r2), ChronologyOrder::Before);
    assert_eq!(book.query_order(r2, r1), ChronologyOrder::After);
    assert_eq!(book.query_order(r2, r3), ChronologyOrder::Incomparable);
    assert_eq!(book.query_order(r1, r3), ChronologyOrder::Incomparable);
}

#[test]
fn english_temporal_shift_flashback_and_meanwhile() {
    let text = "Lin opened the wooden gate.\n\nThree days later, he reached the mountain inn. Meanwhile, Su watched the road.\n\nYears ago, Master visited this valley.";
    let doc = analyze_document(text, &Options::default());
    let events = vec![
        make_event(
            1,
            0,
            text,
            &doc,
            "opened",
            ContextZone::Narrative,
            Factuality::Actual,
        ),
        make_event(
            1,
            1,
            text,
            &doc,
            "reached",
            ContextZone::Narrative,
            Factuality::Actual,
        ),
        make_event(
            1,
            2,
            text,
            &doc,
            "watched",
            ContextZone::Narrative,
            Factuality::Actual,
        ),
        make_event(
            1,
            3,
            text,
            &doc,
            "visited",
            ContextZone::Narrative,
            Factuality::Actual,
        ),
    ];

    let build = build_chapter_graph(1, text, &doc, &events);
    assert!(!build.dag.has_cycle());
    assert_eq!(build.dag.query_order(0, 1), ChronologyOrder::Before);
    assert_eq!(build.dag.query_order(1, 2), ChronologyOrder::Simultaneous);
    assert_eq!(build.dag.query_order(0, 2), ChronologyOrder::Before);
    assert_eq!(build.dag.query_order(2, 3), ChronologyOrder::Incomparable);
    assert_eq!(build.dag.query_order(0, 3), ChronologyOrder::Incomparable);
}

#[test]
fn simultaneous_bridge_transitivity_and_degenerate_relation_hygiene() {
    let text = "甲打開門，乙隨後進屋。丙站在院中，丁接著離開。";
    let mut doc = analyze_document(text, &Options::default());
    let events = vec![
        make_event(
            1,
            0,
            text,
            &doc,
            "打開",
            ContextZone::Narrative,
            Factuality::Actual,
        ),
        make_event(
            1,
            1,
            text,
            &doc,
            "進屋",
            ContextZone::Narrative,
            Factuality::Actual,
        ),
        make_event(
            1,
            2,
            text,
            &doc,
            "站在",
            ContextZone::Narrative,
            Factuality::Actual,
        ),
        make_event(
            1,
            3,
            text,
            &doc,
            "離開",
            ContextZone::Narrative,
            Factuality::Actual,
        ),
    ];

    // Add valid Simultaneous `乙 ~ 丙` (1 ~ 2), bridging `0 < 1 ~ 2 < 3` => `0 < 3`!
    doc.relations.push(Relation {
        kind: RelationKind::Parallel,
        level: RelationLevel::Clause,
        connector: find_char_span(text, "站在"),
        correlate: None,
        arg1: events[1].clause_span,
        arg2: events[2].clause_span,
    });

    // Add degenerate overlapping / empty-after-strip relations that must be ignored cleanly.
    doc.relations.push(Relation {
        kind: RelationKind::Sequence,
        level: RelationLevel::Clause,
        connector: events[3].clause_span,
        correlate: None,
        arg1: events[3].clause_span,
        arg2: events[0].clause_span,
    });
    doc.relations.push(Relation {
        kind: RelationKind::Sequence,
        level: RelationLevel::Clause,
        connector: Span {
            start_char: 0,
            end_char: 1,
        },
        correlate: None,
        arg1: Span {
            start_char: 0,
            end_char: 5,
        },
        arg2: Span {
            start_char: 2,
            end_char: 8,
        },
    });

    let build = build_chapter_graph(1, text, &doc, &events);
    assert!(!build.dag.has_cycle());
    assert_eq!(build.conflict_cycles_broken, 0);
    assert_eq!(build.dag.query_order(1, 2), ChronologyOrder::Simultaneous);
    assert_eq!(build.dag.query_order(0, 3), ChronologyOrder::Before);
    assert_eq!(
        build.event_dag_layers,
        vec![Some(0), Some(1), Some(1), Some(2)]
    );
}

#[test]
fn contrast_and_condition_do_not_assert_chronological_before() {
    let text = "雖然天色已暗，但是林遠還是趕往鎮口。";
    let doc = analyze_document(text, &Options::default());
    let events = vec![
        make_event(
            1,
            0,
            text,
            &doc,
            "暗",
            ContextZone::Narrative,
            Factuality::Actual,
        ),
        make_event(
            1,
            1,
            text,
            &doc,
            "趕往",
            ContextZone::Narrative,
            Factuality::Actual,
        ),
    ];

    let build = build_chapter_graph(1, text, &doc, &events);
    assert!(!build.dag.has_cycle());
    assert!(
        build
            .edges
            .iter()
            .any(|e| { e.kind == EdgeKind::Contrast && e.status == EdgeStatus::Supported })
    );
    assert_eq!(build.dag.query_order(0, 1), ChronologyOrder::Incomparable);
}
