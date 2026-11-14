use faberline_nlp::doc::{EntityKind, Options, RelationKind, analyze_document};
use faberline_nlp_narrative::{
    AliasEntry, ArgumentBasis, EdgeKind, EventRole, Factuality, MentionKind, SpeakerBasis,
    WorkVocabulary, analyze_chapter_narrative,
};

fn span_str(text: &str, start: usize, end: usize) -> String {
    text.chars().skip(start).take(end - start).collect()
}

fn ja_vocab() -> WorkVocabulary {
    WorkVocabulary {
        actors: vec![
            AliasEntry {
                surface: "美咲".to_string(),
                canonical: "美咲".to_string(),
            },
            AliasEntry {
                surface: "健太".to_string(),
                canonical: "健太".to_string(),
            },
        ],
        places: vec![AliasEntry {
            surface: "図書館".to_string(),
            canonical: "図書館".to_string(),
        }],
        objects: vec![AliasEntry {
            surface: "手紙".to_string(),
            canonical: "手紙".to_string(),
        }],
        domain_terms: Vec::new(),
    }
}

fn annotate_ja(text: &str, vocab: &WorkVocabulary) -> faberline_nlp::doc::Document {
    let mut opts = vocab.to_doc_options();
    opts.cpu_syntax = true;
    analyze_document(text, &opts)
}

#[test]
fn ja_honorific_entities_and_colloquial_discourse_layers() {
    let text = "美咲さんは図書館で健太くんと会った。佐藤先生も来た。でも、雨が降った。";
    let opts = Options {
        cpu_syntax: true,
        ..Options::default()
    };
    let doc = analyze_document(text, &opts);

    let names: Vec<String> = doc
        .entities
        .iter()
        .filter(|e| e.kind == EntityKind::Name)
        .map(|e| span_str(text, e.span.start_char, e.span.end_char))
        .collect();
    assert!(
        names.iter().any(|n| n == "美咲さん"),
        "expected 美咲さん as Name entity, got {names:?}"
    );
    assert!(
        names.iter().any(|n| n == "健太くん"),
        "expected 健太くん as Name entity, got {names:?}"
    );
    assert!(
        names.iter().any(|n| n == "佐藤先生"),
        "expected 佐藤先生 as Name entity, got {names:?}"
    );

    assert!(
        doc.relations
            .iter()
            .any(|r| r.kind == RelationKind::Contrast
                && span_str(text, r.connector.start_char, r.connector.end_char) == "でも"),
        "expected Contrast relation for でも, got {:?}",
        doc.relations
    );
}

#[test]
fn ja_dialogue_attribution_and_romance_scene() {
    let vocab = ja_vocab();
    let text = concat!(
        "「遅くなってごめんね」と美咲さんは囁いた。\n",
        "健太くんは微笑んだ。「大丈夫だよ」と彼は答えた。\n",
        "「もっとそばに来て」美咲は小さく喘いだ。彼女はそっと目を閉じた。\n",
        "健太は言った。「ずっとそばにいるよ」",
    );
    let doc = annotate_ja(text, &vocab);
    let narrative = analyze_chapter_narrative(1, Some("第一章".into()), text, &doc, &vocab)
        .expect("analyze_chapter_narrative should succeed");

    assert_eq!(narrative.dialogues.len(), 4);

    assert_eq!(
        narrative.dialogues[0].basis,
        SpeakerBasis::PostQuoteSpeechVerb
    );
    assert_eq!(
        narrative.dialogues[0].speaker_canonical.as_deref(),
        Some("美咲")
    );

    assert_eq!(
        narrative.dialogues[1].basis,
        SpeakerBasis::PostQuoteSpeechVerb
    );
    let d1_speaker = narrative.dialogues[1]
        .speaker_span
        .map(|s| span_str(text, s.start_char, s.end_char));
    assert_eq!(d1_speaker.as_deref(), Some("彼"));

    assert_eq!(
        narrative.dialogues[2].basis,
        SpeakerBasis::PostQuoteSpeechVerb
    );
    assert_eq!(
        narrative.dialogues[2].speaker_canonical.as_deref(),
        Some("美咲")
    );

    assert_eq!(
        narrative.dialogues[3].basis,
        SpeakerBasis::PreQuoteSpeechVerb
    );
    assert_eq!(
        narrative.dialogues[3].speaker_canonical.as_deref(),
        Some("健太")
    );
}

#[test]
fn ja_pronoun_gender_separation_and_alias_normalization() {
    let vocab = ja_vocab();
    let text = concat!(
        "美咲ちゃんは手紙を開いた。健太くんは窓の外を見た。彼女は静かに微笑んだ。\n",
        "美咲さんは部屋を出て、廊下を歩いた。",
    );
    let doc = annotate_ja(text, &vocab);
    let narrative = analyze_chapter_narrative(1, None, text, &doc, &vocab)
        .expect("analyze_chapter_narrative should succeed");

    let kanojo = narrative
        .mentions
        .iter()
        .find(|m| {
            m.kind == MentionKind::Pronoun
                && span_str(text, m.span.start_char, m.span.end_char) == "彼女"
        })
        .expect("expected 彼女 pronoun mention");
    let ant_span = kanojo
        .resolved_antecedent
        .expect("彼女 should resolve to 美咲ちゃん, not 健太くん");
    let ant_text = span_str(text, ant_span.start_char, ant_span.end_char);
    assert!(
        ant_text.contains("美咲"),
        "expected 彼女 to resolve to 美咲, got {ant_text}"
    );

    let misaki_mentions: Vec<_> = narrative
        .mentions
        .iter()
        .filter(|m| m.canonical.as_deref() == Some("美咲"))
        .collect();
    assert!(
        misaki_mentions.len() >= 2,
        "expected normalized 美咲 mentions across 美咲ちゃん/美咲さん, got {misaki_mentions:?}"
    );

    let pro_drop_hits: Vec<_> = narrative
        .mentions
        .iter()
        .filter(|m| m.kind == MentionKind::ZeroPronounProDrop)
        .collect();
    assert!(
        !pro_drop_hits.is_empty(),
        "expected pro-drop subject inheritance in `美咲さんは部屋を出て、廊下を歩いた。`"
    );
}

#[test]
fn ja_factuality_passive_and_timeline_graph() {
    let vocab = ja_vocab();
    let text = concat!(
        "美咲は手紙を読まなかった。健太は明日帰りたい。\n",
        "もし雨が降ったら、試合は中止になる。\n",
        "健太は美咲に押された。\n",
        "雨が降ったので、美咲は傘をさした。同時に健太は駅へ走った。\n",
        "以前、二人はここで出会った。",
    );
    let doc = annotate_ja(text, &vocab);
    let narrative = analyze_chapter_narrative(2, None, text, &doc, &vocab)
        .expect("analyze_chapter_narrative should succeed");

    assert!(
        narrative
            .events
            .iter()
            .any(|e| e.factuality == Factuality::Negated),
        "expected Negated event for 読まなかった: {:?}",
        narrative.events
    );
    assert!(
        narrative
            .events
            .iter()
            .any(|e| e.factuality == Factuality::ModalOrFuture),
        "expected ModalOrFuture event for 帰りたい: {:?}",
        narrative.events
    );
    assert!(
        narrative
            .events
            .iter()
            .any(|e| e.factuality == Factuality::Hypothetical),
        "expected Hypothetical event for もし/降ったら: {:?}",
        narrative.events
    );

    let passive_ev = narrative.events.iter().find(|e| {
        span_str(text, e.clause_span.start_char, e.clause_span.end_char).contains("押された")
    });
    if let Some(ev) = passive_ev {
        let has_patient_kenta = ev.arguments.iter().any(|a| {
            a.role == EventRole::Patient
                && a.basis == ArgumentBasis::PassiveBei
                && span_str(text, a.span.start_char, a.span.end_char).contains("健太")
        });
        assert!(has_patient_kenta, "expected passive patient 健太 in {ev:?}");
    }

    assert!(
        narrative.edges.iter().any(|e| e.kind == EdgeKind::Cause),
        "expected Cause edge for ので, got {:?}",
        narrative.edges
    );
    assert!(
        narrative
            .edges
            .iter()
            .any(|e| e.kind == EdgeKind::Simultaneous),
        "expected Simultaneous edge for 同時に, got {:?}",
        narrative.edges
    );
}
