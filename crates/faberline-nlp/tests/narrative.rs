use faberline_nlp::{Lexicon, NarrativeAnalyzer, Span, analyze, text::Locale};

fn original(text: &str, span: Span) -> String {
    text.chars()
        .skip(span.start_char)
        .take(span.end_char - span.start_char)
        .collect()
}

#[test]
fn attributes_nested_dialogue_without_looking_across_lines() -> anyhow::Result<()> {
    let lexicon = Lexicon {
        actors: &[("陳東", "陳東"), ("陈东", "陳東"), ("𠮷", "𠮷")],
        ..Lexicon::default()
    };
    for (locale, quote, tag, unbound_quote) in [
        (
            Locale::ChineseTaiwan,
            "「先走。記住『等我！』」",
            "陳東說道。",
            "「別回頭！」",
        ),
        (
            Locale::ChineseChina,
            "“先走。记住‘等我！’”",
            "陈东说道。",
            "“别回头！”",
        ),
    ] {
        let analyzer = NarrativeAnalyzer::new(locale, &lexicon)?;
        let text = format!("{quote}{tag}𠮷抬手。{unbound_quote}\r\n{tag}");
        let result = analyzer.analyze(&text, "");
        let dialogue = result
            .sections
            .iter()
            .flat_map(|section| &section.units)
            .filter(|unit| unit.kind == "dialogue")
            .map(|unit| (original(&text, unit.evidence), unit.speaker.as_deref()))
            .collect::<Vec<_>>();
        assert_eq!(
            dialogue,
            [
                (quote.to_owned(), Some("陳東")),
                (unbound_quote.to_owned(), None)
            ]
        );
        let action = result
            .events
            .iter()
            .find(|event| event.actor.as_deref() == Some("𠮷"))
            .unwrap();
        let start_char = quote.chars().count() + tag.chars().count();
        assert_eq!(
            action.evidence[0],
            Span {
                start_char,
                end_char: start_char + 4,
            }
        );
        assert_eq!(original(&text, action.evidence[0]), "𠮷抬手。");
    }
    Ok(())
}

#[test]
fn keeps_speaker_tags_and_retains_text_with_crossed_quotes() -> anyhow::Result<()> {
    let lexicon = Lexicon {
        actors: &[("岑棠", "岑棠")],
        ..Lexicon::default()
    };
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &lexicon)?;
    for text in [
        "「先走！」岑棠說道。",
        "「先走！」岑棠揮手，低聲說道。",
        "（「先走！」）岑棠說道。",
    ] {
        let result = analyzer.analyze(text, "");
        let first_quote = result
            .sections
            .iter()
            .flat_map(|section| &section.units)
            .find(|unit| unit.kind == "dialogue")
            .unwrap();
        assert_eq!(original(text, first_quote.evidence), "「先走！」");
        assert_eq!(first_quote.speaker.as_deref(), Some("岑棠"), "{text}");
    }

    // A tag immediately introducing the next quote does not prove the first
    // quote's speaker. Preserve both written quote spans and the explicit tag.
    let text = "「先走！」岑棠說道：「不要回頭！」";
    let result = analyzer.analyze(text, "");
    let quotes: Vec<_> = result
        .sections
        .iter()
        .flat_map(|section| &section.units)
        .filter(|unit| unit.kind == "dialogue")
        .collect();
    assert_eq!(quotes.len(), 2);
    assert_eq!(original(text, quotes[0].evidence), "「先走！」");
    assert_eq!(quotes[0].speaker.as_deref(), None);
    assert_eq!(original(text, quotes[1].evidence), "「不要回頭！」");
    assert_eq!(quotes[1].speaker.as_deref(), Some("岑棠"));

    let text = "「『甲」乙』岑棠抬手。";
    let result = analyzer.analyze(text, "");
    assert!(
        result.events.is_empty(),
        "damaged text must not establish an actor or event"
    );
    let scoped = faberline_nlp::scoped_written_observations::inspect_with_analyzer(
        text, "", &analyzer, &lexicon,
    )
    .unwrap();
    assert_eq!(scoped.segments.len(), 1);
    let report = &scoped.segments[0].report;
    assert_eq!(report.quotations.len(), 1);
    assert_eq!(
        report.quotations[0].form,
        faberline_nlp::safe_written_observations::QuotationForm::IncompleteOrDamaged,
    );
    let quote = report.quotations[0].source;
    assert_eq!(&text[quote.byte_start..quote.byte_end], text);
    assert_eq!(report.quote_scan_diagnostics.len(), 1);
    let diagnostic = &report.quote_scan_diagnostics[0];
    assert_eq!(diagnostic.status, "unexamined");
    assert_eq!(diagnostic.reason, "crossed_delimiter");
    let suffix = diagnostic.source;
    assert_eq!(
        &text[suffix.byte_start..suffix.byte_end],
        "」乙』岑棠抬手。"
    );
    assert_eq!(
        suffix.scalars.start_char,
        text[..suffix.byte_start].chars().count()
    );
    assert_eq!(suffix.scalars.end_char, text.chars().count());
    Ok(())
}

#[test]
fn carries_a_pronoun_past_a_line_footer_and_ignores_the_chapter_footer() -> anyhow::Result<()> {
    let lexicon = Lexicon {
        actors: &[("陳東", "陳東"), ("沈牧", "沈牧")],
        objects: &[("令牌", "令牌")],
        footers: &["<!end-chapter>"],
        line_footers: &["<LINE>"],
        ..Lexicon::default()
    };
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &lexicon)?;
    let text = "\u{feff}試章\r\n陳東抬手。<LINE>沈牧收起令牌。\r\n他取出令牌。<!END-CHAPTER>\r\n沈牧抬手。";
    let result = analyzer.analyze(text, "試章");
    let units = result
        .sections
        .iter()
        .flat_map(|section| &section.units)
        .map(|unit| original(text, unit.evidence))
        .collect::<Vec<_>>();
    assert_eq!(units, ["陳東抬手。", "他取出令牌。"]);
    assert_eq!(result.events.len(), 2);
    let continued = &result.events[1];
    assert_eq!(continued.actor.as_deref(), Some("陳東"));
    assert_eq!(continued.object.as_deref(), Some("令牌"));
    assert_eq!(original(text, continued.evidence[1]), "陳東");
    Ok(())
}

#[test]
fn protects_a_place_token_that_contains_an_actor_name() -> anyhow::Result<()> {
    let lexicon = Lexicon {
        actors: &[("青木", "青木")],
        places: &["青木城"],
        ..Lexicon::default()
    };
    let text = "青木城出現。青木抬手。";
    let legacy = analyze(text, "", &lexicon);
    assert_eq!(legacy.events[0].actor.as_deref(), Some("青木"));

    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &lexicon)?;
    let result = analyzer.analyze(text, "");
    assert_eq!(result.events.len(), 2);
    assert_eq!(result.events[0].actor, None);
    assert_eq!(result.events[1].actor.as_deref(), Some("青木"));
    Ok(())
}

#[test]
fn uses_locale_sentence_breaks_in_event_analysis() -> anyhow::Result<()> {
    let lexicon = Lexicon {
        actors: &[("青木", "青木"), ("沈牧", "沈牧")],
        ..Lexicon::default()
    };
    let text = "青木抬手. 沈牧出現。";
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &lexicon)?;
    let result = analyzer.analyze(text, "");
    let events = result
        .events
        .iter()
        .map(|event| (event.actor.as_deref(), original(text, event.evidence[0])))
        .collect::<Vec<_>>();
    assert_eq!(
        events,
        [
            (Some("青木"), "青木抬手.".to_owned()),
            (Some("沈牧"), "沈牧出現。".to_owned()),
        ]
    );
    assert_eq!(analyze(text, "", &lexicon).events.len(), 1);
    Ok(())
}

#[test]
fn links_a_passive_action_to_the_transformed_object() -> anyhow::Result<()> {
    let lexicon = Lexicon {
        actors: &[("岑棠", "岑棠")],
        ..Lexicon::default()
    };
    for (locale, transformation, action) in [
        (Locale::ChineseTaiwan, "一撮礦砂凝成薄片，", "隨即被取出。"),
        (Locale::ChineseChina, "一撮矿砂凝成薄片，", "随即被取出。"),
    ] {
        let source = format!("𠮷。岑棠抬手，{transformation}{action}");
        let analyzer = NarrativeAnalyzer::new(locale, &lexicon)?;
        let result = analyzer.analyze(&source, "");
        let event = result
            .events
            .iter()
            .find(|event| event.action == action)
            .unwrap();
        assert_eq!(event.actor.as_deref(), Some("岑棠"));
        assert_eq!(event.object.as_deref(), Some("薄片"));
        assert_eq!(event.status, "occurred");
        assert_eq!(original(&source, event.evidence[0]), action);
        assert_eq!(original(&source, event.evidence[1]), "岑棠");
        assert_eq!(original(&source, event.evidence[2]), transformation);
        assert_eq!(
            analyze(&source, "", &lexicon).events.last().unwrap().object,
            None
        );
    }
    Ok(())
}

#[test]
fn requires_a_linked_transformation_and_an_omitted_passive_object() -> anyhow::Result<()> {
    let lexicon = Lexicon {
        actors: &[("岑棠", "岑棠")],
        ..Lexicon::default()
    };
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &lexicon)?;
    for source in [
        "岑棠抬手，一撮礦砂凝成薄片。隨即被取出。",
        "岑棠抬手，一撮礦砂凝成薄片；隨即被取出。",
        "岑棠抬手，一撮礦砂未凝成薄片，隨即被取出。",
        "岑棠抬手，一撮礦砂差點凝成薄片，隨即被取出。",
        "岑棠抬手，「一撮礦砂凝成薄片！」隨即被取出。",
        "岑棠抬手，一撮礦砂凝成薄片，陶珠被取出。",
        "岑棠抬手，一撮礦砂凝成薄片，隨即取出。",
        "岑棠抬手，一撮礦砂堆在桌上，隨即被取出。",
    ] {
        let result = analyzer.analyze(source, "");
        assert_eq!(result.events.last().unwrap().object, None, "{source}");
    }
    Ok(())
}

#[test]
fn reports_a_completed_first_person_action_with_source_evidence() -> anyhow::Result<()> {
    let lexicon = Lexicon {
        actors: &[("顧梨", "顧梨"), ("顾梨", "顧梨")],
        objects: &[("紙條", "紙條"), ("纸条", "紙條")],
        ..Lexicon::default()
    };
    for (locale, quote, claim, tag, speaker) in [
        (
            Locale::ChineseTaiwan,
            "「門邊風起，讓我碰巧取出了一張紙條……」",
            "讓我碰巧取出了一張紙條……",
            "顧梨停了一會，沒再說下去。",
            "顧梨",
        ),
        (
            Locale::ChineseChina,
            "“门边风起，让我碰巧取出了一张纸条……”",
            "让我碰巧取出了一张纸条……",
            "顾梨停了一会，没再说下去。",
            "顾梨",
        ),
    ] {
        let source = format!("𠮷。{quote}{tag}");
        let analyzer = NarrativeAnalyzer::new(locale, &lexicon)?;
        let result = analyzer.analyze(&source, "");
        let event = result
            .events
            .iter()
            .find(|event| event.status == "reported")
            .unwrap();
        // A following cessation tag is written evidence, not proof that the
        // named person spoke or that the named object participated.
        assert_eq!(event.actor.as_deref(), None);
        assert_eq!(event.object.as_deref(), None);
        assert_eq!(event.evidence.len(), 2);
        assert_eq!(original(&source, event.evidence[0]), claim);
        assert_eq!(original(&source, event.evidence[1]), tag);
        assert!(tag.starts_with(speaker));
        assert!(
            result
                .sections
                .iter()
                .flat_map(|section| &section.units)
                .any(|unit| {
                    unit.speaker.is_none() && original(&source, unit.evidence) == quote
                })
        );
        assert!(
            analyze(&source, "", &lexicon)
                .events
                .iter()
                .all(|event| event.status != "reported")
        );
    }
    Ok(())
}

#[test]
fn excludes_speech_without_a_directly_attributed_completed_action() -> anyhow::Result<()> {
    let lexicon = Lexicon {
        actors: &[("顧梨", "顧梨"), ("寧衡", "寧衡")],
        ..Lexicon::default()
    };
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &lexicon)?;
    for source in [
        "「我取出了救生繩。」",
        "「他取出了救生繩！」顧梨說道。",
        "「我明天要取出救生繩。」顧梨說道。",
        "「我取出了救生繩嗎？」顧梨說道。",
        "「我取出了救生繩，對嗎？」顧梨說道。",
        "「我取出過期的救生繩。」顧梨說道。",
        "「如果浪太大，我取出了救生繩。」顧梨說道。",
        "「我取出了救生繩。」顧梨聽寧衡說道。",
        "「我取出了救生繩。」\r\n顧梨說道。",
    ] {
        let result = analyzer.analyze(source, "");
        assert!(
            result.events.iter().all(|event| event.status != "reported"),
            "{source}"
        );
    }
    Ok(())
}

#[test]
fn assigns_each_quote_to_its_own_preposed_speaker() -> anyhow::Result<()> {
    let lexicon = Lexicon {
        actors: &[("顧梨", "顧梨"), ("寧衡", "寧衡")],
        objects: &[("救生繩", "救生繩")],
        ..Lexicon::default()
    };
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &lexicon)?;
    let source = "顧梨說道：「請快來幫忙」，寧衡答道：「我取出了救生繩。」";
    let result = analyzer.analyze(source, "");
    let dialogue: Vec<_> = result
        .sections
        .iter()
        .flat_map(|section| &section.units)
        .filter(|unit| unit.kind == "dialogue")
        .map(|unit| (original(source, unit.evidence), unit.speaker.as_deref()))
        .collect();
    assert_eq!(
        dialogue,
        [
            ("「請快來幫忙」".to_owned(), Some("顧梨")),
            ("「我取出了救生繩。」".to_owned(), Some("寧衡")),
        ]
    );

    let reported: Vec<_> = result
        .events
        .iter()
        .filter(|event| event.status == "reported")
        .collect();
    assert_eq!(reported.len(), 1);
    let event = reported[0];
    assert_eq!(event.actor.as_deref(), Some("寧衡"));
    assert_eq!(event.object.as_deref(), Some("救生繩"));
    let claim_start = "顧梨說道：「請快來幫忙」，寧衡答道：「".chars().count();
    let speaker_start = "顧梨說道：「請快來幫忙」，".chars().count();
    assert_eq!(
        event.evidence,
        [
            Span {
                start_char: claim_start,
                end_char: claim_start + "我取出了救生繩。".chars().count(),
            },
            Span {
                start_char: speaker_start,
                end_char: speaker_start + "寧衡".chars().count(),
            },
        ]
    );
    Ok(())
}

#[test]
fn rejects_question_dream_and_unspoken_claims() -> anyhow::Result<()> {
    let lexicon = Lexicon {
        actors: &[("顧梨", "顧梨")],
        ..Lexicon::default()
    };
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &lexicon)?;
    let reported: Vec<_> = [
        "「我取出了救生繩嗎。」顧梨說道。",
        "顧梨夢見自己說道：「我取出了救生繩。」",
        "顧梨本想開口說「我取出了救生繩。」卻忍住了。",
    ]
    .into_iter()
    .filter(|source| {
        analyzer
            .analyze(source, "")
            .events
            .iter()
            .any(|event| event.status == "reported")
    })
    .collect();
    assert!(reported.is_empty(), "{reported:?}");

    let source = "「我取出了救生繩。」顧梨說道。";
    let result = analyzer.analyze(source, "");
    let event = result
        .events
        .iter()
        .find(|event| event.status == "reported")
        .unwrap();
    assert_eq!(event.actor.as_deref(), Some("顧梨"));
    assert_eq!(original(source, event.evidence[0]), "我取出了救生繩。");
    assert_eq!(original(source, event.evidence[1]), "顧梨");
    Ok(())
}

#[test]
fn carries_the_adjacent_named_actor_after_a_detached_continuation() -> anyhow::Result<()> {
    let lexicon = Lexicon {
        actors: &[("岑棠", "岑棠")],
        ..Lexicon::default()
    };
    for (locale, source, action) in [
        (
            Locale::ChineseTaiwan,
            "𠮷。岑棠收起羅盤。隨後，把銅板放下。",
            "把銅板放下。",
        ),
        (
            Locale::ChineseChina,
            "𠮷。岑棠收起罗盘。随后，把铜板放下。",
            "把铜板放下。",
        ),
    ] {
        let analyzer = NarrativeAnalyzer::new(locale, &lexicon)?;
        let result = analyzer.analyze(source, "");
        let continued = result.events.last().unwrap();
        assert_eq!(continued.status, "occurred");
        assert_eq!(continued.actor.as_deref(), Some("岑棠"));
        assert_eq!(original(source, continued.evidence[0]), action);
        assert_eq!(original(source, continued.evidence[1]), "岑棠");
        assert_eq!(continued.evidence[1].start_char, 2);
        assert_eq!(
            analyze(source, "", &lexicon).events.last().unwrap().actor,
            None
        );
    }
    Ok(())
}

#[test]
fn detached_continuation_stops_at_subject_and_context_boundaries() -> anyhow::Result<()> {
    let lexicon = Lexicon {
        actors: &[("岑棠", "岑棠"), ("葉淞", "葉淞")],
        ..Lexicon::default()
    };
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &lexicon)?;
    for (source, actor) in [
        ("岑棠收起羅盤。隨後，葉淞把銅門推開。", Some("葉淞")),
        ("岑棠收起羅盤。隨後，一名守衛把銅門推開。", None),
        ("岑棠收起羅盤。隨後，出現一名守衛。", None),
        ("岑棠收起羅盤。隨後，把銅門推開的是一名守衛。", None),
        ("岑棠收起羅盤。「請等一下！」隨後，把銅門推開。", None),
        ("岑棠收起羅盤。天色轉暗。隨後，把銅門推開。", None),
        ("岑棠收起羅盤。\n隨後，把銅門推開。", None),
        ("岑棠準備收起羅盤。隨後，把銅門推開。", None),
        ("岑棠收起羅盤。隨後，把銅門推開才能通行。", None),
        ("岑棠收起羅盤。隨後，把銅門推開才可以通行。", None),
        ("岑棠收起羅盤。隨後，把銅門推開了嗎。", None),
    ] {
        let result = analyzer.analyze(source, "");
        assert_eq!(
            result.events.last().unwrap().actor.as_deref(),
            actor,
            "{source}"
        );
    }
    Ok(())
}

#[test]
fn disposal_frame_anchors_the_verb_after_a_lexical_patient() -> anyhow::Result<()> {
    let lexicon = Lexicon {
        actors: &[("喬汐", "喬汐"), ("簡悠", "簡悠"), ("裴湛", "裴舟")],
        objects: &[("放下號", "試航機"), ("纸匣", "紙匣"), ("紙匣", "紙匣")],
        ..Lexicon::default()
    };
    for (locale, clause, actor, verb, patient, canonical) in [
        (
            Locale::ChineseTaiwan,
            "喬汐把一台放下號放下了。",
            "喬汐",
            "放下",
            "放下號",
            "試航機",
        ),
        (
            Locale::ChineseTaiwan,
            "簡悠將裴湛推開了。",
            "簡悠",
            "推開",
            "裴湛",
            "裴舟",
        ),
        (
            Locale::ChineseChina,
            "簡悠将纸匣取出了。",
            "簡悠",
            "取出",
            "纸匣",
            "紙匣",
        ),
        (
            Locale::ChineseTaiwan,
            "忽然簡悠把紙匣取出了。",
            "簡悠",
            "取出",
            "紙匣",
            "紙匣",
        ),
        (
            Locale::ChineseTaiwan,
            "簡悠已經把紙匣取出了。",
            "簡悠",
            "取出",
            "紙匣",
            "紙匣",
        ),
    ] {
        let analyzer = NarrativeAnalyzer::new(locale, &lexicon)?;
        let source = format!("𠮷。{clause}");
        let result = analyzer.analyze(&source, "");
        assert_eq!(result.events.len(), 1, "{source}");
        let event = &result.events[0];
        assert_eq!(event.status, "occurred");
        assert_eq!(event.actor.as_deref(), Some(actor));
        assert_eq!(event.object.as_deref(), Some(canonical));
        assert_eq!(
            event
                .evidence
                .iter()
                .map(|span| original(&source, *span))
                .collect::<Vec<_>>(),
            [clause, actor, verb, patient]
        );
        assert_eq!(event.evidence[0].start_char, 2);
        let verb_start = source[..source.rfind(verb).unwrap()].chars().count();
        assert_eq!(
            event.evidence[2],
            Span {
                start_char: verb_start,
                end_char: verb_start + verb.chars().count(),
            }
        );
    }
    Ok(())
}

#[test]
fn disposal_completion_does_not_require_a_named_actor() -> anyhow::Result<()> {
    let lexicon = Lexicon {
        actors: &[("簡悠", "簡悠"), ("裴湛", "裴湛")],
        objects: &[("紙匣", "紙匣"), ("纸匣", "紙匣")],
        ..Lexicon::default()
    };
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &lexicon)?;
    for (source, actor_surface, verb, patient) in [
        ("一名匠人把紙匣取出了。", "一名匠人", "取出", "紙匣"),
        ("一名守衛将裴湛推開了。", "一名守衛", "推開", "裴湛"),
    ] {
        let result = analyzer.analyze(source, "");
        let event = result.events.first().unwrap();
        assert_eq!(event.status, "occurred", "{source}");
        assert_eq!(event.actor, None, "{source}");
        assert_eq!(event.object.as_deref(), Some(patient), "{source}");
        assert_eq!(
            event
                .evidence
                .iter()
                .map(|span| original(source, *span))
                .collect::<Vec<_>>(),
            [source, actor_surface, verb, patient]
        );
    }

    for source in [
        "簡悠把紙匣取出了嗎。",
        "簡悠把紙匣取出吧。",
        "簡悠沒把紙匣取出。",
        "若簡悠把紙匣取出了。",
        "「我把紙匣取出了。」簡悠說道。",
    ] {
        let result = analyzer.analyze(source, "");
        assert!(
            result.events.iter().all(|event| event.status != "occurred"),
            "{source}"
        );
    }
    for source in [
        "簡悠抬手，一名匠人把紙匣取出了。",
        "簡悠和裴湛把紙匣取出了。",
    ] {
        let result = analyzer.analyze(source, "");
        let event = result.events.last().unwrap();
        assert_eq!(event.status, "occurred", "{source}");
        assert_eq!(event.actor, None, "{source}");
    }
    let source = "簡悠抬手，連忙把紙匣取出。";
    let (result, frames) = analyzer.analyze_with_clause_frames(source, "");
    let frame = &frames[0];
    assert_eq!(frame.actor_surface, None);
    assert_eq!(
        result.events[frame.event_index].actor.as_deref(),
        Some("簡悠")
    );
    assert_eq!(
        result.events[frame.event_index].object.as_deref(),
        Some("紙匣")
    );
    let (result, frames) = analyzer.analyze_with_clause_frames("請把紙匣取出。", "");
    assert_eq!(frames[0].actor_surface, None);
    assert_ne!(result.events[frames[0].event_index].status, "occurred");
    for source in ["簡悠将取出纸匣。", "簡悠一把取出紙匣。"] {
        let result = analyzer.analyze(source, "");
        assert!(
            result.events.iter().all(|event| event.evidence.len() == 1),
            "{source}"
        );
    }
    Ok(())
}

#[test]
fn disposal_result_requires_the_same_patient_in_the_next_clause() -> anyhow::Result<()> {
    let lexicon = Lexicon {
        actors: &[("澄漪", "澄漪")],
        objects: &[("薄陶盒", "薄陶盒"), ("帆布", "帆布")],
        ..Lexicon::default()
    };
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &lexicon)?;
    let source = "𠮷。澄漪抬手。澄漪把薄陶盒打開了，薄陶盒裂成兩片薄屑。";
    let (result, frames) = analyzer.analyze_with_clause_frames(source, "");
    let [frame] = frames.as_slice() else {
        panic!("expected one structured clause");
    };
    assert_eq!(frame.event_index, 1);
    assert_eq!(original(source, frame.actor_surface.unwrap()), "澄漪");
    assert_eq!(original(source, frame.predicate), "打開");
    assert_eq!(original(source, frame.patient), "薄陶盒");
    assert_eq!(original(source, frame.result.unwrap()), "兩片薄屑");
    let event = &result.events[frame.event_index];
    assert_eq!(event.actor.as_deref(), Some("澄漪"));
    assert_eq!(event.object.as_deref(), Some("薄陶盒"));
    assert_eq!(original(source, event.evidence[0]), "澄漪把薄陶盒打開了，");
    assert!(event.action.contains("裂成兩片薄屑"));
    assert!(
        event
            .evidence
            .iter()
            .any(|span| original(source, *span) == "兩片薄屑")
    );

    let source = "澄漪把帆布割開成兩片。";
    let (result, frames) = analyzer.analyze_with_clause_frames(source, "");
    let frame = &frames[0];
    assert_eq!(original(source, frame.result.unwrap()), "兩片");
    assert_eq!(
        result.events[frame.event_index].object.as_deref(),
        Some("帆布")
    );

    for source in [
        "澄漪把薄陶盒打開了。薄陶盒裂成兩片薄屑。",
        "澄漪把薄陶盒打開了，旁邊的瓷扣裂成兩片薄屑。",
        "澄漪把薄陶盒打開了，薄陶盒蓋裂成兩片薄屑。",
        "澄漪把薄陶盒打開了，薄陶盒可能裂成兩片薄屑。",
        "澄漪把薄陶盒打開了，薄陶盒裂成兩片薄屑嗎？",
        "澄漪把薄陶盒打開了兩次。",
    ] {
        let result = analyzer.analyze(source, "");
        let event = result.events.first().unwrap();
        assert_eq!(event.object.as_deref(), Some("薄陶盒"), "{source}");
        assert_eq!(event.evidence.len(), 4, "{source}");
        assert!(!event.action.contains("兩片薄屑"), "{source}");
    }
    Ok(())
}

#[test]
fn disposal_frames_abstain_on_embedded_and_compound_subjects() -> anyhow::Result<()> {
    let lexicon = Lexicon {
        actors: &[("岑棠", "岑棠"), ("葉淞", "葉淞")],
        objects: &[("紙條", "紙條")],
        ..Lexicon::default()
    };
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &lexicon)?;
    for (text, status) in [
        ("岑棠承認葉淞把紙條撕開了。", "uncertain"),
        ("岑棠勸葉淞把紙條撕開了再離去。", "uncertain"),
        ("岑棠承認郁冉把紙條撕開了。", "uncertain"),
        ("一名守衛請陌生人把紙條撕開了。", "uncertain"),
        ("岑棠和葉淞把紙條撕開了。", "occurred"),
    ] {
        let (analysis, frames) = analyzer.analyze_with_clause_frames(text, "");
        let [frame] = frames.as_slice() else {
            panic!("expected one disposal action for {text}");
        };
        let event = &analysis.events[frame.event_index];
        assert_eq!(event.status, status, "{text}");
        assert_eq!(event.actor, None, "{text}");
        assert_eq!(frame.actor_surface, None, "{text}");
        assert_eq!(original(text, frame.predicate), "撕開");
        assert_eq!(original(text, frame.patient), "紙條");
    }
    Ok(())
}

#[test]
fn disposal_frames_keep_actors_for_modifiers_and_continuing_pronouns() -> anyhow::Result<()> {
    let lexicon = Lexicon {
        actors: &[("岑棠", "岑棠")],
        objects: &[("紙條", "紙條")],
        ..Lexicon::default()
    };
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &lexicon)?;
    for (text, subject) in [
        ("岑棠抬手，同樣將紙條收起后，便轉身離去。", None),
        ("岑棠抬手。\r\n然后他將紙條收起。", Some("他")),
    ] {
        let (analysis, frames) = analyzer.analyze_with_clause_frames(text, "");
        let [frame] = frames.as_slice() else {
            panic!("expected one disposal action for {text}");
        };
        let event = &analysis.events[frame.event_index];
        assert_eq!(event.status, "occurred", "{text}");
        assert_eq!(event.actor.as_deref(), Some("岑棠"), "{text}");
        assert_eq!(
            frame.actor_surface.map(|span| original(text, span)),
            subject.map(str::to_owned),
        );
        assert!(
            event
                .evidence
                .iter()
                .any(|&span| original(text, span) == "岑棠")
        );
    }
    Ok(())
}

#[test]
fn discovers_quote_local_acts_and_a_postposed_speaker() -> anyhow::Result<()> {
    use faberline_nlp::SpeechActKind::{Question, Statement};

    let lexicon = Lexicon {
        actors: &[("岑棠", "岑棠"), ("寧衡", "寧衡"), ("宁衡", "寧衡")],
        ..Lexicon::default()
    };
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &lexicon)?;
    let source =
        "𠮷。岑棠說道：「雨已停。」寧衡問道：「橋面能走嗎？」\r\n「我會先等一等。」寧衡說道。";
    let acts = analyzer.discover_speech_acts(source, "");
    assert_eq!(acts.len(), 3);
    for (act, (kind, cue, speaker, payload)) in acts.iter().zip([
        (Statement, "說道", "岑棠", "雨已停。"),
        (Question, "問道", "寧衡", "橋面能走嗎？"),
        (Statement, "說道", "寧衡", "我會先等一等。"),
    ]) {
        assert_eq!(act.kind, kind);
        assert_eq!(original(source, act.cue), cue);
        assert_eq!(act.speaker.as_deref(), Some(speaker));
        assert_eq!(original(source, act.speaker_surface.unwrap()), speaker);
        assert_eq!(original(source, act.payload), payload);
    }
    assert!(acts[0].cue.end_char <= acts[0].payload.start_char);
    assert!(acts[2].payload.end_char <= acts[2].cue.start_char);

    let analyzer = NarrativeAnalyzer::new(Locale::ChineseChina, &lexicon)?;
    let source = "𠮷。“桥面能走吗？”宁衡问道。";
    let acts = analyzer.discover_speech_acts(source, "");
    let [act] = acts.as_slice() else {
        panic!("expected one question");
    };
    assert_eq!(act.kind, Question);
    assert_eq!(original(source, act.cue), "问道");
    assert_eq!(act.speaker.as_deref(), Some("寧衡"));
    assert_eq!(original(source, act.speaker_surface.unwrap()), "宁衡");
    assert_eq!(original(source, act.payload), "桥面能走吗？");
    Ok(())
}

#[test]
fn discovers_explicit_unquoted_acts_and_keeps_an_unknown_speaker_surface() -> anyhow::Result<()> {
    use faberline_nlp::SpeechActKind::{Decision, Refusal, Request, Warning};

    let lexicon = Lexicon {
        actors: &[("寧衡", "寧衡"), ("岑棠", "岑棠")],
        ..Lexicon::default()
    };
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &lexicon)?;
    let source = "寧衡要求把卷宗留在原處。岑棠拒絕交出鑰匙。\r\n一名守衛警告橋面已鬆動。岑棠決定明天去送信。";
    let acts = analyzer.discover_speech_acts(source, "");
    assert_eq!(acts.len(), 4);
    for (act, (kind, cue, speaker, surface, payload)) in acts.iter().zip([
        (Request, "要求", Some("寧衡"), "寧衡", "把卷宗留在原處。"),
        (Refusal, "拒絕", Some("岑棠"), "岑棠", "交出鑰匙。"),
        (Warning, "警告", None, "一名守衛", "橋面已鬆動。"),
        (Decision, "決定", Some("岑棠"), "岑棠", "明天去送信。"),
    ]) {
        assert_eq!(act.kind, kind);
        assert_eq!(original(source, act.cue), cue);
        assert_eq!(act.speaker.as_deref(), speaker);
        assert_eq!(original(source, act.speaker_surface.unwrap()), surface);
        assert_eq!(original(source, act.payload), payload);
    }

    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &Lexicon::default())?;
    let source = "一名守衛警告橋面已鬆動。";
    let acts = analyzer.discover_speech_acts(source, "");
    let [act] = acts.as_slice() else {
        panic!("expected one warning without a name list");
    };
    assert_eq!(act.kind, Warning);
    assert_eq!(act.speaker, None);
    assert_eq!(original(source, act.speaker_surface.unwrap()), "一名守衛");
    Ok(())
}

#[test]
fn a_spoken_hypothetical_payload_is_not_a_legacy_story_event() -> anyhow::Result<()> {
    use faberline_nlp::SpeechActKind::Statement;

    let lexicon = Lexicon {
        actors: &[("寧衡", "寧衡")],
        ..Lexicon::default()
    };
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &lexicon)?;
    let payload = "如果雨再大一點，我明天打算取出救生繩。";
    let source = format!("寧衡說道：「{payload}」");
    let acts = analyzer.discover_speech_acts(&source, "");
    let [act] = acts.as_slice() else {
        panic!("expected one actually spoken statement");
    };
    assert_eq!(act.kind, Statement);
    assert_eq!(original(&source, act.cue), "說道");
    assert_eq!(original(&source, act.payload), payload);
    assert!(analyze(&source, "", &lexicon).events.is_empty());
    assert!(analyzer.analyze(&source, "").events.is_empty());
    Ok(())
}

#[test]
fn excludes_unasserted_speech_and_keeps_the_next_actual_quote() -> anyhow::Result<()> {
    use faberline_nlp::SpeechActKind::Question;

    let lexicon = Lexicon {
        actors: &[("岑棠", "岑棠"), ("寧衡", "寧衡")],
        ..Lexicon::default()
    };
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &lexicon)?;
    for source in [
        "如果寧衡拒絕交出鑰匙，岑棠就關上房門。",
        "寧衡打算要求重新檢查橋面。",
        "寧衡沒有警告橋面已經鬆動。",
        "如果寧衡說道：「木窗已關上。」",
        "如果船靠岸，寧衡說道：「木窗已關上。」",
        "寧衡打算說道：「木窗已關上。」",
        "寧衡沒有說道：「木窗已關上。」",
        "岑棠夢見寧衡說道：「木窗已關上。」",
    ] {
        assert!(
            analyzer.discover_speech_acts(source, "").is_empty(),
            "{source}"
        );
    }

    let source = "岑棠打算說道：「門已關上。」寧衡問道：「窗子呢？」";
    let acts = analyzer.discover_speech_acts(source, "");
    let [act] = acts.as_slice() else {
        panic!("expected only the actual second speech act");
    };
    assert_eq!(act.kind, Question);
    assert_eq!(act.speaker.as_deref(), Some("寧衡"));
    assert_eq!(original(source, act.cue), "問道");
    assert_eq!(original(source, act.payload), "窗子呢？");
    Ok(())
}

#[test]
fn discovers_performed_quote_acts_without_a_work_name_list() -> anyhow::Result<()> {
    use faberline_nlp::SpeechActKind::{Decision, Refusal, Request, Statement, Warning};

    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &Lexicon::default())?;
    for (source, kind, surface) in [
        ("旅人說道：「請留下卷宗。」", Request, "旅人"),
        ("旅人說道：「我請求留下卷宗。」", Request, "旅人"),
        ("「我拒絕交出卷宗！」匠人說道。", Refusal, "匠人"),
        ("「我拒絕交出卷宗！」匠人森然說道。", Refusal, "匠人"),
        ("「小心碎石！」他低低的說道。", Warning, "他"),
        ("他說道：「小心碎石！」", Warning, "他"),
        ("他說道：「我警告你別上橋。」", Warning, "他"),
        ("守衛說道：「我決定明天離去。」", Decision, "守衛"),
        ("旅人說道：「我拒絕了那個提議。」", Statement, "旅人"),
        ("守衛警告：「小心碎石！」", Warning, "守衛"),
    ] {
        let acts = analyzer.discover_speech_acts(source, "");
        let [act] = acts.as_slice() else {
            panic!("expected one act for {source}");
        };
        assert_eq!(act.kind, kind, "{source}");
        assert_eq!(act.speaker, None, "{source}");
        assert_eq!(original(source, act.speaker_surface.unwrap()), surface);
    }
    let across_line = "「我拒絕交出卷宗！」\r\n匠人說道。";
    assert!(analyzer.discover_speech_acts(across_line, "").is_empty());
    let lexicon = Lexicon {
        line_footers: &["匠人說道"],
        ..Lexicon::default()
    };
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &lexicon)?;
    let footer = "「我拒絕交出卷宗！」匠人說道。";
    assert!(analyzer.discover_speech_acts(footer, "").is_empty());
    Ok(())
}

fn linking_span(source: &str, text: &str, occurrence: usize) -> Span {
    let (start, _) = source.match_indices(text).nth(occurrence).unwrap();
    let start_char = source[..start].chars().count();
    Span {
        start_char,
        end_char: start_char + text.chars().count(),
    }
}

#[test]
fn links_adjacent_inferred_patient_and_result_spans() -> anyhow::Result<()> {
    use faberline_nlp::{ClauseCandidate, ClauseScope};

    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &Lexicon::default())?;
    for source in [
        "𠮷。寧衡推開木門，木門因此裂成了兩截。",
        "𠮷。寧衡推開木門，因此木門裂成了兩截。",
    ] {
        let linked = analyzer.link_clause_candidates(
            source,
            "",
            &[
                ClauseCandidate::new(linking_span(source, "推開", 0)),
                ClauseCandidate::new(linking_span(source, "裂成", 0)),
            ],
        );
        assert!(linked.rejected.is_empty(), "{source}");
        let [action, result] = linked.clauses.as_slice() else {
            panic!("expected two clauses for {source}");
        };
        assert_eq!((action.input_index, result.input_index), (0, 1));
        assert_eq!(action.scope, ClauseScope::Narrated);
        assert_eq!(result.scope, ClauseScope::Narrated);
        assert!(original(source, action.clause).starts_with("寧衡推開木門"));
        assert!(original(source, result.clause).ends_with("裂成了兩截。"));
        let [edge] = linked.results.as_slice() else {
            panic!("expected one explicit result link for {source}");
        };
        assert_eq!((edge.action, edge.result_clause), (0, 1));
        assert!(edge.explicit_cause);
        assert_eq!(edge.cue, linking_span(source, "因此", 0));
        assert_eq!(edge.patient_before, linking_span(source, "木門", 0));
        assert_eq!(edge.patient_after, linking_span(source, "木門", 1));
        assert_eq!(edge.result, Some(linking_span(source, "兩截", 0)));
    }
    Ok(())
}

#[test]
fn retains_supplied_roles_and_links_in_source_order_after_rejecting_bad_spans() -> anyhow::Result<()>
{
    use faberline_nlp::ClauseCandidate;

    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &Lexicon::default())?;
    let source = "𠮷。寧衡推開木門，木門因而裂成兩截。";
    let actor = linking_span(source, "寧衡", 0);
    let before = linking_span(source, "木門", 0);
    let after = linking_span(source, "木門", 1);
    let outcome = linking_span(source, "兩截", 0);
    let action_predicate = linking_span(source, "推開", 0);
    let result_predicate = linking_span(source, "裂成", 0);
    let mut action = ClauseCandidate::new(action_predicate);
    action.actor = Some(actor);
    action.patient = Some(before);
    let mut result = ClauseCandidate::new(result_predicate);
    result.theme = Some(after);
    result.result = Some(outcome);
    let out_of_bounds = ClauseCandidate::new(Span {
        start_char: 0,
        end_char: source.chars().count() + 1,
    });
    let mut crossed = ClauseCandidate::new(result_predicate);
    crossed.actor = Some(actor);
    let empty = ClauseCandidate::new(Span {
        start_char: action_predicate.start_char,
        end_char: action_predicate.start_char,
    });
    let reversed = ClauseCandidate::new(Span {
        start_char: result_predicate.end_char,
        end_char: result_predicate.start_char,
    });

    let linked = analyzer.link_clause_candidates(
        source,
        "",
        &[result, out_of_bounds, action, crossed, empty, reversed],
    );
    assert_eq!(linked.rejected, [1, 3, 4, 5]);
    let [result, action] = linked.clauses.as_slice() else {
        panic!("expected both valid inputs to survive");
    };
    assert_eq!((result.input_index, action.input_index), (0, 2));
    assert_eq!(result.candidate.theme, Some(after));
    assert_eq!(result.candidate.result, Some(outcome));
    assert_eq!(action.candidate.actor, Some(actor));
    assert_eq!(action.candidate.patient, Some(before));
    let [edge] = linked.results.as_slice() else {
        panic!("expected one result link independent of input order");
    };
    assert_eq!((edge.action, edge.result_clause), (1, 0));
    assert_eq!(edge.cue, linking_span(source, "因而", 0));
    assert_eq!((edge.patient_before, edge.patient_after), (before, after));
    assert_eq!(edge.result, Some(outcome));
    Ok(())
}

#[test]
fn scopes_each_clause_without_borrowing_assertion_from_its_neighbors() -> anyhow::Result<()> {
    use faberline_nlp::{
        ClauseCandidate,
        ClauseScope::{Conditional, Dream, Narrated, Negated, Planned, Quoted, Reported},
    };

    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &Lexicon::default())?;
    for (source, predicate, scope, marker) in [
        ("寧衡推開木門。", "推開", Narrated, None),
        ("寧衡不但推開木門，還關上窗。", "推開", Narrated, None),
        (
            "寧衡說道：「別動！」，隨即推開木門。",
            "推開",
            Narrated,
            None,
        ),
        ("寧衡說道：「我推開了木門。」", "推開", Quoted, None),
        (
            "若寧衡推開木門，木門就會裂成兩截。",
            "裂成",
            Conditional,
            Some("若"),
        ),
        ("寧衡打算推開木門。", "推開", Planned, Some("打算")),
        ("寧衡沒有推開木門。", "推開", Negated, Some("沒有")),
        ("據說寧衡推開木門。", "推開", Reported, Some("據說")),
        ("寧衡夢見自己推開木門。", "推開", Dream, Some("夢見")),
        (
            "寧衡夢見自己推開木門。醒來後，寧衡打開木盒。",
            "打開",
            Narrated,
            None,
        ),
    ] {
        let linked = analyzer.link_clause_candidates(
            source,
            "",
            &[ClauseCandidate::new(linking_span(source, predicate, 0))],
        );
        let [clause] = linked.clauses.as_slice() else {
            panic!("expected one candidate to survive for {source}");
        };
        assert_eq!(clause.scope, scope, "{source}");
        if let Some(marker) = marker {
            assert!(
                original(source, clause.scope_evidence.unwrap()).contains(marker),
                "{source}"
            );
        }
    }
    Ok(())
}

#[test]
fn rejects_result_links_without_local_asserted_cause_and_same_patient() -> anyhow::Result<()> {
    use faberline_nlp::ClauseCandidate;

    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &Lexicon::default())?;
    for source in [
        "寧衡推開木門，後門因此裂成兩截。",
        "寧衡推開木門，木門隨後裂成兩截。",
        "寧衡推開木門，門外有人咳嗽，木門因此裂成兩截。",
        "寧衡推開木門，木門因年久失修而裂成兩截。",
        "寧衡因此推開木門，木門裂成兩截。",
        "寧衡推開木門，木門裂成兩截，因此岑棠來看。",
        "寧衡推開木門，木門因此沒有裂成兩截。",
        "寧衡沒有把木門推開，木門因此裂成兩截。",
        "寧衡打算把木門推開，木門因此裂成兩截。",
        "若寧衡推開木門，木門因此裂成兩截。",
        "據說寧衡推開木門，木門因此裂成兩截。",
        "寧衡推開木門，岑棠說道：「木門因此裂成兩截。」",
        "岑棠說道：「事情是這樣。\n寧衡推開木門，木門因此裂成兩截。\n後來有人來修。」",
        "寧衡推開木門。\r\n木門因此裂成兩截。",
        "寧衡推開木門，木門因此裂成兩截嗎？",
    ] {
        let linked = analyzer.link_clause_candidates(
            source,
            "",
            &[
                ClauseCandidate::new(linking_span(source, "推開", 0)),
                ClauseCandidate::new(linking_span(source, "裂成", 0)),
            ],
        );
        assert_eq!(linked.clauses.len(), 2, "{source}");
        assert!(linked.results.is_empty(), "{source}");
    }
    Ok(())
}

#[test]
fn clause_roles_anchor_only_an_explicit_local_actor() -> anyhow::Result<()> {
    use faberline_nlp::ClauseCandidate;

    let lexicon = Lexicon {
        actors: &[("寧衡", "寧衡"), ("宁衡", "寧衡")],
        ..Lexicon::default()
    };
    for (locale, source, predicate, actor) in [
        (
            Locale::ChineseTaiwan,
            "𠮷。寧衡推開木門。",
            "推開",
            Some("寧衡"),
        ),
        (
            Locale::ChineseChina,
            "据说宁衡推开木门。",
            "推开",
            Some("宁衡"),
        ),
        (
            Locale::ChineseTaiwan,
            "寧衡沒有推開木門。",
            "推開",
            Some("寧衡"),
        ),
        (Locale::ChineseTaiwan, "寧衡的同伴推開木門。", "推開", None),
        (
            Locale::ChineseTaiwan,
            "接著寧衡一推開木門。",
            "推開",
            Some("寧衡"),
        ),
        (
            Locale::ChineseTaiwan,
            "一名守衛便推開木門。",
            "推開",
            Some("一名守衛"),
        ),
        (
            Locale::ChineseTaiwan,
            "寧衡站定。又推開木門。",
            "推開",
            None,
        ),
    ] {
        let analyzer = NarrativeAnalyzer::new(locale, &lexicon)?;
        let linked = analyzer.link_clause_candidates(
            source,
            "",
            &[ClauseCandidate::new(linking_span(source, predicate, 0))],
        );
        assert_eq!(
            linked.clauses[0].actor_evidence,
            actor.map(|name| linking_span(source, name, 0)),
            "{source}"
        );
    }
    Ok(())
}

#[test]
fn clause_proposals_cannot_hide_negation_or_turn_a_noun_into_a_cue() -> anyhow::Result<()> {
    use faberline_nlp::{ClauseCandidate, ClauseScope};

    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &Lexicon::default())?;
    let source = "寧衡沒有推開木門，木門因此裂成兩截。";
    let mut actor_masks_cue = ClauseCandidate::new(linking_span(source, "推開", 0));
    actor_masks_cue.actor = Some(linking_span(source, "寧衡沒有", 0));
    for action in [
        actor_masks_cue,
        ClauseCandidate::new(linking_span(source, "沒有推開", 0)),
    ] {
        let linked = analyzer.link_clause_candidates(
            source,
            "",
            &[
                action,
                ClauseCandidate::new(linking_span(source, "裂成", 0)),
            ],
        );
        assert_eq!(linked.clauses[0].scope, ClauseScope::Negated);
        assert!(linked.results.is_empty());
    }
    let source = "寧衡要求驗血，驗血結果顯示陰性。";
    let linked = analyzer.link_clause_candidates(
        source,
        "",
        &[
            ClauseCandidate::new(linking_span(source, "要求", 0)),
            ClauseCandidate::new(linking_span(source, "顯示", 0)),
        ],
    );
    assert_eq!(linked.clauses.len(), 2);
    assert!(linked.results.is_empty());
    Ok(())
}

#[test]
fn aligned_disposal_proposals_reuse_local_role_and_result_evidence() -> anyhow::Result<()> {
    use faberline_nlp::{ClauseCandidate, ClauseScope};

    let lexicon = Lexicon {
        actors: &[("岑棠", "岑棠")],
        objects: &[("紙條", "紙條")],
        ..Lexicon::default()
    };
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &lexicon)?;
    let source = "岑棠把一張紙條輕輕撕開成兩片。";
    let action = ClauseCandidate::new(linking_span(source, "撕開", 0));
    let partial = ClauseCandidate::new(linking_span(source, "撕", 0));
    let mut conflicting = action;
    conflicting.patient = Some(linking_span(source, "一張", 0));
    let linked = analyzer.link_clause_candidates(source, "", &[action, partial, conflicting]);
    assert!(linked.rejected.is_empty());
    let [aligned, partial, conflicting] = linked.clauses.as_slice() else {
        panic!("expected all three source proposals");
    };
    assert_eq!(
        aligned.actor_evidence,
        Some(linking_span(source, "岑棠", 0))
    );
    assert_eq!(
        aligned.patient_evidence,
        Some(linking_span(source, "紙條", 0))
    );
    assert_eq!(
        aligned.result_evidence,
        Some(linking_span(source, "兩片", 0))
    );
    assert_eq!(partial.actor_evidence, None);
    assert_eq!(partial.patient_evidence, None);
    assert_eq!(conflicting.patient_evidence, None);
    let source = "岑棠沒有把一張紙條輕輕撕開。";
    let action = ClauseCandidate::new(linking_span(source, "撕開", 0));
    let linked = analyzer.link_clause_candidates(source, "", &[action]);
    assert_eq!(linked.clauses[0].scope, ClauseScope::Negated);
    assert_eq!(
        linked.clauses[0].scope_evidence,
        Some(linking_span(source, "沒有", 0))
    );
    Ok(())
}

#[test]
fn an_explicit_disposal_result_keeps_its_own_evidence() -> anyhow::Result<()> {
    use faberline_nlp::ClauseCandidate;

    let lexicon = Lexicon {
        actors: &[("岑棠", "岑棠")],
        objects: &[("薄陶盒", "薄陶盒")],
        ..Lexicon::default()
    };
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &lexicon)?;
    let source = "岑棠把薄陶盒打開了，薄陶盒裂成兩片薄屑。";
    let action = ClauseCandidate::new(linking_span(source, "打開", 0));
    let result = ClauseCandidate::new(linking_span(source, "裂", 0));
    let linked = analyzer.link_clause_candidates(source, "", &[action, result]);
    let [edge] = linked.results.as_slice() else {
        panic!("expected one explicit disposal result");
    };
    assert!(!edge.explicit_cause);
    assert_eq!(edge.cue, linking_span(source, "裂成", 0));
    assert_eq!(edge.patient_before, linking_span(source, "薄陶盒", 0));
    assert_eq!(edge.patient_after, linking_span(source, "薄陶盒", 1));
    assert_eq!(edge.result, Some(linking_span(source, "兩片薄屑", 0)));
    assert_eq!(linked.clauses[0].result_evidence, None);
    let inside_result = ClauseCandidate::new(linking_span(source, "薄屑", 0));
    let linked = analyzer.link_clause_candidates(source, "", &[action, inside_result]);
    assert!(linked.results.is_empty());
    Ok(())
}

#[test]
fn terminal_resultative_predicates_keep_written_result_separate() -> anyhow::Result<()> {
    use faberline_nlp::ClauseCandidate;

    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &Lexicon::default())?;
    for (source, verb, patient, result) in [
        (
            "將兩具尸體燒成了灰燼。",
            "燒成",
            Some("兩具尸體"),
            Some("灰燼"),
        ),
        ("把它壓成了薄板。", "壓成", Some("它"), Some("薄板")),
        ("湖水化為了冰層。", "化為", None, Some("冰層")),
        ("將壓成了薄板。", "壓成", None, None),
        ("把兩片泥坯壓成薄板之前。", "壓成", None, None),
        ("將那些泥坯瞬間壓成了薄板。", "壓成", None, None),
        ("把兩片泥坯壓成了薄板嗎？", "壓成", None, None),
    ] {
        let linked = analyzer.link_clause_candidates(
            source,
            "",
            &[ClauseCandidate::new(linking_span(source, verb, 0))],
        );
        let clause = &linked.clauses[0];
        assert_eq!(
            clause.patient_evidence,
            patient.map(|word| linking_span(source, word, 0)),
            "{source}"
        );
        assert_eq!(
            clause.result_evidence,
            result.map(|word| linking_span(source, word, 0)),
            "{source}"
        );
        assert!(linked.results.is_empty());
    }
    Ok(())
}

#[test]
fn written_complements_have_their_own_span_inside_the_predicate() -> anyhow::Result<()> {
    use faberline_nlp::{ClauseCandidate, ClauseScope};

    for (locale, source, verb, complement, result, scope) in [
        (
            Locale::ChineseTaiwan,
            "𠮷。把一張紙條壓成了薄片。",
            "壓成",
            "成",
            Some("薄片"),
            ClauseScope::Narrated,
        ),
        (
            Locale::ChineseChina,
            "一名守卫关上窗。",
            "关上",
            "上",
            None,
            ClauseScope::Narrated,
        ),
        (
            Locale::ChineseChina,
            "湖水化为了冰层。",
            "化为",
            "为",
            Some("冰层"),
            ClauseScope::Narrated,
        ),
        (
            Locale::ChineseTaiwan,
            "若岑棠敲破玻璃。",
            "敲破",
            "破",
            None,
            ClauseScope::Conditional,
        ),
    ] {
        let analyzer = NarrativeAnalyzer::new(locale, &Lexicon::default())?;
        let mut candidate = ClauseCandidate::new(linking_span(source, verb, 0));
        for proposed in [None, Some(linking_span(source, complement, 0))] {
            candidate.result = proposed;
            let linked = analyzer.link_clause_candidates(source, "", &[candidate]);
            assert!(linked.rejected.is_empty(), "{source}");
            let clause = &linked.clauses[0];
            assert_eq!(clause.candidate.result, proposed);
            assert_eq!(
                clause.result_complement,
                Some(linking_span(source, complement, 0))
            );
            assert_eq!(
                clause.result_evidence,
                result.map(|value| linking_span(source, value, 0))
            );
            assert_eq!(clause.scope, scope);
        }
    }
    Ok(())
}

#[test]
fn a_nested_proposal_does_not_verify_a_lexical_lookalike() -> anyhow::Result<()> {
    use faberline_nlp::ClauseCandidate;

    let lexicon = Lexicon {
        objects: &[("打開號", "打開號")],
        ..Lexicon::default()
    };
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &lexicon)?;
    for (source, verb, suffix) in [
        ("岑棠離開了。", "離開", "開"),
        ("岑棠走上台階。", "走上", "上"),
        ("岑棠把一台打開號收起了。", "打開", "開"),
    ] {
        let mut candidate = ClauseCandidate::new(linking_span(source, verb, 0));
        candidate.result = Some(linking_span(source, suffix, 0));
        let linked = analyzer.link_clause_candidates(source, "", &[candidate]);
        assert!(linked.rejected.is_empty(), "{source}");
        assert_eq!(linked.clauses[0].result_complement, None, "{source}");
    }
    let source = "岑棠推開窗。";
    let mut invalid = ClauseCandidate::new(linking_span(source, "推開", 0));
    for range in ["推", "推開", "開窗"] {
        invalid.result = Some(linking_span(source, range, 0));
        let linked = analyzer.link_clause_candidates(source, "", &[invalid]);
        assert_eq!(linked.rejected, [0], "{range}");
    }
    Ok(())
}

#[test]
fn local_result_references_keep_the_written_demonstrative_or_pronoun() -> anyhow::Result<()> {
    use faberline_nlp::ClauseCandidate;

    let lexicon = Lexicon {
        actors: &[("岑棠", "岑棠")],
        objects: &[("陶片", "陶片")],
        ..Lexicon::default()
    };
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &lexicon)?;
    for (source, mention, cause) in [
        (
            "匠人把陶片推開了，這塊陶片隨即裂成了兩半。",
            "這塊陶片",
            false,
        ),
        ("匠人把陶片推開了，這塊陶片因此裂成兩半。", "這塊陶片", true),
        ("匠人把陶片推開了，它隨即裂成了兩半。", "它", false),
        ("匠人推開陶片，此陶片裂成了兩半。", "此陶片", false),
    ] {
        let candidates = [
            ClauseCandidate::new(linking_span(source, "推開", 0)),
            ClauseCandidate::new(linking_span(source, "裂成", 0)),
        ];
        let linked = analyzer.link_clause_candidates(source, "", &candidates);
        let [edge] = linked.results.as_slice() else {
            panic!("expected one result reference for {source}");
        };
        assert!(edge.patient_reference);
        assert_eq!(edge.explicit_cause, cause);
        assert_eq!(edge.patient_before, linking_span(source, "陶片", 0));
        assert_eq!(edge.patient_after, linking_span(source, mention, 0));
        assert_eq!(
            edge.cue,
            linking_span(source, if cause { "因此" } else { "裂成" }, 0)
        );
        assert_eq!(edge.result, Some(linking_span(source, "兩半", 0)));
    }
    for source in [
        "匠人把陶片推開了，這塊木板裂成了兩半。",
        "匠人把陶片推開了，那塊陶片裂成了兩半。",
        "匠人把陶片推開了。這塊陶片裂成了兩半。",
        "匠人把陶片推開了，這個裂成了兩半。",
        "匠人把陶片推開了，此刻裂成了兩半。",
        "匠人把一塊陶片推開了，這兩塊陶片隨即裂成了兩半。",
        "匠人把兩塊陶片推開了，這塊陶片隨即裂成了兩半。",
        "匠人沒有把陶片推開，這塊陶片裂成了兩半。",
        "岑棠把陶片推開了，它裂成了兩半。",
        "匠人把兩塊陶片推開了，它裂成了兩半。",
        "木板吱呀作響，匠人把陶片推開了，它裂成了兩半。",
        "匠人把陶片推開到木板旁，它裂成了兩半。",
        "匠人把陶片推開了，岑棠說：「這塊陶片裂成了兩半。」",
    ] {
        let linked = analyzer.link_clause_candidates(
            source,
            "",
            &[
                ClauseCandidate::new(linking_span(source, "推開", 0)),
                ClauseCandidate::new(linking_span(source, "裂成", 0)),
            ],
        );
        assert!(linked.results.is_empty(), "{source}");
    }
    let source = "匠人把陶片推開了，它隨即裂成了兩半。";
    let mut contested = ClauseCandidate::new(linking_span(source, "裂成", 0));
    contested.actor = Some(linking_span(source, "它", 0));
    let linked = analyzer.link_clause_candidates(
        source,
        "",
        &[
            ClauseCandidate::new(linking_span(source, "推開", 0)),
            contested,
        ],
    );
    assert!(linked.results.is_empty());
    Ok(())
}

#[test]
fn ordinary_precedence_anchors_the_source_cue_and_candidate_indices() -> anyhow::Result<()> {
    use faberline_nlp::ClauseCandidate;

    let lexicon = Lexicon {
        actors: &[("岑棠", "岑棠")],
        objects: &[("木門", "木門")],
        ..Lexicon::default()
    };
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &lexicon)?;
    for (source, cue, paired, reversed) in [
        ("𠮷。岑棠關上木門，隨後守衛熄滅燈火。", "隨後", None, false),
        ("岑棠關上木門。守衛随后熄滅燈火。", "随后", None, false),
        (
            "岑棠關上木門，緊接著，守衛熄滅燈火。",
            "緊接著",
            None,
            false,
        ),
        ("岑棠關上木門之後，守衛熄滅燈火。", "之後", None, false),
        ("岑棠關上木門之前，守衛熄滅燈火。", "之前", None, true),
        ("岑棠先關上木門，再熄滅燈火。", "再", Some("先"), false),
    ] {
        let before = ClauseCandidate::new(linking_span(source, "關上", 0));
        let after = ClauseCandidate::new(linking_span(source, "熄滅", 0));
        let invalid = ClauseCandidate::new(Span {
            start_char: 0,
            end_char: 0,
        });
        let candidates = [after, invalid, before];
        let (linked, order) =
            analyzer.link_clause_candidates_with_precedence(source, "", &candidates);
        assert_eq!(
            linked,
            analyzer.link_clause_candidates(source, "", &candidates)
        );
        assert!(linked.results.is_empty());
        let [edge] = order.as_slice() else {
            panic!("expected one temporal relation for {source}");
        };
        let input_order = (
            linked.clauses[edge.before].input_index,
            linked.clauses[edge.after].input_index,
        );
        assert_eq!(
            input_order,
            if reversed { (0, 2) } else { (2, 0) },
            "{source}"
        );
        assert_eq!(edge.cue, linking_span(source, cue, 0));
        assert_eq!(
            edge.paired_cue,
            paired.map(|word| linking_span(source, word, 0))
        );
    }
    Ok(())
}

#[test]
fn ordinary_precedence_abstains_at_ambiguous_scope_and_anchor_boundaries() -> anyhow::Result<()> {
    use faberline_nlp::ClauseCandidate;

    let lexicon = Lexicon {
        actors: &[("岑棠", "岑棠")],
        objects: &[
            ("木門", "木門"),
            ("熄滅燈火號", "熄滅燈火號"),
            ("木門之前", "木門之前"),
        ],
        ..Lexicon::default()
    };
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &lexicon)?;
    for source in [
        "岑棠關上木門，守衛熄滅燈火。",
        "岑棠關上木門，守衛因此熄滅燈火。",
        "岑棠關上木門，守衛才熄滅燈火。",
        "如果岑棠關上木門，隨後守衛熄滅燈火。",
        "岑棠若關上木門，隨後守衛熄滅燈火。",
        "每當岑棠關上木門，守衛隨後熄滅燈火。",
        "岑棠關上木門，隨後守衛熄滅燈火才能通行。",
        "岑棠沒有關上木門，隨後守衛熄滅燈火。",
        "岑棠沒有立即關上木門，隨後守衛熄滅燈火。",
        "岑棠拒絕關上木門，隨後守衛熄滅燈火。",
        "岑棠關上木門，隨後守衛可能熄滅燈火。",
        "岑棠說他關上木門，隨後守衛熄滅燈火。",
        "有人說，岑棠關上木門，隨後守衛熄滅燈火。",
        "岑棠記下關上木門的建議，隨後守衛熄滅燈火。",
        "岑棠關上木門，隨後熄滅燈火號進港。",
        "岑棠關上木門，「隨後守衛熄滅燈火。」",
        "岑棠關上木門。\r\n\r\n\r\n隨後守衛熄滅燈火。",
        "岑棠關上木門，另一邊，隨後守衛熄滅燈火。",
        "岑棠關上木門。次日，隨後守衛熄滅燈火。",
        "岑棠關上木門，守衛同時熄滅燈火。",
        "岑棠一面關上木門，隨後守衛熄滅燈火。",
        "岑棠關上木門，隨後守衛熄滅燈火了嗎？",
        "岑棠關上木門的聲音消失之後，守衛熄滅燈火。",
        "岑棠關上木門之前，守衛熄滅燈火。",
        "岑棠關上木門，隨後的安排是守衛熄滅燈火。",
        "先關上木門，再熄滅燈火。",
    ] {
        let (_, order) = analyzer.link_clause_candidates_with_precedence(
            source,
            "",
            &[
                ClauseCandidate::new(linking_span(source, "關上", 0)),
                ClauseCandidate::new(linking_span(source, "熄滅", 0)),
            ],
        );
        assert!(order.is_empty(), "{source}");
    }
    let source = "岑棠先關上木門，隨後守衛熄滅燈火。";
    let (_, order) = analyzer.link_clause_candidates_with_precedence(
        source,
        "",
        &[
            ClauseCandidate::new(linking_span(source, "關上", 0)),
            ClauseCandidate::new(linking_span(source, "關", 0)),
            ClauseCandidate::new(linking_span(source, "熄滅", 0)),
        ],
    );
    assert!(order.is_empty());
    Ok(())
}

#[test]
fn ordinary_precedence_accepts_a_grounded_cue_across_one_paragraph_gap() -> anyhow::Result<()> {
    use faberline_nlp::ClauseCandidate;

    let lexicon = Lexicon {
        actors: &[("岑棠", "岑棠")],
        objects: &[("木門", "木門")],
        ..Lexicon::default()
    };
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &lexicon)?;
    for (source, cue, reversed) in [
        ("𠮷。岑棠關上木門。\r\n隨後守衛熄滅燈火。", "隨後", false),
        (
            "岑棠關上木門。\n\n下一刻，守衛便熄滅燈火。",
            "下一刻",
            false,
        ),
        ("岑棠關上木門了。\r\n\r\n守衛隨即熄滅燈火。", "隨即", false),
        (
            "岑棠關上了木門。\n\n片刻后，守衛熄滅燈火。",
            "片刻后",
            false,
        ),
        (
            "岑棠關上了木門。\n\n過了半晌，守衛熄滅燈火。",
            "過了半晌",
            false,
        ),
        ("岑棠關上木門之前，\n\n守衛熄滅燈火。", "之前", true),
        ("岑棠關上木門，然後守衛便熄滅燈火。", "然後", false),
    ] {
        let (linked, order) = analyzer.link_clause_candidates_with_precedence(
            source,
            "",
            &[
                ClauseCandidate::new(linking_span(source, "關上", 0)),
                ClauseCandidate::new(linking_span(source, "熄滅", 0)),
            ],
        );
        let [edge] = order.as_slice() else {
            panic!("expected one temporal relation for {source}");
        };
        assert!(linked.results.is_empty());
        assert_eq!(
            (edge.before, edge.after),
            if reversed { (1, 0) } else { (0, 1) }
        );
        assert_eq!(edge.cue, linking_span(source, cue, 0));
        assert_eq!(edge.paired_cue, None);
    }
    for (noun, expected) in [("光影", 1), ("光球", 0)] {
        let source = format!("一團光影浮現而出。\n\n隨即{noun}凝聚成圓環。");
        let (_, order) = analyzer.link_clause_candidates_with_precedence(
            &source,
            "",
            &[
                ClauseCandidate::new(linking_span(&source, "浮現", 0)),
                ClauseCandidate::new(linking_span(&source, "凝聚", 0)),
            ],
        );
        assert_eq!(order.len(), expected, "{source}");
        if let Some(edge) = order.first() {
            assert_eq!(edge.cue, linking_span(&source, "隨即", 0));
        }
    }
    Ok(())
}

#[test]
fn cross_paragraph_precedence_does_not_guess_the_prior_action() -> anyhow::Result<()> {
    use faberline_nlp::ClauseCandidate;

    let lexicon = Lexicon {
        actors: &[("岑棠", "岑棠")],
        objects: &[("木門", "木門")],
        line_footers: &["本行略"],
        ..Lexicon::default()
    };
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &lexicon)?;
    for source in [
        "岑棠關上木門。\n\n\n隨後守衛熄滅燈火。",
        "岑棠關上木門。\n本行略\n隨後守衛熄滅燈火。",
        "岑棠關上木門。\n\n夜色沉沉。\n\n隨後守衛熄滅燈火。",
        "岑棠關上木門。\n\n另一邊，隨即守衛熄滅燈火。",
        "岑棠關上木門。\n\n與此同時，守衛熄滅燈火。",
        "岑棠關上木門。\n\n三年前，守衛熄滅燈火。",
        "岑棠關上木門。\n\n「下一刻，守衛熄滅燈火。」",
        "有人聲稱，岑棠關上木門。\n\n下一刻，守衛熄滅燈火。",
        "據傳，岑棠關上木門。\n\n下一刻，守衛熄滅燈火。",
        "假如岑棠關上木門。\n\n下一刻，守衛熄滅燈火。",
        "岑棠等候關上木門的時機。\n\n隨後守衛熄滅燈火。",
        "岑棠關上木門後又在門外敲鐘。\n\n隨後守衛熄滅燈火。",
        "岑棠關上木門\n\n下一刻，守衛熄滅燈火。",
        "岑棠關上木門，隨後，\n\n守衛熄滅燈火。",
    ] {
        let (_, order) = analyzer.link_clause_candidates_with_precedence(
            source,
            "",
            &[
                ClauseCandidate::new(linking_span(source, "關上", 0)),
                ClauseCandidate::new(linking_span(source, "熄滅", 0)),
            ],
        );
        assert!(order.is_empty(), "{source}");
    }
    for source in [
        "岑棠關上木門後又敲鐘。\n\n隨後守衛熄滅燈火。",
        "岑棠關上木門後又敲鐘之前，\n\n守衛熄滅燈火。",
    ] {
        let mut prior = ClauseCandidate::new(linking_span(source, "關上", 0));
        prior.patient = Some(linking_span(source, "木門後又敲鐘", 0));
        let (_, order) = analyzer.link_clause_candidates_with_precedence(
            source,
            "",
            &[prior, ClauseCandidate::new(linking_span(source, "熄滅", 0))],
        );
        assert!(order.is_empty(), "{source}");
    }
    for (source, head) in [
        ("雨聲。\n\n下一刻，守衛熄滅燈火。", "雨聲"),
        ("岑棠四十歲了。\n\n下一刻，守衛熄滅燈火。", "四十歲"),
        (
            "關上木門的時機。\n\n下一刻，守衛熄滅燈火。",
            "關上木門的時機",
        ),
    ] {
        let (_, order) = analyzer.link_clause_candidates_with_precedence(
            source,
            "",
            &[
                ClauseCandidate::new(linking_span(source, head, 0)),
                ClauseCandidate::new(linking_span(source, "熄滅", 0)),
            ],
        );
        assert!(order.is_empty(), "{source}");
    }
    Ok(())
}

#[test]
fn cue_first_keeps_written_markers_without_any_proposals() -> anyhow::Result<()> {
    use faberline_nlp::{ClauseOrderCueKind, ClauseScope};

    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &Lexicon::default())?;
    for (source, marker, kind) in [
        (
            "𠮷。隨後，守衛打開窗戶。",
            "隨後",
            ClauseOrderCueKind::Sequence,
        ),
        (
            "不知過了多久後，守衛打開窗戶。",
            "不知過了多久後",
            ClauseOrderCueKind::Elapsed,
        ),
        (
            "過了片刻之後，守衛打開窗戶。",
            "過了片刻之後",
            ClauseOrderCueKind::Elapsed,
        ),
        (
            "𠮷。一盞茶的工夫之後，守衛打開窗戶。",
            "一盞茶的工夫之後",
            ClauseOrderCueKind::Elapsed,
        ),
        (
            "三年前，守衛打開窗戶。",
            "三年前",
            ClauseOrderCueKind::Reference,
        ),
        (
            "三點五分鐘後，守衛打開窗戶。",
            "三點五分鐘後",
            ClauseOrderCueKind::Elapsed,
        ),
    ] {
        let (linked, cues) = analyzer.link_clause_candidates_with_cues(source, "", &[]);
        assert!(linked.clauses.is_empty(), "{source}");
        assert!(linked.rejected.is_empty(), "{source}");
        let [cue] = cues.as_slice() else {
            panic!("expected one written cue for {source}");
        };
        assert_eq!(cue.kind, kind);
        assert_eq!(cue.cue, linking_span(source, marker, 0));
        assert_eq!(cue.paired_cue, None);
        assert_eq!(cue.scope, ClauseScope::Narrated);
        assert_eq!((cue.before, cue.after), (None, None));
    }
    Ok(())
}

#[test]
fn cue_first_keeps_narrator_time_separate_and_binds_written_pair() -> anyhow::Result<()> {
    use faberline_nlp::{ClauseCandidate, ClauseOrderCueKind};

    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &Lexicon::default())?;
    let source = "匠人關上木門。三年前，守衛打開窗戶。";
    let (_, cues) = analyzer.link_clause_candidates_with_cues(
        source,
        "",
        &[
            ClauseCandidate::new(linking_span(source, "關上", 0)),
            ClauseCandidate::new(linking_span(source, "打開", 0)),
        ],
    );
    assert_eq!(cues.len(), 1);
    assert_eq!(cues[0].kind, ClauseOrderCueKind::Reference);
    assert_eq!((cues[0].before, cues[0].after), (None, None));

    let source = "匠人先關上木門，再打開窗戶。";
    let (_, cues) = analyzer.link_clause_candidates_with_cues(
        source,
        "",
        &[
            ClauseCandidate::new(linking_span(source, "關上", 0)),
            ClauseCandidate::new(linking_span(source, "打開", 0)),
        ],
    );
    assert_eq!(cues.len(), 1);
    assert_eq!(cues[0].kind, ClauseOrderCueKind::FirstThen);
    assert_eq!(cues[0].paired_cue, Some(linking_span(source, "先", 0)));
    assert_eq!(cues[0].cue, linking_span(source, "再", 0));
    assert_eq!((cues[0].before, cues[0].after), (Some(0), Some(1)));
    Ok(())
}

#[test]
fn cue_first_resolves_only_present_and_unambiguous_endpoints() -> anyhow::Result<()> {
    use faberline_nlp::{ClauseCandidate, ClauseOrderCueKind};

    let lexicon = Lexicon {
        actors: &[("岑棠", "岑棠")],
        objects: &[("木門", "木門"), ("窗戶", "窗戶")],
        ..Lexicon::default()
    };
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &lexicon)?;
    let source = "岑棠關上木門。\r\n\r\n隨後守衛打開窗戶。";
    let prior = ClauseCandidate::new(linking_span(source, "關上", 0));
    let following = ClauseCandidate::new(linking_span(source, "打開", 0));
    for (proposals, expected) in [
        (vec![prior], (Some(0), None)),
        (vec![following], (None, Some(0))),
        (vec![prior, prior, following], (None, Some(2))),
        (vec![prior, following, following], (Some(0), None)),
    ] {
        let (linked, cues) = analyzer.link_clause_candidates_with_cues(source, "", &proposals);
        assert_eq!(linked.clauses.len(), proposals.len());
        assert!(linked.rejected.is_empty());
        let [cue] = cues.as_slice() else {
            panic!("expected one cue regardless of missing or duplicate proposals");
        };
        assert_eq!(cue.kind, ClauseOrderCueKind::Sequence);
        assert_eq!(cue.cue, linking_span(source, "隨後", 0));
        assert_eq!((cue.before, cue.after), expected);
    }
    Ok(())
}

#[test]
fn cue_first_binds_the_exact_reverse_order_heads() -> anyhow::Result<()> {
    use faberline_nlp::{ClauseCandidate, ClauseOrderCueKind, ClauseScope};

    let lexicon = Lexicon {
        actors: &[("岑棠", "岑棠")],
        objects: &[("木門", "木門"), ("窗戶", "窗戶")],
        ..Lexicon::default()
    };
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &lexicon)?;
    let source = "岑棠關上了木門之前，守衛打開窗戶。";
    let exact_prior = ClauseCandidate::new(linking_span(source, "關上", 0));
    let exact_following = ClauseCandidate::new(linking_span(source, "打開", 0));
    let prior_prefix = ClauseCandidate::new(linking_span(source, "關", 0));
    let following_with_object = ClauseCandidate::new(linking_span(source, "打開窗戶", 0));
    let (linked, cues) = analyzer.link_clause_candidates_with_cues(
        source,
        "",
        &[
            following_with_object,
            exact_prior,
            prior_prefix,
            exact_following,
        ],
    );
    assert_eq!(linked.clauses.len(), 4);
    let [cue] = cues.as_slice() else {
        panic!("expected one attached before cue");
    };
    assert_eq!(cue.kind, ClauseOrderCueKind::Before);
    assert_eq!(cue.scope, ClauseScope::Narrated);
    assert_eq!(cue.cue, linking_span(source, "之前", 0));
    assert_eq!(cue.paired_cue, None);
    assert_eq!((cue.before, cue.after), (Some(3), Some(1)));

    let (_, partial) = analyzer.link_clause_candidates_with_cues(source, "", &[exact_prior]);
    let [cue] = partial.as_slice() else {
        panic!("expected the before cue without the following proposal");
    };
    assert_eq!((cue.before, cue.after), (None, Some(0)));
    Ok(())
}

#[test]
fn cue_first_does_not_turn_titles_or_quoted_words_into_narrated_order() -> anyhow::Result<()> {
    use faberline_nlp::{ClauseCandidate, ClauseScope};

    let lexicon = Lexicon {
        actors: &[("岑棠", "岑棠")],
        objects: &[("隨後號", "隨後號"), ("天亮之前", "天亮之前")],
        ..Lexicon::default()
    };
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &lexicon)?;
    for source in [
        "岑棠把寫著「隨後」的牌子取出。",
        "岑棠閱讀天亮之前，守衛打開窗戶。",
        "岑棠關上木門，隨後號進港。",
        "岑棠接著繩子，守衛打開窗戶。",
        "先祖關上木門，再打開窗戶。",
        "匠人先關上木門，守衛再三打開窗戶。",
    ] {
        let (_, cues) = analyzer.link_clause_candidates_with_cues(source, "", &[]);
        assert!(cues.is_empty(), "{source}");
    }
    let title = "隨後守衛打開窗戶。";
    let (_, cues) = analyzer.link_clause_candidates_with_cues(title, title, &[]);
    assert!(cues.is_empty());

    let source = "岑棠說道：「隨後守衛打開窗戶。」";
    let proposal = ClauseCandidate::new(linking_span(source, "打開", 0));
    let (_, cues) = analyzer.link_clause_candidates_with_cues(source, "", &[proposal]);
    assert!(cues.iter().all(|cue| {
        cue.scope != ClauseScope::Narrated && (cue.before.is_none() || cue.after.is_none())
    }));
    Ok(())
}

#[test]
fn cue_first_leaves_unwritten_endpoints_open() -> anyhow::Result<()> {
    use faberline_nlp::{ClauseCandidate, ClauseOrderCueKind};

    let lexicon = Lexicon {
        actors: &[("岑棠", "岑棠")],
        objects: &[("木門", "木門"), ("事故記錄", "事故記錄")],
        ..Lexicon::default()
    };
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &lexicon)?;
    for (source, marker, prior, kind, expected) in [
        (
            "岑棠拿出事故記錄。事後守衛打開窗戶。",
            "事後",
            "拿出",
            ClauseOrderCueKind::Reference,
            (None, None),
        ),
        (
            "岑棠關上木門。以後，守衛打開窗戶。",
            "以後",
            "關上",
            ClauseOrderCueKind::Reference,
            (None, None),
        ),
        (
            "岑棠關上木門。隨後守衛打開窗戶的時間還沒到。",
            "隨後",
            "關上",
            ClauseOrderCueKind::Sequence,
            (Some(0), None),
        ),
    ] {
        let (_, cues) = analyzer.link_clause_candidates_with_cues(
            source,
            "",
            &[
                ClauseCandidate::new(linking_span(source, prior, 0)),
                ClauseCandidate::new(linking_span(source, "打開", 0)),
            ],
        );
        let [cue] = cues.as_slice() else {
            panic!("expected one exact cue for {source}");
        };
        assert_eq!(cue.kind, kind, "{source}");
        assert_eq!(cue.cue, linking_span(source, marker, 0));
        assert_eq!((cue.before, cue.after), expected, "{source}");
    }
    Ok(())
}

#[test]
fn cue_first_preserves_whole_event_bound_constructions() -> anyhow::Result<()> {
    use faberline_nlp::{ClauseCandidate, ClauseOrderCueKind};

    let lexicon = Lexicon {
        actors: &[("岑棠", "岑棠")],
        objects: &[("木門", "木門"), ("木屋", "木屋")],
        ..Lexicon::default()
    };
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &lexicon)?;
    for (source, opening, marker, construction, head, next, kind, expected) in [
        (
            "𠮷。岑棠在關上了木門之前，守衛打開了窗戶。",
            Some("在"),
            "之前",
            "在關上了木門之前",
            "關上",
            "打開",
            ClauseOrderCueKind::Before,
            (Some(1), Some(0)),
        ),
        (
            "在岑棠關上木門之後守衛打開了窗戶。",
            Some("在"),
            "之後",
            "在岑棠關上木門之後",
            "關上",
            "打開",
            ClauseOrderCueKind::After,
            (Some(0), Some(1)),
        ),
        (
            "守衛走進了木屋後，岑棠取出羅盤。",
            None,
            "後",
            "走進了木屋後",
            "走進",
            "取出",
            ClauseOrderCueKind::After,
            (Some(0), Some(1)),
        ),
    ] {
        let candidates = [
            ClauseCandidate::new(linking_span(source, head, 0)),
            ClauseCandidate::new(linking_span(source, next, 0)),
        ];
        let (_, cues) = analyzer.link_clause_candidates_with_cues(source, "", &candidates);
        let (_, empty) = analyzer.link_clause_candidates_with_cues(source, "", &[]);
        let [cue] = cues.as_slice() else {
            panic!("expected one event-bound cue for {source}");
        };
        let [unbound] = empty.as_slice() else {
            panic!("expected the same written cue without proposals for {source}");
        };
        assert_eq!(cue.kind, kind, "{source}");
        assert_eq!(cue.cue, linking_span(source, marker, 0));
        assert_eq!(
            cue.paired_cue,
            opening.map(|word| linking_span(source, word, 0))
        );
        assert_eq!(
            cue.construction,
            Some(linking_span(source, construction, 0))
        );
        assert_eq!((cue.before, cue.after), expected, "{source}");
        assert_eq!(unbound.construction, cue.construction);
        assert_eq!((unbound.before, unbound.after), (None, None));
    }
    Ok(())
}

#[test]
fn cue_first_pairs_completed_actions_and_keeps_other_pairs_open() -> anyhow::Result<()> {
    use faberline_nlp::{ClauseCandidate, ClauseOrderCueKind};

    let lexicon = Lexicon {
        actors: &[("岑棠", "岑棠")],
        objects: &[("木門", "木門")],
        ..Lexicon::default()
    };
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &lexicon)?;
    for (source, first, second, construction, realized) in [
        (
            "岑棠剛關上了木門，守衛就打開了窗戶。",
            "剛",
            "就",
            "剛關上了木門，守衛就",
            true,
        ),
        (
            "岑棠方一關上了木門，守衛便打開了窗戶。",
            "方一",
            "便",
            "方一關上了木門，守衛便",
            true,
        ),
        (
            "岑棠一關上木門，守衛就打開窗戶。",
            "一",
            "就",
            "一關上木門，守衛就",
            false,
        ),
    ] {
        let (_, cues) = analyzer.link_clause_candidates_with_cues(
            source,
            "",
            &[
                ClauseCandidate::new(linking_span(source, "關上", 0)),
                ClauseCandidate::new(linking_span(source, "打開", 0)),
            ],
        );
        let [cue] = cues.as_slice() else {
            panic!("expected one paired cue for {source}");
        };
        assert_eq!(cue.kind, ClauseOrderCueKind::FirstThen);
        assert_eq!(cue.paired_cue, Some(linking_span(source, first, 0)));
        assert_eq!(cue.cue, linking_span(source, second, 0));
        assert_eq!(
            cue.construction,
            Some(linking_span(source, construction, 0))
        );
        let expected = if realized {
            (Some(0), Some(1))
        } else {
            (None, None)
        };
        assert_eq!((cue.before, cue.after), expected, "{source}");
    }
    Ok(())
}

#[test]
fn cue_first_rejects_spatial_landmarks_and_nonactual_pairs() -> anyhow::Result<()> {
    use faberline_nlp::ClauseCandidate;

    let lexicon = Lexicon {
        actors: &[("岑棠", "岑棠")],
        objects: &[("木門", "木門"), ("木桌", "木桌")],
        ..Lexicon::default()
    };
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &lexicon)?;
    for (source, head) in [
        ("岑棠走到木門前，守衛打開窗戶。", "走到"),
        ("岑棠走到木門之前，守衛打開窗戶。", "走到"),
        ("岑棠推到木桌後，守衛打開窗戶。", "推到"),
        ("守衛站在木門之前，岑棠打開窗戶。", "站在"),
    ] {
        let (_, cues) = analyzer.link_clause_candidates_with_cues(
            source,
            "",
            &[
                ClauseCandidate::new(linking_span(source, head, 0)),
                ClauseCandidate::new(linking_span(source, "打開", 0)),
            ],
        );
        assert!(cues.is_empty(), "{source}");
    }
    let source = "岑棠聽到消息之後，守衛打開窗戶。";
    let (_, cues) = analyzer.link_clause_candidates_with_cues(source, "", &[]);
    let [cue] = cues.as_slice() else {
        panic!("expected the written marker after a non-spatial 到");
    };
    assert_eq!(cue.cue, linking_span(source, "之後", 0));
    for source in [
        "如果岑棠剛關上了木門，守衛就打開了窗戶。",
        "岑棠剛想關上木門，守衛就打開窗戶。",
        "「岑棠剛關上了木門，守衛就打開了窗戶。」",
        "岑棠剛關上了木門，守衛就打開了窗戶的時機還沒到。",
        "岑棠剛關上了木門的傳聞四處流傳，守衛就打開了窗戶。",
        "一人關上了木門，守衛就打開了窗戶。",
        "岑棠剛關上了木門，就守衛打開了窗戶的說法而言，情況未明。",
    ] {
        let (_, cues) = analyzer.link_clause_candidates_with_cues(
            source,
            "",
            &[
                ClauseCandidate::new(linking_span(source, "關上", 0)),
                ClauseCandidate::new(linking_span(source, "打開", 0)),
            ],
        );
        assert!(
            cues.iter()
                .all(|cue| cue.before.is_none() || cue.after.is_none()),
            "{source}"
        );
    }
    let source = "在岑棠關上木門之前，守衛打開了窗戶。岑棠最終沒有關門。";
    let (_, cues) = analyzer.link_clause_candidates_with_cues(
        source,
        "",
        &[
            ClauseCandidate::new(linking_span(source, "關上", 0)),
            ClauseCandidate::new(linking_span(source, "打開", 0)),
        ],
    );
    let [cue] = cues.as_slice() else {
        panic!("expected the construction before an unresolved action");
    };
    assert_eq!((cue.before, cue.after), (Some(1), None));
    Ok(())
}

#[test]
fn ranked_cues_offer_an_unlisted_action_with_written_evidence() -> anyhow::Result<()> {
    use faberline_nlp::ClauseCandidate;

    let lexicon = Lexicon {
        actors: &[("岑棠", "岑棠")],
        objects: &[("羅盤", "羅盤"), ("燈火", "燈火")],
        ..Lexicon::default()
    };
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &lexicon)?;
    let source = "岑棠收起羅盤。隨後守衛熄滅了燈火。";
    let candidates = [
        ClauseCandidate::new(linking_span(source, "收起", 0)),
        ClauseCandidate::new(linking_span(source, "熄滅", 0)),
    ];
    let (linked, cues) = analyzer.link_clause_candidates_with_ranked_cues(source, "", &candidates);
    let [cue] = cues.as_slice() else {
        panic!("expected one cue")
    };
    assert_eq!((cue.order.before, cue.order.after), (Some(0), None));
    let [option] = cue.after_alternatives.as_slice() else {
        panic!("expected one written action")
    };
    assert_eq!(option.clause, 1);
    assert_eq!(option.head, linking_span(source, "熄滅", 0));
    assert_eq!(option.completion, Some(linking_span(source, "了", 0)));
    assert_eq!(option.argument, Some(linking_span(source, "燈火", 0)));
    assert!(!option.lexical_head);
    assert!(!option.eligible);
    assert_eq!(linked.clauses[option.clause].candidate, candidates[1]);
    let (old, orders) = analyzer.link_clause_candidates_with_cues(source, "", &candidates);
    assert_eq!(linked, old);
    assert_eq!(orders, [cue.order]);
    Ok(())
}

#[test]
fn ranked_cues_split_lexical_heads_from_wider_proposals_and_completion() -> anyhow::Result<()> {
    use faberline_nlp::ClauseCandidate;

    let lexicon = Lexicon {
        objects: &[("窗戶", "窗戶"), ("地圖", "地圖"), ("過濾器", "過濾器")],
        ..Lexicon::default()
    };
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &lexicon)?;
    for (source, full, shorter, head, argument, chosen) in [
        (
            "隨後守衛打開了窗戶。",
            "打開了窗戶",
            "打開",
            "打開",
            "窗戶",
            1,
        ),
        ("隨後守衛發現了地圖。", "發現了", "發現", "發現", "地圖", 0),
    ] {
        let (_, cues) = analyzer.link_clause_candidates_with_ranked_cues(
            source,
            "",
            &[
                ClauseCandidate::new(linking_span(source, full, 0)),
                ClauseCandidate::new(linking_span(source, shorter, 0)),
            ],
        );
        let [cue] = cues.as_slice() else {
            panic!("expected one cue for {source}")
        };
        assert_eq!(cue.order.after, Some(chosen));
        let [exact, alternative] = cue.after_alternatives.as_slice() else {
            panic!("expected two ranked source ranges for {source}")
        };
        assert_eq!(exact.clause, chosen);
        assert!(exact.eligible);
        assert!(!alternative.eligible);
        for option in [exact, alternative] {
            assert!(option.lexical_head);
            assert_eq!(option.head, linking_span(source, head, 0));
            assert_eq!(option.completion, Some(linking_span(source, "了", 0)));
            assert_eq!(option.argument, Some(linking_span(source, argument, 0)));
        }
    }
    let source = "追了上去。隨後守衛收起地圖。";
    let (_, cues) = analyzer.link_clause_candidates_with_ranked_cues(
        source,
        "",
        &[
            ClauseCandidate::new(linking_span(source, "追了上去", 0)),
            ClauseCandidate::new(linking_span(source, "收起", 0)),
        ],
    );
    let [cue] = cues.as_slice() else {
        panic!("expected one cue")
    };
    assert_eq!((cue.order.before, cue.order.after), (Some(0), Some(1)));
    let [option] = cue.before_alternatives.as_slice() else {
        panic!("expected the written directional action")
    };
    assert_eq!(option.head, linking_span(source, "追", 0));
    assert_eq!(option.completion, Some(linking_span(source, "了", 0)));
    assert_eq!(option.argument, None);
    let source = "隨後守衛收起過濾器。";
    let (_, cues) = analyzer.link_clause_candidates_with_ranked_cues(
        source,
        "",
        &[ClauseCandidate::new(linking_span(source, "收起", 0))],
    );
    let [option] = cues[0].after_alternatives.as_slice() else {
        panic!("expected the written object")
    };
    assert_eq!(option.completion, None);
    assert_eq!(option.argument, Some(linking_span(source, "過濾器", 0)));
    Ok(())
}

#[test]
fn ranked_cues_explain_an_uncompleted_before_anchor_and_competing_inputs() -> anyhow::Result<()> {
    use faberline_nlp::ClauseCandidate;

    let lexicon = Lexicon {
        actors: &[
            ("陳東", "陳東"),
            ("林卓", "林卓"),
            ("了塵", "了塵"),
            ("塵", "塵"),
        ],
        objects: &[("石室", "石室"), ("地圖", "地圖")],
        ..Lexicon::default()
    };
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &lexicon)?;
    let source = "陳東走出石室之前，林卓收起了地圖。";
    let (_, cues) = analyzer.link_clause_candidates_with_ranked_cues(
        source,
        "",
        &[
            ClauseCandidate::new(linking_span(source, "走出", 0)),
            ClauseCandidate::new(linking_span(source, "收起", 0)),
        ],
    );
    let [cue] = cues.as_slice() else {
        panic!("expected one before cue")
    };
    assert_eq!((cue.order.before, cue.order.after), (Some(1), None));
    let [option] = cue.after_alternatives.as_slice() else {
        panic!("expected an unresolved action")
    };
    assert_eq!(option.head, linking_span(source, "走出", 0));
    assert_eq!(option.argument, Some(linking_span(source, "石室", 0)));
    assert_eq!(option.completion, None);
    assert!(!option.eligible);

    let source = "陳東找到了塵之前，林卓收起地圖。";
    let (_, cues) = analyzer.link_clause_candidates_with_ranked_cues(
        source,
        "",
        &[
            ClauseCandidate::new(linking_span(source, "找到了", 0)),
            ClauseCandidate::new(linking_span(source, "收起", 0)),
        ],
    );
    let [cue] = cues.as_slice() else {
        panic!("expected one before cue with a named argument")
    };
    assert_eq!((cue.order.before, cue.order.after), (Some(1), None));
    let [option] = cue.after_alternatives.as_slice() else {
        panic!("expected the written action with an unresolved completion")
    };
    assert_eq!(option.argument, Some(linking_span(source, "了塵", 0)));
    assert_eq!(option.completion, None);
    assert!(!option.eligible);

    let source = "隨後林卓收起了地圖。";
    let action = ClauseCandidate::new(linking_span(source, "收起", 0));
    let (_, cues) = analyzer.link_clause_candidates_with_ranked_cues(source, "", &[action; 7]);
    let [cue] = cues.as_slice() else {
        panic!("expected one sequence cue")
    };
    assert_eq!(cue.order.after, None);
    assert_eq!(cue.after_alternatives.len(), 5);
    assert!(cue.after_alternatives.iter().all(|option| option.eligible));
    Ok(())
}

#[test]
fn ranked_cues_leave_nonactual_and_spatial_readings_out_of_endpoint_lists() -> anyhow::Result<()> {
    use faberline_nlp::{ClauseCandidate, ClauseOrderCueKind};

    let lexicon = Lexicon {
        actors: &[("岑棠", "岑棠")],
        objects: &[("燈火", "燈火")],
        ..Lexicon::default()
    };
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &lexicon)?;
    for (source, head) in [
        ("隨後岑棠四十歲了。", "四十歲"),
        ("隨後守衛熄滅了燈火的傳聞四處流傳。", "熄滅"),
        ("如果岑棠收起了燈火，隨後守衛熄滅了燈火。", "熄滅"),
        ("隨後守衛沒有熄滅燈火。", "熄滅"),
        ("隨後守衛假裝熄滅了燈火。", "假裝熄滅"),
        ("隨後守衛謊稱熄滅了燈火。", "謊稱熄滅"),
    ] {
        let (_, cues) = analyzer.link_clause_candidates_with_ranked_cues(
            source,
            "",
            &[ClauseCandidate::new(linking_span(source, head, 0))],
        );
        assert!(
            cues.iter()
                .all(|cue| cue.before_alternatives.is_empty() && cue.after_alternatives.is_empty()),
            "{source}"
        );
    }
    let source = "岑棠走到陌生巷口之前，守衛收起燈火。";
    let (_, cues) = analyzer.link_clause_candidates_with_ranked_cues(source, "", &[]);
    let [cue] = cues.as_slice() else {
        panic!("expected the ambiguous written reference")
    };
    assert_eq!(cue.order.kind, ClauseOrderCueKind::Reference);
    assert_eq!((cue.order.before, cue.order.after), (None, None));
    Ok(())
}

#[test]
fn verified_heads_rank_unlisted_actions_on_each_side_without_binding() -> anyhow::Result<()> {
    use faberline_nlp::{ClauseCandidate, VerifiedClauseHead};

    let lexicon = Lexicon {
        objects: &[("羅盤", "羅盤"), ("燈火", "燈火")],
        ..Lexicon::default()
    };
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &lexicon)?;
    let source = "𠮷。守衛拂拭了羅盤。隨後工人熄滅燈火。";
    let invalid = Span {
        start_char: usize::MAX,
        end_char: usize::MAX,
    };
    let left = linking_span(source, "拂拭", 0);
    let right = linking_span(source, "熄滅", 0);
    let candidates = [
        ClauseCandidate::new(invalid),
        ClauseCandidate::new(right),
        ClauseCandidate::new(linking_span(source, "拂拭了羅盤", 0)),
    ];
    let (plain, baseline) =
        analyzer.link_clause_candidates_with_ranked_cues(source, "", &candidates);
    let [base_cue] = baseline.as_slice() else {
        panic!("expected one written sequence cue")
    };
    assert!(base_cue.before_alternatives.is_empty());
    assert!(base_cue.after_alternatives.is_empty());

    let (linked, ranked) = analyzer.link_clause_candidates_with_ranked_cues_and_heads(
        source,
        "",
        &candidates,
        &[
            VerifiedClauseHead {
                input_index: 2,
                head: left,
            },
            VerifiedClauseHead {
                input_index: 1,
                head: right,
            },
        ],
    );
    assert_eq!(linked, plain);
    assert_eq!(linked.rejected, [0]);
    let [cue] = ranked.as_slice() else {
        panic!("expected the same written cue")
    };
    assert_eq!(cue.order, base_cue.order);
    assert_eq!((cue.order.before, cue.order.after), (None, None));
    let [before] = cue.before_alternatives.as_slice() else {
        panic!("expected the action before the cue")
    };
    let [after] = cue.after_alternatives.as_slice() else {
        panic!("expected the action after the cue")
    };
    assert_eq!((before.clause, before.head), (1, left));
    assert_eq!(before.completion, Some(linking_span(source, "了", 0)));
    assert_eq!(before.argument, Some(linking_span(source, "羅盤", 0)));
    assert_eq!((after.clause, after.head), (0, right));
    assert_eq!(after.completion, None);
    assert_eq!(after.argument, Some(linking_span(source, "燈火", 0)));
    for option in [before, after] {
        assert!(!option.lexical_head);
        assert!(!option.eligible);
    }

    let invalid_hints = [
        VerifiedClauseHead {
            input_index: usize::MAX,
            head: right,
        },
        VerifiedClauseHead {
            input_index: 0,
            head: right,
        },
        VerifiedClauseHead {
            input_index: 1,
            head: invalid,
        },
        VerifiedClauseHead {
            input_index: 1,
            head: left,
        },
        VerifiedClauseHead {
            input_index: 1,
            head: linking_span(source, "熄滅燈火", 0),
        },
        VerifiedClauseHead {
            input_index: 2,
            head: linking_span(source, "拭", 0),
        },
    ];
    assert_eq!(
        analyzer.link_clause_candidates_with_ranked_cues_and_heads(
            source,
            "",
            &candidates,
            &invalid_hints,
        ),
        (plain, baseline)
    );
    Ok(())
}

#[test]
fn verified_head_ranks_the_later_action_in_a_before_construction() -> anyhow::Result<()> {
    use faberline_nlp::{ClauseCandidate, ClauseOrderCueKind, VerifiedClauseHead};

    let lexicon = Lexicon {
        objects: &[("羅盤", "羅盤"), ("燈火", "燈火")],
        ..Lexicon::default()
    };
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &lexicon)?;
    let source = "在守衛拂拭了羅盤之前，工人收起燈火。";
    let head = linking_span(source, "拂拭", 0);
    let candidates = [
        ClauseCandidate::new(linking_span(source, "拂拭了羅盤", 0)),
        ClauseCandidate::new(linking_span(source, "收起", 0)),
    ];
    let (plain, baseline) =
        analyzer.link_clause_candidates_with_ranked_cues(source, "", &candidates);
    let (linked, ranked) = analyzer.link_clause_candidates_with_ranked_cues_and_heads(
        source,
        "",
        &candidates,
        &[VerifiedClauseHead {
            input_index: 0,
            head,
        }],
    );
    assert_eq!(linked, plain);
    let ([base_cue], [cue]) = (baseline.as_slice(), ranked.as_slice()) else {
        panic!("expected one written before cue in both outputs")
    };
    assert_eq!(cue.order, base_cue.order);
    assert_eq!(cue.order.kind, ClauseOrderCueKind::Before);
    assert_eq!(cue.order.cue, linking_span(source, "之前", 0));
    assert_eq!((cue.order.before, cue.order.after), (Some(1), None));
    assert!(base_cue.after_alternatives.is_empty());
    let [option] = cue.after_alternatives.as_slice() else {
        panic!("expected the action written before the cue on its later side")
    };
    assert_eq!((option.clause, option.head), (0, head));
    assert_eq!(option.completion, Some(linking_span(source, "了", 0)));
    assert_eq!(option.argument, Some(linking_span(source, "羅盤", 0)));
    assert!(!option.lexical_head);
    assert!(!option.eligible);
    Ok(())
}

#[test]
fn verified_heads_keep_local_subject_scope_and_landmark_guards() -> anyhow::Result<()> {
    use faberline_nlp::{ClauseCandidate, VerifiedClauseHead};

    let lexicon = Lexicon {
        objects: &[("羅盤", "羅盤"), ("燈火", "燈火")],
        ..Lexicon::default()
    };
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &lexicon)?;
    for (source, head) in [
        ("隨後守衛抬手熄滅燈火。", "熄滅"),
        ("隨後工人收起羅盤，守衛熄滅燈火。", "熄滅"),
        ("工人說：「隨後守衛熄滅燈火。」", "熄滅"),
        ("隨後守衛沒有熄滅燈火。", "熄滅"),
        ("隨後熄滅了燈火的婦人離開了。", "熄滅"),
        ("隨後守衛羅盤掉了。", "羅盤"),
        ("守衛站在陌生窗台之前，工人收起羅盤。", "站"),
    ] {
        let head = linking_span(source, head, 0);
        let candidates = [ClauseCandidate::new(head)];
        let baseline = analyzer.link_clause_candidates_with_ranked_cues(source, "", &candidates);
        let actual = analyzer.link_clause_candidates_with_ranked_cues_and_heads(
            source,
            "",
            &candidates,
            &[VerifiedClauseHead {
                input_index: 0,
                head,
            }],
        );
        assert_eq!(actual, baseline, "{source}");
    }

    let source = "隨後守衛熄滅了燈火的傳聞四處流傳。";
    let head = linking_span(source, "熄滅", 0);
    let proposal = linking_span(source, "熄滅了燈火的傳聞", 0);
    let (linked, cues) = analyzer.link_clause_candidates_with_ranked_cues_and_heads(
        source,
        "",
        &[ClauseCandidate::new(proposal)],
        &[VerifiedClauseHead {
            input_index: 0,
            head,
        }],
    );
    assert!(linked.rejected.is_empty());
    assert!(cues[0].after_alternatives.is_empty());
    Ok(())
}

#[test]
fn verified_subjects_rank_unlisted_subjects_without_binding() -> anyhow::Result<()> {
    use faberline_nlp::{ClauseCandidate, VerifiedClauseEvidence, VerifiedClauseHead};

    let lexicon = Lexicon {
        objects: &[("羅盤", "羅盤"), ("燈火", "燈火")],
        ..Lexicon::default()
    };
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &lexicon)?;
    let source = "郵差拂拭羅盤。隨後顧青禾便熄滅燈火。";
    let before = linking_span(source, "拂拭", 0);
    let after = linking_span(source, "熄滅", 0);
    let subject = linking_span(source, "郵差", 0);
    let candidates = [
        ClauseCandidate {
            actor: Some(subject),
            ..ClauseCandidate::new(before)
        },
        ClauseCandidate::new(after),
    ];
    let heads = [
        VerifiedClauseHead {
            input_index: 0,
            head: before,
        },
        VerifiedClauseHead {
            input_index: 1,
            head: after,
        },
    ];
    let (plain, baseline) =
        analyzer.link_clause_candidates_with_ranked_cues_and_heads(source, "", &candidates, &heads);
    let (linked, ranked) = analyzer.link_clause_candidates_with_ranked_cues_and_evidence(
        source,
        "",
        &candidates,
        &[
            VerifiedClauseEvidence {
                head: heads[0],
                subject: Some(subject),
            },
            VerifiedClauseEvidence {
                head: heads[1],
                subject: Some(linking_span(source, "顧青禾", 0)),
            },
        ],
    );
    assert_eq!(linked, plain);
    let ([base_cue], [cue]) = (baseline.as_slice(), ranked.as_slice()) else {
        panic!("expected the same written sequence cue")
    };
    assert_eq!(cue.order, base_cue.order);
    assert!(base_cue.before_alternatives.is_empty());
    assert!(base_cue.after_alternatives.is_empty());
    let ([left], [right]) = (
        cue.before_alternatives.as_slice(),
        cue.after_alternatives.as_slice(),
    ) else {
        panic!("expected an advisory action on each side")
    };
    assert_eq!((left.clause, left.head), (0, before));
    assert_eq!((right.clause, right.head), (1, after));
    assert!([left, right].iter().all(|option| !option.eligible));

    for (source, subject) in [
        ("隨後郵差演練熄滅燈火。", "郵差"),
        ("隨後郵差熄滅燈火。", "熄滅"),
        ("隨後另一邊的郵差熄滅燈火。", "另一邊的郵差"),
    ] {
        let head = linking_span(source, "熄滅", 0);
        let (linked, cues) = analyzer.link_clause_candidates_with_ranked_cues_and_evidence(
            source,
            "",
            &[ClauseCandidate::new(head)],
            &[VerifiedClauseEvidence {
                head: VerifiedClauseHead {
                    input_index: 0,
                    head,
                },
                subject: Some(linking_span(source, subject, 0)),
            }],
        );
        assert!(linked.rejected.is_empty(), "{source}");
        assert!(cues[0].after_alternatives.is_empty(), "{source}");
    }
    Ok(())
}

#[test]
fn attached_before_cue_requires_completion_for_the_later_action() -> anyhow::Result<()> {
    use faberline_nlp::{ClauseCandidate, ClauseOrderCueKind};

    let lexicon = Lexicon {
        actors: &[("岑棠", "岑棠")],
        objects: &[("舊地圖", "舊地圖"), ("窗戶", "窗戶")],
        ..Lexicon::default()
    };
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &lexicon)?;
    for (source, completed) in [
        (
            "岑棠先收起舊地圖之前，守衛打開了窗戶。岑棠最終沒有收起。",
            false,
        ),
        ("岑棠先收起了舊地圖之前，守衛打開了窗戶。", true),
    ] {
        let (_, cues) = analyzer.link_clause_candidates_with_ranked_cues(
            source,
            "",
            &[
                ClauseCandidate::new(linking_span(source, "收起", 0)),
                ClauseCandidate::new(linking_span(source, "打開", 0)),
            ],
        );
        let [cue] = cues.as_slice() else {
            panic!("expected one before cue for {source}")
        };
        assert_eq!(cue.order.kind, ClauseOrderCueKind::Before);
        assert_eq!(cue.order.before, Some(1));
        assert_eq!(cue.order.after, completed.then_some(0));
        let [option] = cue.after_alternatives.as_slice() else {
            panic!("expected the written action as an alternative for {source}")
        };
        assert_eq!(option.clause, 0);
        assert_eq!(option.completion.is_some(), completed);
        assert_eq!(option.eligible, completed);
    }
    Ok(())
}

#[test]
fn ranked_sequence_can_look_past_one_local_description() -> anyhow::Result<()> {
    use faberline_nlp::{ClauseCandidate, ClauseOrderCueKind, VerifiedClauseHead};

    let lexicon = Lexicon {
        objects: &[("羅盤", "羅盤"), ("窗戶", "窗戶")],
        ..Lexicon::default()
    };
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &lexicon)?;
    for (source, predicate, head, lexical) in [
        (
            "守衛收起了羅盤。屋內仍然寂靜。隨後工人打開了窗戶。",
            "收起",
            "收起",
            true,
        ),
        (
            "守衛拂拭了羅盤，屋內一片昏暗，隨後工人打開了窗戶。",
            "拂拭了羅盤",
            "拂拭",
            false,
        ),
        (
            "守衛收起了羅盤。屋內仍然寂靜。隨後，工人打開了窗戶。",
            "收起",
            "收起",
            true,
        ),
    ] {
        let head = linking_span(source, head, 0);
        let candidates = [
            ClauseCandidate::new(linking_span(source, predicate, 0)),
            ClauseCandidate::new(linking_span(source, "打開", 0)),
        ];
        let verified = (!lexical).then_some(VerifiedClauseHead {
            input_index: 0,
            head,
        });
        let (linked, ranked) = analyzer.link_clause_candidates_with_ranked_cues_and_heads(
            source,
            "",
            &candidates,
            verified.as_slice(),
        );
        let (strict_linked, strict) =
            analyzer.link_clause_candidates_with_cues(source, "", &candidates);
        assert_eq!(linked, strict_linked, "{source}");
        assert!(linked.rejected.is_empty(), "{source}");
        let [cue] = ranked.as_slice() else {
            panic!("expected one sequence cue for {source}");
        };
        assert_eq!(strict, [cue.order], "{source}");
        assert_eq!(cue.order.kind, ClauseOrderCueKind::Sequence, "{source}");
        assert_eq!(cue.order.cue, linking_span(source, "隨後", 0), "{source}");
        assert_eq!(
            (cue.order.before, cue.order.after),
            (None, Some(1)),
            "{source}"
        );
        let [earlier] = cue.before_alternatives.as_slice() else {
            panic!("expected an earlier advisory action for {source}");
        };
        assert_eq!((earlier.clause, earlier.head), (0, head), "{source}");
        assert_eq!(
            earlier.completion,
            Some(linking_span(source, "了", 0)),
            "{source}"
        );
        assert_eq!(
            earlier.argument,
            Some(linking_span(source, "羅盤", 0)),
            "{source}"
        );
        assert_eq!(earlier.lexical_head, lexical, "{source}");
        assert!(!earlier.eligible, "{source}");
    }
    Ok(())
}

#[test]
fn ranked_sequence_does_not_skip_actions_or_boundaries() -> anyhow::Result<()> {
    use faberline_nlp::{ClauseCandidate, ClauseOrderCueKind, VerifiedClauseHead};

    let lexicon = Lexicon {
        objects: &[("羅盤", "羅盤"), ("窗戶", "窗戶")],
        ..Lexicon::default()
    };
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &lexicon)?;
    for source in [
        "守衛收起了羅盤。工人敲了三下鐘。隨後工人打開了窗戶。",
        "守衛收起了羅盤，屋內忽然安靜下來，隨後工人打開了窗戶。",
        "守衛收起了羅盤。工人說：「屋內仍然寂靜。」隨後工人打開了窗戶。",
        "在睡夢中，守衛收起了羅盤，屋內仍然寂靜，隨後工人打開了窗戶。",
        "守衛收起了羅盤。屋內仍然寂靜。隨後另一邊，工人打開了窗戶。",
        "守衛收起了羅盤。屋內仍然寂靜。窗外一片昏暗。隨後工人打開了窗戶。",
        "守衛收起了羅盤。\n屋內仍然寂靜。隨後工人打開了窗戶。",
        "守衛收起了羅盤。屋內仍然寂靜。\n隨後工人打開了窗戶。",
    ] {
        let head = linking_span(source, "收起", 0);
        let candidates = [
            ClauseCandidate::new(head),
            ClauseCandidate::new(linking_span(source, "打開", 0)),
        ];
        let (linked, ranked) = analyzer.link_clause_candidates_with_ranked_cues_and_heads(
            source,
            "",
            &candidates,
            &[VerifiedClauseHead {
                input_index: 0,
                head,
            }],
        );
        assert!(linked.rejected.is_empty(), "{source}");
        let [cue] = ranked.as_slice() else {
            panic!("expected one sequence cue for {source}");
        };
        assert_eq!(cue.order.kind, ClauseOrderCueKind::Sequence, "{source}");
        assert_eq!(cue.order.before, None, "{source}");
        assert!(cue.before_alternatives.is_empty(), "{source}");
    }
    Ok(())
}

#[test]
fn ranked_evidence_distinguishes_local_placement_and_completion() -> anyhow::Result<()> {
    use faberline_nlp::{
        ClauseCandidate, ClauseOrderCueKind, VerifiedClauseEvidence, VerifiedClauseHead,
    };

    let lexicon = Lexicon {
        objects: &[("羅盤", "羅盤"), ("燈火", "燈火"), ("窗戶", "窗戶")],
        ..Lexicon::default()
    };
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &lexicon)?;
    for (source, subject, use_prior) in [
        ("守衛拂拭羅盤。郵差隨後熄滅燈火。", "郵差", true),
        ("守衛拂拭羅盤。\n郵差隨後熄滅燈火。", "郵差", false),
        ("守衛拂拭羅盤。\u{2028}郵差隨後熄滅燈火。", "郵差", false),
        ("守衛拂拭羅盤。郵差\u{b}隨後熄滅燈火。", "郵差", false),
        ("守衛拂拭羅盤。郵差隨後\u{c}熄滅燈火。", "郵差", false),
        ("守衛拂拭羅盤。郵差抬手隨後熄滅燈火。", "郵差", false),
        (
            "守衛拂拭羅盤。剛打開了窗戶的郵差隨後熄滅燈火。",
            "剛打開了窗戶的郵差",
            false,
        ),
    ] {
        let earlier = linking_span(source, "拂拭", 0);
        let later = linking_span(source, "熄滅", 0);
        let candidates = [ClauseCandidate::new(earlier), ClauseCandidate::new(later)];
        let heads = [
            VerifiedClauseHead {
                input_index: 0,
                head: earlier,
            },
            VerifiedClauseHead {
                input_index: 1,
                head: later,
            },
        ];
        let (plain, base) = analyzer.link_clause_candidates_with_ranked_cues_and_heads(
            source,
            "",
            &candidates,
            &heads,
        );
        let (linked, ranked) = analyzer.link_clause_candidates_with_ranked_cues_and_evidence(
            source,
            "",
            &candidates,
            &[
                VerifiedClauseEvidence {
                    head: heads[0],
                    subject: None,
                },
                VerifiedClauseEvidence {
                    head: heads[1],
                    subject: Some(linking_span(source, subject, 0)),
                },
            ],
        );
        assert_eq!(linked, plain, "{source}");
        let ([base], [cue]) = (base.as_slice(), ranked.as_slice()) else {
            panic!("expected one written cue for {source}");
        };
        assert!(base.before_alternatives.is_empty(), "{source}");
        assert_eq!(cue.order, base.order, "{source}");
        assert_eq!(
            cue.before_alternatives.len(),
            usize::from(use_prior),
            "{source}"
        );
        if let Some(option) = cue.before_alternatives.first() {
            assert_eq!((option.clause, option.head), (0, earlier), "{source}");
            assert!(!option.eligible, "{source}");
        }
    }

    for (source, listed) in [
        ("守衛已經拂拭羅盤，隨後工人熄滅燈火。", true),
        ("守衛早已拂拭羅盤，隨後工人熄滅燈火。", true),
        ("守衛抬手已經拂拭羅盤，隨後工人熄滅燈火。", false),
        ("守衛假裝拂拭羅盤，隨後工人熄滅燈火。", false),
    ] {
        let head = linking_span(source, "拂拭", 0);
        let candidates = [ClauseCandidate::new(head)];
        let (plain, baseline) =
            analyzer.link_clause_candidates_with_ranked_cues(source, "", &candidates);
        let (linked, ranked) = analyzer.link_clause_candidates_with_ranked_cues_and_evidence(
            source,
            "",
            &candidates,
            &[VerifiedClauseEvidence {
                head: VerifiedClauseHead {
                    input_index: 0,
                    head,
                },
                subject: None,
            }],
        );
        assert_eq!(linked, plain, "{source}");
        let ([base], [cue]) = (baseline.as_slice(), ranked.as_slice()) else {
            panic!("expected one sequence cue for {source}");
        };
        assert_eq!(cue.order, base.order, "{source}");
        assert_eq!(
            cue.before_alternatives.len(),
            usize::from(listed),
            "{source}"
        );
        if let Some(option) = cue.before_alternatives.first() {
            assert_eq!(option.head, head, "{source}");
            assert_eq!(option.completion, None, "{source}");
            assert!(!option.eligible, "{source}");
        }
    }

    for (source, completed) in [
        ("守衛剛收起羅盤，工人就打開了窗戶。", false),
        ("守衛剛收起了羅盤，工人就打開了窗戶。", true),
    ] {
        let earlier = linking_span(source, "收起", 0);
        let (_, cues) = analyzer.link_clause_candidates_with_ranked_cues(
            source,
            "",
            &[
                ClauseCandidate::new(earlier),
                ClauseCandidate::new(linking_span(source, "打開", 0)),
            ],
        );
        let [cue] = cues.as_slice() else {
            panic!("expected one written pair for {source}");
        };
        assert_eq!(cue.order.kind, ClauseOrderCueKind::FirstThen);
        assert_eq!(cue.order.before, completed.then_some(0));
        assert_eq!(cue.order.after, completed.then_some(1));
        let ([before], [after]) = (
            cue.before_alternatives.as_slice(),
            cue.after_alternatives.as_slice(),
        ) else {
            panic!("expected both written actions for {source}");
        };
        assert_eq!((before.clause, before.head), (0, earlier));
        assert_eq!(before.completion.is_some(), completed);
        assert!(after.completion.is_some());
        assert_eq!(before.eligible, completed);
    }
    Ok(())
}

#[test]
fn cue_diagnostics_keep_sparse_probe_order_and_original_windows() -> anyhow::Result<()> {
    use faberline_nlp::{
        ClauseCandidate, ClauseOrderAdvisory, ClauseOrderBinding, ClauseOrderProbe, ClauseOrderSide,
    };

    let lexicon = Lexicon {
        objects: &[("羅盤", "羅盤"), ("窗戶", "窗戶")],
        ..Lexicon::default()
    };
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &lexicon)?;
    let source = "𠮷。守衛收起了羅盤。隨後工人打開了窗戶。三年後，守衛離開了。";
    let invalid = Span {
        start_char: usize::MAX,
        end_char: usize::MAX,
    };
    let sequence = linking_span(source, "隨後", 0);
    assert_eq!(sequence.start_char, 10);
    let candidates = [
        ClauseCandidate::new(invalid),
        ClauseCandidate::new(linking_span(source, "打開", 0)),
        ClauseCandidate::new(linking_span(source, "收起", 0)),
        ClauseCandidate::new(linking_span(source, "離開", 0)),
    ];
    let probe = |cue, side, input_index| ClauseOrderProbe {
        cue,
        side,
        input_index,
    };
    let after = probe(sequence, ClauseOrderSide::After, 1);
    let probes = [
        after,
        probe(
            linking_span(source, "三年後", 0),
            ClauseOrderSide::Before,
            3,
        ),
        probe(sequence, ClauseOrderSide::Before, 1),
        probe(sequence, ClauseOrderSide::After, 0),
        probe(sequence, ClauseOrderSide::Before, 2),
        after,
        probe(sequence, ClauseOrderSide::After, usize::MAX),
        probe(invalid, ClauseOrderSide::Before, 2),
    ];
    let (linked, cues, diagnostics) =
        analyzer.link_clause_candidates_with_cue_diagnostics(source, "", &candidates, &[], &probes);
    let baseline = analyzer.link_clause_candidates_with_ranked_cues(source, "", &candidates);
    assert_eq!((linked, cues), baseline);
    assert_eq!(baseline.0.rejected, [0]);
    let [
        later,
        reference,
        outside,
        rejected,
        earlier,
        repeated,
        unknown,
        missing,
    ] = diagnostics.as_slice()
    else {
        panic!("expected one diagnostic per requested probe");
    };
    assert!(
        diagnostics
            .iter()
            .zip(&probes)
            .all(|(row, probe)| row.probe == *probe)
    );
    assert_eq!(later, repeated);
    assert_eq!(reference.advisory, ClauseOrderAdvisory::ReferenceOnly);
    assert!(reference.windows.is_empty());
    assert_eq!(outside.advisory, ClauseOrderAdvisory::OutsideUnit);
    assert_eq!(outside.accepted_clause, Some(0));
    assert_eq!(rejected.advisory, ClauseOrderAdvisory::CandidateRejected);
    assert_eq!(rejected.accepted_clause, None);
    assert_eq!(unknown.advisory, ClauseOrderAdvisory::UnknownCandidate);
    assert_eq!(unknown.accepted_clause, None);
    assert_eq!(missing.advisory, ClauseOrderAdvisory::CueNotFound);
    assert!(missing.windows.is_empty());
    for row in [reference, outside, rejected, unknown, missing] {
        assert_eq!(row.binding, ClauseOrderBinding::NotEvaluated);
        assert_eq!(row.rank, None);
    }
    for (row, accepted, unit, window) in [
        (earlier, 1, "守衛收起了羅盤。", "守衛收起了羅盤"),
        (later, 0, "隨後工人打開了窗戶。", "工人打開了窗戶。"),
    ] {
        assert_eq!(row.accepted_clause, Some(accepted));
        assert_eq!(row.advisory, ClauseOrderAdvisory::Admitted);
        assert_eq!(row.binding, ClauseOrderBinding::Bound);
        assert_eq!(row.rank, Some(0));
        let [observed] = row.windows.as_slice() else {
            panic!("expected one source window for {unit}");
        };
        assert_eq!(original(source, observed.unit), unit);
        assert_eq!(original(source, observed.window), window);
        assert_eq!(observed.written_head, None);
        assert_eq!(observed.required_head, None);
        assert!(!observed.advisory_only);
    }
    Ok(())
}

#[test]
fn cue_diagnostics_report_before_windows_in_chronological_order() -> anyhow::Result<()> {
    use faberline_nlp::{
        ClauseCandidate, ClauseOrderAdvisory, ClauseOrderBinding, ClauseOrderProbe, ClauseOrderSide,
    };

    let lexicon = Lexicon {
        places: &["石室"],
        objects: &[("羅盤", "羅盤"), ("窗戶", "窗戶")],
        ..Lexicon::default()
    };
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &lexicon)?;
    for (source, written, required, binding) in [
        (
            "𠮷。在守衛收起羅盤之前，工人打開了窗戶。",
            "收起",
            "收起",
            ClauseOrderBinding::MissingCompletion,
        ),
        (
            "𠮷。在守衛回到了石室之前，工人打開了窗戶。",
            "回到了",
            "回到",
            ClauseOrderBinding::Bound,
        ),
    ] {
        let written_head = linking_span(source, written, 0);
        let required_head = linking_span(source, required, 0);
        let candidates = [
            ClauseCandidate::new(written_head),
            ClauseCandidate::new(linking_span(source, "打開", 0)),
        ];
        let cue = linking_span(source, "之前", 0);
        let probes = [
            ClauseOrderProbe {
                cue,
                side: ClauseOrderSide::Before,
                input_index: 1,
            },
            ClauseOrderProbe {
                cue,
                side: ClauseOrderSide::After,
                input_index: 0,
            },
        ];
        let (linked, cues, diagnostics) = analyzer.link_clause_candidates_with_cue_diagnostics(
            source,
            "",
            &candidates,
            &[],
            &probes,
        );
        assert_eq!(
            (linked, cues),
            analyzer.link_clause_candidates_with_ranked_cues(source, "", &candidates),
            "{source}"
        );
        let [earlier, later] = diagnostics.as_slice() else {
            panic!("expected both chronological sides for {source}");
        };
        assert_eq!(earlier.accepted_clause, Some(1), "{source}");
        assert_eq!(earlier.binding, ClauseOrderBinding::Bound, "{source}");
        assert_eq!(later.accepted_clause, Some(0), "{source}");
        assert_eq!(later.advisory, ClauseOrderAdvisory::Admitted, "{source}");
        assert_eq!(later.binding, binding, "{source}");
        let ([following], [anchored]) = (earlier.windows.as_slice(), later.windows.as_slice())
        else {
            panic!("expected one window on each side for {source}");
        };
        assert_eq!(original(source, following.window), "工人打開了窗戶。");
        assert_eq!(following.written_head, None, "{source}");
        assert_eq!(anchored.written_head, Some(written_head), "{source}");
        assert_eq!(anchored.required_head, Some(required_head), "{source}");
        assert_eq!(anchored.window.end_char, cue.start_char, "{source}");
        assert!(original(source, anchored.unit).ends_with("之前，"));
    }
    Ok(())
}

#[test]
fn cue_diagnostics_report_both_attempts_when_crossing_one_description() -> anyhow::Result<()> {
    use faberline_nlp::{
        ClauseCandidate, ClauseOrderAdvisory, ClauseOrderBinding, ClauseOrderProbe, ClauseOrderSide,
    };

    let lexicon = Lexicon {
        objects: &[("羅盤", "羅盤")],
        ..Lexicon::default()
    };
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &lexicon)?;
    let source = "守衛收起了羅盤。屋內仍然寂靜。隨後工人打開窗戶。";
    let candidates = [ClauseCandidate::new(linking_span(source, "收起", 0))];
    let (linked, ranked, diagnostics) = analyzer.link_clause_candidates_with_cue_diagnostics(
        source,
        "",
        &candidates,
        &[],
        &[ClauseOrderProbe {
            cue: linking_span(source, "隨後", 0),
            side: ClauseOrderSide::Before,
            input_index: 0,
        }],
    );
    assert_eq!(
        (linked, ranked),
        analyzer.link_clause_candidates_with_ranked_cues(source, "", &candidates)
    );
    let [row] = diagnostics.as_slice() else {
        panic!("expected one diagnostic");
    };
    assert_eq!(row.advisory, ClauseOrderAdvisory::Admitted);
    assert_eq!(row.binding, ClauseOrderBinding::AdvisoryOnly);
    let [immediate, earlier] = row.windows.as_slice() else {
        panic!("expected the immediate and advisory windows");
    };
    assert_eq!(original(source, immediate.window), "屋內仍然寂靜");
    assert!(!immediate.advisory_only);
    assert_eq!(original(source, earlier.window), "守衛收起了羅盤");
    assert!(earlier.advisory_only);
    Ok(())
}

#[test]
fn cue_diagnostics_distinguish_admission_from_the_display_limit() -> anyhow::Result<()> {
    use faberline_nlp::{
        ClauseCandidate, ClauseOrderAdvisory, ClauseOrderBinding, ClauseOrderProbe, ClauseOrderSide,
    };

    let lexicon = Lexicon {
        objects: &[("窗戶", "窗戶")],
        ..Lexicon::default()
    };
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &lexicon)?;
    let source = "隨後工人打開了窗戶。";
    let candidates = [ClauseCandidate::new(linking_span(source, "打開", 0)); 7];
    let cue = linking_span(source, "隨後", 0);
    let probes = [
        ClauseOrderProbe {
            cue,
            side: ClauseOrderSide::After,
            input_index: 6,
        },
        ClauseOrderProbe {
            cue,
            side: ClauseOrderSide::After,
            input_index: 0,
        },
        ClauseOrderProbe {
            cue,
            side: ClauseOrderSide::Before,
            input_index: 0,
        },
    ];
    let (_, ranked, diagnostics) =
        analyzer.link_clause_candidates_with_cue_diagnostics(source, "", &candidates, &[], &probes);
    assert_eq!(ranked[0].after_alternatives.len(), 5);
    assert_eq!(ranked[0].order.after, None);
    let [omitted, displayed, no_window] = diagnostics.as_slice() else {
        panic!("expected one diagnostic per requested probe");
    };
    for row in [omitted, displayed] {
        assert_eq!(row.advisory, ClauseOrderAdvisory::Admitted);
        assert_eq!(row.binding, ClauseOrderBinding::Competing);
    }
    assert_eq!(omitted.rank, None);
    assert_eq!(displayed.rank, Some(0));
    assert_eq!(no_window.advisory, ClauseOrderAdvisory::NoWindow);
    assert!(no_window.windows.is_empty());
    Ok(())
}

#[test]
fn cue_diagnostics_distinguish_verified_placement_tail_and_scope_filters() -> anyhow::Result<()> {
    use faberline_nlp::{
        ClauseCandidate, ClauseOrderAdvisory, ClauseOrderBinding, ClauseOrderProbe,
        ClauseOrderSide, VerifiedClauseEvidence, VerifiedClauseHead,
    };

    let lexicon = Lexicon {
        objects: &[("燈火", "燈火"), ("羅盤", "羅盤"), ("窗戶", "窗戶")],
        ..Lexicon::default()
    };
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &lexicon)?;
    for (source, word, reason) in [
        (
            "隨後工人演練熄滅燈火。",
            "熄滅",
            ClauseOrderAdvisory::SubjectBridge,
        ),
        (
            "隨後守衛熄滅燈火又走出石室。",
            "熄滅",
            ClauseOrderAdvisory::Tail,
        ),
        (
            "守衛收起羅盤隨後熄滅燈火。",
            "收起",
            ClauseOrderAdvisory::OutsideAnchor,
        ),
        (
            "隨後守衛假裝熄滅了燈火。",
            "熄滅",
            ClauseOrderAdvisory::ScopeOrLexicalGuard,
        ),
        (
            "守衛說：「隨後工人打開了窗戶。」",
            "打開",
            ClauseOrderAdvisory::NonNarrated,
        ),
    ] {
        let head = linking_span(source, word, 0);
        let (linked, ranked, diagnostics) = analyzer.link_clause_candidates_with_cue_diagnostics(
            source,
            "",
            &[ClauseCandidate::new(head)],
            &[VerifiedClauseEvidence {
                head: VerifiedClauseHead {
                    input_index: 0,
                    head,
                },
                subject: None,
            }],
            &[ClauseOrderProbe {
                cue: linking_span(source, "隨後", 0),
                side: ClauseOrderSide::After,
                input_index: 0,
            }],
        );
        assert!(linked.rejected.is_empty(), "{source}");
        let ([cue], [row]) = (ranked.as_slice(), diagnostics.as_slice()) else {
            panic!("expected one cue and one diagnostic for {source}");
        };
        assert!(cue.after_alternatives.is_empty(), "{source}");
        assert_eq!(row.accepted_clause, Some(0), "{source}");
        assert_eq!(row.advisory, reason, "{source}");
        assert_eq!(row.rank, None, "{source}");
        assert_eq!(row.binding, ClauseOrderBinding::NotEvaluated, "{source}");
        assert_eq!(
            row.windows.len(),
            usize::from(reason != ClauseOrderAdvisory::NonNarrated),
            "{source}"
        );
    }
    Ok(())
}

#[test]
fn short_paired_cues_keep_a_proven_relative_action_out_of_strict_links() -> anyhow::Result<()> {
    use faberline_nlp::{
        ClauseCandidate, ClauseOrderAdvisory, ClauseOrderBinding, ClauseOrderCueKind,
        ClauseOrderProbe, ClauseOrderSide,
    };

    let lexicon = Lexicon {
        objects: &[("羅盤", "羅盤"), ("木匠做的匣子", "木匠做的匣子")],
        ..Lexicon::default()
    };
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &lexicon)?;
    for (source, right_argument, binding) in [
        (
            "𠮷。守衛剛收起了羅盤，就打開了窗戶的工人離開了。",
            None,
            ClauseOrderBinding::UngroundedTail,
        ),
        (
            "𠮷。守衛剛收起了羅盤，工人就打開了窗戶的工人離開了。",
            None,
            ClauseOrderBinding::UngroundedTail,
        ),
        (
            "𠮷。守衛剛收起了羅盤，工人就打開了窗戶。",
            None,
            ClauseOrderBinding::Bound,
        ),
        (
            "𠮷。守衛剛收起了羅盤，工人就打開了木匠做的匣子。",
            Some("木匠做的匣子"),
            ClauseOrderBinding::Bound,
        ),
        (
            "𠮷。守衛剛收起了羅盤，工人就打開了工人的窗戶。",
            None,
            ClauseOrderBinding::Bound,
        ),
        (
            "𠮷。守衛剛收起了羅盤，工人就打開了工人做的木匣。",
            None,
            ClauseOrderBinding::Bound,
        ),
    ] {
        let before = linking_span(source, "收起", 0);
        let after = linking_span(source, "打開", 0);
        let first = linking_span(source, "剛", 0);
        let second = linking_span(source, "就", 0);
        let candidates = [ClauseCandidate::new(before), ClauseCandidate::new(after)];
        let probes = [
            ClauseOrderProbe {
                cue: second,
                side: ClauseOrderSide::Before,
                input_index: 0,
            },
            ClauseOrderProbe {
                cue: second,
                side: ClauseOrderSide::After,
                input_index: 1,
            },
        ];
        let (linked, ranked, rows) = analyzer.link_clause_candidates_with_cue_diagnostics(
            source,
            "",
            &candidates,
            &[],
            &probes,
        );
        let (plain_linked, plain_ranked) =
            analyzer.link_clause_candidates_with_ranked_cues(source, "", &candidates);
        let (strict_linked, strict) =
            analyzer.link_clause_candidates_with_cues(source, "", &candidates);
        assert_eq!(linked, plain_linked, "{source}");
        assert_eq!(linked, strict_linked, "{source}");
        assert_eq!(ranked, plain_ranked, "{source}");
        assert!(linked.rejected.is_empty(), "{source}");
        let [cue] = ranked.as_slice() else {
            panic!("expected one written pair for {source}");
        };
        assert_eq!(strict, [cue.order], "{source}");
        assert_eq!(cue.order.kind, ClauseOrderCueKind::FirstThen);
        assert_eq!(cue.order.cue, second);
        assert_eq!(cue.order.paired_cue, Some(first));
        assert_eq!(
            cue.order.construction,
            Some(Span {
                start_char: first.start_char,
                end_char: second.end_char,
            })
        );
        let bound = binding == ClauseOrderBinding::Bound;
        assert_eq!(cue.order.before, Some(0), "{source}");
        assert_eq!(cue.order.after, bound.then_some(1), "{source}");
        let ([left], [right]) = (
            cue.before_alternatives.as_slice(),
            cue.after_alternatives.as_slice(),
        ) else {
            panic!("expected both written alternatives for {source}");
        };
        assert_eq!((left.clause, left.head), (0, before));
        assert_eq!(left.completion, Some(linking_span(source, "了", 0)));
        assert_eq!(left.argument, Some(linking_span(source, "羅盤", 0)));
        assert_eq!((right.clause, right.head), (1, after));
        assert_eq!(right.completion, Some(linking_span(source, "了", 1)));
        assert_eq!(
            right.argument,
            right_argument.map(|text| linking_span(source, text, 0)),
            "{source}"
        );
        assert_eq!((left.eligible, right.eligible), (true, bound));
        assert_eq!(rows.len(), probes.len(), "{source}");
        for (index, row) in rows.iter().enumerate() {
            assert_eq!(row.probe, probes[index]);
            assert_eq!(row.accepted_clause, Some(index));
            assert_eq!(row.advisory, ClauseOrderAdvisory::Admitted);
            assert_eq!(row.rank, Some(0));
            let expected = if index == 0 {
                ClauseOrderBinding::Bound
            } else {
                binding
            };
            assert_eq!(row.binding, expected, "{source}");
        }
    }
    Ok(())
}

#[test]
fn verified_single_character_action_ranks_with_its_subject() -> anyhow::Result<()> {
    use faberline_nlp::{
        ClauseCandidate, ClauseOrderAdvisory, ClauseOrderBinding, ClauseOrderProbe,
        ClauseOrderSide::{After, Before},
        VerifiedClauseEvidence, VerifiedClauseHead,
    };

    let lexicon = Lexicon {
        objects: &[("木門", "木門"), ("木匠做的箱子", "木匠做的箱子")],
        ..Lexicon::default()
    };
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &lexicon)?;
    let following = [
        ("青銅燈一晃。", "青銅燈", "晃", true),
        ("青銅燈一晃動。", "青銅燈", "晃動", false),
        ("青銅燈翻轉一晃。", "青銅燈", "晃", false),
        ("青銅燈一晃的瞬間，天色忽明。", "青銅燈", "晃", false),
        ("郵差一敲木門的那聲悶響嚇哭了孩子。", "郵差", "敲", false),
        ("郵差一敲木匠做的箱子。", "郵差", "敲", true),
        ("郵差假裝一晃。", "郵差", "晃", false),
        ("\u{b}青銅燈一晃。", "青銅燈", "晃", false),
        ("青銅\u{85}燈一晃。", "青銅\u{85}燈", "晃", false),
        ("青銅燈一\u{2028}晃。", "青銅燈", "晃", false),
        ("郵差一敲\u{b}木門。", "郵差", "敲", false),
    ];
    let cases = std::iter::once((
        "𠮷。陶罐一顫之後，守衛打開木門。".to_owned(),
        "陶罐",
        "顫",
        "之後",
        Before,
        true,
    ))
    .chain(
        following
            .into_iter()
            .map(|(tail, subject, verb, admitted)| {
                (
                    format!("守衛打開木門。隨即{tail}"),
                    subject,
                    verb,
                    "隨即",
                    After,
                    admitted,
                )
            }),
    );
    for (source, subject, verb, marker, side, admitted) in cases {
        let source = source.as_str();
        let head = linking_span(source, verb, 0);
        let cue = linking_span(source, marker, 0);
        let candidates = [ClauseCandidate::new(head)];
        let verified_head = VerifiedClauseHead {
            input_index: 0,
            head,
        };
        let evidence = [VerifiedClauseEvidence {
            head: verified_head,
            subject: Some(linking_span(source, subject, 0)),
        }];
        let probe = ClauseOrderProbe {
            cue,
            side,
            input_index: 0,
        };
        let no_evidence = analyzer.link_clause_candidates_with_ranked_cues(source, "", &candidates);
        let head_only = analyzer.link_clause_candidates_with_ranked_cues_and_heads(
            source,
            "",
            &candidates,
            &[verified_head],
        );
        let (plain, strict) = analyzer.link_clause_candidates_with_cues(source, "", &candidates);
        let (linked, ranked, rows) = analyzer.link_clause_candidates_with_cue_diagnostics(
            source,
            "",
            &candidates,
            &evidence,
            &[probe],
        );
        assert_eq!(head_only, no_evidence, "{source}");
        assert_eq!(linked, plain, "{source}");
        assert_eq!(linked, head_only.0, "{source}");
        assert!(ranked.iter().map(|item| item.order).eq(strict), "{source}");
        let (plain_linked, plain_ranked) = analyzer
            .link_clause_candidates_with_ranked_cues_and_evidence(
                source,
                "",
                &candidates,
                &evidence,
            );
        assert_eq!(linked, plain_linked, "{source}");
        assert_eq!(ranked, plain_ranked, "{source}");

        let [row] = rows.as_slice() else {
            panic!("expected one diagnostic for {source}");
        };
        assert_eq!(row.probe, probe);
        assert_eq!(row.accepted_clause, Some(0));
        assert_eq!(
            row.advisory == ClauseOrderAdvisory::Admitted,
            admitted,
            "{source}"
        );
        if admitted {
            assert_eq!(row.binding, ClauseOrderBinding::AdvisoryOnly, "{source}");
            assert_eq!(row.rank, Some(0), "{source}");
            let item = ranked.iter().find(|item| item.order.cue == cue).unwrap();
            let alternatives = match side {
                Before => &item.before_alternatives,
                After => &item.after_alternatives,
            };
            let [option] = alternatives.as_slice() else {
                panic!("expected the verified action for {source}");
            };
            assert_eq!((option.clause, option.head), (0, head));
            assert_eq!(option.completion, None);
            assert!(!option.eligible);
            assert!(!option.lexical_head);
        } else {
            assert_eq!(row.rank, None, "{source}");
            assert_eq!(ranked, head_only.1, "{source}");
        }
    }
    Ok(())
}

#[test]
fn discovers_completed_reading_and_checking_with_exact_source_roles() -> anyhow::Result<()> {
    use faberline_nlp::ClauseCandidate;

    let lexicon = Lexicon {
        actors: &[("阿梨", "林梨"), ("小罗", "羅青")],
        objects: &[
            ("名冊", "名冊"),
            ("名册", "名冊"),
            ("清單", "清單"),
            ("老師留下的名冊", "名冊"),
            ("木門", "木門"),
            ("信", "信"),
        ],
        ..Lexicon::default()
    };
    let traditional = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &lexicon)?;
    let simplified = NarrativeAnalyzer::new(Locale::ChineseChina, &lexicon)?;
    for (analyzer, alias, canonical, predicate, completion, object) in [
        (&traditional, "阿梨", "林梨", "翻閱", Some("了"), "名冊"),
        (&traditional, "阿梨", "林梨", "核對", Some("了"), "一張清單"),
        (&traditional, "阿梨", "林梨", "讀", Some("了"), "一封信"),
        (&traditional, "阿梨", "林梨", "讀完", None, "名冊"),
        (
            &traditional,
            "阿梨",
            "林梨",
            "查閱",
            Some("了"),
            "老師留下的名冊",
        ),
        (&simplified, "小罗", "羅青", "查阅", Some("了"), "名册"),
    ] {
        let clause = format!(
            "{alias}{predicate}{}{object}。",
            completion.unwrap_or_default()
        );
        let source = format!("𠮷。\r\n{clause}");
        let actions = analyzer.discover_reading_actions(&source, "");
        let [action] = actions.as_slice() else {
            panic!("expected one reading action for {source}");
        };
        assert_eq!(action.clause, linking_span(&source, &clause, 0));
        assert_eq!(action.actor, canonical);
        assert_eq!(action.actor_surface, linking_span(&source, alias, 0));
        assert_eq!(action.predicate, linking_span(&source, predicate, 0));
        assert_eq!(action.object, linking_span(&source, object, 0));
        assert_eq!(
            action.completion,
            completion.map(|text| linking_span(&source, text, 0))
        );
    }

    let source = "阿梨翻閱了名冊。隨後阿梨打開了木門。";
    assert_eq!(traditional.discover_reading_actions(source, "").len(), 1);
    let analysis = traditional.analyze(source, "");
    let [opening] = analysis.events.as_slice() else {
        panic!("expected only the existing opening event in default analysis");
    };
    assert_eq!(opening.action, "隨後阿梨打開了木門。");
    let (_, cues) = traditional.link_clause_candidates_with_cues(
        source,
        "",
        &[
            ClauseCandidate::new(linking_span(source, "翻閱", 0)),
            ClauseCandidate::new(linking_span(source, "打開", 0)),
        ],
    );
    let [cue] = cues.as_slice() else {
        panic!("expected one sequence cue");
    };
    assert_eq!((cue.before, cue.after), (None, Some(1)));
    Ok(())
}

#[test]
fn reading_discovery_requires_direct_narrated_action_and_grounded_roles() -> anyhow::Result<()> {
    let lexicon = Lexicon {
        actors: &[("阿梨", "林梨"), ("小羅", "羅青")],
        objects: &[
            ("名冊", "名冊"),
            ("了然", "了然"),
            ("然", "然"),
            ("煙火", "煙火"),
        ],
        ..Lexicon::default()
    };
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &lexicon)?;
    for source in [
        "守衛說：「阿梨翻閱了名冊。」",
        "守衛說：「等等。\n阿梨翻閱了名冊。」",
        "守衛說：「等等。\n阿梨翻閱了名冊。",
        "守衛寫道：\n阿梨翻閱了名冊。",
        "如果停電，阿梨讀完名冊。",
        "據說，阿梨翻閱了名冊。",
        "守衛聲稱，阿梨翻閱了名冊。",
        "守衛夢見，阿梨翻閱了名冊。",
        "阿梨打算讀完名冊。",
        "阿梨沒有讀完名冊。",
        "阿梨翻閱了名冊的記錄遺失了。",
        "小羅看見阿梨翻閱了名冊。",
        "阿梨讓小羅讀完名冊。",
        "守衛翻閱了名冊。",
        "阿梨翻閱了航海日誌。",
        "阿梨查閱了然。",
        "阿梨翻閱名冊。",
        "阿梨讀完名冊，務必交回。",
        "阿梨讀完名冊！",
        "阿梨翻閱了名冊，就能找到線索。",
        "阿梨翻閱名冊了。",
        "阿梨看完煙火。",
    ] {
        assert!(
            analyzer.discover_reading_actions(source, "").is_empty(),
            "{source}"
        );
    }
    Ok(())
}

#[test]
fn duration_observations_keep_written_actor_verb_and_time() -> anyhow::Result<()> {
    let lexicon = Lexicon {
        actors: &[("阿蘭", "林蘭"), ("小羅", "羅青")],
        objects: &[("木門", "木門")],
        ..Lexicon::default()
    };
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &lexicon)?;
    for (source, clause, predicate, duration, subject, actor) in [
        (
            "𠮷。阿蘭看了半晌。",
            "阿蘭看了半晌。",
            "看",
            "半晌",
            Some("阿蘭"),
            Some("林蘭"),
        ),
        (
            "阿蘭打開木門。\r\n看了片刻。",
            "看了片刻。",
            "看",
            "片刻",
            None,
            None,
        ),
        (
            "她觀看了片刻。",
            "她觀看了片刻。",
            "觀看",
            "片刻",
            Some("她"),
            None,
        ),
        (
            "阿蘭翻閱了一會兒。",
            "阿蘭翻閱了一會兒。",
            "翻閱",
            "一會兒",
            Some("阿蘭"),
            Some("林蘭"),
        ),
        (
            "隨後阿蘭仔細看了半晌。",
            "隨後阿蘭仔細看了半晌。",
            "看",
            "半晌",
            Some("阿蘭"),
            Some("林蘭"),
        ),
        (
            "阿蘭仔細讀了片刻。",
            "阿蘭仔細讀了片刻。",
            "讀",
            "片刻",
            Some("阿蘭"),
            Some("林蘭"),
        ),
        (
            "阿蘭細看了半晌。",
            "阿蘭細看了半晌。",
            "細看",
            "半晌",
            Some("阿蘭"),
            Some("林蘭"),
        ),
        (
            "阿蘭看了半晌，接著起身。",
            "阿蘭看了半晌，",
            "看",
            "半晌",
            Some("阿蘭"),
            Some("林蘭"),
        ),
        (
            "阿蘭看了半晌，也不知看出了什麼。",
            "阿蘭看了半晌，",
            "看",
            "半晌",
            Some("阿蘭"),
            Some("林蘭"),
        ),
    ] {
        let observations = analyzer.discover_duration_observations(source, "");
        let [observation] = observations.as_slice() else {
            panic!("expected one bounded observation for {source}");
        };
        assert_eq!(observation.clause, linking_span(source, clause, 0));
        assert_eq!(observation.predicate, linking_span(source, predicate, 0));
        assert_eq!(observation.completion, linking_span(source, "了", 0));
        assert_eq!(observation.duration, linking_span(source, duration, 0));
        assert_eq!(observation.actor.as_deref(), actor);
        assert_eq!(
            observation.actor_surface,
            subject.map(|text| linking_span(source, text, 0))
        );
        assert_eq!(observation.object, None);
    }

    let source = "阿蘭看了片刻，小羅看了半晌。";
    let observations = analyzer.discover_duration_observations(source, "");
    let [first, second] = observations.as_slice() else {
        panic!("expected a separate observation for each written actor");
    };
    assert_eq!(first.actor.as_deref(), Some("林蘭"));
    assert_eq!(second.actor.as_deref(), Some("羅青"));
    assert_eq!(second.clause, linking_span(source, "小羅看了半晌。", 0));
    Ok(())
}

#[test]
fn duration_observations_reject_nonactual_and_continuing_looks() -> anyhow::Result<()> {
    let lexicon = Lexicon {
        actors: &[("阿蘭", "林蘭"), ("小羅", "羅青")],
        ..Lexicon::default()
    };
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &lexicon)?;
    for source in [
        "守衛說：「阿蘭看了半晌。」",
        "守衛說：「等等。\n阿蘭看了半晌。」",
        "如果燈亮了，阿蘭看了片刻，便能認出方向。",
        "阿蘭原想看了半晌再走。",
        "原本打算，阿蘭看了半晌。",
        "阿蘭看了半晌，接著就可以起身。",
        "阿蘭看了半晌，接著她務必起身。",
        "阿蘭看了多久？",
        "阿蘭看了半晌嗎。",
        "阿蘭看了三遍。",
        "阿蘭看了半晌的記錄遺失了。",
        "阿蘭讓小羅看了半晌。",
        "阿蘭盯著窗外，看了半晌。",
        "阿蘭繼續看了半晌。",
    ] {
        assert!(
            analyzer
                .discover_duration_observations(source, "")
                .is_empty(),
            "{source}"
        );
    }

    let source = "阿蘭看了半晌，繼續看了片刻。";
    let observations = analyzer.discover_duration_observations(source, "");
    let [first] = observations.as_slice() else {
        panic!("expected only the original observation");
    };
    assert_eq!(first.duration, linking_span(source, "半晌", 0));
    Ok(())
}

#[test]
fn postposed_speech_keeps_exact_speaker_roles_and_scope() -> anyhow::Result<()> {
    use faberline_nlp::SpeechActKind;

    let lexicon = Lexicon {
        actors: &[("阿蘭", "林蘭"), ("明", "羅明"), ("自明", "林自明")],
        ..Lexicon::default()
    };
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &lexicon)?;
    for (tag, surface, speaker, cue) in [
        ("阿蘭眼珠一轉說道", "阿蘭", Some("林蘭"), "說道"),
        ("灰袍青年說道", "灰袍青年", None, "說道"),
        ("白眉長老說道", "白眉長老", Some("白眉長老"), "說道"),
        ("白眉長老不慌不忙說道", "白眉長老", None, "說道"),
        ("阿蘭則不慌不忙的說道", "阿蘭", Some("林蘭"), "說道"),
        ("這句話從自明口中說出", "自明", Some("林自明"), "說出"),
    ] {
        let source = format!("𠮷。「快走。」{tag}。");
        let acts = analyzer.discover_speech_acts(&source, "");
        let [act] = acts.as_slice() else {
            panic!("expected one postposed speech act for {source}");
        };
        assert_eq!(act.kind, SpeechActKind::Statement, "{source}");
        assert_eq!(act.speaker.as_deref(), speaker, "{source}");
        assert_eq!(act.speaker_surface, Some(linking_span(&source, surface, 0)));
        assert_eq!(act.cue, linking_span(&source, cue, 0));
        assert_eq!(act.payload, linking_span(&source, "快走。", 0));
    }

    for source in [
        "「快走。」阿蘭安慰少女說道。",
        "「快走。」阿蘭安慰老者說道。",
        "「快走。」這句話未能從自明口中說出。",
        "「快走。」這句話若從自明口中說出，情況會不同。",
        "「快走。」這句話不是從自明口中說出。",
        "灰袍青年說道：「快走。」",
        "阿蘭眼珠一轉說道：「快走。」",
        "「快走。」\u{2028}灰袍青年說道。",
    ] {
        assert!(
            analyzer.discover_speech_acts(source, "").is_empty(),
            "{source}"
        );
    }
    for source in [
        "阿蘭說道：「快走。」安慰少女說道。",
        "阿蘭說道：「快走。」灰袍青年說道。",
    ] {
        let [act] = analyzer
            .discover_speech_acts(source, "")
            .try_into()
            .unwrap();
        assert_eq!(act.speaker.as_deref(), Some("林蘭"));
        assert_eq!(act.speaker_surface, Some(linking_span(source, "阿蘭", 0)));
    }
    Ok(())
}

#[test]
fn postposed_number_quote_uses_the_written_source_not_the_reaction_subject() -> anyhow::Result<()> {
    let lexicon = Lexicon {
        actors: &[
            ("阿蘭", "林蘭"),
            ("阿兰", "林蘭"),
            ("明", "羅明"),
            ("自明", "林自明"),
        ],
        ..Lexicon::default()
    };
    for (locale, payload, tag, cue) in [
        (
            Locale::ChineseTaiwan,
            "只要三十元。",
            "一個令阿蘭吃驚的數字從自明口中說了出來",
            "說了出來",
        ),
        (
            Locale::ChineseChina,
            "只要 25 元。",
            "这个让阿兰难以置信的数字从自明口中说了出来",
            "说了出来",
        ),
        (
            Locale::ChineseTaiwan,
            "七。",
            "一個使她驚訝的數字從自明口中說出",
            "說出",
        ),
        (
            Locale::ChineseTaiwan,
            "七。",
            "一個數字從自明口中說出",
            "說出",
        ),
    ] {
        let analyzer = NarrativeAnalyzer::new(locale, &lexicon)?;
        let source = format!("𠮷。\u{201c}{payload}\u{201d}{tag}。");
        let [act] = analyzer
            .discover_speech_acts(&source, "")
            .try_into()
            .unwrap();
        assert_eq!(act.speaker.as_deref(), Some("林自明"), "{source}");
        assert_eq!(act.speaker_surface, Some(linking_span(&source, "自明", 0)));
        assert_eq!(act.cue, linking_span(&source, cue, 0));
        assert_eq!(act.payload, linking_span(&source, payload, 0));
    }

    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &lexicon)?;
    for source in [
        "「快走。」一個令阿蘭吃驚的數字從自明口中說出。",
        "「萬一有雨？」一個令阿蘭吃驚的數字從自明口中說出。",
        "「三十。」另一個令阿蘭吃驚的數字從自明口中說出。",
        "「三十。」一個令阿蘭吃驚的數字沒能從自明口中說出。",
        "「三十。」一個令阿蘭吃驚的數字從阿蘭和自明口中說出。",
        "「三十。」\n一個令阿蘭吃驚的數字從自明口中說出。",
    ] {
        assert!(
            analyzer.discover_speech_acts(source, "").is_empty(),
            "{source}"
        );
    }
    Ok(())
}

#[test]
fn duration_observations_bound_location_manner_and_modality() -> anyhow::Result<()> {
    let lexicon = Lexicon {
        actors: &[("阿蘭", "林蘭"), ("小羅", "羅青"), ("默默", "季默")],
        places: &["長廊", "窗", "工地"],
        ..Lexicon::default()
    };
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &lexicon)?;
    for (source, surface, actor) in [
        ("阿蘭在窗邊靜靜地看了片刻。", Some("阿蘭"), Some("林蘭")),
        ("阿蘭在窗邊靜靜的看了片刻。", Some("阿蘭"), Some("林蘭")),
        ("仔细的看了片刻。", None, None),
        ("在長廊阿蘭仔細看了片刻。", Some("阿蘭"), Some("林蘭")),
        ("在長廊仔細看了片刻。", None, None),
        ("她在窗前看了片刻。", Some("她"), None),
        ("在長廊守衛看了片刻。", Some("守衛"), None),
        ("默默看了片刻。", Some("默默"), Some("季默")),
        ("默默於工地看了片刻。", Some("默默"), Some("季默")),
    ] {
        let observations = analyzer.discover_duration_observations(source, "");
        let [observation] = observations.as_slice() else {
            panic!("expected one observation for {source}");
        };
        assert_eq!(observation.clause, linking_span(source, source, 0));
        assert_eq!(observation.actor.as_deref(), actor, "{source}");
        assert_eq!(
            observation.actor_surface,
            surface.map(|text| linking_span(source, text, 0)),
            "{source}"
        );
        assert_eq!(observation.predicate, linking_span(source, "看", 0));
        assert_eq!(observation.completion, linking_span(source, "了", 0));
        assert_eq!(observation.duration, linking_span(source, "片刻", 0));
        assert_eq!(observation.object, None);
    }

    for source in [
        "阿蘭在陌生巷口看了片刻。",
        "阿蘭走到長廊看了片刻。",
        "阿蘭在長廊站定看了片刻。",
        "郵差在長廊看了片刻。",
        "阿蘭的看了片刻。",
        "阿蘭漫不經心的看了片刻。",
        "差一點，阿蘭在長廊看了半晌。",
        "本可以，阿蘭在長廊看了半晌。",
        "假裝，阿蘭在長廊看了半晌。",
        "她認真的假裝，靜靜的看了片刻。",
    ] {
        assert!(
            analyzer
                .discover_duration_observations(source, "")
                .is_empty(),
            "{source}"
        );
    }

    let nominal = "她認真的觀察很有用，她看了片刻。";
    let observations = analyzer.discover_duration_observations(nominal, "");
    let [observation] = observations.as_slice() else {
        panic!("expected one observation for {nominal}");
    };
    assert_eq!(observation.predicate, linking_span(nominal, "看", 0));
    assert_eq!(
        observation.actor_surface,
        Some(linking_span(nominal, "她", 1))
    );

    let actual = "小羅在窗邊靜靜看了片刻";
    for nonactual in [
        "阿蘭差一點在長廊看了半晌",
        "阿蘭在長廊本可以看了半晌",
        "阿蘭假裝在長廊看了半晌",
        "阿蘭差一點靜靜的看了半晌",
        "阿蘭本可以靜靜的看了半晌",
        "阿蘭假裝靜靜的看了半晌",
    ] {
        for source in [
            format!("{nonactual}，{actual}。"),
            format!("{actual}，{nonactual}。"),
        ] {
            let observations = analyzer.discover_duration_observations(&source, "");
            let [observation] = observations.as_slice() else {
                panic!("expected only the actual observation for {source}");
            };
            assert_eq!(observation.actor.as_deref(), Some("羅青"), "{source}");
            assert_eq!(
                observation.actor_surface,
                Some(linking_span(&source, "小羅", 0))
            );
            assert_eq!(observation.duration, linking_span(&source, "片刻", 0));
        }
    }
    Ok(())
}
