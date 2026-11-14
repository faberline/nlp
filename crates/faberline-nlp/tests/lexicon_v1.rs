use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use faberline_nlp::Span;
use faberline_nlp::doc::{Document, Lang, Options, Reason, SemanticClass, Upos, analyze_document};
use faberline_nlp::lexicon;
use serde_json::Value;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn slice(text: &str, s: Span) -> String {
    text.chars()
        .skip(s.start_char)
        .take(s.end_char - s.start_char)
        .collect()
}

fn find_token<'a>(doc: &'a Document, text: &str, surface: &str) -> &'a faberline_nlp::doc::Token {
    doc.tokens
        .iter()
        .find(|t| slice(text, t.span) == surface)
        .unwrap_or_else(|| panic!("missing token {surface:?} in {text:?}"))
}

#[test]
fn license_audit_and_receipt_integrity() {
    let licenses_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("lexicon/LICENSES");
    assert!(licenses_dir.is_dir(), "missing lexicon/LICENSES directory");

    let forbidden = ["BY-SA", "BY-NC", "BY-ND", "GPL"];
    let mut license_files = HashSet::new();
    for entry in fs::read_dir(&licenses_dir).expect("read lexicon/LICENSES") {
        let entry = entry.expect("license dir entry");
        let path = entry.path();
        if path.is_file() {
            let name = path
                .file_name()
                .and_then(|s| s.to_str())
                .expect("utf8 filename")
                .to_owned();
            let body = fs::read_to_string(&path).expect("read license text");
            let upper = body.to_uppercase();
            for bad in forbidden {
                assert!(
                    !upper.contains(bad),
                    "forbidden license marker {bad:?} found in {}",
                    path.display()
                );
            }
            license_files.insert(name);
        }
    }
    assert!(
        license_files.len() >= 6,
        "expected >= 6 license files, got {:?}",
        license_files
    );

    let receipt_path = workspace_root().join("docs/receipts/lexicon-2026-11-14.json");
    let receipt_raw = fs::read_to_string(&receipt_path).expect("read lexicon receipt");
    let receipt_upper = receipt_raw.to_uppercase();
    for bad in forbidden {
        assert!(
            !receipt_upper.contains(bad),
            "forbidden license marker {bad:?} found in {}",
            receipt_path.display()
        );
    }

    let receipt: Value = serde_json::from_str(&receipt_raw).expect("valid receipt JSON");
    let sources = receipt["sources"]
        .as_array()
        .expect("receipt.sources must be an array");
    assert!(!sources.is_empty());
    for src in sources {
        let lic_file = src["license_file"]
            .as_str()
            .expect("source.license_file must be a string");
        assert!(
            license_files.contains(lic_file),
            "receipt references missing license file {lic_file:?}"
        );
    }
}

#[test]
fn stats_and_binary_loaded() {
    let st = lexicon::stats();
    assert!(st.total_bytes > 1_000_000, "unexpectedly small lexicon");
    assert!(st.zh_hant > 50_000);
    assert!(st.zh_hans > 50_000);
    assert!(st.en_us > 10_000);
    assert!(st.en_gb > 10_000);
    assert!(st.en > 30_000);
    assert!(st.ja > 50_000);
    assert!(st.ko > 50_000);
    assert!(st.ja_inflect_rules > 400);
    assert!(st.ko_morph_rules > 100);
}

#[test]
fn default_options_backwards_compatibility() {
    let def = Options::default();
    assert!(!def.enrich_morphology);
    assert!(!def.first_party_zh_segmenter);
    assert_eq!(serde_json::to_string(&def).unwrap(), "{}");

    let text = "She went to Beijing. 我們開發軟體。";
    let doc = analyze_document(text, &def);
    let json = serde_json::to_value(&doc).unwrap();
    for tok in json["tokens"].as_array().unwrap() {
        let obj = tok.as_object().unwrap();
        assert!(!obj.contains_key("lemma"));
        assert!(!obj.contains_key("canonical_variant"));
        assert!(!obj.contains_key("upos_candidates"));
        assert!(!obj.contains_key("sem_class"));
        assert!(!obj.contains_key("morphemes"));
    }
}

#[test]
fn zh_hant_and_zh_hans_segmentation_variants_and_semantics() {
    let text = "我們在台北使用軟體與記憶體。我们在北京使用软件与内存。";
    let opts = Options {
        enrich_morphology: true,
        first_party_zh_segmenter: true,
        ..Options::default()
    };
    let doc = analyze_document(text, &opts);
    let surfs: Vec<String> = doc.tokens.iter().map(|t| slice(text, t.span)).collect();
    assert!(surfs.contains(&"軟體".to_owned()), "{surfs:?}");
    assert!(surfs.contains(&"記憶體".to_owned()), "{surfs:?}");
    assert!(surfs.contains(&"软件".to_owned()), "{surfs:?}");
    assert!(surfs.contains(&"内存".to_owned()), "{surfs:?}");

    assert_eq!(
        find_token(&doc, text, "軟體").canonical_variant.as_deref(),
        Some("软件")
    );
    assert_eq!(
        find_token(&doc, text, "软件").canonical_variant.as_deref(),
        Some("軟體")
    );
    assert_eq!(
        find_token(&doc, text, "記憶體")
            .canonical_variant
            .as_deref(),
        Some("内存")
    );
    assert_eq!(
        find_token(&doc, text, "内存").canonical_variant.as_deref(),
        Some("記憶體")
    );

    let bj = find_token(&doc, text, "北京");
    assert_eq!(bj.sem_class, Some(SemanticClass::Location));

    // Verify `options.lexicon` protection still works with `first_party_zh_segmenter: true`.
    let custom_text = "他拿出了青竹木劍。";
    let custom_opts = Options {
        lexicon: vec!["青竹木劍".to_owned()],
        first_party_zh_segmenter: true,
        ..Options::default()
    };
    let custom_doc = analyze_document(custom_text, &custom_opts);
    let custom_surfs: Vec<String> = custom_doc
        .tokens
        .iter()
        .map(|t| slice(custom_text, t.span))
        .collect();
    assert!(
        custom_surfs.contains(&"青竹木劍".to_owned()),
        "{custom_surfs:?}"
    );
}

#[test]
fn en_us_and_en_gb_lemma_variants_candidates_and_semantics() {
    let text = "The doctor went to academia and saw the colour of mice.";
    let opts = Options {
        enrich_morphology: true,
        ..Options::default()
    };
    let doc = analyze_document(text, &opts);

    let went = find_token(&doc, text, "went");
    assert_eq!(went.lang, Lang::En);
    assert_eq!(went.lemma.as_deref(), Some("go"));

    let mice = find_token(&doc, text, "mice");
    assert_eq!(mice.lemma.as_deref(), Some("mouse"));

    let colour = find_token(&doc, text, "colour");
    assert_eq!(colour.canonical_variant.as_deref(), Some("color"));

    let doctor = find_token(&doc, text, "doctor");
    assert_eq!(doctor.sem_class, Some(SemanticClass::Person));

    let acad = find_token(&doc, text, "academia");
    assert_eq!(acad.sem_class, Some(SemanticClass::Organization));

    // US -> GB direction
    let us_text = "The color was bright.";
    let us_doc = analyze_document(us_text, &opts);
    assert_eq!(
        find_token(&us_doc, us_text, "color")
            .canonical_variant
            .as_deref(),
        Some("colour")
    );

    // `upos_candidates` populated when `upos.is_none()` and >1 candidate exists.
    let amb_text = "Open the door.";
    let amb_doc = analyze_document(amb_text, &opts);
    let open_tok = find_token(&amb_doc, amb_text, "Open");
    assert_eq!(open_tok.upos, None);
    assert!(
        open_tok.upos_candidates.len() > 1,
        "expected >1 upos_candidates for 'Open', got {:?}",
        open_tok.upos_candidates
    );
}

#[test]
fn ja_segmentation_pos_lemma_and_semantics() {
    let text = "東京駅で3冊の本を買った。";
    let opts = Options {
        enrich_morphology: true,
        ..Options::default()
    };
    let doc = analyze_document(text, &opts);
    let surfs: Vec<String> = doc.tokens.iter().map(|t| slice(text, t.span)).collect();
    assert!(surfs.contains(&"で".to_owned()), "{surfs:?}");
    assert!(surfs.contains(&"3".to_owned()), "{surfs:?}");
    assert!(surfs.contains(&"本".to_owned()), "{surfs:?}");
    assert!(surfs.contains(&"を".to_owned()), "{surfs:?}");
    assert!(surfs.contains(&"買っ".to_owned()), "{surfs:?}");
    assert!(surfs.contains(&"た".to_owned()), "{surfs:?}");

    let katta_stem = find_token(&doc, text, "買っ");
    assert_eq!(katta_stem.upos, Some(Upos::Verb));
    assert_eq!(katta_stem.lemma.as_deref(), Some("買う"));

    let ta = find_token(&doc, text, "た");
    assert_eq!(ta.upos, Some(Upos::Aux));

    let wo = find_token(&doc, text, "を");
    assert_eq!(wo.upos, Some(Upos::Adp));

    // No UnsupportedLanguage for Japanese tokens.
    assert!(
        doc.unknown
            .iter()
            .all(|u| u.reason != Reason::UnsupportedLanguage),
        "{:?}",
        doc.unknown
    );
}

#[test]
fn ko_eojeol_pos_lemma_and_morpheme_spans() {
    let text = "서울역에서 3권의 책을 샀다.";
    let opts = Options {
        enrich_morphology: true,
        ..Options::default()
    };
    let doc = analyze_document(text, &opts);

    let book = find_token(&doc, text, "책을");
    assert_eq!(book.lang, Lang::Ko);
    assert_eq!(book.lemma.as_deref(), Some("책"));
    assert_eq!(book.morphemes.len(), 2, "{:?}", book.morphemes);
    assert_eq!(book.morphemes[0].surface, "책");
    assert_eq!(book.morphemes[1].surface, "을");
    assert_eq!(book.morphemes[0].span.start_char, book.span.start_char);
    assert_eq!(
        book.morphemes[0].span.end_char,
        book.morphemes[1].span.start_char
    );
    assert_eq!(book.morphemes[1].span.end_char, book.span.end_char);

    let satda = find_token(&doc, text, "샀다");
    assert_eq!(satda.lang, Lang::Ko);
    assert_eq!(satda.lemma.as_deref(), Some("사다"));
    assert!(!satda.morphemes.is_empty(), "{:?}", satda.morphemes);
    assert_eq!(
        satda.morphemes.first().unwrap().span.start_char,
        satda.span.start_char
    );
    assert_eq!(
        satda.morphemes.last().unwrap().span.end_char,
        satda.span.end_char
    );
    for w in satda.morphemes.windows(2) {
        assert_eq!(w[0].span.end_char, w[1].span.start_char);
    }

    // No UnsupportedLanguage for Korean tokens.
    assert!(
        doc.unknown
            .iter()
            .all(|u| u.reason != Reason::UnsupportedLanguage),
        "{:?}",
        doc.unknown
    );
}

#[test]
fn ja_colloquial_inflections_and_gloss_lookup() {
    let opts = Options {
        enrich_morphology: true,
        ..Options::default()
    };

    for (text, expected_head, expected_lemma) in [
        ("食べちゃった", "食べ", "食べる"),
        ("濡れてる", "濡れ", "濡れる"),
        ("喘いでいた", "喘い", "喘ぐ"),
        ("見ないで", "見", "見る"),
    ] {
        let doc = analyze_document(text, &opts);
        let head = find_token(&doc, text, expected_head);
        assert_eq!(head.upos, Some(Upos::Verb), "text={text}");
        assert_eq!(head.lemma.as_deref(), Some(expected_lemma), "text={text}");
    }

    assert_eq!(
        lexicon::ja::lookup_ja_gloss("手紙", Some(Upos::Noun)),
        Some("信")
    );
    assert_eq!(lexicon::ja::lookup_ja_gloss("大丈夫", None), Some("沒關係"));
    assert_eq!(
        lexicon::ja::lookup_ja_gloss("気持ちいい", Some(Upos::Adj)),
        Some("舒服")
    );
    assert_eq!(
        lexicon::ja::lookup_ja_gloss("愛撫する", Some(Upos::Verb)),
        Some("愛撫")
    );
}

#[test]
fn ja_and_zh_adult_colloquial_and_custom_lexicon() {
    let st = lexicon::stats();
    assert!(st.ja >= 70_500, "ja={}", st.ja);
    assert!(st.zh_hant >= 69_400, "zh_hant={}", st.zh_hant);
    assert!(st.zh_hans >= 73_700, "zh_hans={}", st.zh_hans);
    assert!(
        st.ja_inflect_rules >= 1_000,
        "ja_inflect={}",
        st.ja_inflect_rules
    );

    let zh_text = "她輕聲嬌喘並愛撫著他。她轻声娇喘并爱抚着他。";
    let zh_opts = Options {
        enrich_morphology: true,
        first_party_zh_segmenter: true,
        ..Options::default()
    };
    let zh_doc = analyze_document(zh_text, &zh_opts);
    let zh_surfs: Vec<String> = zh_doc
        .tokens
        .iter()
        .map(|t| slice(zh_text, t.span))
        .collect();
    assert!(zh_surfs.contains(&"嬌喘".to_owned()), "{zh_surfs:?}");
    assert!(zh_surfs.contains(&"愛撫".to_owned()), "{zh_surfs:?}");
    assert!(zh_surfs.contains(&"娇喘".to_owned()), "{zh_surfs:?}");
    assert!(zh_surfs.contains(&"爱抚".to_owned()), "{zh_surfs:?}");
    assert_eq!(
        find_token(&zh_doc, zh_text, "愛撫")
            .canonical_variant
            .as_deref(),
        Some("爱抚")
    );
    assert_eq!(
        find_token(&zh_doc, zh_text, "爱抚")
            .canonical_variant
            .as_deref(),
        Some("愛撫")
    );

    let ja_custom = "彼女は魔導書をそっと開いた。";
    let chars: Vec<char> = ja_custom.chars().collect();
    let segs = lexicon::ja::segment_ja_with_lexicon(&chars, 0, &["魔導書".to_owned()]);
    let seg_surfs: Vec<String> = segs.iter().map(|s| slice(ja_custom, s.span)).collect();
    assert!(seg_surfs.contains(&"魔導書".to_owned()), "{seg_surfs:?}");

    let ja_opts = Options {
        enrich_morphology: true,
        lexicon: vec!["魔導書".to_owned()],
        ..Options::default()
    };
    let ja_doc = analyze_document(ja_custom, &ja_opts);
    let ja_surfs: Vec<String> = ja_doc
        .tokens
        .iter()
        .map(|t| slice(ja_custom, t.span))
        .collect();
    assert!(ja_surfs.contains(&"魔導書".to_owned()), "{ja_surfs:?}");
}
