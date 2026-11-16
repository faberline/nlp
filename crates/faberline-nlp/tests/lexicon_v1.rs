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

    let forbidden = ["BY-SA", "BY-NC", "BY-ND", "GPL", "CC-BY", "CC BY"];
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

    let receipt_path = workspace_root().join("docs/receipts/lexicon-2026-11-16.json");
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

    fn assert_sha256_fields_valid(val: &Value, path: &str) {
        match val {
            Value::Object(map) => {
                for (k, v) in map {
                    let next = format!("{path}.{k}");
                    if k == "sha256" || k.ends_with("_sha256") {
                        let hex = v
                            .as_str()
                            .unwrap_or_else(|| panic!("{next} must be a string"));
                        assert!(
                            hex.len() == 64
                                && hex.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')),
                            "{next} must be 64-char lowercase hex, got {hex:?}"
                        );
                    } else {
                        assert_sha256_fields_valid(v, &next);
                    }
                }
            }
            Value::Array(arr) => {
                for (i, item) in arr.iter().enumerate() {
                    assert_sha256_fields_valid(item, &format!("{path}[{i}]"));
                }
            }
            _ => {}
        }
    }
    assert_sha256_fields_valid(&receipt, "receipt");
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

#[test]
fn ja_shinjitai_to_traditional_mapping() {
    use lexicon::ja::{map_ja_shinjitai, map_ja_shinjitai_char};

    let pairs = [
        ('毎', '每'),
        ('塩', '鹽'),
        ('発', '發'),
        ('験', '驗'),
        ('鉄', '鐵'),
        ('転', '轉'),
        ('経', '經'),
        ('権', '權'),
        ('観', '觀'),
        ('聴', '聽'),
        ('覚', '覺'),
        ('訳', '譯'),
        ('説', '說'),
        ('対', '對'),
        ('専', '專'),
        ('応', '應'),
        ('労', '勞'),
        ('営', '營'),
        ('団', '團'),
        ('囲', '圍'),
        ('圧', '壓'),
        ('増', '增'),
        ('写', '寫'),
        ('払', '拂'),
        ('択', '擇'),
        ('拠', '據'),
        ('拡', '擴'),
        ('撃', '擊'),
        ('収', '收'),
        ('単', '單'),
        ('参', '參'),
        ('双', '雙'),
        ('号', '號'),
        ('広', '廣'),
        ('楽', '樂'),
        ('変', '變'),
        ('帰', '歸'),
        ('売', '賣'),
        ('続', '續'),
        ('残', '殘'),
        ('遅', '遲'),
        ('辺', '邊'),
        ('図', '圖'),
        ('読', '讀'),
        ('国', '國'),
        ('学', '學'),
        ('気', '氣'),
        ('声', '聲'),
        ('体', '體'),
        ('髪', '髮'),
        ('黒', '黑'),
        ('恋', '戀'),
        ('悪', '惡'),
        ('絵', '繪'),
        ('昼', '晝'),
        ('豊', '豐'),
        ('静', '靜'),
        ('触', '觸'),
        ('届', '屆'),
        ('絶', '絕'),
        ('抜', '拔'),
        ('揺', '搖'),
        ('乱', '亂'),
        ('余', '餘'),
        ('亀', '龜'),
        ('茎', '莖'),
        ('顔', '顏'),
        ('薬', '藥'),
        ('痴', '癡'),
        ('沢', '澤'),
        ('実', '實'),
        ('宝', '寶'),
        ('剣', '劍'),
        ('戦', '戰'),
        ('竜', '龍'),
        ('桜', '櫻'),
        ('雑', '雜'),
        ('産', '產'),
        ('値', '值'),
        ('隠', '隱'),
        ('陥', '陷'),
        ('践', '踐'),
        ('逓', '遞'),
        ('鋭', '銳'),
        ('隷', '隸'),
        ('頬', '頰'),
        ('歳', '歲'),
        ('脱', '脫'),
        ('査', '查'),
        ('税', '稅'),
        ('艶', '豔'),
        ('槙', '槇'),
    ];
    for (src, dst) in pairs {
        assert_eq!(map_ja_shinjitai_char(src), dst, "pair {src} -> {dst}");
    }

    assert_eq!(map_ja_shinjitai("経済発展"), "經濟發展");
    assert_eq!(map_ja_shinjitai("図書館の経験"), "圖書館の經驗");
    // Pure glyph conversion only: '駅' -> '驛' (not lexical '站'); '台', '着', '偽', '戻' stay unmapped.
    assert_eq!(map_ja_shinjitai("東京駅"), "東京驛");
    assert_eq!(map_ja_shinjitai("台風到着"), "台風到着");
    assert_eq!(map_ja_shinjitai("偽物と戻る"), "偽物と戻る");
    assert_eq!(map_ja_shinjitai("艶"), "豔");
    assert_eq!(map_ja_shinjitai("槙"), "槇");
}

#[test]
fn c1_hygiene_en_and_zh_pos_lemma_and_entity_regressions() {
    let opts = Options {
        enrich_morphology: true,
        first_party_zh_segmenter: true,
        ..Options::default()
    };

    // EN: non-comparative -er/-ier nouns must not truncate to "-i"/"-", and derivational
    // / hyphenated / VarCon terms must retain accurate UPOS.
    for (word, expected_pos) in [
        ("polariser", Upos::Noun),
        ("polarizer", Upos::Noun),
        ("polarisable", Upos::Adj),
        ("remineralize", Upos::Verb),
        ("co-operate", Upos::Verb),
    ] {
        let hit = lexicon::en::analyze_en_token(word);
        assert!(
            hit.upos_candidates.contains(&expected_pos),
            "{word} expected {expected_pos:?}, got {hit:?}"
        );
        assert_ne!(hit.lemma.as_deref(), Some("polari"), "{word}: {hit:?}");
    }

    let en_text = "A polariser and polarizer are polarisable if we remineralize and co-operate, while studies stopped.";
    let en_doc = analyze_document(en_text, &opts);
    assert_eq!(
        find_token(&en_doc, en_text, "polariser")
            .canonical_variant
            .as_deref(),
        Some("polarizer")
    );
    assert_eq!(
        find_token(&en_doc, en_text, "polarizer")
            .canonical_variant
            .as_deref(),
        Some("polariser")
    );
    assert_ne!(
        find_token(&en_doc, en_text, "polariser").lemma.as_deref(),
        Some("polari")
    );
    assert_ne!(
        find_token(&en_doc, en_text, "polarizer").lemma.as_deref(),
        Some("polari")
    );

    // Near-negative: regular inflections still lemmatize cleanly, and comparatives keep ADJ.
    assert_eq!(
        find_token(&en_doc, en_text, "studies").lemma.as_deref(),
        Some("study")
    );
    assert_eq!(
        find_token(&en_doc, en_text, "stopped").lemma.as_deref(),
        Some("stop")
    );
    assert!(
        lexicon::en::analyze_en_token("smaller")
            .upos_candidates
            .contains(&Upos::Adj)
    );
    assert!(
        lexicon::en::analyze_en_token("bigger")
            .upos_candidates
            .contains(&Upos::Adj)
    );

    // Additional EN VarCon & derivational POS/lemma regressions.
    for (word, expected_pos, expected_lemma) in [
        ("anonymized", Upos::Verb, Some("anonymize")),
        ("anonymises", Upos::Verb, Some("anonymise")),
        ("sterilisable", Upos::Adj, None),
        ("mediaevally", Upos::Adv, None),
        ("anarchize", Upos::Verb, None),
        ("photosynthesizes", Upos::Verb, Some("photosynthesize")),
        ("baptised", Upos::Verb, Some("baptise")),
    ] {
        let hit = lexicon::en::analyze_en_token(word);
        assert!(
            hit.upos_candidates.contains(&expected_pos),
            "{word} expected {expected_pos:?}, got {hit:?}"
        );
        if let Some(lem) = expected_lemma {
            assert_eq!(hit.lemma.as_deref(), Some(lem), "{word}: {hit:?}");
        }
    }
    assert!(
        lexicon::en::analyze_en_token("baptised")
            .upos_candidates
            .contains(&Upos::Adj)
    );

    // ZH: non-person nouns/idioms must not be mislabeled SemanticClass::Person.
    let zh_hant_text = "古建築與司法權、粟米、蜀錦、清單、優越感、封建禮教、相生相剋，以及李白、杜甫、諸葛亮、呂布、史可法、小桂子、阿芳、老王、北京、花蓮、波士頓、沃爾夫斯堡、柴科夫斯基、耶魯大學、中國人民銀行、美國政府。";
    let zh_hant_doc = analyze_document(zh_hant_text, &opts);
    for non_person in [
        "古建築",
        "司法權",
        "粟米",
        "蜀錦",
        "清單",
        "優越感",
        "封建禮教",
        "相生相剋",
    ] {
        let tok = find_token(&zh_hant_doc, zh_hant_text, non_person);
        assert_ne!(
            tok.sem_class,
            Some(SemanticClass::Person),
            "{non_person} must not be Person: {tok:?}"
        );
    }
    for person in [
        "李白",
        "杜甫",
        "諸葛亮",
        "呂布",
        "史可法",
        "小桂子",
        "阿芳",
        "老王",
    ] {
        let tok = find_token(&zh_hant_doc, zh_hant_text, person);
        assert_eq!(
            tok.sem_class,
            Some(SemanticClass::Person),
            "{person} must remain Person: {tok:?}"
        );
    }
    for loc in ["北京", "花蓮", "波士頓", "沃爾夫斯堡"] {
        let tok = find_token(&zh_hant_doc, zh_hant_text, loc);
        assert_eq!(
            tok.sem_class,
            Some(SemanticClass::Location),
            "{loc} must remain Location: {tok:?}"
        );
    }
    for org in ["耶魯大學", "中國人民銀行", "美國政府"] {
        let tok = find_token(&zh_hant_doc, zh_hant_text, org);
        assert_eq!(
            tok.sem_class,
            Some(SemanticClass::Organization),
            "{org} must remain Organization: {tok:?}"
        );
    }
    let tchaikovsky = lexicon::lookup_zh("柴科夫斯基").expect("柴科夫斯基 in lexicon");
    assert_eq!(
        tchaikovsky.decided_upos,
        Some(Upos::Propn),
        "柴科夫斯基 expected Propn, got {tchaikovsky:?}"
    );
    let wetland = lexicon::lookup_zh("湿地").expect("湿地 in lexicon");
    assert_eq!(
        wetland.decided_upos,
        Some(Upos::Noun),
        "湿地 expected Noun, got {wetland:?}"
    );

    let zh_hans_text = "古建筑与司法权、粟米、蜀锦、清单、优越感、封建礼教、相生相克、雷霆万钧。";
    let zh_hans_doc = analyze_document(zh_hans_text, &opts);
    for non_person in [
        "古建筑",
        "司法权",
        "粟米",
        "蜀锦",
        "清单",
        "优越感",
        "封建礼教",
        "相生相克",
        "雷霆万钧",
    ] {
        let tok = find_token(&zh_hans_doc, zh_hans_text, non_person);
        assert_ne!(
            tok.sem_class,
            Some(SemanticClass::Person),
            "{non_person} must not be Person: {tok:?}"
        );
    }
}
