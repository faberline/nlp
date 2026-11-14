use faberline_nlp::Span;
use faberline_nlp::doc::{
    Document, Lang, OpaqueKind, Options, Reason, RelationLevel, Upos, analyze_document,
};

fn slice(text: &str, s: Span) -> String {
    text.chars()
        .skip(s.start_char)
        .take(s.end_char - s.start_char)
        .collect()
}

fn sentence_texts(text: &str, doc: &Document) -> Vec<String> {
    doc.segmentation
        .sentences
        .iter()
        .map(|s| slice(text, s.span))
        .collect()
}

fn clause_texts(text: &str, doc: &Document) -> Vec<String> {
    doc.segmentation
        .clauses
        .iter()
        .map(|c| slice(text, c.span))
        .collect()
}

fn token_texts(text: &str, doc: &Document) -> Vec<String> {
    doc.tokens.iter().map(|t| slice(text, t.span)).collect()
}

struct CorpusCell {
    id: &'static str,
    variant: &'static str,
    genre: &'static str,
    dominant_lang: Lang,
    allowed_langs: &'static [Lang],
    text: &'static str,
}

const CELLS: &[CorpusCell] = &[
    // -------------------------------------------------------------------------
    // zh-Hant (4 genres)
    // -------------------------------------------------------------------------
    CorpusCell {
        id: "zh-Hant_novel",
        variant: "zh-Hant",
        genre: "novel",
        dominant_lang: Lang::Zh,
        allowed_langs: &[Lang::Zh],
        text: "村中，秦得參的一家，尤其是窮困的慘痛，當他生下的時候，他父親早就死了。他在世，雖曾贌得幾畝田地耕作，他死了後，只剩下可憐的妻兒。若能得到業主的恩恤，田地繼續租給他們，雇用工人替她們種作，猶可得稍少利頭，以維持生計。但是富家人，誰肯讓他們的利益，給人家享。若然就不能成其富戶了。\n\n好容易，到得參九歲的那一年，他母就遣他，去替人家看牛，做長工。這時候，他後父已不大顧到家內，雖然他們母子倆，自己的勞力，經已可免凍餒的威脅。",
    },
    CorpusCell {
        id: "zh-Hant_essay",
        variant: "zh-Hant",
        genre: "essay",
        dominant_lang: Lang::Zh,
        allowed_langs: &[Lang::Zh],
        text: "我與父親不相見已二年餘了，我最不能忘記的是他的背影。\n\n那年冬天，祖母死了，父親的差使也交卸了，正是禍不單行的日子，我從北京到徐州，打算跟着父親奔喪回家。到徐州見着父親，看見滿院狼藉的東西，又想起祖母，不禁簌簌地流下眼淚。父親說，「事已如此，不必難過，好在天無絕人之路！」\n\n我說道，「爸爸，你走吧。」他望車外看了看，說，「我買幾個橘子去。你就在此地，不要走動。」",
    },
    CorpusCell {
        id: "zh-Hant_social",
        variant: "zh-Hant",
        genre: "social",
        dominant_lang: Lang::Zh,
        allowed_langs: &[Lang::Zh, Lang::Mixed],
        text: "笑死 根本就沒差好嗎==\n你是不是沒出過社會？？？\nwww 肥宅不要幻想好不好 (´・ω・`)\n推樓上：真的笑死，這波操作直接原地起飛……\n※ 發信站: 批踢踢實業坊(ptt.cc), 來自: 114.32.0.1 (臺灣)",
    },
    CorpusCell {
        id: "zh-Hant_paper",
        variant: "zh-Hant",
        genre: "paper",
        dominant_lang: Lang::Mixed,
        allowed_langs: &[Lang::Zh, Lang::Mixed],
        text: "本研究之目的在於運用多種數值指標及實驗資料來評估 CKIP、Jieba、MONPA 等三種廣泛應用於臺灣自然語言處理產學界的正體中文斷詞器。我們特別針對運算效能、資源需求等等面向，檢驗其應用於大型語言文字資料集時，處理斷詞、詞性標註及命名實體辨識等工作之成效。實驗結果顯示，MONPA 利用圖形運算加速器（GPU）進行批次處理斷詞時，可以大幅度縮減巨量中文資料的運算時間。",
    },
    // -------------------------------------------------------------------------
    // zh-Hans (4 genres)
    // -------------------------------------------------------------------------
    CorpusCell {
        id: "zh-Hans_novel",
        variant: "zh-Hans",
        genre: "novel",
        dominant_lang: Lang::Zh,
        allowed_langs: &[Lang::Zh],
        text: "一\n\n今天晚上，很好的月光。\n\n我不见他，已是三十多年；今天见了，精神分外爽快。才知道以前的三十多年，全是发昏；然而须十分小心。不然，那赵家的狗，何以看我两眼呢？\n\n我怕得有理。",
    },
    CorpusCell {
        id: "zh-Hans_essay",
        variant: "zh-Hans",
        genre: "essay",
        dominant_lang: Lang::Zh,
        allowed_langs: &[Lang::Zh],
        text: "这几天心里颇不宁静。今晚在院子里坐着乘凉，忽然想起日日走过的荷塘，在这满月的光里，总该另有一番样子吧。月亮渐渐地升高了，墙外马路上孩子们的欢笑，已经听不见了；妻在屋里拍着闰儿，迷迷糊糊地哼着眠歌。我悄悄地披了大衫，带上门出去。\n\n虽然是满月，天上却有一层淡淡的云，所以不能朗照；但我以为这恰是到了好处——酣眠固不可少，小睡也别有风味的。",
    },
    CorpusCell {
        id: "zh-Hans_social",
        variant: "zh-Hans",
        genre: "social",
        dominant_lang: Lang::Mixed,
        allowed_langs: &[Lang::Zh, Lang::Mixed],
        text: "哈哈哈笑死我了666\n听君一席话，如听一席话......\n不是吧阿sir，这都能忍？？？\n绝绝子 yyds 我真的会谢，下次别说了好吗",
    },
    CorpusCell {
        id: "zh-Hans_paper",
        variant: "zh-Hans",
        genre: "paper",
        dominant_lang: Lang::Mixed,
        allowed_langs: &[Lang::Zh, Lang::Mixed],
        text: "科学文献作为高质量语料库，为众多自然语言处理（NLP）研究提供了支持[1]。在这项工作中，我们提出了 CSL，一个大规模中文科学文献数据集，其中包含了 39.6 万篇论文的标题、摘要、关键词和所属学术领域。针对机载 InSAR 系统多航迹联合定标问题，提出一种基于加权最优化模型的联合定标算法，有效提高了地形测绘精度。",
    },
    // -------------------------------------------------------------------------
    // ja (4 genres)
    // -------------------------------------------------------------------------
    CorpusCell {
        id: "ja_novel",
        variant: "ja",
        genre: "novel",
        dominant_lang: Lang::Ja,
        allowed_langs: &[Lang::Ja, Lang::Zh],
        text: "一\n\n吾輩は猫である。名前はまだ無い。\nどこで生れたかとんと見当がつかぬ。何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。吾輩はここで始めて人間というものを見た。しかもあとで聞くとそれは書生という人間中で一番獰悪な種族であったそうだ。この書生というのは時々我々を捕えて煮て食うという話である。",
    },
    CorpusCell {
        id: "ja_essay",
        variant: "ja",
        genre: "essay",
        dominant_lang: Lang::Ja,
        allowed_langs: &[Lang::Ja, Lang::Zh],
        text: "一\n\n硝子戸の中から外を見渡すと、霜除をした芭蕉だの、赤い実の結った梅もどきの枝だの、無遠慮に直立した電信柱だのがすぐ眼に着くが、その他にこれと云って数え立てるほどのものはほとんど視線に入って来ない。書斎にいる私の眼界は極めて単調でそうしてまた極めて狭いのである。\nその上私は去年の暮から風邪を引いてほとんど表へ出ずに、毎日この硝子戸の中にばかり坐っているので、世間の様子はちっとも分らない。心持が悪いから読書もあまりしない。私はただ坐ったり寝たりしてその日その日を送っているだけである。",
    },
    CorpusCell {
        id: "ja_social",
        variant: "ja",
        genre: "social",
        dominant_lang: Lang::Ja,
        allowed_langs: &[Lang::Ja, Lang::Mixed],
        text: "雨…（笑）\nんっ！？何かおもろそうなテレビ始まった！ドラマ？\nハッ！もしかして、それを知っての、水…！？ 惚れてまうやろー！！\n納豆ひとつも売ってなくて草\nこれマジでキッツイな伊藤隼也wwwwww\nわ！？定期便にしていたニトリルグローブが欠品になった ヤバいな…",
    },
    CorpusCell {
        id: "ja_paper",
        variant: "ja",
        genre: "paper",
        dominant_lang: Lang::Mixed,
        allowed_langs: &[Lang::Ja, Lang::Mixed],
        text: "質問応答は，自然言語処理における重要な研究テーマの一つである．近年の深層学習技術の発達と言語資源の充実により，質問応答技術は飛躍的な発展を遂げている．しかし，これらの研究は英語を対象としたものがほとんどであり，現状，日本語での質問応答に関する研究はあまり活発には行われていない．この背景を受けて，我々は日本語での質問応答研究を促進するため，日本語のクイズを題材とした質問応答のコンペティション「AI 王」を企画し，これまでに計 3 回実施してきた．本論文では，日本語の質問応答技術における現在の到達点と課題を明らかにすることを目標として，使用したクイズ問題と提出された質問応答システム，さらに比較対象として大規模言語モデルを用いた分析を行い，その結果を報告する．",
    },
    // -------------------------------------------------------------------------
    // en-US (4 genres)
    // -------------------------------------------------------------------------
    CorpusCell {
        id: "en-US_novel",
        variant: "en-US",
        genre: "novel",
        dominant_lang: Lang::En,
        allowed_langs: &[Lang::En],
        text: "Chapter I\n\n\"Tom!\"\nNo answer.\n\"Really?\" she asked. \"You mean Mr. Thatcher and Dr. Robinson saw him on Nov. 13, 2026?\"\nHe called it \"absurd.\" The honor of the town was at stake, and they must organize a search.",
    },
    CorpusCell {
        id: "en-US_essay",
        variant: "en-US",
        genre: "essay",
        dominant_lang: Lang::En,
        allowed_langs: &[Lang::En],
        text: "Self-Reliance\n\nTrust thyself: every heart vibrates to that iron string. Nothing is at last sacred but the integrity of your own mind (Emerson, 1841). Society everywhere is in conspiracy against the manhood of every one of its members; it honors conformity and penalizes originality.",
    },
    CorpusCell {
        id: "en-US_social",
        variant: "en-US",
        genre: "social",
        dominant_lang: Lang::En,
        allowed_langs: &[Lang::En],
        text: "tbh i waited 45 mins at the DMV and the guy just said \"system is down\"??? ngl i almost lost it lmao...\nanyway i'm going home to order tacos, don't @ me",
    },
    CorpusCell {
        id: "en-US_paper",
        variant: "en-US",
        genre: "paper",
        dominant_lang: Lang::En,
        allowed_langs: &[Lang::En],
        text: "1. Introduction\n\nLarge language models (Vaswani et al., 2017) have revolutionized natural language processing. In Sec. 2, we analyze tokenization behavior across 12 corpora from the U.S. National Library of Medicine, e.g., PMC Open Access Subset. Our experiments demonstrate a 4.2% reduction in out-of-vocabulary tokens (p < 0.01).",
    },
    // -------------------------------------------------------------------------
    // en-GB (4 genres)
    // -------------------------------------------------------------------------
    CorpusCell {
        id: "en-GB_novel",
        variant: "en-GB",
        genre: "novel",
        dominant_lang: Lang::En,
        allowed_langs: &[Lang::En],
        text: "Chapter I\n\n‘Really?’ she asked. ‘You mean Mr Holmes and Dr Watson saw Prof Moriarty at St James’s on 13 November 2026?’\nHe called it ‘absurd’. The honour of the firm, Baker & Co Ltd, was at stake; they must organise an inquiry.\n'What's that?' asked Mrs Hudson. 'It's a telegram from Scotland Yard.'",
    },
    CorpusCell {
        id: "en-GB_essay",
        variant: "en-GB",
        genre: "essay",
        dominant_lang: Lang::En,
        allowed_langs: &[Lang::En],
        text: "Politics and the English Language\n\nMost people who bother with the matter at all would admit that the English language is in a bad way, but it is generally assumed that we cannot by conscious action do anything about it (Orwell, 1946). Our civilisation is decadent and our language—so the argument runs—must inevitably share in the general collapse. He called such fatalism ‘nonsense’; it is neither inevitable nor excusable.",
    },
    CorpusCell {
        id: "en-GB_social",
        variant: "en-GB",
        genre: "social",
        dominant_lang: Lang::En,
        allowed_langs: &[Lang::En],
        text: "mate i waited 40 mins for the 08:15 train at Euston and the announcer just said 'signalling fault'...\npeak British Rail behaviour tbh, couldn't organise a piss-up in a brewery, innit? 'sorry for the inconvenience' my arse",
    },
    CorpusCell {
        id: "en-GB_paper",
        variant: "en-GB",
        genre: "paper",
        dominant_lang: Lang::En,
        allowed_langs: &[Lang::En],
        text: "1. Introduction\n\nRecent advances in computational linguistics (Derczynski et al., 2016) have transformed corpus analysis across UKRI-funded repositories. In collaboration with Oxford University Press Ltd and Dr Smith at the University of Cambridge, we characterise morphological variation. As noted in Fig. 1, British orthography (e.g. ‘honour’, ‘organise’) exhibits distinct tokenisation patterns (p < 0.01).",
    },
    // -------------------------------------------------------------------------
    // ko (4 genres)
    // -------------------------------------------------------------------------
    CorpusCell {
        id: "ko_novel",
        variant: "ko",
        genre: "novel",
        dominant_lang: Lang::Ko,
        allowed_langs: &[Lang::Ko],
        text: "새침하게 흐린 품이 눈이 올 듯하더니 눈은 아니 오고 얼다가 만 비가 추적추적 내리었다. 이날이야말로 동소문 안에서 인력거꾼 노릇을 하는 김 첨지에게는 오래간만에도 닥친 운수 좋은 날이었다.",
    },
    CorpusCell {
        id: "ko_essay",
        variant: "ko",
        genre: "essay",
        dominant_lang: Lang::Ko,
        allowed_langs: &[Lang::Ko],
        text: "“네 소원이 무엇이냐?” 하고 하느님이 물으시면, 나는 서슴지 않고 “내 소원은 대한 독립이오” 하고 대답할 것이다.\n\n“그다음 소원은 무엇이냐?” 하면, 나는 또 “우리나라의 독립이오” 할 것이요, 또, “그다음 소원이 무엇이냐?” 하는 세 번째 물음에도, 나는 더욱 소리를 높여서, “나의 소원은 우리나라 대한의 완전한 자주독립이오.” 하고 대답할 것이다.",
    },
    CorpusCell {
        id: "ko_social",
        variant: "ko",
        genre: "social",
        dominant_lang: Lang::Ko,
        allowed_langs: &[Lang::Ko, Lang::Mixed],
        text: "지루하지는 않은데 완전 막장임... 돈주고 보기에는....\n3D만 아니었어도 별 다섯 개 줬을텐데.. 왜 3D로 나와서 제 심기를 불편하게 하죠??\n별 재미도없는거 우려먹어 .... 챔프에서 방송 몇번했더라 ? ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ\n설마 ㅈ 현정 작가 아니지??\n일베충들 ㅂㄷ거리는것봐라 ㅉㅉ",
    },
    CorpusCell {
        id: "ko_paper",
        variant: "ko",
        genre: "paper",
        dominant_lang: Lang::Mixed,
        allowed_langs: &[Lang::Ko, Lang::Mixed],
        text: "최근 언어모델(LM)은 문서 작성, 요약, 번역 등 자연어 생성 작업에서 탁월한 성능을 발휘하고 있다. 그러나 생성된 텍스트의 사실성을 보장하는 문제는 여전히 도전 과제로 남아있다. 신뢰할 수 있는 인공지능 시스템을 구축하기 위해서는 언어모델이 생성하는 문서가 얼마나 사실적인 정보로 구성되어 있는지 체계적으로 검증하는 일이 필수적이다. 아직 한국어와 같이 비영어권 언어에서는 이러한 기술의 적용과 검증이 부족한 실정이다.\n\n본 논문에서는 한국어로 생성된 문서의 사실성을 검증하기 위한 시스템인 KorFactScore를 제안한다. LM의 사실성을 실험적으로 평가하고 LM과 검색기의 결합이 검증 성능에 미치는 영향을 분석하였다. 한국어 위키피디아 데이터를 활용해 한국인과 외국인 인물에 대한 약력 데이터를 생성하고, 이를 평가하기 위해 다양한 LLM(GPT-3.5, GPT-4, Gemini, EXAONE, LLaMA)과 검색기(Cross-encoder, BM25)를 적용하였다.",
    },
];

fn assert_valid_span(cell_id: &str, kind: &str, span: Span, text_chars: usize) {
    assert!(
        span.start_char < span.end_char && span.end_char <= text_chars,
        "[{cell_id}] invalid {kind} span {span:?} (text_chars={text_chars})"
    );
}

fn verify_cell_invariants(cell: &CorpusCell, options: &Options) {
    let doc = analyze_document(cell.text, options);
    let chars: Vec<char> = cell.text.chars().collect();
    assert_eq!(
        doc.text_chars,
        chars.len(),
        "[{}] text_chars mismatch",
        cell.id
    );

    // Determinism check
    let doc2 = analyze_document(cell.text, options);
    assert_eq!(doc, doc2, "[{}] non-deterministic output", cell.id);

    // 1. Valid spans for Paragraph, Sentence, Clause, Opaque, Token, Unknown
    assert!(
        !doc.segmentation.paragraphs.is_empty(),
        "[{}] expected non-empty paragraphs",
        cell.id
    );
    assert!(
        !doc.segmentation.sentences.is_empty(),
        "[{}] expected non-empty sentences",
        cell.id
    );

    for p in &doc.segmentation.paragraphs {
        assert_valid_span(cell.id, "paragraph", p.span, doc.text_chars);
        assert!(
            cell.allowed_langs.contains(&p.lang),
            "[{}] unexpected paragraph lang {:?} for {:?}",
            cell.id,
            p.lang,
            slice(cell.text, p.span)
        );
    }
    assert!(
        doc.segmentation
            .paragraphs
            .iter()
            .any(|p| p.lang == cell.dominant_lang)
            || doc
                .segmentation
                .sentences
                .iter()
                .any(|s| s.lang == cell.dominant_lang),
        "[{}] dominant lang {:?} not observed in paragraphs or sentences",
        cell.id,
        cell.dominant_lang
    );

    for s in &doc.segmentation.sentences {
        assert_valid_span(cell.id, "sentence", s.span, doc.text_chars);
        assert!(
            cell.allowed_langs.contains(&s.lang),
            "[{}] unexpected sentence lang {:?} for {:?}",
            cell.id,
            s.lang,
            slice(cell.text, s.span)
        );
    }
    for c in &doc.segmentation.clauses {
        assert_valid_span(cell.id, "clause", c.span, doc.text_chars);
    }
    for o in &doc.segmentation.opaque {
        assert_valid_span(cell.id, "opaque", o.span, doc.text_chars);
    }
    for t in &doc.tokens {
        assert_valid_span(cell.id, "token", t.span, doc.text_chars);
    }
    for u in &doc.unknown {
        assert_valid_span(cell.id, "unknown", u.span, doc.text_chars);
    }

    // 2. Non-overlapping sorted tokens & exact char coverage inside supported sentences
    for w in doc.tokens.windows(2) {
        assert!(
            w[0].span.end_char <= w[1].span.start_char,
            "[{}] tokens overlap or unsorted: {:?} vs {:?}",
            cell.id,
            w[0],
            w[1]
        );
    }

    let unsupported: Vec<Span> = doc
        .unknown
        .iter()
        .filter(|u| u.layer == "segmentation" && u.reason == Reason::UnsupportedLanguage)
        .map(|u| u.span)
        .collect();

    for (sent_idx, s) in doc.segmentation.sentences.iter().enumerate() {
        if unsupported.contains(&s.span) {
            assert!(
                doc.tokens.iter().all(|t| t.sentence != sent_idx),
                "[{}] unsupported sentence emitted tokens",
                cell.id
            );
            continue;
        }
        for (char_idx, ch) in chars
            .iter()
            .enumerate()
            .take(s.span.end_char)
            .skip(s.span.start_char)
        {
            if ch.is_whitespace() {
                continue;
            }
            let covering = doc
                .tokens
                .iter()
                .filter(|t| {
                    t.sentence == sent_idx
                        && t.span.start_char <= char_idx
                        && char_idx < t.span.end_char
                })
                .count();
            assert_eq!(
                covering, 1,
                "[{}] char index {char_idx} ({ch:?}) in sentence {sent_idx} covered by {covering} tokens",
                cell.id
            );
        }
    }

    // 3. 1-to-1 correspondence between tokens with upos == None and Unknown entries with layer == "pos"
    let undecided_spans: Vec<Span> = doc
        .tokens
        .iter()
        .filter(|t| t.upos.is_none())
        .map(|t| t.span)
        .collect();
    let pos_unknown_spans: Vec<Span> = doc
        .unknown
        .iter()
        .filter(|u| u.layer == "pos")
        .map(|u| u.span)
        .collect();
    assert_eq!(
        undecided_spans, pos_unknown_spans,
        "[{}] mismatch between upos==None tokens and layer==\"pos\" unknowns",
        cell.id
    );

    // 4. Layer 3 (entities) invariants: valid spans, sentence bounds, sorted non-overlapping
    for e in &doc.entities {
        assert_valid_span(cell.id, "entity", e.span, doc.text_chars);
        assert!(
            e.sentence < doc.segmentation.sentences.len(),
            "[{}] entity sentence index {} out of bounds",
            cell.id,
            e.sentence
        );
        let s_span = doc.segmentation.sentences[e.sentence].span;
        assert!(
            e.span.start_char >= s_span.start_char && e.span.end_char <= s_span.end_char,
            "[{}] entity {:?} outside its sentence {:?}",
            cell.id,
            e.span,
            s_span
        );
    }
    for w in doc.entities.windows(2) {
        assert!(
            w[0].span.end_char <= w[1].span.start_char,
            "[{}] entities overlap or unsorted: {:?} vs {:?}",
            cell.id,
            w[0],
            w[1]
        );
    }

    // 5. Layer 7 (relations) invariants: valid spans, unit membership, non-overlapping args, sorted
    let sent_spans: Vec<Span> = doc.segmentation.sentences.iter().map(|s| s.span).collect();
    let clause_spans: Vec<Span> = doc.segmentation.clauses.iter().map(|c| c.span).collect();
    for r in &doc.relations {
        assert_valid_span(cell.id, "relation.connector", r.connector, doc.text_chars);
        assert_valid_span(cell.id, "relation.arg1", r.arg1, doc.text_chars);
        assert_valid_span(cell.id, "relation.arg2", r.arg2, doc.text_chars);
        if let Some(corr) = r.correlate {
            assert_valid_span(cell.id, "relation.correlate", corr, doc.text_chars);
        }
        let unit_spans = match r.level {
            RelationLevel::Clause => &clause_spans,
            RelationLevel::Sentence => &sent_spans,
        };
        assert!(
            unit_spans.contains(&r.arg1),
            "[{}] relation arg1 {:?} is not a {:?} unit",
            cell.id,
            r.arg1,
            r.level
        );
        assert!(
            unit_spans.contains(&r.arg2),
            "[{}] relation arg2 {:?} is not a {:?} unit",
            cell.id,
            r.arg2,
            r.level
        );
        assert!(
            r.arg1.end_char <= r.arg2.start_char || r.arg2.end_char <= r.arg1.start_char,
            "[{}] relation args overlap: {:?} vs {:?}",
            cell.id,
            r.arg1,
            r.arg2
        );
    }
    for w in doc.relations.windows(2) {
        assert!(
            w[0].connector.start_char <= w[1].connector.start_char,
            "[{}] relations unsorted: {:?} vs {:?}",
            cell.id,
            w[0],
            w[1]
        );
    }

    // 6. ja/ko cells must have 0 UnsupportedLanguage unknowns
    if matches!(cell.variant, "ja" | "ko") {
        assert!(
            !doc.unknown
                .iter()
                .any(|u| u.reason == Reason::UnsupportedLanguage),
            "[{}] unexpected UnsupportedLanguage unknown on {} cell: {:?}",
            cell.id,
            cell.variant,
            doc.unknown
        );
    }
}

#[test]
fn multilingual_24_cell_matrix_invariants() {
    assert_eq!(CELLS.len(), 24, "expected 6 variants x 4 genres = 24 cells");

    let mut seen_pairs = std::collections::BTreeSet::new();
    for cell in CELLS {
        assert!(
            seen_pairs.insert((cell.variant, cell.genre)),
            "duplicate cell ({}, {})",
            cell.variant,
            cell.genre
        );
    }
    assert_eq!(seen_pairs.len(), 24);

    let default_opts = Options::default();
    let norm_trad_opts = Options {
        normalize_traditional: true,
        ..Default::default()
    };

    for cell in CELLS {
        verify_cell_invariants(cell, &default_opts);
        verify_cell_invariants(cell, &norm_trad_opts);
    }

    // Verify Layer 3 (entities) and Layer 7 (relations) extract non-zero results across ja and ko
    for variant in ["ja", "ko"] {
        let mut entity_count = 0usize;
        let mut relation_count = 0usize;
        for cell in CELLS.iter().filter(|c| c.variant == variant) {
            let doc = analyze_document(cell.text, &default_opts);
            entity_count += doc.entities.len();
            relation_count += doc.relations.len();
        }
        assert!(
            entity_count > 0,
            "expected non-zero Layer 3 entities across {variant} real-corpus cells"
        );
        assert!(
            relation_count > 0,
            "expected non-zero Layer 7 relations across {variant} real-corpus cells"
        );
    }
}

#[test]
fn ja_academic_fullwidth_period_jstage() {
    let jstage = CELLS
        .iter()
        .find(|c| c.id == "ja_paper")
        .expect("ja_paper cell missing");
    let doc = analyze_document(jstage.text, &Options::default());
    let sents = sentence_texts(jstage.text, &doc);
    assert_eq!(
        sents.len(),
        5,
        "J-STAGE abstract using U+FF0E fullwidth period should split into 5 sentences, got: {sents:?}"
    );

    let fullwidth_period_tokens: Vec<_> = doc
        .tokens
        .iter()
        .filter(|t| slice(jstage.text, t.span) == "．")
        .collect();
    assert_eq!(
        fullwidth_period_tokens.len(),
        5,
        "expected 5 fullwidth period tokens"
    );
    for t in fullwidth_period_tokens {
        assert_eq!(
            t.upos,
            Some(Upos::Punct),
            "fullwidth period token `．` should have Upos::Punct"
        );
    }

    let decimal_text = "円周率は約３．１４である。";
    let decimal_doc = analyze_document(decimal_text, &Options::default());
    assert_eq!(
        sentence_texts(decimal_text, &decimal_doc),
        vec!["円周率は約３．１４である。"],
        "fullwidth decimal `３．１４` must NOT split sentences"
    );
}

#[test]
fn en_us_vs_en_gb_academic_and_dialogue() {
    let academic = "1. Introduction\n\nLarge language models (Vaswani et al., 2017) have revolutionized natural language processing. In Sec. 2, we analyze tokenization behavior across 12 corpora from the U.S. National Library of Medicine, e.g., PMC Open Access Subset.\n\n2.1 Experimental Setup\n\nSee Ch. 3, Vol. 5, and p. 14 for U.K. Parliament benchmarks.";
    let doc = analyze_document(academic, &Options::default());

    let headings: Vec<String> = doc
        .segmentation
        .opaque
        .iter()
        .filter(|o| o.kind == OpaqueKind::Heading)
        .map(|o| slice(academic, o.span))
        .collect();
    assert_eq!(
        headings,
        vec!["1. Introduction", "2.1 Experimental Setup"],
        "numbered section headings should be OpaqueKind::Heading"
    );

    let sents = sentence_texts(academic, &doc);
    assert_eq!(
        sents,
        vec![
            "Large language models (Vaswani et al., 2017) have revolutionized natural language processing.",
            "In Sec. 2, we analyze tokenization behavior across 12 corpora from the U.S. National Library of Medicine, e.g., PMC Open Access Subset.",
            "See Ch. 3, Vol. 5, and p. 14 for U.K. Parliament benchmarks.",
        ],
        "academic abbreviations (Sec., U.S., Ch., Vol., p., U.K.) must not cause false sentence splits"
    );

    // en-GB single-quote dialogue, logical punctuation, and dotless titles
    let gb_cases = [
        "‘Really?’ she asked.",
        "'What's that?' asked Mrs Hudson.",
        "He called it ‘absurd’.",
        "‘You mean Mr Holmes and Dr Watson saw Prof Moriarty at St James’s on 13 November 2026?’",
    ];
    for case in gb_cases {
        let d = analyze_document(case, &Options::default());
        assert_eq!(
            sentence_texts(case, &d),
            vec![case],
            "en-GB dialogue/dotless title should stay 1 sentence: {case}"
        );
        assert_eq!(d.segmentation.sentences[0].lang, Lang::En);
    }

    // en-US double-quote dialogue and US date (Nov. 13, 2026)
    let us_dialogue = "\"Really?\" she asked.";
    let d_us = analyze_document(us_dialogue, &Options::default());
    assert_eq!(sentence_texts(us_dialogue, &d_us), vec![us_dialogue]);

    let us_date = "They met on Nov. 13, 2026 in Boston.";
    let d_date = analyze_document(us_date, &Options::default());
    assert_eq!(sentence_texts(us_date, &d_date), vec![us_date]);
    assert_eq!(
        clause_texts(us_date, &d_date),
        vec![us_date],
        "US date `Nov. 13, 2026` must not split `, 2026` into a separate clause"
    );
    let toks = token_texts(us_date, &d_date);
    assert!(
        toks.contains(&"Nov.".to_string()),
        "abbreviated month `Nov.` should stay 1 token, got: {toks:?}"
    );
}

#[test]
fn ko_essay_quotative_continuation_and_paren_commas() {
    let kim_koo_1 = "“네 소원이 무엇이냐?” 하고 하느님이 물으시면, 나는 서슴지 않고 “내 소원은 대한 독립이오” 하고 대답할 것이다.";
    let doc1 = analyze_document(kim_koo_1, &Options::default());
    assert_eq!(
        sentence_texts(kim_koo_1, &doc1),
        vec![kim_koo_1],
        "Korean quotative continuation `“...?” 하고` must stay 1 sentence"
    );

    let kim_koo_2 = "“나의 소원은 우리나라 대한의 완전한 자주독립이오.” 하고 대답할 것이다.";
    let doc2 = analyze_document(kim_koo_2, &Options::default());
    assert_eq!(
        sentence_texts(kim_koo_2, &doc2),
        vec![kim_koo_2],
        "Korean quotative continuation `“....” 하고` must stay 1 sentence"
    );

    let paren_list = "LLM(GPT-3.5, GPT-4, Gemini, EXAONE, LLaMA)을 활용하여 평가하였다.";
    let doc3 = analyze_document(paren_list, &Options::default());
    assert_eq!(
        clause_texts(paren_list, &doc3),
        vec![paren_list],
        "commas inside parentheses `LLM(GPT-3.5, ...)` must not split clauses"
    );
}

#[test]
fn ja_social_slang_and_soseki_particles() {
    let slang = "雨…（笑）";
    let doc_slang = analyze_document(slang, &Options::default());
    assert_eq!(
        sentence_texts(slang, &doc_slang),
        vec![slang],
        "`雨…（笑）` must stay 1 sentence"
    );
    assert_eq!(
        doc_slang.segmentation.sentences[0].lang,
        Lang::Ja,
        "`雨…（笑）` should be classified as Lang::Ja"
    );

    let wagahai = "吾輩は猫である。名前はまだ無い。";
    let doc_wagahai = analyze_document(wagahai, &Options::default());
    let toks_wagahai = token_texts(wagahai, &doc_wagahai);
    assert!(
        toks_wagahai.windows(3).any(|w| w == ["名前", "は", "まだ"]),
        "expected `名前 | は | まだ` (not `はま | だ`), got: {toks_wagahai:?}"
    );

    let garasudo = "私はただ坐ったり";
    let doc_garasudo = analyze_document(garasudo, &Options::default());
    let toks_garasudo = token_texts(garasudo, &doc_garasudo);
    assert!(
        toks_garasudo.windows(3).any(|w| w == ["私", "は", "ただ"]),
        "expected `私 | は | ただ` (not `はた | だ`), got: {toks_garasudo:?}"
    );
}

#[test]
fn zh_hant_normalize_traditional_and_ptt() {
    let beiying = "到徐州見着父親，看見滿院狼藉的東西，又想起祖母，不禁簌簌地流下眼淚。";
    let opts = Options {
        normalize_traditional: true,
        ..Default::default()
    };
    let doc = analyze_document(beiying, &opts);
    let toks = token_texts(beiying, &doc);
    assert!(
        toks.contains(&"狼藉".to_string()),
        "`normalize_traditional: true` must keep `狼藉` as a single token, got: {toks:?}"
    );

    let ptt = "推樓上：真的笑死\n※ 發信站: 批踢踢實業坊(ptt.cc), 來自: 114.32.0.1 (臺灣)";
    let doc_ptt = analyze_document(ptt, &Options::default());
    let list_items: Vec<String> = doc_ptt
        .segmentation
        .opaque
        .iter()
        .filter(|o| o.kind == OpaqueKind::ListItem)
        .map(|o| slice(ptt, o.span))
        .collect();
    assert!(
        list_items
            .iter()
            .any(|s| s.starts_with("※ 發信站: 批踢踢實業坊(ptt.cc)")),
        "PTT `※ 發信站:` line should be tagged as OpaqueKind::ListItem, got: {list_items:?}"
    );
}
