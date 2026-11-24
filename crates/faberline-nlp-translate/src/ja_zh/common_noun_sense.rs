//! Resolve common homographic nouns from their own Japanese syntactic context.

use super::{
    Atom, ClauseAlignment, Document, Span, TargetVariant, Upos, dasu_case_link,
    localize_close_quote, localize_open_quote, nominal_possession,
};

const MARKER: &str = "__ja_common_noun_sense:";

fn supported(surface: &str) -> bool {
    matches!(
        surface,
        "ソフト" | "アルバム" | "ドラマ" | "ポスト" | "バッテリー" | "カウンター" | "ケーブル"
    )
}

fn boundary(atom: &Atom) -> bool {
    atom.is_locked
        || matches!(
            atom.surface.as_str(),
            "「" | "」" | "『" | "』" | "“" | "”" | "‘" | "’" | "。" | "！" | "？"
        )
}

fn like(atom: &Atom, words: &[&str]) -> bool {
    words.contains(&atom.surface.as_str())
        || atom
            .lemma
            .as_deref()
            .is_some_and(|lemma| words.contains(&lemma))
}

// Preserve the noun's original surface and grammatical identity for rendering.
pub(super) fn bind(atoms: &mut [Atom], doc: &Document, chars: &[char]) {
    let readings: Vec<_> = atoms
        .iter()
        .enumerate()
        .filter_map(|(index, noun)| {
            if noun.is_locked
                || noun.upos != Some(Upos::Noun)
                || noun.head.is_none()
                || !supported(&noun.surface)
            {
                return None;
            }
            let original = doc.tokens.get(noun.doc_tok_idx?)?;
            if original.span != noun.span
                || original.upos != noun.upos
                || !chars
                    .get(noun.span.start_char..noun.span.end_char)?
                    .iter()
                    .copied()
                    .eq(noun.surface.chars())
            {
                return None;
            }

            let start = atoms[..index]
                .iter()
                .rposition(boundary)
                .map_or(0, |i| i + 1);
            let end = atoms[index + 1..]
                .iter()
                .position(boundary)
                .map_or(atoms.len(), |i| index + 1 + i);
            let context = Context {
                noun,
                local: &atoms[start..end],
            };
            Some((index, context.reading().unwrap_or("")))
        })
        .collect();

    for (index, gloss) in readings {
        atoms[index].lemma = Some(format!("{MARKER}{gloss}"));
    }
}

pub(super) fn render<'a>(surface: &'a str, lemma: Option<&'a str>) -> Option<&'a str> {
    if !supported(surface) {
        return None;
    }
    let gloss = lemma?.strip_prefix(MARKER)?;
    Some(if gloss.is_empty() { surface } else { gloss })
}

pub(super) fn quoted_definition(
    atoms: &[Atom],
    doc: &Document,
    sentence: usize,
    chars: &[char],
) -> Option<(String, Vec<ClauseAlignment>)> {
    if atoms.iter().any(|atom| atom.is_locked) {
        return None;
    }
    let span = doc.segmentation.sentences.get(sentence)?.span;
    let source: String = chars.get(span.start_char..span.end_char)?.iter().collect();
    let (prefix, quoted) = source
        .strip_prefix("引用した")
        .map_or(("", source.as_str()), |rest| ("引述的", rest));
    let open = quoted.chars().next()?;
    let close = match open {
        '「' => '」',
        '『' => '』',
        _ => return None,
    };
    let (term, rest) = quoted.strip_prefix(open)?.split_once(close)?;
    if !supported(term) {
        return None;
    }
    let definition = rest
        .strip_prefix("という語は")
        .or_else(|| rest.strip_prefix("という言葉は"))?
        .trim_start_matches('、');
    let (location, definition) = definition
        .strip_prefix("ここでは")
        .map_or(("", definition), |rest| ("在這裡", rest));
    let role = match definition {
        "役職の意味です。" | "役職という意味です。" => "職位",
        "役職の意味です" | "役職という意味です" => "職位",
        _ => return None,
    };
    let subject = format!(
        "{prefix}{}{term}{}一詞",
        localize_open_quote(open, TargetVariant::ZhHant),
        localize_close_quote(close, TargetVariant::ZhHant)
    );
    let pivot = span.start_char + source.split_once(close)?.0.chars().count() + 1;
    let subject_end = subject.chars().count();
    let punctuation = if definition.ends_with('。') {
        "。"
    } else {
        ""
    };
    let text = format!("{subject}{location}是{role}的意思{punctuation}");
    let end = text.chars().count();
    let alignments = doc
        .segmentation
        .clauses
        .iter()
        .enumerate()
        .filter(|(_, clause)| clause.sentence == sentence)
        .map(|(index, clause)| ClauseAlignment {
            sentence,
            clause: index,
            source_span: clause.span,
            target_span: Span {
                start_char: if clause.span.start_char >= pivot {
                    subject_end
                } else {
                    0
                },
                end_char: if clause.span.end_char <= pivot {
                    subject_end
                } else {
                    end
                },
            },
        })
        .collect();
    Some((text, alignments))
}

pub(super) fn render_transfer(source: &[Atom], doc: &Document, sentence: usize) -> Option<String> {
    if !source.iter().any(|atom| supported(&atom.surface))
        || source.iter().any(|atom| atom.is_locked)
    {
        return None;
    }
    let predicate = source
        .iter()
        .rposition(|atom| atom.upos == Some(Upos::Verb))?;
    let verb = &source[predicate];
    if !like(verb, &["入れる", "投函する", "インストールする"]) {
        return None;
    }
    let governor = verb.doc_tok_idx?;
    let ending = &source[predicate + 1..];
    if ending.iter().any(|atom| atom.upos != Some(Upos::Aux)) {
        return None;
    }
    let surface: String = ending.iter().map(|atom| atom.surface.as_str()).collect();
    let aspect = match surface.as_str() {
        "ました" | "た" => "了",
        "" | "ます" => "",
        _ => return None,
    };
    let mut subject = None;
    let mut object = None;
    let mut destination = None;
    let mut start = 0;
    for (index, pair) in source[..predicate].windows(2).enumerate() {
        let [noun, particle] = pair else { continue };
        if noun.head != Some(governor) || !dasu_case_link(noun, particle) {
            continue;
        }
        let slot = match (noun.deprel.as_str(), particle.surface.as_str()) {
            ("nsubj", "は" | "が") => &mut subject,
            ("obj", "を") => &mut object,
            ("obl", "に" | "へ") => &mut destination,
            _ => return None,
        };
        if slot.replace((source.get(start..=index)?, noun)).is_some() {
            return None;
        }
        start = index + 2;
    }
    if start != predicate {
        return None;
    }
    let (object, item) = object?;
    let (destination, place) = destination?;
    let item_reading = render(&item.surface, item.lemma.as_deref());
    let place_reading = render(&place.surface, place.lemma.as_deref());
    let install = item_reading == Some("軟體")
        && like(verb, &["入れる", "インストールする"])
        && like(place, &["パソコン", "コンピューター", "端末"]);
    let noun_phrase =
        |atoms| nominal_possession::noun_phrase(atoms, doc, sentence, TargetVariant::ZhHant);
    let subject = match subject {
        Some((subject, _)) => noun_phrase(subject)?,
        None => String::new(),
    };
    let item = noun_phrase(object)?;
    let place = noun_phrase(destination)?;
    if install {
        return Some(format!("{subject}在{place}安裝{aspect}{item}"));
    }
    let action = match place_reading {
        Some("相簿") if like(verb, &["入れる"]) => "放進",
        Some("郵筒") if like(verb, &["入れる", "投函する"]) => "投進",
        _ => return None,
    };
    Some(format!("{subject}把{item}{action}{aspect}{place}"))
}

struct Context<'a> {
    noun: &'a Atom,
    local: &'a [Atom],
}

impl Context<'_> {
    fn head(&self) -> Option<&Atom> {
        let head = self.noun.head?;
        self.local
            .iter()
            .find(|atom| atom.doc_tok_idx == Some(head))
    }

    fn predicate(&self, role: &str, words: &[&str]) -> bool {
        self.head().is_some_and(|head| {
            head.upos == Some(Upos::Verb) && self.noun.deprel == role && like(head, words)
        })
    }

    fn case_of(&self, noun: &Atom, choices: &[&str]) -> bool {
        noun.doc_tok_idx.is_some()
            && self
                .local
                .iter()
                .any(|case| dasu_case_link(noun, case) && choices.contains(&case.surface.as_str()))
    }

    fn case(&self, choices: &[&str]) -> bool {
        self.case_of(self.noun, choices)
    }

    fn modifier(&self, words: &[&str]) -> bool {
        self.local.iter().any(|atom| {
            atom.head == self.noun.doc_tok_idx
                && matches!(atom.deprel.as_str(), "amod" | "nmod" | "compound")
                && like(atom, words)
        })
    }

    fn relatives(&self) -> impl Iterator<Item = &Atom> {
        self.local.iter().filter(|atom| {
            atom.head == self.noun.doc_tok_idx
                && atom.upos == Some(Upos::Verb)
                && matches!(atom.deprel.as_str(), "acl" | "acl:relcl")
        })
    }

    fn relative(&self, words: &[&str]) -> bool {
        self.relatives().any(|atom| like(atom, words))
    }

    fn relative_argument(&self, verb: &[&str], term: &[&str], roles: &[&str]) -> bool {
        self.relatives().any(|predicate| {
            predicate.doc_tok_idx.is_some()
                && like(predicate, verb)
                && self.local.iter().any(|atom| {
                    atom.head == predicate.doc_tok_idx
                        && roles.contains(&atom.deprel.as_str())
                        && like(atom, term)
                })
        })
    }

    fn peers(&self, role: &str) -> impl Iterator<Item = &Atom> {
        self.local.iter().filter(move |atom| {
            self.noun.head.is_some()
                && atom.doc_tok_idx != self.noun.doc_tok_idx
                && atom.head == self.noun.head
                && atom.deprel == role
        })
    }

    fn peer(&self, words: &[&str], role: &str, particles: &[&str]) -> bool {
        self.peers(role)
            .any(|atom| like(atom, words) && self.case_of(atom, particles))
    }

    fn reading(&self) -> Option<&'static str> {
        match self.noun.surface.as_str() {
            "ソフト" => self.soft(),
            "アルバム" => self.album(),
            "ドラマ" => self.drama(),
            "ポスト" => self.post(),
            "バッテリー" => self.battery(),
            "カウンター" => self.counter(),
            "ケーブル" => self.cable(),
            _ => None,
        }
    }

    fn soft(&self) -> Option<&'static str> {
        let install = self.predicate("obj", &["入れる", "インストールする", "インストール"])
            && self.case(&["を"])
            && self.peer(&["パソコン", "コンピューター", "端末"], "obl", &["に"]);
        let software = self.modifier(&["ゲーム", "業務用"])
            || install
            || self.predicate("obj", &["起動する", "起動"]);
        let game = self.noun.deprel == "nmod"
            && self
                .head()
                .is_some_and(|head| like(head, &["試合", "大会"]));
        let play = self.predicate("obj", &["する", "プレーする", "プレー"])
            && self.peer(
                &["公園", "校庭", "球場", "グラウンド", "運動場"],
                "obl",
                &["で"],
            );
        select(&[
            (software, "軟體"),
            (
                self.predicate("obj", &["食べる", "舐める"]) && self.case(&["を"]),
                "霜淇淋",
            ),
            (game || play, "壘球"),
        ])
    }

    fn album(&self) -> Option<&'static str> {
        let photo = self.modifier(&["写真", "卒業"])
            || (self.case(&["に"])
                && self.predicate("obl", &["入れる", "貼る", "貼り付ける", "載せる"])
                && self.peer(&["写真"], "obj", &["を"]));
        let music = self.modifier(&["歌手", "アーティスト", "バンド", "音楽"])
            || self.predicate("obj", &["聴く", "試聴する", "試聴"]);
        select(&[(photo, "相簿"), (music, "音樂專輯")])
    }

    fn drama(&self) -> Option<&'static str> {
        let writing = &["書く", "書き足す", "書き直す"];
        let script = (self.predicate("obj", writing) && self.case(&["を"]))
            || (self.predicate("obl", writing) && self.case(&["に"]));
        let events = &["再会", "出会い", "試合", "事件", "逆転", "出来事", "結末"];
        let event = self.case(&["に"])
            && self.predicate("obl", &["なる"])
            && (self.peer(events, "nsubj", &["は", "が"])
                || self.relative_argument(&["見守る", "見届ける"], events, &["nsubj"]));
        let stage = self.modifier(&["舞台"])
            || self.predicate("obj", &["上演する", "上演"])
            || self.relative(&["上演する", "上演"])
            || self.relative_argument(&["する"], &["上演"], &["compound", "obl"]);
        let television = self.modifier(&["テレビ"])
            || self.peer(&["テレビ"], "obl", &["で"])
            || self.predicate("obj", &["放送する", "放送"]);
        select(&[
            (script, "劇本"),
            (event, "戲劇性事件"),
            (stage && !script, "戲劇"),
            (television, "電視劇"),
        ])
    }

    fn post(&self) -> Option<&'static str> {
        let mail = self.predicate("obl", &["入れる", "投函", "投函する"])
            && self.case(&["に", "へ"])
            && self.peer(&["手紙", "葉書", "はがき", "郵便"], "obj", &["を"]);
        let vacancy = self.predicate("nsubj", &["空く", "空いている"])
            && self.modifier(&["店長", "部長", "社長", "代表", "委員長"]);
        let job = self.modifier(&["役職"])
            || vacancy
            || (self.case(&["に"]) && self.predicate("obl", &["就く"]));
        let goal = self.modifier(&["ゴール", "ゴールバー"])
            && self.predicate("obl", &["当たる", "当てる", "かすめる"]);
        select(&[(mail, "郵筒"), (job, "職位"), (goal, "球門柱")])
    }

    fn battery(&self) -> Option<&'static str> {
        let unit = self.modifier(&["投手", "捕手", "ピッチャー", "キャッチャー", "野球"]);
        let duration = self.predicate("nsubj", &["使える", "持つ"])
            && self.peers("obl").any(|atom| self.case_of(atom, &["から"]))
            && self.peers("obl").any(|atom| self.case_of(atom, &["まで"]));
        let electricity = self.modifier(&["充電", "予備", "車", "携帯", "スマホ"])
            || self.predicate("obj", &["充電", "充電する", "買う", "放電する", "放電"])
            || duration;
        select(&[(unit, "投捕搭檔"), (electricity, "電池")])
    }

    fn counter(&self) -> Option<&'static str> {
        let desk = self.modifier(&["窓口", "受付", "レジ"])
            || (self.case(&["で"])
                && (self.predicate("obl", &["払う", "支払う"])
                    || (self.predicate("obl", &["買う"])
                        && self.peer(&["切符", "券", "チケット"], "obj", &["を"]))));
        let device = self.relative(&["数える", "計測する", "カウントする"])
            || self.modifier(&["人数", "回数", "計数"]);
        let response = self.predicate("obj", &["仕掛ける", "放つ", "返す"])
            && self.peer(&["攻撃"], "obl", &["に"]);
        select(&[(desk, "櫃檯"), (device, "計數器"), (response, "反擊")])
    }

    fn cable(&self) -> Option<&'static str> {
        let transport = self.predicate("obl", &["乗る", "乗り込む"])
            && self.case(&["に"])
            && self.modifier(&["山", "山頂", "索道"]);
        let structural = self.modifier(&["鋼", "鋼製", "ワイヤー"])
            || self.relative_argument(&["支える"], &["橋", "橋梁"], &["obj"]);
        let wire = self.modifier(&["電源", "ＵＳＢ", "USB", "充電", "通信"])
            || self.relative(&["つなぐ", "繋ぐ", "接続する", "接続"]);
        let placed = self.predicate("obj", &["置く"]) && self.peer(&["床", "机"], "obl", &["に"]);
        select(&[
            (transport, "纜車"),
            (structural, "纜索"),
            (wire || (placed && !structural), "纜線"),
        ])
    }
}

fn select(choices: &[(bool, &'static str)]) -> Option<&'static str> {
    let mut selected = choices
        .iter()
        .filter_map(|&(matched, gloss)| matched.then_some(gloss));
    let choice = selected.next();
    if selected.next().is_some() {
        None
    } else {
        choice
    }
}

#[cfg(test)]
mod tests {
    use crate::{TranslateOptions, TranslationGlossary, analyze_and_translate};
    use faberline_nlp::doc::model::Backend;
    use serde_json::Value;

    use super::*;

    fn adopted_bank() -> Value {
        serde_json::from_str(include_str!(
            "../../tests/data/ja_zh/common_noun_sense_dev_v1.json"
        ))
        .unwrap()
    }

    fn run(bank: &Value, id: &str) -> crate::TranslatedDocument {
        let item = bank["items"]
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row["id"] == id)
            .unwrap();
        assert_eq!(item["visibility"], "development-visible");
        assert_eq!(item["heldout"], false);
        let glossary: TranslationGlossary = serde_json::from_value(
            item.pointer("/fixture/glossary")
                .unwrap_or(&bank["fixture"]["glossary"])
                .clone(),
        )
        .unwrap();
        let mut options = TranslateOptions::new(TargetVariant::ZhHant);
        options.source_hint = Some(TargetVariant::Ja);
        analyze_and_translate(
            item["jp"].as_str().unwrap(),
            &options,
            &glossary,
            None,
            Backend::Onnx,
        )
        .unwrap()
    }

    #[test]
    fn adopted_noun_senses_are_selected_by_their_own_clause() {
        let bank = adopted_bank();
        for (id, wanted, unwanted) in [
            ("N03", "霜淇淋", "軟體"),
            ("N06", "壘球", "軟體"),
            ("N08", "音樂專輯", "相簿"),
            ("N10", "相簿", "音樂專輯"),
            ("N13", "劇本", "電視劇"),
            ("N14", "戲劇性事件", "連續劇"),
            ("N16", "劇本", "電視劇"),
            ("N19", "球門柱", "郵筒"),
            ("N24", "投捕搭檔", "電池"),
            ("N26", "電池", "投捕搭檔"),
            ("N29", "反擊", "櫃檯"),
            ("N31", "櫃檯", "計數器"),
            ("N34", "纜索", "纜車"),
            ("N39", "電池", "投捕搭檔"),
        ] {
            let result = run(&bank, id);
            assert!(
                result.target_text.contains(wanted),
                "{id}: {}",
                result.target_text
            );
            assert!(
                !result.target_text.contains(unwanted),
                "{id}: {}",
                result.target_text
            );
        }
        let cable = run(&bank, "N37").target_text;
        assert!(
            cable
                .find("纜線")
                .zip(cable.find("纜車"))
                .is_some_and(|(wire, car)| wire < car)
        );
    }

    #[test]
    fn adopted_weak_context_and_protected_spans_are_preserved() {
        let bank = adopted_bank();
        for (id, term) in [
            ("N05", "ソフト"),
            ("N09", "アルバム"),
            ("N15", "ドラマ"),
            ("N20", "ポスト"),
            ("N25", "バッテリー"),
            ("N30", "カウンター"),
            ("N36", "ケーブル"),
            ("N38", "アルバム"),
        ] {
            assert!(run(&bank, id).target_text.contains(term), "{id}");
        }
        let code = run(&bank, "N32").target_text;
        let guarded = format!("{0}カウンター.reset(){0}", char::from(96));
        assert!(code.contains("計數器") && code.contains(&guarded));
        let definition = run(&bank, "N22").target_text;
        assert!(definition.contains("『ポスト』") && definition.contains("職位"));

        let locked = run(&bank, "N40");
        assert!(locked.target_text.contains("戲劇作品"));
        let [hit] = locked.glossary_hits.as_slice() else {
            panic!("expected one caller glossary hit");
        };
        assert_eq!(hit.canonical, "ドラマ");
        assert_eq!(
            hit.source_span,
            Span {
                start_char: 0,
                end_char: 3
            }
        );
    }
}
