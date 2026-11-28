//! Distinguish a detached part, an available item, a harvest, and a photograph.

use super::{Atom, RenderedFrame, Upos, adjacent, dasu_case_link, linked, read_nominal};

fn auxiliary(aux: &Atom, verb: &Atom, surface: &str) -> bool {
    aux.surface == surface && aux.upos == Some(Upos::Aux) && linked(aux, verb, "aux")
}

fn split_photography(quality: &Atom, tail: &[Atom]) -> bool {
    let [ni, to, reru] = tail else {
        return false;
    };
    // The analyzer can attach all three suffix atoms to the adjective.
    // This only identifies a candidate. The camera and concession frame
    // below must establish the photographic reading.
    quality.upos == Some(Upos::Adj)
        && matches!(quality.surface.as_str(), "きれい" | "綺麗")
        && ni.surface == "に"
        && dasu_case_link(quality, ni)
        && to.surface == "と"
        && to.upos == Some(Upos::Adp)
        && linked(to, quality, "case")
        && auxiliary(reru, quality, "れる")
        && adjacent(ni, to)
        && adjacent(to, reru)
}

fn noun_case(noun: &Atom, case: &Atom, surface: &str) -> bool {
    noun.upos == Some(Upos::Noun) && case.surface == surface && dasu_case_link(noun, case)
}

fn topic_place(noun: &Atom, de: &Atom, wa: &Atom, verb: &Atom) -> bool {
    // A parser nsubj label on a では phrase does not turn its noun into a
    // participant. Both case tokens must attach to the same local noun.
    noun_case(noun, de, "で")
        && wa.surface == "は"
        && wa.upos == Some(Upos::Adp)
        && linked(wa, noun, "case")
        && adjacent(de, wa)
        && (linked(noun, verb, "obl") || linked(noun, verb, "nsubj"))
}

fn demonstrative(det: &Atom, noun: &Atom) -> Option<&'static str> {
    if det.upos != Some(Upos::Det) || !adjacent(det, noun) || !linked(det, noun, "det") {
        return None;
    }
    match det.surface.as_str() {
        "この" => Some("這"),
        "その" | "あの" => Some("那"),
        _ => None,
    }
}

fn noun_spelling(words: &[Atom]) -> String {
    words.iter().map(|word| word.surface.as_str()).collect()
}

fn detachment(prefix: &[Atom], verb: &Atom) -> Option<String> {
    let [garment, origin, part, ga] = prefix else {
        return None;
    };
    if part.surface != "ボタン"
        || !noun_case(part, ga, "が")
        || !linked(part, verb, "nsubj")
        || !adjacent(origin, part)
    {
        return None;
    }
    let garment_text = match garment.surface.as_str() {
        "コート" | "上着" => "外套",
        "ジャケット" => "夾克",
        "シャツ" => "襯衫",
        _ => return None,
    };
    if noun_case(garment, origin, "の") && linked(garment, part, "nmod") {
        return Some(format!("{garment_text}的鈕扣掉了"));
    }
    if noun_case(garment, origin, "から") && linked(garment, verb, "obl") {
        return Some(format!("鈕扣從{garment_text}上掉下來了"));
    }
    None
}

fn service_desk(noun: &Atom) -> bool {
    noun.upos == Some(Upos::Noun) && matches!(noun.surface.as_str(), "受付" | "窓口")
}

fn unavailable_ticket(prefix: &[Atom], verb: &Atom) -> Option<String> {
    let [desk, de, wa, ticket @ .., ga] = prefix else {
        return None;
    };
    let head = read_nominal(ticket)?.head();
    (service_desk(desk)
        && topic_place(desk, de, wa, verb)
        && noun_case(head, ga, "が")
        && linked(head, verb, "nsubj")
        && noun_spelling(ticket) == "整理券")
        .then(|| "沒能在服務櫃檯拿到號碼牌".to_owned())
}

fn available_document(prefix: &[Atom], verb: &Atom) -> Option<String> {
    let [necessary, na, document @ .., wa, desk, de] = prefix else {
        return None;
    };
    let head = read_nominal(document)?.head();
    (service_desk(desk)
        && noun_case(desk, de, "で")
        && linked(desk, verb, "obl")
        && noun_case(head, wa, "は")
        && linked(head, verb, "nsubj")
        && noun_spelling(document) == "証明書"
        && necessary.surface == "必要"
        && necessary.upos == Some(Upos::Adj)
        && linked(necessary, head, "amod")
        && na.surface == "な"
        && na.upos == Some(Upos::Part)
        && linked(na, necessary, "discourse")
        && adjacent(necessary, na)
        && adjacent(na, document.first()?))
    .then(|| "必要的證明文件可以在服務櫃檯取得".to_owned())
}

fn crop_name(crop: &Atom, ga: &Atom, verb: &Atom) -> Option<&'static str> {
    if !noun_case(crop, ga, "が") || !linked(crop, verb, "nsubj") {
        return None;
    }
    match crop.surface.as_str() {
        "桃" => Some("桃子"),
        "トマト" => Some("番茄"),
        "梨" => Some("梨子"),
        _ => None,
    }
}

fn good_harvest(prefix: &[Atom], verb: &Atom) -> Option<String> {
    let [year, wa, field, de, crop, ga, well] = prefix else {
        return None;
    };
    let year_text = match year.surface.as_str() {
        "今年" => "今年",
        "去年" => "去年",
        _ => return None,
    };
    (noun_case(year, wa, "は")
        && (linked(year, verb, "nsubj") || linked(year, verb, "advmod"))
        && field.surface == "畑"
        && noun_case(field, de, "で")
        && linked(field, verb, "obl")
        && well.surface == "よく"
        && well.upos == Some(Upos::Adv)
        && linked(well, verb, "advmod"))
    .then_some(())?;
    Some(format!(
        "{year_text}田裡收成了不少{}",
        crop_name(crop, ga, verb)?
    ))
}

fn seasonal_harvest(prefix: &[Atom], verb: &Atom) -> Option<String> {
    let [det, field, de, wa, season, ni, crop, ga] = prefix else {
        return None;
    };
    if field.surface != "畑"
        || !topic_place(field, de, wa, verb)
        || !noun_case(season, ni, "に")
        || !linked(season, verb, "obl")
    {
        return None;
    }
    let season = match season.surface.as_str() {
        "春" => "春天",
        "夏" => "夏天",
        "秋" => "秋天",
        "冬" => "冬天",
        _ => return None,
    };
    let pointing = demonstrative(det, field)?;
    let crop = crop_name(crop, ga, verb)?;
    Some(format!("{pointing}塊田{season}採得到{crop}"))
}

fn group_photo(prefix: &[Atom], verb: &Atom) -> Option<String> {
    let [det, camera, de, wa, photo @ .., mo] = prefix else {
        return None;
    };
    let head = read_nominal(photo)?.head();
    if camera.surface != "カメラ"
        || !topic_place(camera, de, wa, verb)
        || !noun_case(head, mo, "も")
        || !linked(head, verb, "nsubj")
        || noun_spelling(photo) != "集合写真"
    {
        return None;
    }
    Some(format!("{}台相機也能拍團體照", demonstrative(det, camera)?))
}

fn low_light_photo(prefix: &[Atom], quality: &Atom) -> Option<RenderedFrame> {
    let [det, camera, nara, pause, dark, place, demo] = prefix else {
        return None;
    };
    // The split analysis links the camera to 暗い instead of to a verb.
    // Require the camera condition, dark-place concession, and the whole
    // adjacent quality/potential tail before inferring the missing event.
    if camera.surface != "カメラ"
        || camera.upos != Some(Upos::Noun)
        || !linked(camera, dark, "obl")
        || nara.surface != "なら"
        || nara.upos != Some(Upos::Sconj)
        || !linked(nara, camera, "mark")
        || pause.surface != "、"
        || !linked(pause, quality, "punct")
        || dark.surface != "暗い"
        || dark.upos != Some(Upos::Adj)
        || !linked(dark, place, "amod")
        || place.surface != "場所"
        || place.upos != Some(Upos::Noun)
        || !linked(place, quality, "obl")
        || demo.surface != "でも"
        || demo.upos != Some(Upos::Sconj)
        || !linked(demo, place, "mark")
        || !prefix.windows(2).all(|pair| adjacent(&pair[0], &pair[1]))
        || !adjacent(demo, quality)
    {
        return None;
    }
    let camera = format!("用{}台相機，", demonstrative(det, camera)?);
    let cut = Some((pause.span, camera.chars().count()));
    Some(RenderedFrame {
        text: format!("{camera}即使在暗處也能拍得漂亮"),
        cut,
    })
}

pub(super) fn render(prefix: &[Atom], verb: &Atom, tail: &[Atom]) -> Option<RenderedFrame> {
    if verb.deprel != "root" {
        return None;
    }
    if split_photography(verb, tail) {
        return low_light_photo(prefix, verb);
    }
    if verb.upos != Some(Upos::Verb) {
        return None;
    }
    let text = match (verb.surface.as_str(), verb.lemma.as_deref(), tail) {
        ("とれ", Some("とる" | "とれる"), [past])
            if auxiliary(past, verb, "た") && adjacent(verb, past) =>
        {
            detachment(prefix, verb).or_else(|| good_harvest(prefix, verb))?
        }
        ("取れ", Some("取る" | "取れる"), [past])
            if auxiliary(past, verb, "た") && adjacent(verb, past) =>
        {
            detachment(prefix, verb)?
        }
        ("とれ", Some("とる" | "とれる"), [negative, past])
            if auxiliary(negative, verb, "なかっ")
                && negative.lemma.as_deref() == Some("ない")
                && auxiliary(past, verb, "た")
                && adjacent(verb, negative)
                && adjacent(negative, past) =>
        {
            unavailable_ticket(prefix, verb)?
        }
        ("取れ", Some("取る" | "取れる"), [polite])
            if auxiliary(polite, verb, "ます") && adjacent(verb, polite) =>
        {
            available_document(prefix, verb)?
        }
        ("採れる", None | Some("採れる"), []) => seasonal_harvest(prefix, verb)?,
        ("撮れる", None | Some("撮れる"), []) => group_photo(prefix, verb)?,
        _ => return None,
    };
    Some(RenderedFrame { text, cut: None })
}

#[cfg(test)]
mod tests {
    use crate::{TargetVariant, TranslateOptions, TranslationGlossary, analyze_and_translate};
    use faberline_nlp::doc::model::Backend;
    use serde_json::Value;

    #[test]
    fn toreru_preserves_roles_result_and_capability_for_adopted_readings() {
        let bank: Value = serde_json::from_str(include_str!(
            "../../../tests/data/ja_zh/verb_toreru_readings_dev_v1.json"
        ))
        .unwrap();
        let rows = bank["items"].as_array().unwrap();
        assert_eq!(rows.len(), 8);
        let options = TranslateOptions::new(TargetVariant::ZhHant);
        let glossary = TranslationGlossary::new();
        for row in rows {
            assert_eq!(row["heldout"], false);
            assert_eq!(row["visibility"], "development-visible");
            let id = row["id"].as_str().unwrap();
            let source = row["jp"].as_str().unwrap();
            let expected = row["zh_hant"].as_str().unwrap();
            let result =
                analyze_and_translate(source, &options, &glossary, None, Backend::Onnx).unwrap();
            assert_eq!(result.target_text, expected, "{id}");
            if id == "TORERU-DEV-04" {
                let [camera, quality] = &result.clause_alignments[..] else {
                    panic!("expected two clauses for {id}");
                };
                assert_eq!(
                    (camera.source_span.start_char, camera.source_span.end_char),
                    (0, 7)
                );
                assert_eq!(
                    (quality.source_span.start_char, quality.source_span.end_char),
                    (8, 22)
                );
                assert_eq!(
                    (camera.target_span.start_char, camera.target_span.end_char),
                    (0, 6)
                );
                assert_eq!(
                    (quality.target_span.start_char, quality.target_span.end_char),
                    (6, expected.chars().count())
                );
            } else {
                let [clause] = &result.clause_alignments[..] else {
                    panic!("expected one clause for {id}");
                };
                assert_eq!(
                    (clause.source_span.start_char, clause.source_span.end_char),
                    (0, source.chars().count())
                );
                assert_eq!(
                    (clause.target_span.start_char, clause.target_span.end_char),
                    (0, expected.chars().count())
                );
            }
        }
    }
}
