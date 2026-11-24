//! Preserve contextual event settings and unresolved noun topics.

use super::{
    Atom, Document, TargetVariant, Upos, attached_to, boundary, finite_affirmative, like,
    localize_close_quote, localize_open_quote, noun_phrase, render, render_temporal_made_atom,
    supported,
};

pub(super) fn weather_food(
    atoms: &[Atom],
    doc: &Document,
    sentence: usize,
) -> Option<(String, usize, usize)> {
    let end = atoms.iter().position(|atom| atom.deprel == "root")?;
    let eat = &atoms[end];
    let aspect = finite_affirmative(eat, &atoms[end + 1..])?;
    let [weather, day, time_case, remainder @ ..] = &atoms[..end] else {
        return None;
    };
    if atoms.iter().any(boundary)
        || !like(eat, &["食べる"])
        || day.surface != "日"
        || day.doc_tok_idx.is_none()
        || weather.upos != Some(Upos::Adj)
        || weather.head != day.doc_tok_idx
        || weather.deprel != "amod"
        || !attached_to(day, time_case, eat, "obl", "に")
    {
        return None;
    }
    let day_text = match weather.lemma.as_deref().unwrap_or(&weather.surface) {
        "暑い" => "炎熱的日子裡",
        "寒い" => "寒冷的日子裡",
        _ => return None,
    };
    let (source_cut, remainder) = match remainder {
        [pause, rest @ ..] if matches!(pause.surface.as_str(), "、" | "，" | ",") => {
            (pause.span.end_char, rest)
        }
        _ => (time_case.span.end_char, remainder),
    };
    let [setting @ .., place, location_case, food, object_case] = remainder else {
        return None;
    };
    if !attached_to(place, location_case, eat, "obl", "で")
        || !attached_to(food, object_case, eat, "obj", "を")
        || render(&food.surface, food.lemma.as_deref()) != Some("霜淇淋")
    {
        return None;
    }
    let location = if setting.is_empty() && place.surface == "駅前" {
        "車站前".to_owned()
    } else {
        noun_phrase(&remainder[..setting.len() + 1], doc, sentence)?
    };
    let prefix = format!("{day_text}，");
    let target_cut = prefix.chars().count();
    Some((
        format!("{prefix}在{location}吃{aspect}霜淇淋"),
        source_cut,
        target_cut,
    ))
}

fn genitive_of(modifier: &Atom, particle: &Atom, head: &Atom) -> bool {
    attached_to(modifier, particle, head, "nmod", "の")
}

pub(super) fn purchases(atoms: &[Atom]) -> Option<String> {
    if atoms.iter().any(boundary) {
        return None;
    }
    let end = atoms.iter().position(|atom| atom.deprel == "root")?;
    let buy = &atoms[end];
    if !like(buy, &["買う"]) {
        return None;
    }
    let aspect = finite_affirmative(buy, &atoms[end + 1..])?;
    match &atoms[..end] {
        [
            station,
            first,
            window,
            second,
            counter,
            position,
            ticket,
            object,
        ] if station.surface == "駅"
            && window.surface == "窓口"
            && genitive_of(station, first, window)
            && genitive_of(window, second, counter)
            && render(&counter.surface, counter.lemma.as_deref()) == Some("櫃檯")
            && attached_to(counter, position, buy, "obl", "で")
            && like(ticket, &["切符"])
            && attached_to(ticket, object, buy, "obj", "を") =>
        {
            Some(format!("在車站的售票櫃檯買{aspect}車票"))
        }
        [performance, genitive, ticket, object]
            if performance.surface == "ライブ"
                && genitive_of(performance, genitive, ticket)
                && like(ticket, &["チケット"])
                && attached_to(ticket, object, buy, "obj", "を") =>
        {
            Some(format!("買{aspect}現場演出的門票"))
        }
        _ => None,
    }
}

pub(super) fn quoted_search(atoms: &[Atom], doc: &Document, sentence: usize) -> Option<String> {
    let [
        open,
        label,
        close,
        topic,
        store,
        location,
        search,
        tail @ ..,
    ] = atoms
    else {
        return None;
    };
    let caps = match (open.surface.as_str(), close.surface.as_str()) {
        ("「", "」") => ('「', '」'),
        ("『", "』") => ('『', '』'),
        _ => return None,
    };
    if atoms.iter().any(|atom| atom.is_locked)
        || !like(search, &["探す"])
        || label.upos != Some(Upos::Noun)
        || label.doc_tok_idx.is_none()
        || label.head != search.doc_tok_idx
        || label.deprel != "obl"
        || topic.surface != "は"
        || topic.head != search.doc_tok_idx
        || !atoms[..4]
            .windows(2)
            .all(|pair| pair[0].span.end_char == pair[1].span.start_char)
        || !attached_to(store, location, search, "obl", "で")
    {
        return None;
    }
    let aspect = finite_affirmative(search, tail)?;
    let location = noun_phrase(std::slice::from_ref(store), doc, sentence)?;
    let open = localize_open_quote(caps.0, TargetVariant::ZhHant);
    let close = localize_close_quote(caps.1, TargetVariant::ZhHant);
    Some(format!(
        "在{location}裡找{aspect}{open}{}{close}",
        label.surface
    ))
}

fn simple_copula(copula: &Atom, predicate: &Atom) -> bool {
    predicate.deprel == "root"
        && predicate.doc_tok_idx.is_some()
        && copula.head == predicate.doc_tok_idx
        && copula.upos == Some(Upos::Aux)
        && copula.deprel == "cop"
        && matches!(copula.surface.as_str(), "です" | "だ")
}

pub(super) fn unresolved_quality(atoms: &[Atom]) -> Option<String> {
    let [noun, topic, adjective, copula] = atoms else {
        return None;
    };
    if atoms.iter().any(boundary)
        || !supported(&noun.surface)
        || render(&noun.surface, noun.lemma.as_deref()) != Some(noun.surface.as_str())
        || adjective.upos != Some(Upos::Adj)
        || !attached_to(noun, topic, adjective, "nsubj", "は")
        || !simple_copula(copula, adjective)
    {
        return None;
    }
    let property = match adjective.lemma.as_deref().unwrap_or(&adjective.surface) {
        "古い" => "很舊",
        "新しい" => "很新",
        _ => return None,
    };
    Some(format!("{}{property}", noun.surface))
}

pub(super) fn scheduled_game(atoms: &[Atom]) -> Option<String> {
    let [sport, genitive, event, topic, day, copula] = atoms else {
        return None;
    };
    if atoms.iter().any(boundary)
        || !genitive_of(sport, genitive, event)
        || !like(event, &["試合"])
        || !attached_to(event, topic, day, "nsubj", "は")
        || !simple_copula(copula, day)
    {
        return None;
    }
    let sport = match sport.surface.as_str() {
        "野球" => "棒球",
        "サッカー" => "足球",
        _ => return None,
    };
    let schedule = render_temporal_made_atom(day)?;
    Some(format!("{sport}比賽在{schedule}"))
}

pub(super) fn charge_battery(atoms: &[Atom]) -> Option<String> {
    if atoms.iter().any(boundary) {
        return None;
    }
    let end = atoms.iter().position(|atom| atom.deprel == "root")?;
    let root = &atoms[end];
    let aspect = finite_affirmative(root, &atoms[end + 1..])?;
    let [battery, case, stem @ ..] = &atoms[..end] else {
        return None;
    };
    if render(&battery.surface, battery.lemma.as_deref()) != Some("電池") {
        return None;
    }
    let governed = match stem {
        [] if like(root, &["充電する"]) => root,
        [charge]
            if like(charge, &["充電"])
                && like(root, &["する"])
                && charge.head == root.doc_tok_idx
                && charge.deprel == "advcl"
                && charge.span.end_char == root.span.start_char =>
        {
            charge
        }
        _ => return None,
    };
    attached_to(battery, case, governed, "obj", "を").then(|| format!("替電池充{aspect}電"))
}
