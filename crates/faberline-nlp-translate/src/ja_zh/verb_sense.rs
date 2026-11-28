//! Select a verb sense from the grammatical roles of one Japanese event.

use super::{
    Atom, ClauseAlignment, Document, Span, Upos, core_non_punct, dasu_case_link,
    is_demonstrative_person_noun, is_kana, is_te_form_link, localize_punct,
    render_temporal_made_atom, return_endpoint, translate_single_ja_term,
};
use faberline_nlp::lexicon::{SemanticClass, lookup_ja};

mod fall;
mod lament;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Aspect {
    Plain,
    Past,
    Continuing,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TimeKind {
    Repeated,
    Since,
    Until,
    At,
}

#[derive(Clone, Copy)]
struct Nominal<'a>(&'a [Atom]);

impl<'a> Nominal<'a> {
    fn head(self) -> &'a Atom {
        self.0.last().expect("a nominal has a head")
    }

    fn key(self) -> String {
        if self.0.windows(2).all(|pair| {
            pair[0].deprel == "compound" && pair[0].span.end_char == pair[1].span.start_char
        }) || split_friend(self.0)
        {
            self.0.iter().map(|atom| atom.surface.as_str()).collect()
        } else {
            self.head().surface.clone()
        }
    }

    fn person(self) -> bool {
        matches!(
            self.key().as_str(),
            "人" | "友だち" | "友達" | "客" | "客人" | "訪問者"
        ) || self.head().upos == Some(Upos::Pron)
            || is_demonstrative_person_noun(self.head())
            || lookup_ja(&self.key())
                .is_some_and(|entry| entry.sem_class == Some(SemanticClass::Person))
    }

    fn place(self) -> bool {
        return_endpoint::is_place(self.head())
            || matches!(
                self.key().as_str(),
                "玄関"
                    | "教室"
                    | "建物"
                    | "会場"
                    | "室内"
                    | "店内"
                    | "倉庫"
                    | "事務所"
                    | "大学"
                    | "塾"
                    | "診療所"
                    | "クリニック"
            )
    }

    fn text(self) -> String {
        let key = self.key();
        if let Some(text) = nominal_label(&key)
            && (self.0.len() == 1 || key != self.head().surface)
        {
            return text.to_owned();
        }
        let word = nominal_label(&self.head().surface).map_or_else(
            || {
                translate_single_ja_term(
                    &self.head().surface,
                    self.head().lemma.as_deref(),
                    self.head().upos,
                )
            },
            str::to_owned,
        );
        match self.0 {
            [.., no, _] if no.surface == "の" => {
                let owner = Nominal(&self.0[..self.0.len() - 2]).text();
                format!("{owner}的{word}")
            }
            [modifier, _] if modifier.upos == Some(Upos::Det) => {
                let prefix = match modifier.surface.as_str() {
                    "この" => "這",
                    "その" | "あの" => "那",
                    _ => &modifier.surface,
                };
                format!("{prefix}{word}")
            }
            [modifier, _] if modifier.upos == Some(Upos::Adj) => {
                let description = translate_single_ja_term(
                    &modifier.surface,
                    modifier.lemma.as_deref(),
                    modifier.upos,
                );
                let description = description.trim_end_matches('的');
                if word == "風" {
                    format!("{description}{word}")
                } else {
                    format!("{description}的{word}")
                }
            }
            [..] if key != self.head().surface && !split_friend(self.0) => {
                translate_single_ja_term(&key, None, self.head().upos)
            }
            _ => word,
        }
    }
}

fn nominal_label(word: &str) -> Option<&'static str> {
    Some(match word {
        "友だち" | "友達" => "朋友",
        "警備員" => "警衛",
        "客" | "客人" => "客人",
        "毛布" => "毛毯",
        "針穴" => "針孔",
        "糸" | "紐" => "線",
        "ページ" | "頁" => "頁",
        "こと" | "事" => "件事",
        "場" => "現場",
        _ => return None,
    })
}

fn adjacent(a: &Atom, b: &Atom) -> bool {
    a.span.end_char == b.span.start_char
}

fn linked(child: &Atom, parent: &Atom, relation: &str) -> bool {
    parent.doc_tok_idx.is_some() && child.head == parent.doc_tok_idx && child.deprel == relation
}

fn split_friend(atoms: &[Atom]) -> bool {
    matches!(atoms, [first, copula, tail]
        if first.surface == "友" && copula.surface == "だ" && tail.surface == "ち"
            && adjacent(first, copula) && adjacent(copula, tail)
            && linked(first, tail, "acl") && linked(copula, first, "cop"))
}

fn read_nominal(atoms: &[Atom]) -> Option<Nominal<'_>> {
    let head = atoms.last()?;
    if !matches!(head.upos, Some(Upos::Noun | Upos::Pron | Upos::Propn))
        || head.doc_tok_idx.is_none()
    {
        return None;
    }
    let valid = match atoms {
        [_] => true,
        words if split_friend(words) => true,
        [modifier, _] if linked(modifier, head, "amod") || linked(modifier, head, "det") => {
            matches!(modifier.upos, Some(Upos::Adj))
                || matches!(modifier.surface.as_str(), "この" | "その" | "あの")
        }
        [owner @ .., no, _] if no.surface == "の" => read_nominal(owner).is_some_and(|owner| {
            dasu_case_link(owner.head(), no)
                && linked(owner.head(), head, "nmod")
                && adjacent(no, head)
        }),
        words => words.windows(2).enumerate().all(|(index, pair)| {
            matches!(pair[0].upos, Some(Upos::Noun | Upos::Propn))
                && adjacent(&pair[0], &pair[1])
                && pair[0].deprel == "compound"
                && words[index + 1..]
                    .iter()
                    .any(|candidate| linked(&pair[0], candidate, "compound"))
        }),
    };
    valid.then_some(Nominal(atoms))
}

#[derive(Default)]
struct Frame<'a> {
    subject: Option<Nominal<'a>>,
    object: Option<Nominal<'a>>,
    goal: Option<(Nominal<'a>, &'a str)>,
    direction: Option<Nominal<'a>>,
    source: Option<Nominal<'a>>,
    setting: Option<Nominal<'a>>,
    topic: Option<Nominal<'a>>,
    time: Option<(TimeKind, String, Span)>,
    adverb: Option<&'static str>,
    pause: Option<Span>,
}

fn record<T>(slot: &mut Option<T>, item: T) -> Option<()> {
    slot.replace(item).is_none().then_some(())
}

fn time_word(word: &Atom) -> Option<(TimeKind, String)> {
    let recurring = match word.surface.as_str() {
        "毎朝" => "每天早上",
        "毎晩" | "毎夜" => "每天晚上",
        "毎日" => "每天",
        "毎週" => "每週",
        "毎月" => "每個月",
        "毎年" => "每年",
        "平日" => "平日",
        "いつも" => "總是",
        _ => {
            return render_temporal_made_atom(word).map(|value| (TimeKind::At, value));
        }
    };
    Some((TimeKind::Repeated, recurring.to_owned()))
}

fn aiming_at(orientation: &Atom, verb: &Atom) -> bool {
    verb.lemma.as_deref() == Some("吹く")
        && orientation.upos == Some(Upos::Verb)
        && orientation.lemma.as_deref() == Some("向かう")
        && linked(orientation, verb, "advcl")
}

fn actor_link(noun: &Atom, marker: &Atom, verb: &Atom, prefix: &[Atom]) -> bool {
    linked(noun, verb, "nsubj")
        || prefix
            .iter()
            .any(|orientation| aiming_at(orientation, verb) && linked(noun, orientation, "nsubj"))
        || marker.surface == "は"
            && prefix.windows(3).any(|phrase| {
                let [adj, place, case] = phrase else {
                    return false;
                };
                marker.span.end_char <= adj.span.start_char
                    && adj.upos == Some(Upos::Adj)
                    && linked(noun, adj, "nsubj")
                    && linked(adj, place, "amod")
                    && linked(place, verb, "obl")
                    && return_endpoint::is_place(place)
                    && matches!(case.surface.as_str(), "へ" | "に")
                    && dasu_case_link(place, case)
            })
        || noun.deprel == "nsubj"
            && prefix.windows(3).any(|friend| {
                split_friend(friend)
                    && noun.head == friend[0].doc_tok_idx
                    && (friend[2].head == verb.doc_tok_idx
                        || prefix.iter().any(|argument| {
                            linked(&friend[2], argument, "nmod")
                                && argument.head == verb.doc_tok_idx
                        }))
            })
}

fn is_wind(verb: &Atom, prefix: &[Atom]) -> bool {
    verb.lemma.as_deref() == Some("吹く")
        && prefix
            .iter()
            .any(|noun| noun.surface == "風" && linked(noun, verb, "nsubj"))
}

fn read_frame<'a>(prefix: &'a [Atom], verb: &Atom) -> Option<Frame<'a>> {
    let mut frame = Frame::default();
    let mut rest = prefix;
    while let Some(first) = rest.first() {
        if matches!(first.surface.as_str(), "、" | "，") && rest.len() < prefix.len() {
            record(&mut frame.pause, first.span)?;
            rest = &rest[1..];
            continue;
        }
        if linked(first, verb, "advmod") {
            if let Some((kind, time)) = time_word(first) {
                record(&mut frame.time, (kind, time, first.span))?;
                rest = &rest[1..];
                continue;
            }
            if matches!(first.surface.as_str(), "すぐ" | "直ぐ") {
                record(&mut frame.adverb, "立刻")?;
                rest = &rest[1..];
                continue;
            }
        }

        let end = rest.iter().position(|atom| {
            atom.upos == Some(Upos::Adp) && atom.deprel == "case" && atom.surface != "の"
        })?;
        let noun = read_nominal(&rest[..end])?;
        let head = noun.head();
        let marker = &rest[end];
        if !dasu_case_link(head, marker) {
            return None;
        }
        rest = &rest[end + 1..];
        let governed =
            head.head == verb.doc_tok_idx && matches!(head.deprel.as_str(), "obl" | "iobj");
        match marker.surface.as_str() {
            "は" | "が" if actor_link(head, marker, verb, prefix) => {
                record(&mut frame.subject, noun)?
            }
            "を" if linked(head, verb, "obj") => record(&mut frame.object, noun)?,
            "に" | "へ" if governed => {
                let time = (noun.0.len() == 1 && marker.surface == "に")
                    .then(|| time_word(head))
                    .flatten();
                if let Some((kind, time)) = time {
                    record(&mut frame.time, (kind, time, marker.span))?;
                } else {
                    record(&mut frame.goal, (noun, &marker.surface))?;
                }
            }
            "に" | "へ" if verb.lemma.as_deref() == Some("吹く") => {
                let [direction, te, tail @ ..] = rest else {
                    return None;
                };
                if !aiming_at(direction, verb)
                    || !linked(head, direction, "obl")
                    || !linked(te, direction, "mark")
                    || te.upos != Some(Upos::Sconj)
                    || te.surface != "て"
                    || !adjacent(direction, te)
                {
                    return None;
                }
                record(&mut frame.direction, noun)?;
                rest = tail;
            }
            "から"
                if governed
                    || is_wind(verb, prefix)
                        && prefix.iter().any(|adj| {
                            adj.upos == Some(Upos::Adj)
                                && linked(head, adj, "obl")
                                && prefix
                                    .iter()
                                    .any(|wind| wind.surface == "風" && linked(adj, wind, "amod"))
                        }) =>
            {
                if let Some((_, time)) = time_word(head).filter(|_| noun.0.len() == 1) {
                    record(&mut frame.time, (TimeKind::Since, time, marker.span))?;
                } else {
                    record(&mut frame.source, noun)?;
                }
            }
            "まで" if governed => {
                let label = if noun.key() == "最後" {
                    "最後".to_owned()
                } else if noun.0.len() == 1 {
                    time_word(head)?.1
                } else {
                    return None;
                };
                record(&mut frame.time, (TimeKind::Until, label, marker.span))?;
            }
            "で" if governed => record(&mut frame.setting, noun)?,
            "について" if governed => record(&mut frame.topic, noun)?,
            _ => return None,
        }
    }
    Some(frame)
}

enum Sense {
    Retort,
    RushIn,
    CramIn,
    Wind,
    Breath,
    Instrument,
    Frequent,
    TurnPage,
    TurnOver,
    LiftCover,
    LiftEdge,
    PassGate,
    ThreadThrough,
    LetIn,
    Maintain,
    Mail,
    Deliver,
    Dispatch,
}

fn room(noun: Nominal<'_>) -> bool {
    matches!(
        noun.key().as_str(),
        "部屋" | "客間" | "応接室" | "教室" | "室内" | "家" | "店内"
    )
}

fn container(noun: Nominal<'_>) -> bool {
    matches!(
        noun.key().as_str(),
        "かばん"
            | "鞄"
            | "バッグ"
            | "袋"
            | "箱"
            | "引き出し"
            | "ポケット"
            | "容器"
            | "ケース"
            | "封筒"
    )
}

fn first_person(noun: Nominal<'_>) -> bool {
    matches!(
        noun.key().as_str(),
        "私" | "僕" | "俺" | "私たち" | "ぼく" | "わたし"
    )
}

fn speaker_location(noun: Nominal<'_>) -> bool {
    matches!(noun.0, [speaker, no, place] if no.surface == "の"
        && matches!(place.surface.as_str(), "ところ" | "所" | "もと" | "元")
        && first_person(Nominal(std::slice::from_ref(speaker))))
}

fn choose(lemma: &str, frame: &Frame<'_>) -> Option<Sense> {
    let goal = frame.goal.map(|(noun, _)| noun);
    let object_key = frame.object.map(Nominal::key);
    let object = object_key.as_deref();
    let sense = match (lemma, object, goal) {
        ("突っ込む", None, Some(goal)) if goal.place() => Sense::RushIn,
        ("突っ込む", Some(_), Some(goal)) if container(goal) => Sense::CramIn,
        ("突っ込む", None, Some(goal))
            if matches!(
                goal.key().as_str(),
                "冗談" | "発言" | "話" | "ボケ" | "コメント"
            ) && frame.goal.is_some_and(|(_, marker)| marker == "に") =>
        {
            Sense::Retort
        }
        ("吹く", None, None) if frame.subject.is_some_and(|noun| noun.key() == "風") => {
            Sense::Wind
        }
        ("吹く", Some("息"), None) if frame.object.is_some_and(|noun| noun.0.len() == 1) => {
            Sense::Breath
        }
        (
            "吹く",
            Some(
                "フルート" | "笛" | "トランペット" | "クラリネット" | "サックス" | "ハーモニカ"
                | "尺八" | "口笛" | "ラッパ",
            ),
            None,
        ) => Sense::Instrument,
        ("通う", None, Some(goal)) if goal.place() => Sense::Frequent,
        ("めくる", Some("ページ" | "頁" | "紙"), None) => Sense::TurnPage,
        ("めくる", Some("カード" | "トランプ" | "札"), None) => Sense::TurnOver,
        ("めくる", Some("毛布" | "布団" | "シーツ" | "カバー" | "ひざ掛け" | "布"), None) => {
            Sense::LiftCover
        }
        ("めくる", Some("端" | "縁"), None)
            if frame.object.is_some_and(|noun| {
                matches!(noun.0, [owner, no, _] if no.surface == "の"
                && matches!(owner.surface.as_str(), "シール" | "ステッカー" | "紙" | "ラベル"))
            }) =>
        {
            Sense::LiftEdge
        }
        ("通す", Some(_), None)
            if frame.object.is_some_and(Nominal::person)
                && frame.source.is_some_and(|noun| {
                    matches!(noun.key().as_str(), "門" | "入口" | "ゲート" | "改札")
                }) =>
        {
            Sense::PassGate
        }
        (
            "通す",
            Some("糸" | "紐" | "ひも" | "ロープ" | "針金" | "線" | "ケーブル"),
            Some(goal),
        ) if matches!(
            goal.key().as_str(),
            "針穴" | "穴" | "隙間" | "輪" | "管" | "筒"
        ) =>
        {
            Sense::ThreadThrough
        }
        ("通す", Some(_), Some(goal))
            if frame.object.is_some_and(Nominal::person) && room(goal) =>
        {
            Sense::LetIn
        }
        ("通す", Some("無言" | "沈黙"), None)
            if frame.object.is_some_and(|noun| noun.0.len() == 1) =>
        {
            Sense::Maintain
        }
        ("寄こす", Some(_), Some(goal))
            if frame.object.is_some_and(Nominal::person) && speaker_location(goal) =>
        {
            Sense::Dispatch
        }
        ("寄こす", Some("手紙" | "はがき" | "葉書" | "小包" | "荷物" | "書類"), Some(goal))
            if first_person(goal)
                && frame
                    .setting
                    .is_some_and(|noun| matches!(noun.key().as_str(), "郵便" | "郵送")) =>
        {
            Sense::Mail
        }
        (
            "寄こす",
            Some("鍵" | "書類" | "荷物" | "手紙" | "品物" | "資料" | "切符" | "チケット" | "財布"),
            Some(goal),
        ) if first_person(goal) => Sense::Deliver,
        _ => return None,
    };
    Some(sense)
}

struct RenderedFrame {
    text: String,
    cut: Option<(Span, usize)>,
}

fn render_frame(lemma: &str, frame: Frame<'_>, aspect: Aspect) -> Option<RenderedFrame> {
    let sense = choose(lemma, &frame)?;
    let subject = frame.subject.map(Nominal::text).unwrap_or_default();
    let object = || frame.object.map(Nominal::text);
    let goal = || frame.goal.map(|(noun, _)| noun.text());
    let past = if aspect == Aspect::Past { "了" } else { "" };
    let mut timing = String::new();
    let is_repeated = matches!(frame.time, Some((TimeKind::Repeated, _, _)));
    if let Some((kind, time, _)) = &frame.time {
        timing = match kind {
            TimeKind::Repeated | TimeKind::At => time.clone(),
            TimeKind::Since if matches!(sense, Sense::Wind) => format!("從{time}開始"),
            TimeKind::Until if matches!(sense, Sense::Maintain) && time == "最後" => {
                "始終".into()
            }
            TimeKind::Until if matches!(sense, Sense::Maintain) => format!("直到{time}都"),
            _ => return None,
        };
    }
    if frame.direction.is_some() && !matches!(sense, Sense::Breath)
        || frame.source.is_some() && !matches!(sense, Sense::PassGate)
        || frame.setting.is_some() && !matches!(sense, Sense::Mail | Sense::Deliver)
        || frame.topic.is_some() && !matches!(sense, Sense::Maintain)
    {
        return None;
    }
    if is_repeated && !timing.ends_with("總是") {
        timing.push('都');
    }
    let adverb = frame.adverb.unwrap_or("");
    let progressing = if aspect == Aspect::Continuing && !is_repeated {
        "正"
    } else {
        ""
    };
    if frame.pause.is_some() && !matches!(sense, Sense::Breath | Sense::Frequent) {
        return None;
    }
    let mut cut = None;
    let action = match sense {
        Sense::Retort => format!("{progressing}吐槽{past}{}", goal()?),
        Sense::RushIn => format!("{progressing}衝進{past}{}", goal()?),
        Sense::CramIn => format!("{progressing}把{}塞進{past}{}裡", object()?, goal()?),
        Sense::Wind => {
            let action = match (aspect, &frame.time) {
                (Aspect::Past, _) => format!("颳了{subject}"),
                (Aspect::Continuing, Some((TimeKind::Since, _, _))) => {
                    format!("就一直颳著{subject}")
                }
                (Aspect::Continuing, _) => format!("{progressing}颳著{subject}"),
                (Aspect::Plain, _) => format!("颳{subject}"),
            };
            return Some(RenderedFrame {
                text: format!("{timing}{adverb}{action}"),
                cut: None,
            });
        }
        Sense::Breath => {
            let toward = frame
                .direction
                .map(|noun| format!("朝著{}", noun.text()))
                .unwrap_or_default();
            if let Some(pause) = frame.pause {
                if frame.time.is_some()
                    || frame.adverb.is_some()
                    || frame.direction?.head().span.end_char > pause.start_char
                    || frame.object?.0.first()?.span.start_char < pause.end_char
                    || frame
                        .subject
                        .is_some_and(|noun| noun.head().span.end_char > pause.start_char)
                {
                    return None;
                }
                cut = Some((pause, subject.chars().count() + toward.chars().count()));
            }
            format!("{toward}{progressing}吹{past}氣")
        }
        Sense::Instrument => format!("{progressing}吹{past}{}", object()?),
        Sense::Frequent => {
            let repeated = if is_repeated { "" } else { "定期" };
            if let Some(pause) = frame.pause {
                if frame.adverb.is_some()
                    || frame
                        .time
                        .as_ref()
                        .is_none_or(|(_, _, span)| span.end_char > pause.start_char)
                    || frame.goal?.0.0.first()?.span.start_char < pause.end_char
                    || frame
                        .subject
                        .is_some_and(|noun| noun.head().span.end_char > pause.start_char)
                {
                    return None;
                }
                cut = Some((pause, subject.chars().count() + timing.chars().count()));
            }
            format!("{repeated}去{goal}{past}", goal = goal()?)
        }
        Sense::TurnPage => {
            let noun = frame.object?;
            let page = match noun.0 {
                [book, no, _] if no.surface == "の" && book.surface == "本" => "書頁".to_owned(),
                [_] if matches!(noun.key().as_str(), "頁" | "ページ") => "頁".to_owned(),
                _ => noun.text(),
            };
            format!("{progressing}翻{past}{page}")
        }
        Sense::TurnOver => format!("{progressing}把{}翻過來{past}", object()?),
        Sense::LiftCover => format!("{progressing}掀開{past}{}", object()?),
        Sense::LiftEdge => {
            let [owner, _, _] = frame.object?.0 else {
                return None;
            };
            let material = match owner.surface.as_str() {
                "シール" | "ステッカー" => "貼紙",
                "ラベル" => "標籤",
                "紙" => "紙",
                _ => return None,
            };
            format!("{progressing}掀起{past}{material}的邊緣")
        }
        Sense::PassGate => format!(
            "{progressing}讓{}從{}通過{past}",
            object()?,
            frame.source?.text()
        ),
        Sense::ThreadThrough => format!("{progressing}把{}穿過{past}{}", object()?, goal()?),
        Sense::LetIn => format!("{progressing}讓{}進{past}{}", object()?, goal()?),
        Sense::Maintain => {
            let topic = frame
                .topic
                .map(|noun| format!("對{}", noun.text()))
                .unwrap_or_default();
            return Some(RenderedFrame {
                text: format!("{subject}{topic}{timing}{adverb}{progressing}保持沉默"),
                cut: None,
            });
        }
        Sense::Mail => format!("{progressing}把{}郵寄給{}{past}", object()?, goal()?),
        Sense::Deliver => {
            let setting = match frame.setting {
                Some(noun)
                    if noun.key() == "場"
                        && matches!(noun.0, [det, _] if det.surface == "その") =>
                {
                    "當場".to_owned()
                }
                Some(noun) if noun.place() => format!("在{}", noun.text()),
                None => String::new(),
                _ => return None,
            };
            format!("{setting}{progressing}把{}交給{}{past}", object()?, goal()?)
        }
        Sense::Dispatch => {
            let [speaker, _, _] = frame.goal?.0.0 else {
                return None;
            };
            let destination = Nominal(std::slice::from_ref(speaker)).text();
            let person = match frame.object?.0 {
                [role, no, person]
                    if role.surface == "係" && no.surface == "の" && person.surface == "人" =>
                {
                    "工作人員".to_owned()
                }
                _ => object()?,
            };
            format!("{progressing}派{past}{person}到{destination}這裡")
        }
    };
    Some(RenderedFrame {
        text: format!("{subject}{timing}{adverb}{action}"),
        cut,
    })
}

fn aspect(verb: &Atom, tail: &[Atom]) -> Option<Aspect> {
    match tail {
        [] if verb.deprel == "root" && verb.lemma.as_deref() == Some(verb.surface.as_str()) => {
            Some(Aspect::Plain)
        }
        [polite]
            if verb.deprel == "root"
                && polite.surface == "ます"
                && adjacent(verb, polite)
                && linked(polite, verb, "aux") =>
        {
            Some(Aspect::Plain)
        }
        [past]
            if verb.deprel == "root"
                && matches!(past.surface.as_str(), "た" | "だ")
                && adjacent(verb, past)
                && linked(past, verb, "aux") =>
        {
            Some(Aspect::Past)
        }
        [polite, past]
            if verb.deprel == "root"
                && polite.surface == "まし"
                && past.surface == "た"
                && adjacent(verb, polite)
                && adjacent(polite, past)
                && linked(polite, verb, "aux")
                && linked(past, verb, "aux") =>
        {
            Some(Aspect::Past)
        }
        [te, state]
            if state.surface == "いる"
                && state.deprel == "root"
                && is_te_form_link(verb, te, state)
                && linked(te, verb, "mark")
                && linked(verb, state, "advcl") =>
        {
            Some(Aspect::Continuing)
        }
        _ => None,
    }
}

pub(super) fn render(
    atoms: &[Atom],
    doc: &Document,
    source: &[char],
    sentence: usize,
    doc_clauses: &[(usize, Span)],
) -> Option<(String, Vec<ClauseAlignment>)> {
    let finite = core_non_punct(atoms);
    let has_sense_verb = |atom: &Atom| {
        atom.upos == Some(Upos::Verb)
            && (fall::is_predicate(atom)
                || lament::is_predicate(atom)
                || matches!(
                    atom.lemma.as_deref(),
                    Some("突っ込む" | "吹く" | "通う" | "めくる" | "通す" | "寄こす")
                ))
    };
    let mut predicates = finite
        .iter()
        .enumerate()
        .filter(|(_, atom)| has_sense_verb(atom));
    let (index, verb) = predicates.next()?;
    let span = doc.segmentation.sentences.get(sentence)?.span;
    if predicates.next().is_some()
        || atoms.first()?.span.start_char != span.start_char
        || atoms.last()?.span.end_char != span.end_char
        || atoms.windows(2).any(|pair| {
            source
                .get(pair[0].span.end_char..pair[1].span.start_char)
                .is_none_or(|gap| !gap.iter().all(|ch| ch.is_whitespace()))
        })
        || atoms.iter().any(|atom| {
            atom.is_locked
                || atom.doc_tok_idx.is_none_or(|id| {
                    doc.tokens.get(id).is_none_or(|token| {
                        token.sentence != sentence
                            || token.span != atom.span
                            || token.upos != atom.upos
                            || token.lemma != atom.lemma
                    }) || doc.syntax.get(id).is_none_or(|link| {
                        link.token != id || link.head != atom.head || link.deprel != atom.deprel
                    }) || source
                        .get(atom.span.start_char..atom.span.end_char)
                        .is_none_or(|raw| !raw.iter().copied().eq(atom.surface.chars()))
                })
        })
    {
        return None;
    }
    let punctuation = match &atoms[finite.len()..] {
        [] => "",
        [punct] if matches!(punct.surface.as_str(), "。" | "！" | "!") => &punct.surface,
        _ => return None,
    };
    let aspect = aspect(verb, &finite[index + 1..])?;
    let lemma = verb.lemma.as_deref()?;
    let RenderedFrame { mut text, cut } = if fall::is_predicate(verb) {
        fall::render(&finite[..index], verb, aspect)?
    } else if lament::is_predicate(verb) {
        lament::render(&finite[..index], verb, aspect)?
    } else {
        let frame = read_frame(&finite[..index], verb)?;
        render_frame(lemma, frame, aspect)?
    };
    if text.chars().any(is_kana) {
        return None;
    }
    text.push_str(&localize_punct(punctuation));
    let total = text.chars().count();
    let alignment =
        |&(clause, source_span): &(usize, Span), start_char, end_char| ClauseAlignment {
            clause,
            sentence,
            source_span,
            target_span: Span {
                start_char,
                end_char,
            },
        };
    let alignments = match (doc_clauses, cut) {
        ([], _) => Vec::new(),
        ([clause], _) => vec![alignment(clause, 0, total)],
        ([left, right], Some((pause, end)))
            if left.1.end_char == pause.start_char
                && pause.end_char == right.1.start_char
                && end > 0
                && end < total =>
        {
            vec![alignment(left, 0, end), alignment(right, end, total)]
        }
        _ => return None,
    };
    Some((text, alignments))
}

#[cfg(test)]
mod tests {
    use super::{super::build_sentence_atoms, *};
    use crate::{TargetVariant, TranslateOptions, TranslationGlossary, analyze_and_translate};
    use faberline_nlp::doc::{analyze_document, model::Backend};
    use serde_json::Value;

    fn public_bank() -> Vec<Value> {
        let bank: Value = serde_json::from_str(include_str!(
            "../../tests/data/ja_zh/verb_sense_sample_dev_v1.json"
        ))
        .unwrap();
        assert_eq!(bank["status"], "development-visible");
        bank["items"].as_array().unwrap().clone()
    }

    #[test]
    fn selects_a_translation_for_each_adopted_frame() {
        let glossary = TranslationGlossary::new();
        let options = TranslateOptions::new(TargetVariant::ZhHant);
        let expected = [
            ("吐槽", "朋友的玩笑"),
            ("衝進", "房間"),
            ("錢包塞進", "包包"),
            ("從下午開始", "一直颳著強風"),
            ("朝著風車", "吹了氣"),
            ("每天晚上都", "吹長笛"),
            ("每天早上都", "去學校"),
            ("每週都", "去醫院"),
            ("平日都", "去公司"),
            ("翻", "書頁"),
            ("卡片", "翻過來"),
            ("掀開", "毛毯"),
            ("掀起", "貼紙的邊緣"),
            ("讓客人", "從門通過"),
            ("線穿過", "針孔"),
            ("讓朋友", "進了房間"),
            ("對那件事", "始終保持沉默"),
            ("把信郵寄給", "我"),
            ("當場", "把鑰匙交給我"),
            ("派了工作人員", "到我這裡"),
        ];
        let bank = public_bank();
        assert_eq!(bank.len(), expected.len());
        for (row, (first, second)) in bank.iter().zip(expected) {
            assert_eq!(row["heldout"], false);
            let source = row["jp"].as_str().unwrap();
            let result = analyze_and_translate(source, &options, &glossary, None, Backend::Onnx)
                .expect("translate the adopted public Japanese control");
            assert!(
                result.target_text.contains(first) && result.target_text.contains(second),
                "{} -> {}",
                row["id"],
                result.target_text,
            );
            if matches!(row["id"].as_str(), Some("VS-N5-05" | "VS-N5-08")) {
                let [left, right] = result.clause_alignments.as_slice() else {
                    panic!("{} must retain both source clause alignments", row["id"])
                };
                assert!(left.target_span.start_char < left.target_span.end_char);
                assert_eq!(left.target_span.end_char, right.target_span.start_char);
                assert!(right.target_span.start_char < right.target_span.end_char);
            }
        }
    }

    #[test]
    fn falls_back_when_a_required_argument_is_locked_or_loses_its_case() {
        let glossary = TranslationGlossary::new();
        let options = TranslateOptions::new(TargetVariant::ZhHant);
        let doc_options = crate::doc_options_for_translate(&options, &glossary);
        let bank = public_bank();
        for (index, argument) in [
            (2, "を"),
            (5, "を"),
            (7, "に"),
            (10, "を"),
            (14, "を"),
            (19, "へ"),
        ] {
            let source = bank[index]["jp"].as_str().unwrap();
            let chars = source.chars().collect::<Vec<_>>();
            let mut doc = analyze_document(source, &doc_options);
            let arg = doc
                .tokens
                .iter()
                .position(|token| {
                    chars[token.span.start_char..token.span.end_char]
                        .iter()
                        .copied()
                        .eq(argument.chars())
                })
                .unwrap();
            let spans = doc
                .segmentation
                .clauses
                .iter()
                .enumerate()
                .map(|(i, clause)| (i, clause.span))
                .collect::<Vec<_>>();
            let atoms = |doc: &Document| {
                build_sentence_atoms(
                    0,
                    &chars,
                    doc,
                    &doc.tokens.iter().collect::<Vec<_>>(),
                    &crate::lock::SpanLockTable::new(),
                )
            };
            let mut locked = atoms(&doc);
            assert!(
                render(&locked, &doc, &chars, 0, &spans).is_some(),
                "{source}"
            );
            locked[arg].is_locked = true;
            assert!(
                render(&locked, &doc, &chars, 0, &spans).is_none(),
                "{source}"
            );

            doc.syntax[arg].head = None;
            assert!(
                render(&atoms(&doc), &doc, &chars, 0, &spans).is_none(),
                "{source}"
            );
        }
    }
}
