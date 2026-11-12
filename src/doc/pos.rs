//! Layer 2 UPOS tagging of `nlp.doc.v1`: closed-class rules and a small hand-written lexicon.
//! The layer never guesses. A token that no rule or word decides is `Missing`, or `Ambiguous`
//! when the word is on the ambiguous list. See `docs/doc-v1.md`.

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

use super::{Lang, OpaqueKind, Upos};

pub(super) enum Tag {
    Decided(Upos),
    Ambiguous,
    Missing,
}

const PUNCT: &str =
    "!\"'(),-./:;?[\\]_{}，。、；：？！…「」『』（）《》〈〉“”‘’—–～·‧【】〔〕［］｛｝";
const SYM: &str = "$%+<=>@^`|~#*°×÷±©®™§¥€£₩¢←→↑↓";
const ZH_NUMERALS: &str = "〇零一二三四五六七八九十百千萬万億亿兩两";

const EN_DECIDED: &[(Upos, &str)] = &[
    (Upos::Det, "a an the every each"),
    (
        Upos::Pron,
        "i me you he him she her it we us they them my your his its our their myself yourself himself herself itself ourselves themselves someone something anyone anything everyone everything nobody nothing everybody somebody anybody",
    ),
    (
        Upos::Adp,
        "of in on at by with from into onto over under about against between through during without within upon toward towards across",
    ),
    (Upos::Cconj, "and or but nor &"),
    (
        Upos::Sconj,
        "because although though unless whereas if whether",
    ),
    (
        Upos::Aux,
        "is am are was were be been being will would shall should can could may might must 'll 're 've 'm",
    ),
    (Upos::Part, "not n't"),
    (
        Upos::Adv,
        "very too also quite rather always never often sometimes already still here there now then again soon however",
    ),
    (Upos::Intj, "oh ah wow oops hey hello hi hmm"),
    (
        Upos::Num,
        "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million billion",
    ),
];

const EN_AMBIGUOUS: &[&str] = &[
    "this that these those some any no all both either neither another",
    "who whom whose which what do does did have has had to as than so",
    "before after until since while when where why how yet 's 'd",
];

const ZH_DECIDED: &[(Upos, &str)] = &[
    (
        Upos::Pron,
        "我 你 妳 他 她 它 牠 我們 你們 妳們 他們 她們 它們 咱們 自己 大家 誰 這裡 那裡 這裏 那裏 我们 你们 他们 她们 它们 咱们 谁 这里 那里 在下 妾身 老夫 老身 貧道 贫道 本座",
    ),
    (Upos::Det, "每 各 某 該 本 此"),
    (Upos::Part, "的 了 嗎 吗 吧 呢 啊 啦 喔 哦 呀"),
    (
        Upos::Adp,
        "把 被 從 从 對於 对于 關於 关于 為了 为了 按照 根據 根据 除了",
    ),
    (Upos::Cconj, "並且 并且 而且 或者 或 但是 然而 可是"),
    (
        Upos::Sconj,
        "因為 因为 如果 假如 雖然 虽然 即使 儘管 尽管 除非 只要 既然 一旦",
    ),
    (
        Upos::Adv,
        "不 很 也 都 就 才 只 已經 已经 曾經 曾经 非常 太 更 最 十分 常常 一直 剛 剛剛 刚刚 馬上 马上 又 再 終於 终于 一齊 一齐",
    ),
    (Upos::Aux, "能 可以 應該 应该 必須 必须 能夠 能够"),
    (Upos::Verb, "有"),
    (Upos::Intj, "哇 唉 喂 嗨 哈哈"),
];

const ZH_AMBIGUOUS: &[&str] = &[
    "這 这 那 這個 这个 那個 那个 什麼 什么 甚麼 哪 在 對 对 跟 和 與 与 用 到 給 给 比 向",
    "所以 因此 於是 于是 是 會 会 要 沒 没 沒有 没有 還 还 幾 几 著 着 過 过 得 地",
];

fn decided() -> &'static HashMap<&'static str, Upos> {
    static MAP: OnceLock<HashMap<&'static str, Upos>> = OnceLock::new();
    MAP.get_or_init(|| {
        EN_DECIDED
            .iter()
            .chain(ZH_DECIDED)
            .flat_map(|(upos, words)| words.split_whitespace().map(move |w| (w, *upos)))
            .collect()
    })
}

fn ambiguous() -> &'static HashSet<&'static str> {
    static SET: OnceLock<HashSet<&'static str>> = OnceLock::new();
    SET.get_or_init(|| {
        EN_AMBIGUOUS
            .iter()
            .chain(ZH_AMBIGUOUS)
            .flat_map(|words| words.split_whitespace())
            .collect()
    })
}

/// Tag one token. `lang` picks the lookup key: lowercase for English, exact for Chinese.
pub(super) fn tag(text: &str, lang: Lang, opaque: Option<OpaqueKind>) -> Tag {
    match opaque {
        Some(OpaqueKind::Url | OpaqueKind::Citation | OpaqueKind::Code) => {
            return Tag::Decided(Upos::X);
        }
        Some(OpaqueKind::Formula | OpaqueKind::Emoji) => return Tag::Decided(Upos::Sym),
        // Block spans never reach a sentence, so they never reach tagging.
        Some(OpaqueKind::Heading | OpaqueKind::ListItem) => return Tag::Missing,
        None => {}
    }
    let chars: Vec<char> = text.chars().collect();
    if !chars.is_empty() && chars.iter().all(|ch| PUNCT.contains(*ch)) {
        return Tag::Decided(Upos::Punct);
    }
    if !chars.is_empty() && chars.iter().all(|ch| SYM.contains(*ch)) {
        return Tag::Decided(Upos::Sym);
    }
    if is_number(&chars) {
        return Tag::Decided(Upos::Num);
    }
    let en_key;
    let zh_key;
    let key: &str = match lang {
        // A curly apostrophe in a clitic looks up as the straight one.
        Lang::En => {
            en_key = text.to_lowercase().replace('\u{2019}', "'");
            en_key.as_str()
        }
        Lang::Zh => {
            zh_key = super::token::strip_scraper_infix(text);
            zh_key.as_ref()
        }
        Lang::Mixed | Lang::Und => return Tag::Missing,
    };
    if let Some(&upos) = decided().get(key) {
        Tag::Decided(upos)
    } else if ambiguous().contains(key) {
        Tag::Ambiguous
    } else {
        Tag::Missing
    }
}

/// ASCII digits with single `.` or `,` between them, or a run of Chinese numerals.
fn is_number(chars: &[char]) -> bool {
    let digits = chars.first().is_some_and(char::is_ascii_digit)
        && chars.last().is_some_and(char::is_ascii_digit)
        && chars
            .iter()
            .all(|ch| ch.is_ascii_digit() || matches!(ch, '.' | ','))
        && chars
            .windows(2)
            .all(|w| w[0].is_ascii_digit() || w[1].is_ascii_digit());
    digits || (!chars.is_empty() && chars.iter().all(|ch| ZH_NUMERALS.contains(*ch)))
}
