//! Layer 5 of `nlp.doc.v1`: Deterministic predicate-argument semantic frame extraction
//! (`ARG0`, `ARG1`, `ARGM`) across `zh`, `en`, `ja`, `ko`, and `mixed`.
//! See `docs/annotation-guide.md`, section "Layer 5: Semantic Frames & Issue #14 Role Mapping".

use crate::Span;
use crate::doc::{Document, EntityKind, Lang, Upos};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum FrameRole {
    #[serde(rename = "ARG0")]
    Arg0,
    #[serde(rename = "ARG1")]
    Arg1,
    #[serde(rename = "ARGM")]
    Argm,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrameArg {
    pub role: FrameRole,
    /// 0-based index into `Document.tokens` (syntactic head of the argument constituent).
    pub head: usize,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SemanticFrame {
    /// 0-based index into `Document.tokens` of the predicate (`VERB` or predicative `ADJ`).
    pub predicate: usize,
    /// 0-based index into `Document.segmentation.sentences`.
    pub sentence: usize,
    pub span: Span,
    pub args: Vec<FrameArg>,
}

/// Token syntax view indexed by 0-based token index.
#[derive(Clone)]
struct SynInfo<'a> {
    head: Option<usize>,
    deprel: &'a str,
}

pub fn annotate(doc: &mut Document, text: &str) {
    doc.frames.clear();
    if doc.syntax.is_empty() || doc.tokens.is_empty() {
        return;
    }

    let chars: Vec<char> = text.chars().collect();
    let n = doc.tokens.len();

    // Precompute lowercase surface strings for each token.
    let surfaces: Vec<String> = doc
        .tokens
        .iter()
        .map(|t| {
            let start = t.span.start_char.min(chars.len());
            let end = t.span.end_char.min(chars.len());
            chars[start..end].iter().collect()
        })
        .collect();
    let lowers: Vec<String> = surfaces.iter().map(|s| s.to_lowercase()).collect();

    // Map token index -> SynInfo and head -> children list.
    let mut syn: Vec<Option<SynInfo<'_>>> = vec![None; n];
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
    for st in &doc.syntax {
        if st.token < n {
            syn[st.token] = Some(SynInfo {
                head: st.head.filter(|&h| h < n && h != st.token),
                deprel: st.deprel.as_str(),
            });
            if let Some(h) = st.head
                && h < n
                && h != st.token
                && doc.tokens[h].sentence == doc.tokens[st.token].sentence
            {
                children[h].push(st.token);
            }
        }
    }

    // Map token index -> overlapping EntityKind (if any) for numeric/temporal detection.
    let mut tok_entity: Vec<Option<EntityKind>> = vec![None; n];
    for ent in &doc.entities {
        for (i, tok) in doc.tokens.iter().enumerate() {
            if tok.sentence == ent.sentence
                && tok.span.start_char >= ent.span.start_char
                && tok.span.end_char <= ent.span.end_char
            {
                tok_entity[i] = Some(ent.kind);
            }
        }
    }

    // Identify predicate candidates first so we can propagate shared arguments cleanly.
    let mut is_pred = vec![false; n];
    for (i, pred_flag) in is_pred.iter_mut().enumerate() {
        if is_predicate_token(i, doc, &syn, &children) {
            *pred_flag = true;
        }
    }

    // Collect direct arguments per predicate.
    let mut frame_args: HashMap<usize, BTreeSet<(FrameRole, usize)>> = HashMap::new();
    let mut pred_passive: HashMap<usize, bool> = HashMap::new();
    let mut pred_unacc: HashMap<usize, bool> = HashMap::new();

    for (p, &pred_flag) in is_pred.iter().enumerate() {
        if !pred_flag {
            continue;
        }
        let sent = doc.tokens[p].sentence;
        let p_upos = doc.tokens[p].upos;
        let p_lang = doc.tokens[p].lang;
        let p_surf = &surfaces[p];
        let p_low = &lowers[p];

        let passive = is_passive(p, p_lang, p_surf, p_low, &syn, &children, &lowers);
        let causative = is_causative(p, p_lang, &syn, &children, &lowers);
        pred_passive.insert(p, passive);

        let has_disposal = children[p].iter().any(|&c| {
            syn[c].as_ref().is_some_and(|s| s.deprel == "obl")
                && has_case_in(c, &children, &syn, &lowers, &["把", "將", "将"])
        });
        let has_obj_or_ccomp = children[p].iter().any(|&c| {
            syn[c]
                .as_ref()
                .is_some_and(|s| matches!(s.deprel, "obj" | "iobj" | "ccomp"))
        });
        let has_xcomp_adj = children[p].iter().any(|&c| {
            syn[c].as_ref().is_some_and(|s| s.deprel == "xcomp")
                && doc.tokens[c].upos == Some(Upos::Adj)
        });

        let unacc = p_upos == Some(Upos::Adj)
            || is_unaccusative_verb(
                p,
                p_low,
                p_lang,
                has_disposal,
                has_obj_or_ccomp,
                has_xcomp_adj,
                doc,
                &syn,
                &children,
                &lowers,
            );
        pred_unacc.insert(p, unacc);

        let mut args = BTreeSet::new();

        for &c in &children[p] {
            if doc.tokens[c].sentence != sent {
                continue;
            }
            let Some(ref c_syn) = syn[c] else { continue };
            let rel = c_syn.deprel;

            if is_non_arg_upos(doc.tokens[c].upos, rel) {
                // Allow xcomp ADJ/VERB or advcl VERB/ADJ as ARGM/ARG1, but skip ADP/PART/PUNCT/AUX/CCONJ/SCONJ.
                if !matches!(rel, "xcomp" | "ccomp" | "advcl") {
                    continue;
                }
            }

            match rel {
                "nsubj" | "csubj" => {
                    let role = if passive || unacc {
                        FrameRole::Arg1
                    } else {
                        FrameRole::Arg0
                    };
                    if is_valid_arg_head(c, p, sent, doc, &syn) {
                        args.insert((role, c));
                    }
                }
                "obj" => {
                    if is_valid_arg_head(c, p, sent, doc, &syn) {
                        let role = if is_extent_or_duration_obj(
                            c,
                            p_low,
                            &lowers,
                            &children,
                            &syn,
                            &tok_entity,
                            doc,
                        ) {
                            FrameRole::Argm
                        } else {
                            FrameRole::Arg1
                        };
                        args.insert((role, c));
                    }
                }
                "iobj" => {
                    if is_valid_arg_head(c, p, sent, doc, &syn) {
                        if passive
                            && p_lang == Lang::Ja
                            && is_passive_agent_obl(
                                c,
                                p_lang,
                                &children,
                                &syn,
                                &lowers,
                                &surfaces,
                                &tok_entity,
                                doc,
                            )
                        {
                            args.insert((FrameRole::Arg0, c));
                        } else {
                            args.insert((FrameRole::Arg1, c));
                        }
                    }
                }
                "obl" => {
                    if is_valid_arg_head(c, p, sent, doc, &syn) {
                        if passive
                            && is_passive_agent_obl(
                                c,
                                p_lang,
                                &children,
                                &syn,
                                &lowers,
                                &surfaces,
                                &tok_entity,
                                doc,
                            )
                        {
                            args.insert((FrameRole::Arg0, c));
                        } else if (!passive
                            && causative
                            && p_lang == Lang::Ja
                            && has_case_in(c, &children, &syn, &lowers, &["に"])
                            && is_ja_animate_arg(c, doc, &tok_entity, &surfaces))
                            || has_case_in(c, &children, &syn, &lowers, &["把", "將", "将"])
                        {
                            args.insert((FrameRole::Arg1, c));
                        } else {
                            args.insert((FrameRole::Argm, c));
                        }
                    }
                }
                "advmod" => {
                    if is_valid_arg_head(c, p, sent, doc, &syn)
                        && is_frame_advmod(&lowers[c], doc.tokens[c].upos)
                    {
                        args.insert((FrameRole::Argm, c));
                    }
                }
                "advcl" => {
                    if is_valid_arg_head(c, p, sent, doc, &syn) {
                        args.insert((FrameRole::Argm, c));
                    }
                }
                "xcomp" => {
                    if is_valid_arg_head(c, p, sent, doc, &syn) {
                        let role = if doc.tokens[c].upos == Some(Upos::Adj)
                            || is_copular_or_state_xcomp(p_low)
                        {
                            FrameRole::Argm
                        } else {
                            FrameRole::Arg1
                        };
                        args.insert((role, c));
                    }
                }
                "ccomp" if is_valid_arg_head(c, p, sent, doc, &syn) => {
                    args.insert((FrameRole::Arg1, c));
                }
                _ => {}
            }
        }

        frame_args.insert(p, args);
    }

    // Shared subject propagation across `conj`, `xcomp`, `advcl`, and relative-clause `acl` head-noun linking.
    for (p, &pred_flag) in is_pred.iter().enumerate() {
        if !pred_flag {
            continue;
        }
        let sent = doc.tokens[p].sentence;
        let Some(ref p_syn) = syn[p] else { continue };
        let Some(parent) = p_syn.head else { continue };
        if doc.tokens[parent].sentence != sent {
            continue;
        }

        if matches!(p_syn.deprel, "conj" | "xcomp" | "advcl") && is_pred[parent] {
            let p_has_subj = frame_args.get(&p).is_some_and(|args| {
                args.iter().any(|&(r, h)| {
                    matches!(r, FrameRole::Arg0 | FrameRole::Arg1) && is_subj_like(h, p, &syn)
                })
            });
            if !p_has_subj {
                // Find subject to propagate from parent:
                // For object-control `xcomp` (e.g., `請 他[obj] 幫忙[xcomp]`, `helped me[obj] fold`),
                // prefer parent's `obj`/`iobj` if parent is a control verb and has one; otherwise parent's `nsubj`.
                let ctrl_subj =
                    if p_syn.deprel == "xcomp" && is_object_control_verb(&lowers[parent]) {
                        find_child_with_rel(parent, &children, &syn, &["obj", "iobj"])
                            .or_else(|| find_propagated_subj(parent, &syn, &children, &frame_args))
                    } else {
                        find_propagated_subj(parent, &syn, &children, &frame_args)
                    };

                if let Some(subj_head) = ctrl_subj
                    && is_valid_arg_head(subj_head, p, sent, doc, &syn)
                {
                    let passive = *pred_passive.get(&p).unwrap_or(&false);
                    let unacc = *pred_unacc.get(&p).unwrap_or(&false);
                    let role = if passive || unacc {
                        FrameRole::Arg1
                    } else {
                        FrameRole::Arg0
                    };
                    frame_args.entry(p).or_default().insert((role, subj_head));
                }
            }
        } else if p_syn.deprel == "acl"
            && matches!(
                doc.tokens[parent].upos,
                Some(Upos::Noun | Upos::Propn | Upos::Pron)
            )
        {
            // Relative clause `acl` attached to noun `parent`:
            // If `p` is missing `obj` (or is passive/unacc missing `nsubj`), link `parent` as `ARG1` (or `ARG0` if missing `nsubj` and `p` already has `obj`).
            let p_has_obj = children[p]
                .iter()
                .any(|&c| syn[c].as_ref().is_some_and(|s| s.deprel == "obj"));
            let p_has_nsubj = children[p]
                .iter()
                .any(|&c| syn[c].as_ref().is_some_and(|s| s.deprel == "nsubj"));
            let passive = *pred_passive.get(&p).unwrap_or(&false);
            let unacc = *pred_unacc.get(&p).unwrap_or(&false);

            if is_valid_arg_head(parent, p, sent, doc, &syn) {
                if !p_has_obj && !passive && !unacc && p_has_nsubj {
                    frame_args
                        .entry(p)
                        .or_default()
                        .insert((FrameRole::Arg1, parent));
                } else if !p_has_nsubj {
                    let role = if passive || unacc || !p_has_obj {
                        FrameRole::Arg1
                    } else {
                        FrameRole::Arg0
                    };
                    frame_args.entry(p).or_default().insert((role, parent));
                }
            }
        }
    }

    // Build sorted SemanticFrame output.
    let mut frames = Vec::new();
    for (p, &pred_flag) in is_pred.iter().enumerate() {
        if !pred_flag {
            continue;
        }
        let Some(arg_set) = frame_args.get(&p) else {
            continue;
        };
        if arg_set.is_empty() {
            continue;
        }
        let mut args: Vec<FrameArg> = arg_set
            .iter()
            .map(|&(role, head)| FrameArg {
                role,
                head,
                span: doc.tokens[head].span,
            })
            .collect();
        args.sort_by_key(|a| (a.head, a.role));
        frames.push(SemanticFrame {
            predicate: p,
            sentence: doc.tokens[p].sentence,
            span: doc.tokens[p].span,
            args,
        });
    }
    frames.sort_by_key(|f| f.predicate);
    doc.frames = frames;
}

fn is_predicate_token(
    i: usize,
    doc: &Document,
    syn: &[Option<SynInfo<'_>>],
    children: &[Vec<usize>],
) -> bool {
    let upos = doc.tokens[i].upos;
    let Some(ref info) = syn[i] else {
        return false;
    };
    let rel = info.deprel;

    // Exclude auxiliary, copula, adnominal modifiers, compounds, case/mark, etc.
    if matches!(
        rel,
        "aux"
            | "aux:pass"
            | "cop"
            | "amod"
            | "case"
            | "mark"
            | "punct"
            | "cc"
            | "det"
            | "clf"
            | "nummod"
    ) {
        return false;
    }
    if rel == "compound" {
        // Secondary piece of split V-V / V-Adj compound (`讀+完`, `刷+干净`).
        return false;
    }

    // Exclude nominal copular predicates (`是 手藝人` / `is a teacher`).
    let has_cop_child = children[i]
        .iter()
        .any(|&c| syn[c].as_ref().is_some_and(|s| s.deprel == "cop"));
    if has_cop_child && !matches!(upos, Some(Upos::Verb | Upos::Adj)) {
        return false;
    }

    match upos {
        Some(Upos::Verb) => true,
        Some(Upos::Adj) => matches!(
            rel,
            "root" | "conj" | "advcl" | "ccomp" | "parataxis" | "acl" | "xcomp"
        ),
        _ => false,
    }
}

fn is_non_arg_upos(upos: Option<Upos>, _rel: &str) -> bool {
    matches!(
        upos,
        Some(Upos::Adp | Upos::Part | Upos::Punct | Upos::Aux | Upos::Cconj | Upos::Sconj)
    )
}

fn is_valid_arg_head(
    head: usize,
    pred: usize,
    sent: usize,
    doc: &Document,
    syn: &[Option<SynInfo<'_>>],
) -> bool {
    if head == pred || head >= doc.tokens.len() || doc.tokens[head].sentence != sent {
        return false;
    }
    if matches!(
        doc.tokens[head].upos,
        Some(Upos::Adp | Upos::Part | Upos::Punct)
    ) {
        return false;
    }
    if let Some(ref s) = syn[head]
        && matches!(s.deprel, "case" | "mark" | "punct" | "cop")
    {
        return false;
    }
    true
}

fn has_case_in(
    node: usize,
    children: &[Vec<usize>],
    syn: &[Option<SynInfo<'_>>],
    lowers: &[String],
    targets: &[&str],
) -> bool {
    children[node].iter().any(|&c| {
        syn[c]
            .as_ref()
            .is_some_and(|s| matches!(s.deprel, "case" | "mark"))
            && targets.contains(&lowers[c].as_str())
    })
}

fn is_passive(
    p: usize,
    lang: Lang,
    p_surf: &str,
    p_low: &str,
    syn: &[Option<SynInfo<'_>>],
    children: &[Vec<usize>],
    lowers: &[String],
) -> bool {
    // Chinese passive: `被`/`挨`/`遭` as `aux` child, or `被`/`由` as `case` child of `obl`.
    let zh_aux = children[p].iter().any(|&c| {
        syn[c]
            .as_ref()
            .is_some_and(|s| matches!(s.deprel, "aux" | "aux:pass"))
            && matches!(lowers[c].as_str(), "被" | "挨" | "遭")
    });
    let zh_obl = children[p].iter().any(|&c| {
        syn[c].as_ref().is_some_and(|s| s.deprel == "obl")
            && has_case_in(c, children, syn, lowers, &["被", "由"])
    });
    if zh_aux || zh_obl {
        return true;
    }

    // English passive: `be`/`get` aux child + past participle morphology (or `by` obl).
    let en_aux = children[p].iter().any(|&c| {
        syn[c]
            .as_ref()
            .is_some_and(|s| matches!(s.deprel, "aux" | "aux:pass"))
            && matches!(
                lowers[c].as_str(),
                "be" | "is"
                    | "am"
                    | "are"
                    | "was"
                    | "were"
                    | "been"
                    | "being"
                    | "get"
                    | "got"
                    | "gotten"
            )
    });
    if en_aux && !p_low.ends_with("ing") {
        return true;
    }

    // Japanese passive: `れる` / `られる` / `れ` / `られ` as `aux` child.
    if lang == Lang::Ja {
        let ja_aux = children[p].iter().any(|&c| {
            syn[c].as_ref().is_some_and(|s| s.deprel == "aux")
                && matches!(
                    lowers[c].as_str(),
                    "れる" | "られる" | "れ" | "られ" | "れた" | "られた" | "れて" | "られて"
                )
        });
        if ja_aux {
            return true;
        }
    }

    // Korean passive: `-되다` / `-어지다` / `-아지다` / `-받다` / `-당하다` or `지` aux.
    if lang == Lang::Ko {
        if p_surf.contains("되")
            || p_surf.contains("됐")
            || p_surf.contains("받았")
            || p_surf.contains("당했")
            || p_surf.ends_with("졌다")
            || p_surf.ends_with("진다")
        {
            return true;
        }
        let ko_aux = children[p].iter().any(|&c| {
            syn[c].as_ref().is_some_and(|s| s.deprel == "aux")
                && (lowers[c].starts_with('지') || lowers[c].starts_with('되'))
        });
        if ko_aux {
            return true;
        }
    }

    false
}

fn is_causative(
    p: usize,
    lang: Lang,
    syn: &[Option<SynInfo<'_>>],
    children: &[Vec<usize>],
    lowers: &[String],
) -> bool {
    if lang != Lang::Ja {
        return false;
    }
    children[p].iter().any(|&c| {
        syn[c].as_ref().is_some_and(|s| s.deprel == "aux")
            && matches!(
                lowers[c].as_str(),
                "せる" | "させる" | "せ" | "させ" | "せた" | "させた" | "せて" | "させて"
            )
    })
}

fn is_ja_animate_arg(
    tok_idx: usize,
    doc: &Document,
    tok_entity: &[Option<EntityKind>],
    surfaces: &[String],
) -> bool {
    if matches!(
        tok_entity[tok_idx],
        Some(EntityKind::Date | EntityKind::RelativeDate | EntityKind::Time)
    ) {
        return false;
    }
    if tok_entity[tok_idx] == Some(EntityKind::Name) {
        return true;
    }
    let tok = &doc.tokens[tok_idx];
    if matches!(tok.upos, Some(Upos::Propn | Upos::Pron)) {
        return true;
    }
    let surf = &surfaces[tok_idx];
    if let Some(hit) = crate::lexicon::lookup_ja(surf) {
        if matches!(
            hit.sem_class,
            Some(
                crate::lexicon::SemanticClass::Location
                    | crate::lexicon::SemanticClass::Time
                    | crate::lexicon::SemanticClass::Artifact
            )
        ) {
            return false;
        }
        if hit.sem_class == Some(crate::lexicon::SemanticClass::Person) {
            return true;
        }
    }
    surf.ends_with("さん")
        || surf.ends_with("くん")
        || surf.ends_with("ちゃん")
        || surf.ends_with("様")
        || surf.ends_with("さま")
        || surf.ends_with("先生")
        || surf.ends_with("先輩")
        || surf.ends_with("殿")
        || surf.ends_with("氏")
        || surf.ends_with("嬢")
        || matches!(
            surf.as_str(),
            "人" | "男"
                | "女"
                | "少年"
                | "少女"
                | "子供"
                | "子ども"
                | "母"
                | "父"
                | "兄"
                | "姉"
                | "弟"
                | "妹"
                | "夫"
                | "妻"
                | "友達"
                | "友人"
                | "医者"
                | "警察"
                | "犯人"
                | "犬"
                | "猫"
        )
}

#[allow(clippy::too_many_arguments)]
fn is_passive_agent_obl(
    obl: usize,
    lang: Lang,
    children: &[Vec<usize>],
    syn: &[Option<SynInfo<'_>>],
    lowers: &[String],
    surfaces: &[String],
    tok_entity: &[Option<EntityKind>],
    doc: &Document,
) -> bool {
    if has_case_in(
        obl,
        children,
        syn,
        lowers,
        &["被", "由", "by", "によって", "により"],
    ) {
        return true;
    }
    if has_case_in(obl, children, syn, lowers, &["に"]) {
        return lang != Lang::Ja || is_ja_animate_arg(obl, doc, tok_entity, surfaces);
    }
    if lang == Lang::Ko {
        let s = &surfaces[obl];
        if s.ends_with("에게") || s.ends_with("한테") || s.ends_with("에") {
            return true;
        }
        if has_case_in(obl, children, syn, lowers, &["의해", "에게", "한테"]) {
            return true;
        }
    }
    false
}

#[allow(clippy::too_many_arguments)]
fn is_unaccusative_verb(
    p: usize,
    p_low: &str,
    _lang: Lang,
    has_disposal: bool,
    has_obj_or_ccomp: bool,
    has_xcomp_adj: bool,
    doc: &Document,
    syn: &[Option<SynInfo<'_>>],
    children: &[Vec<usize>],
    lowers: &[String],
) -> bool {
    if has_disposal {
        return false;
    }
    if has_xcomp_adj {
        return true;
    }

    const UNACC_LIST: &[&str] = &[
        // Chinese
        "有",
        "存在",
        "出現",
        "出现",
        "发生",
        "發生",
        "傳來",
        "传来",
        "升起",
        "落下",
        "落",
        "搖晃",
        "摇晃",
        "收斂",
        "收敛",
        "穩定",
        "稳定",
        "增加",
        "減少",
        "减少",
        "下降",
        "上升",
        "漲高",
        "涨高",
        "顯得",
        "显得",
        "变得",
        "變得",
        "停",
        "下",
        "存放",
        "超过",
        "超過",
        "结束",
        "結束",
        "靜置",
        "静置",
        "縮短",
        "缩短",
        "保持",
        "饱和",
        "飽和",
        "中断",
        "中斷",
        "到",
        "放晴",
        "清澈见底",
        "清澈見底",
        "分成",
        "用於",
        "用于",
        "开展",
        "開展",
        // English
        "appear",
        "appeared",
        "appears",
        "arrive",
        "arrived",
        "arrives",
        "fall",
        "fell",
        "fallen",
        "rise",
        "rose",
        "risen",
        "converge",
        "converged",
        "converges",
        "drop",
        "dropped",
        "occur",
        "occurred",
        "occurs",
        "happen",
        "happened",
        "happens",
        "remain",
        "remained",
        "remains",
        "stay",
        "stayed",
        "stays",
        "stand",
        "stood",
        "stands",
        "sit",
        "sat",
        "sits",
        "glow",
        "glowed",
        "drift",
        "drifted",
        "stream",
        "streamed",
        "echo",
        "echoed",
        "land",
        "landed",
        "ring",
        "rang",
        "chime",
        "chimed",
        "swing",
        "swung",
        "gather",
        "gathered",
        "begin",
        "began",
        "begun",
        "start",
        "started",
        "starts",
        "dry",
        "dried",
        "burn",
        "burned",
        "last",
        "lasted",
        "compile",
        "compiles",
        "compiled",
        "peak",
        "peaked",
        "decline",
        "declined",
        "taste",
        "tastes",
        "tasted",
        "seem",
        "seemed",
        "seems",
        "become",
        "became",
        "exceed",
        "exceeded",
        "exceeds",
        "reach",
        "reached",
        "reaches",
        // Japanese
        "ある",
        "いる",
        "降る",
        "落ちる",
        "増える",
        "減る",
        "始まる",
        "終わる",
        "残る",
        "届く",
        // Korean
        "있다",
        "없다",
        "내리다",
        "떨어지다",
        "늘다",
        "줄다",
        "시작되다",
        "끝나다",
        "남다",
    ];

    if UNACC_LIST.contains(&p_low) {
        // Note: transitive uses with a non-extent obj (e.g. `reached the station`) are active unless extent obj.
        if !has_obj_or_ccomp {
            return true;
        }
        // Check if all obj children are extent/margin objs (e.g. `exceeded ninety percent`).
        let all_extent = children[p]
            .iter()
            .filter(|&&c| syn[c].as_ref().is_some_and(|s| s.deprel == "obj"))
            .all(|&c| {
                lowers[c] == "percent"
                    || lowers[c].starts_with("百分之")
                    || lowers[c].contains("毫秒")
                    || lowers[c].starts_with('`')
            });
        if all_extent {
            return true;
        }
    }

    // Check if there's an nsubj and no obj/ccomp, and nsubj is inanimate (not PRON/PROPN).
    if !has_obj_or_ccomp && let Some(subj) = find_child_with_rel(p, children, syn, &["nsubj"]) {
        let subj_low = lowers[subj].as_str();
        let is_animate_pron = matches!(
            subj_low,
            "我" | "你"
                | "他"
                | "她"
                | "我們"
                | "我们"
                | "你們"
                | "你们"
                | "他們"
                | "他们"
                | "她們"
                | "她们"
                | "i"
                | "you"
                | "he"
                | "she"
                | "we"
                | "they"
                | "who"
        );
        let is_person_or_agent_noun = is_animate_pron
            || doc.tokens[subj].upos == Some(Upos::Propn)
            || doc.tokens[subj]
                .sem_class
                .is_some_and(|sc| matches!(sc, crate::lexicon::SemanticClass::Person));
        if !is_person_or_agent_noun && doc.tokens[subj].upos == Some(Upos::Noun) {
            // Inanimate subject of an intransitive verb without obj -> likely unaccusative/state unless clearly agentive.
            const AGENTIVE_INTRANS: &[&str] = &[
                "站", "走", "跑", "笑", "哭", "睡", "坐", "看", "聽", "听", "說", "说", "巡查",
                "工作", "學習", "学习", "休息", "出發", "出发", "walk", "walked", "run", "ran",
                "sleep", "slept", "trot", "trotted", "smile", "smiled", "laugh", "laughed", "work",
                "worked", "wait", "waited", "pause", "paused", "slip", "slipped", "给出", "給出",
            ];
            if !AGENTIVE_INTRANS.contains(&p_low) {
                return true;
            }
        }
    }

    false
}

fn is_extent_or_duration_obj(
    obj: usize,
    p_low: &str,
    lowers: &[String],
    children: &[Vec<usize>],
    syn: &[Option<SynInfo<'_>>],
    tok_entity: &[Option<EntityKind>],
    doc: &Document,
) -> bool {
    let w = lowers[obj].as_str();
    if w == "percent"
        || w.starts_with("百分之")
        || w.ends_with("毫秒")
        || w.ends_with("小時")
        || w.ends_with("小时")
        || w.ends_with("分鐘")
        || w.ends_with("分钟")
    {
        return true;
    }
    if matches!(
        tok_entity[obj],
        Some(EntityKind::Quantity | EntityKind::Time | EntityKind::Number)
    ) && matches!(
        p_low,
        "超过"
            | "超過"
            | "增加"
            | "減少"
            | "减少"
            | "下降"
            | "上升"
            | "exceed"
            | "exceeded"
            | "exceeds"
            | "reach"
            | "reached"
            | "需"
            | "花"
            | "等"
            | "等了"
            | "last"
            | "lasted"
    ) {
        return true;
    }
    // Check if obj has a nummod child and is a unit/extent noun (`percent`, `毫秒`, `小時`).
    let has_nummod = children[obj]
        .iter()
        .any(|&c| syn[c].as_ref().is_some_and(|s| s.deprel == "nummod"));
    if has_nummod
        && matches!(
            w,
            "percent" | "毫秒" | "秒" | "分鐘" | "分钟" | "小時" | "小时" | "天" | "年"
        )
    {
        return true;
    }
    if doc.tokens[obj].upos == Some(Upos::X)
        && w.starts_with('`')
        && matches!(p_low, "超过" | "超過" | "exceed" | "exceeded")
    {
        return true;
    }
    false
}

fn is_frame_advmod(w: &str, upos: Option<Upos>) -> bool {
    if matches!(
        w,
        "都" | "均"
            | "皆"
            | "也"
            | "就"
            | "才"
            | "在"
            | "不"
            | "没"
            | "沒"
            | "未"
            | "并未"
            | "並未"
            | "是否"
            | "還要"
            | "还要"
            | "not"
            | "n't"
            | "as"
    ) {
        return false;
    }
    if matches!(upos, Some(Upos::Part | Upos::Aux)) {
        return false;
    }
    true
}

fn is_copular_or_state_xcomp(p_low: &str) -> bool {
    matches!(
        p_low,
        "保持"
            | "remain"
            | "remained"
            | "remains"
            | "stay"
            | "stayed"
            | "stays"
            | "taste"
            | "tastes"
            | "tasted"
            | "smell"
            | "smells"
            | "smelled"
            | "grow"
            | "grew"
            | "paint"
            | "painted"
            | "seem"
            | "seemed"
            | "become"
            | "became"
            | "start"
            | "starts"
            | "started"
    )
}

fn is_object_control_verb(p_low: &str) -> bool {
    matches!(
        p_low,
        "請" | "请"
            | "讓"
            | "让"
            | "叫"
            | "派"
            | "幫"
            | "帮"
            | "幫助"
            | "帮助"
            | "勸"
            | "劝"
            | "ask"
            | "asked"
            | "tell"
            | "told"
            | "help"
            | "helped"
            | "make"
            | "made"
            | "makes"
            | "let"
            | "allow"
            | "allowed"
            | "force"
            | "forced"
    )
}

fn find_child_with_rel(
    head: usize,
    children: &[Vec<usize>],
    syn: &[Option<SynInfo<'_>>],
    rels: &[&str],
) -> Option<usize> {
    children[head]
        .iter()
        .copied()
        .find(|&c| syn[c].as_ref().is_some_and(|s| rels.contains(&s.deprel)))
}

fn find_propagated_subj(
    parent: usize,
    syn: &[Option<SynInfo<'_>>],
    children: &[Vec<usize>],
    frame_args: &HashMap<usize, BTreeSet<(FrameRole, usize)>>,
) -> Option<usize> {
    if let Some(s) = find_child_with_rel(parent, children, syn, &["nsubj", "csubj"]) {
        return Some(s);
    }
    if let Some(args) = frame_args.get(&parent)
        && let Some(&(_, h)) = args.iter().find(|&&(r, h)| {
            matches!(r, FrameRole::Arg0 | FrameRole::Arg1) && is_subj_like(h, parent, syn)
        })
    {
        return Some(h);
    }
    None
}

fn is_subj_like(arg_head: usize, _pred: usize, syn: &[Option<SynInfo<'_>>]) -> bool {
    syn[arg_head]
        .as_ref()
        .is_some_and(|s| matches!(s.deprel, "nsubj" | "csubj"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::doc::{
        Clause, Options, Paragraph, SCHEMA, Segmentation, Sentence, SyntaxToken, Token,
        syntax::Score,
    };
    use std::collections::HashSet;

    fn parse_upos(s: &str) -> Option<Upos> {
        match s {
            "ADJ" => Some(Upos::Adj),
            "ADP" => Some(Upos::Adp),
            "ADV" => Some(Upos::Adv),
            "AUX" => Some(Upos::Aux),
            "CCONJ" => Some(Upos::Cconj),
            "DET" => Some(Upos::Det),
            "INTJ" => Some(Upos::Intj),
            "NOUN" => Some(Upos::Noun),
            "NUM" => Some(Upos::Num),
            "PART" => Some(Upos::Part),
            "PRON" => Some(Upos::Pron),
            "PROPN" => Some(Upos::Propn),
            "PUNCT" => Some(Upos::Punct),
            "SCONJ" => Some(Upos::Sconj),
            "SYM" => Some(Upos::Sym),
            "VERB" => Some(Upos::Verb),
            "X" => Some(Upos::X),
            _ => None,
        }
    }

    fn parse_lang(s: &str) -> Lang {
        match s {
            "zh" => Lang::Zh,
            "en" => Lang::En,
            "ja" => Lang::Ja,
            "ko" => Lang::Ko,
            "mixed" => Lang::Mixed,
            _ => Lang::Und,
        }
    }

    fn build_doc_from_tokens(
        toks: &[&str],
        upos_strs: &[&str],
        heads_1based: &[usize],
        deprels: &[&str],
        lang: Lang,
    ) -> (Document, String) {
        let mut text = String::new();
        let mut tokens = Vec::new();
        let mut syntax = Vec::new();
        let mut char_pos = 0usize;
        for i in 0..toks.len() {
            if i > 0 {
                text.push(' ');
                char_pos += 1;
            }
            let len = toks[i].chars().count();
            let span = Span {
                start_char: char_pos,
                end_char: char_pos + len,
            };
            text.push_str(toks[i]);
            char_pos += len;
            tokens.push(Token {
                span,
                lang,
                sentence: 0,
                upos: parse_upos(upos_strs[i]),
                lemma: None,
                canonical_variant: None,
                upos_candidates: Vec::new(),
                sem_class: None,
                morphemes: Vec::new(),
            });
            let head = if heads_1based[i] == 0 {
                None
            } else {
                Some(heads_1based[i] - 1)
            };
            syntax.push(SyntaxToken {
                token: i,
                sentence: 0,
                span,
                head,
                deprel: deprels[i].to_owned(),
                score: Score::from_confidence(1.0),
            });
        }
        let full_span = Span {
            start_char: 0,
            end_char: char_pos,
        };
        let doc = Document {
            schema: SCHEMA.to_owned(),
            text_chars: char_pos,
            text_sha256: String::new(),
            options: Options::default(),
            segmentation: Segmentation {
                paragraphs: vec![Paragraph {
                    span: full_span,
                    lang,
                }],
                sentences: vec![Sentence {
                    span: full_span,
                    lang,
                    paragraph: 0,
                }],
                clauses: vec![Clause {
                    span: full_span,
                    lang,
                    sentence: 0,
                }],
                opaque: Vec::new(),
            },
            tokens,
            entities: Vec::new(),
            syntax,
            frames: Vec::new(),
            coref: Vec::new(),
            zero_mentions: Vec::new(),
            relations: Vec::new(),
            unknown: Vec::new(),
        };
        (doc, text)
    }

    #[test]
    fn empty_syntax_leaves_frames_empty() {
        let mut doc = crate::doc::analyze_document("Hello world.", &Options::default());
        annotate(&mut doc, "Hello world.");
        assert!(doc.frames.is_empty());
    }

    #[test]
    fn ja_active_passive_ditransitive() {
        // Active + ditransitive: 太郎 が 花子 に 本 を 渡した 。
        // 0:太郎(nsubj->6) 1:が(case->0) 2:花子(iobj->6) 3:に(case->2) 4:本(obj->6) 5:を(case->4) 6:渡した(root) 7:。(punct->6)
        let (mut doc, text) = build_doc_from_tokens(
            &["太郎", "が", "花子", "に", "本", "を", "渡した", "。"],
            &[
                "PROPN", "ADP", "PROPN", "ADP", "NOUN", "ADP", "VERB", "PUNCT",
            ],
            &[7, 1, 7, 3, 7, 5, 0, 7],
            &[
                "nsubj", "case", "iobj", "case", "obj", "case", "root", "punct",
            ],
            Lang::Ja,
        );
        annotate(&mut doc, &text);
        assert_eq!(doc.frames.len(), 1);
        let f = &doc.frames[0];
        assert_eq!(f.predicate, 6);
        assert_eq!(
            f.args.iter().map(|a| (a.role, a.head)).collect::<Vec<_>>(),
            vec![
                (FrameRole::Arg0, 0),
                (FrameRole::Arg1, 2),
                (FrameRole::Arg1, 4),
            ]
        );

        // Passive: 本 が 太郎 によって 読ま れた 。
        // 0:本(nsubj->4) 1:が(case->0) 2:太郎(obl->4) 3:によって(case->2) 4:読ま(root) 5:れた(aux->4) 6:。(punct->4)
        let (mut doc_p, text_p) = build_doc_from_tokens(
            &["本", "が", "太郎", "によって", "読ま", "れた", "。"],
            &["NOUN", "ADP", "PROPN", "ADP", "VERB", "AUX", "PUNCT"],
            &[5, 1, 5, 3, 0, 5, 5],
            &["nsubj", "case", "obl", "case", "root", "aux", "punct"],
            Lang::Ja,
        );
        annotate(&mut doc_p, &text_p);
        assert_eq!(doc_p.frames.len(), 1);
        assert_eq!(
            doc_p.frames[0]
                .args
                .iter()
                .map(|a| (a.role, a.head))
                .collect::<Vec<_>>(),
            vec![(FrameRole::Arg1, 0), (FrameRole::Arg0, 2)]
        );
    }

    #[test]
    fn ko_active_passive_ditransitive() {
        // Active + ditransitive: 민수가 지수에게 책을 주었다 .
        // 0:민수가(nsubj->3) 1:지수에게(iobj->3) 2:책을(obj->3) 3:주었다(root) 4:.(punct->3)
        let (mut doc, text) = build_doc_from_tokens(
            &["민수가", "지수에게", "책을", "주었다", "."],
            &["PROPN", "PROPN", "NOUN", "VERB", "PUNCT"],
            &[4, 4, 4, 0, 4],
            &["nsubj", "iobj", "obj", "root", "punct"],
            Lang::Ko,
        );
        annotate(&mut doc, &text);
        assert_eq!(doc.frames.len(), 1);
        assert_eq!(
            doc.frames[0]
                .args
                .iter()
                .map(|a| (a.role, a.head))
                .collect::<Vec<_>>(),
            vec![
                (FrameRole::Arg0, 0),
                (FrameRole::Arg1, 1),
                (FrameRole::Arg1, 2),
            ]
        );

        // Passive: 책이 민수에게 읽혔다 / 읽혀졌다 (or 선택되었다)
        // 0:보고서가(nsubj->2) 1:팀장에게(obl->2) 2:승인되었다(root) 3:.(punct->2)
        let (mut doc_p, text_p) = build_doc_from_tokens(
            &["보고서가", "팀장에게", "승인되었다", "."],
            &["NOUN", "NOUN", "VERB", "PUNCT"],
            &[3, 3, 0, 3],
            &["nsubj", "obl", "root", "punct"],
            Lang::Ko,
        );
        annotate(&mut doc_p, &text_p);
        assert_eq!(doc_p.frames.len(), 1);
        assert_eq!(
            doc_p.frames[0]
                .args
                .iter()
                .map(|a| (a.role, a.head))
                .collect::<Vec<_>>(),
            vec![(FrameRole::Arg1, 0), (FrameRole::Arg0, 1)]
        );
    }

    #[test]
    fn gold_300_frame_f1() {
        let gold_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/gold_layers4_6_v1.jsonl"
        );
        let raw = std::fs::read_to_string(gold_path).expect("read gold_layers4_6_v1.jsonl");
        let mut gold_set: HashSet<(String, usize, FrameRole, usize)> = HashSet::new();
        let mut pred_set: HashSet<(String, usize, FrameRole, usize)> = HashSet::new();

        for line in raw.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let v: serde_json::Value = serde_json::from_str(line).unwrap();
            let id = v["id"].as_str().unwrap().to_owned();
            let lang = parse_lang(v["lang"].as_str().unwrap());
            let toks: Vec<&str> = v["tokens"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_str().unwrap())
                .collect();
            let upos: Vec<&str> = v["upos"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_str().unwrap())
                .collect();
            let heads: Vec<usize> = v["heads"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as usize)
                .collect();
            let deprels: Vec<&str> = v["deprels"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_str().unwrap())
                .collect();

            // Reconstruct sentence indices from gold `sentences` token spans if present.
            let mut tok_sent = vec![0usize; toks.len()];
            let mut sentences_vec = Vec::new();
            let mut clauses_vec = Vec::new();
            let mut text = String::new();
            let mut char_pos = 0usize;
            let mut tok_spans = Vec::with_capacity(toks.len());
            for (i, t) in toks.iter().enumerate() {
                if i > 0 {
                    text.push(' ');
                    char_pos += 1;
                }
                let len = t.chars().count();
                let sp = Span {
                    start_char: char_pos,
                    end_char: char_pos + len,
                };
                tok_spans.push(sp);
                text.push_str(t);
                char_pos += len;
            }
            if let Some(sents) = v["sentences"].as_array() {
                for (si, s) in sents.iter().enumerate() {
                    let s0 = s[0].as_u64().unwrap() as usize;
                    let s1 = s[1].as_u64().unwrap() as usize;
                    for ti in s0..s1 {
                        if ti < tok_sent.len() {
                            tok_sent[ti] = si;
                        }
                    }
                    let sp = Span {
                        start_char: tok_spans[s0].start_char,
                        end_char: tok_spans[s1 - 1].end_char,
                    };
                    sentences_vec.push(Sentence {
                        span: sp,
                        lang,
                        paragraph: 0,
                    });
                    clauses_vec.push(Clause {
                        span: sp,
                        lang,
                        sentence: si,
                    });
                }
            }
            if sentences_vec.is_empty() {
                let sp = Span {
                    start_char: 0,
                    end_char: char_pos,
                };
                sentences_vec.push(Sentence {
                    span: sp,
                    lang,
                    paragraph: 0,
                });
                clauses_vec.push(Clause {
                    span: sp,
                    lang,
                    sentence: 0,
                });
            }

            let tokens: Vec<Token> = (0..toks.len())
                .map(|i| Token {
                    span: tok_spans[i],
                    lang,
                    sentence: tok_sent[i],
                    upos: parse_upos(upos[i]),
                    lemma: None,
                    canonical_variant: None,
                    upos_candidates: Vec::new(),
                    sem_class: None,
                    morphemes: Vec::new(),
                })
                .collect();
            let syntax: Vec<SyntaxToken> = (0..toks.len())
                .map(|i| SyntaxToken {
                    token: i,
                    sentence: tok_sent[i],
                    span: tok_spans[i],
                    head: if heads[i] == 0 {
                        None
                    } else {
                        Some(heads[i] - 1)
                    },
                    deprel: deprels[i].to_owned(),
                    score: Score::from_confidence(1.0),
                })
                .collect();

            let mut doc = Document {
                schema: SCHEMA.to_owned(),
                text_chars: char_pos,
                text_sha256: String::new(),
                options: Options::default(),
                segmentation: Segmentation {
                    paragraphs: vec![Paragraph {
                        span: Span {
                            start_char: 0,
                            end_char: char_pos,
                        },
                        lang,
                    }],
                    sentences: sentences_vec,
                    clauses: clauses_vec,
                    opaque: Vec::new(),
                },
                tokens,
                entities: Vec::new(),
                syntax,
                frames: Vec::new(),
                coref: Vec::new(),
                zero_mentions: Vec::new(),
                relations: Vec::new(),
                unknown: Vec::new(),
            };

            annotate(&mut doc, &text);

            let mut prev_pred: Option<usize> = None;
            for pf in &doc.frames {
                assert!(!pf.args.is_empty(), "{id}: empty frame args");
                if let Some(pp) = prev_pred {
                    assert!(pp < pf.predicate, "{id}: frames not sorted");
                }
                prev_pred = Some(pf.predicate);
                let mut seen_args = HashSet::new();
                let mut prev_key: Option<(usize, FrameRole)> = None;
                for a in &pf.args {
                    assert_ne!(a.head, pf.predicate, "{id}: self-arg");
                    assert_eq!(
                        doc.tokens[a.head].sentence, pf.sentence,
                        "{id}: cross-sentence arg"
                    );
                    assert!(
                        !matches!(doc.tokens[a.head].upos, Some(Upos::Adp | Upos::Part)),
                        "{id}: arg head is ADP/PART"
                    );
                    assert!(seen_args.insert((a.role, a.head)), "{id}: duplicate arg");
                    if let Some(pk) = prev_key {
                        assert!(pk <= (a.head, a.role), "{id}: args not sorted");
                    }
                    prev_key = Some((a.head, a.role));
                }
            }

            if let Some(gframes) = v["frames"].as_array() {
                for gf in gframes {
                    let p = gf["predicate"].as_u64().unwrap() as usize - 1;
                    for a in gf["args"].as_array().unwrap() {
                        let role = match a["role"].as_str().unwrap() {
                            "ARG0" => FrameRole::Arg0,
                            "ARG1" => FrameRole::Arg1,
                            "ARGM" => FrameRole::Argm,
                            other => panic!("unexpected role {other}"),
                        };
                        let h = a["head"].as_u64().unwrap() as usize - 1;
                        gold_set.insert((id.clone(), p, role, h));
                    }
                }
            }

            for pf in &doc.frames {
                for a in &pf.args {
                    pred_set.insert((id.clone(), pf.predicate, a.role, a.head));
                }
            }
        }

        let tp = gold_set.intersection(&pred_set).count();
        let p = tp as f64 / pred_set.len() as f64;
        let r = tp as f64 / gold_set.len() as f64;
        let f1 = 2.0 * p * r / (p + r);
        eprintln!(
            "GOLD 300 Frame P={p:.4} R={r:.4} F1={f1:.4} (tp={tp}, pred={}, gold={})",
            pred_set.len(),
            gold_set.len()
        );
        assert!(f1 >= 0.85, "Frame F1 {f1:.4} < 0.85");
    }
}
