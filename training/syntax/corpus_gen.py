#!/usr/bin/env python3
"""Generate a neutral, multi-genre multilingual (zh-TW / zh-CN / en / mixed-TW / mixed-CN) corpus.

Produces ~8,000+ verified items (~7,200 train, ~800 dev) in `training/.work/syntax/` without
committing any corpus files to git. Every item is segmented by the local `nlp_doc` binary
(Layers 1 and 2), projected onto `nlp_doc`'s exact token spans, and checked by `check_annotation`:
  1. `tokens` and `sentences` match `analyze_document` 100%,
  2. `upos` tags are in the 17 UD v2 tags and agree 100% with Layer-2 decided tags,
  3. `heads` and `deprels` form a single-root, cycle-free tree in the 37 UD v2 relation scheme,
  4. Authentic Taiwan (`zh-TW` / `mixed-TW`), Mainland (`zh-CN` / `mixed-CN`), and English (`en`)
     usages across `novel`, `paper`, and `casual` genres,
  5. ZERO texts overlap `training/probe/probe_texts.jsonl`, `training/silver/sample_dryrun.jsonl`,
     or `tests/data/gold_layers4_6_v1.jsonl` (if present).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "training" / "silver"))
from validate_merge import check_annotation  # noqa: E402

# Priority when jieba fuses two sibling modifiers/arguments with the same external head
# (e.g. `他` nsubj + `馬上` advmod -> `他馬上` nsubj; `它` nsubj + `會` aux -> `它會` nsubj).
REL_PRIORITY: dict[str, int] = {
    "root": 100,
    "nsubj": 90,
    "csubj": 89,
    "obj": 88,
    "iobj": 87,
    "obl": 85,
    "ccomp": 84,
    "xcomp": 83,
    "advcl": 82,
    "acl": 81,
    "conj": 80,
    "parataxis": 78,
    "appos": 77,
    "nmod": 75,
    "amod": 72,
    "compound": 70,
    "flat": 69,
    "nummod": 68,
    "det": 65,
    "expl": 64,
    "vocative": 62,
    "discourse": 60,
    "advmod": 50,
    "aux": 45,
    "cop": 44,
    "case": 40,
    "mark": 39,
    "cc": 38,
    "clf": 35,
    "fixed": 30,
    "dep": 20,
    "punct": 10,
}


@dataclass
class Phrase:
    """A subtree with 0-based token list, internal relative heads, and designated root index."""

    tokens: list[str]
    upos: list[str]
    heads: list[int | None]
    deprels: list[str]
    root_idx: int


def word_phrase(
    toks: list[tuple[str, str]], head_pos: int = -1, internal_rel: str = "compound"
) -> Phrase:
    n = len(toks)
    h_idx = n - 1 if head_pos < 0 else head_pos
    heads: list[int | None] = []
    rels: list[str] = []
    for i in range(n):
        if i == h_idx:
            heads.append(None)
            rels.append("root")
        else:
            heads.append(h_idx)
            rels.append(internal_rel)
    return Phrase(
        tokens=[t for t, _ in toks],
        upos=[u for _, u in toks],
        heads=heads,
        deprels=rels,
        root_idx=h_idx,
    )


def W(text: str, upos: str) -> Phrase:
    return word_phrase([(text, upos)], head_pos=0)


def assemble(
    root: Phrase,
    left_children: list[tuple[Phrase, str]],
    right_children: list[tuple[Phrase, str]],
) -> Phrase:
    tokens: list[str] = []
    upos: list[str] = []
    heads: list[int | None] = []
    deprels: list[str] = []
    child_roots: list[int] = []

    for phrase, rel in left_children:
        offset = len(tokens)
        tokens.extend(phrase.tokens)
        upos.extend(phrase.upos)
        for i, h in enumerate(phrase.heads):
            if i == phrase.root_idx:
                heads.append(None)
                deprels.append(rel)
                child_roots.append(offset + i)
            else:
                assert h is not None
                heads.append(offset + h)
                deprels.append(phrase.deprels[i])

    root_offset = len(tokens)
    new_root_idx = root_offset + root.root_idx
    tokens.extend(root.tokens)
    upos.extend(root.upos)
    for i, h in enumerate(root.heads):
        if i == root.root_idx:
            heads.append(None)
            deprels.append("root")
        else:
            assert h is not None
            heads.append(root_offset + h)
            deprels.append(root.deprels[i])

    for phrase, rel in right_children:
        offset = len(tokens)
        tokens.extend(phrase.tokens)
        upos.extend(phrase.upos)
        for i, h in enumerate(phrase.heads):
            if i == phrase.root_idx:
                heads.append(None)
                deprels.append(rel)
                child_roots.append(offset + i)
            else:
                assert h is not None
                heads.append(offset + h)
                deprels.append(phrase.deprels[i])

    for c_idx in child_roots:
        heads[c_idx] = new_root_idx

    return Phrase(
        tokens=tokens, upos=upos, heads=heads, deprels=deprels, root_idx=new_root_idx
    )


def is_cjk_or_fullwidth(s: str) -> bool:
    for ch in s:
        cp = ord(ch)
        if (
            (0x3000 <= cp <= 0x303F)
            or (0x3400 <= cp <= 0x4DBF)
            or (0x4E00 <= cp <= 0x9FFF)
            or (0xF900 <= cp <= 0xFAFF)
            or (0xFE30 <= cp <= 0xFFEF)
            or ch in "「」『』（）《》【】—…"
        ):
            return True
    return False


def render_surface_with_spans(
    tokens: list[str], space_cjk_ascii: bool = True
) -> tuple[str, list[tuple[int, int]]]:
    """Render token sequence to natural surface text and return exact char spans per token."""
    no_space_before = {
        ".",
        ",",
        "?",
        "!",
        ";",
        ":",
        "'s",
        "n't",
        "'ll",
        "'re",
        "'ve",
        "'m",
        "'d",
        ")",
        "]",
        "%",
        "-",
        "/",
    }
    no_space_after = {"(", "[", "$", "US$", "NT$", "¥", "-", "/"}
    cjk_punct_before = {"，", "。", "？", "！", "；", "：", "、", "」", "）", "》", "”"}
    cjk_punct_after = {"，", "。", "？", "！", "；", "：", "、", "「", "（", "《", "“"}

    buf: list[str] = []
    spans: list[tuple[int, int]] = []
    pos = 0
    for i, tok in enumerate(tokens):
        if i > 0:
            prev = tokens[i - 1]
            need_space = True
            if tok in no_space_before or prev in no_space_after:
                need_space = False
            elif tok in cjk_punct_before or prev in cjk_punct_after:
                need_space = False
            elif is_cjk_or_fullwidth(tok) and is_cjk_or_fullwidth(prev):
                need_space = False
            elif is_cjk_or_fullwidth(tok) or is_cjk_or_fullwidth(prev):
                need_space = space_cjk_ascii
            if need_space:
                buf.append(" ")
                pos += 1
        start = pos
        buf.append(tok)
        pos += len(tok)
        spans.append((start, pos))
    return "".join(buf), spans


def finalize_sentences(
    sent_phrases: list[Phrase],
    genre: str,
    lang: str,
    variant: str,
    space_cjk_ascii: bool = True,
) -> dict:
    all_tokens: list[str] = []
    all_spans: list[tuple[int, int]] = []
    all_upos: list[str] = []
    all_heads_1based: list[int] = []
    all_deprels: list[str] = []
    sent_ranges: list[list[int]] = []

    sep = " " if lang == "en" else ""
    parts: list[str] = []
    char_cursor = 0

    for idx_s, sp in enumerate(sent_phrases):
        if idx_s > 0 and sep:
            parts.append(sep)
            char_cursor += len(sep)
        base = len(all_tokens)
        n = len(sp.tokens)
        sent_ranges.append([base, base + n])
        all_tokens.extend(sp.tokens)
        all_upos.extend(sp.upos)
        for i, h in enumerate(sp.heads):
            if i == sp.root_idx:
                all_heads_1based.append(0)
                all_deprels.append("root")
            else:
                assert h is not None
                all_heads_1based.append(base + h + 1)
                all_deprels.append(sp.deprels[i])
        s_txt, s_spans = render_surface_with_spans(sp.tokens, space_cjk_ascii=space_cjk_ascii)
        parts.append(s_txt)
        for cs, ce in s_spans:
            all_spans.append((char_cursor + cs, char_cursor + ce))
        char_cursor += len(s_txt)

    return {
        "genre": genre,
        "lang": lang,
        "variant": variant,
        "text": "".join(parts),
        "tokens": all_tokens,
        "token_spans": all_spans,
        "upos": all_upos,
        "sentences": sent_ranges,
        "heads": all_heads_1based,
        "deprels": all_deprels,
        "frames": [],
        "coref": [],
    }


def project_to_repo_tokens(
    cand: dict,
    repo_toks: list[str],
    repo_spans: list[tuple[int, int]],
    repo_upos: list[str | None],
    repo_sents: list[list[int]],
) -> dict | None:
    """Project atomic gold dependency tree onto `nlp_doc` token spans."""
    if len(repo_sents) != len(cand["sentences"]):
        return None

    # Verify sentence character boundaries align cleanly
    for (gs, ge), (rs, re_) in zip(cand["sentences"], repo_sents):
        if ge <= gs or re_ <= rs:
            return None
        g_start = cand["token_spans"][gs][0]
        g_end = cand["token_spans"][ge - 1][1]
        r_start = repo_spans[rs][0]
        r_end = repo_spans[re_ - 1][1]
        if g_start != r_start or g_end != r_end:
            return None

    # Fast path: 1-to-1 exact token match
    if repo_toks == cand["tokens"] and repo_sents == cand["sentences"]:
        merged_upos = [
            ru if ru is not None else cu for cu, ru in zip(cand["upos"], repo_upos)
        ]
        out = dict(cand)
        out.pop("token_spans", None)
        out["upos"] = merged_upos
        return out

    # General span-based projection: handle jieba multi-word fusions (`他馬上`, `很乾淨`, `花園裡`,
    # `借走`, `在家`) and intra-word splits (`研究` + `員`, `筆記` + `本`, `錯` + `誤率`).
    g_spans: list[tuple[int, int]] = cand["token_spans"]
    g_heads: list[int] = cand["heads"]  # 1-based, 0 = root
    g_rels: list[str] = cand["deprels"]
    g_upos: list[str] = cand["upos"]
    n_g = len(g_spans)
    n_r = len(repo_spans)

    # First expand any atomic token `g` that is cleanly partitioned into k >= 2 repo tokens
    # (e.g., `研究員` -> `研究` + `員`, head at rightmost piece with `compound`/`amod`).
    exp_spans: list[tuple[int, int]] = []
    exp_upos: list[str] = []
    exp_orig_head_g: list[int | None] = []  # 0 = root, positive 1-based g idx, or -1 internal
    exp_internal_target: list[int | None] = []
    exp_rels: list[str] = []
    g_to_head_exp: dict[int, int] = {}

    r_ptr = 0
    for g_idx in range(n_g):
        gs, ge = g_spans[g_idx]
        while r_ptr < n_r and repo_spans[r_ptr][1] <= gs:
            r_ptr += 1
        # Check if repo tokens starting at r_ptr partition [gs, ge)
        sub_r: list[int] = []
        k = r_ptr
        while k < n_r and repo_spans[k][0] >= gs and repo_spans[k][1] <= ge:
            sub_r.append(k)
            k += 1
        if len(sub_r) >= 2 and repo_spans[sub_r[0]][0] == gs and repo_spans[sub_r[-1]][1] == ge:
            # Clean split of atomic token g_idx into sub_r!
            gu = g_upos[g_idx]
            sub_rel = "compound" if gu in ("NOUN", "PROPN", "VERB", "PRON") else ("amod" if gu == "ADJ" else "flat")
            head_local = 0 if gu == "PRON" else (len(sub_r) - 1)
            head_e = len(exp_spans) + head_local
            g_to_head_exp[g_idx + 1] = head_e
            for pos_i, r_i in enumerate(sub_r):
                exp_spans.append(repo_spans[r_i])
                exp_upos.append(repo_upos[r_i] or ("PART" if gu == "PRON" and pos_i > 0 else gu))
                if pos_i == head_local:
                    exp_orig_head_g.append(g_heads[g_idx])
                    exp_internal_target.append(None)
                    exp_rels.append(g_rels[g_idx])
                else:
                    exp_orig_head_g.append(-1)
                    exp_internal_target.append(head_e)
                    exp_rels.append(sub_rel)
        else:
            e_idx = len(exp_spans)
            g_to_head_exp[g_idx + 1] = e_idx
            exp_spans.append((gs, ge))
            exp_upos.append(g_upos[g_idx])
            exp_orig_head_g.append(g_heads[g_idx])
            exp_internal_target.append(None)
            exp_rels.append(g_rels[g_idx])

    n_e = len(exp_spans)
    e_heads: list[int] = []  # 0-based into exp, or -1 for root
    for e_i in range(n_e):
        og = exp_orig_head_g[e_i]
        if og == -1:
            it = exp_internal_target[e_i]
            assert it is not None
            e_heads.append(it)
        elif og == 0:
            e_heads.append(-1)
        else:
            assert og is not None
            e_heads.append(g_to_head_exp[og])

    # Now every expanded token `e_i` must fall completely inside a single `repo_tok` `R(e_i)`!
    e_to_r: list[int] = []
    r_groups: list[list[int]] = [[] for _ in range(n_r)]
    r_idx = 0
    for e_i, (es, ee) in enumerate(exp_spans):
        while r_idx < n_r and repo_spans[r_idx][1] <= es:
            r_idx += 1
        if r_idx >= n_r:
            return None
        rs, re_ = repo_spans[r_idx]
        if es < rs or ee > re_:
            # Cross-boundary split/merge in jieba -> skip cleanly
            return None
        e_to_r.append(r_idx)
        r_groups[r_idx].append(e_i)

    if any(len(grp) == 0 for grp in r_groups):
        return None

    proj_heads: list[int] = []
    proj_rels: list[str] = []
    proj_upos: list[str] = []

    for r_i, grp in enumerate(r_groups):
        ext: list[tuple[int, int, str]] = []  # (e_i, target_r_or_-1, rel)
        for e_i in grp:
            eh = e_heads[e_i]
            t_r = -1 if eh == -1 else e_to_r[eh]
            if t_r != r_i:
                ext.append((e_i, t_r, exp_rels[e_i]))
        if not ext:
            return None
        # All external edges from fused group must attach to the same external head token
        target_set = {t_r for _, t_r, _ in ext}
        if len(target_set) != 1:
            return None
        ext.sort(key=lambda item: REL_PRIORITY.get(item[2], 0), reverse=True)
        best_e, best_target_r, best_rel = ext[0]
        proj_heads.append(0 if best_target_r == -1 else best_target_r + 1)
        proj_rels.append("root" if best_target_r == -1 else best_rel)
        proj_upos.append(repo_upos[r_i] if repo_upos[r_i] is not None else exp_upos[best_e])

    out = dict(cand)
    out.pop("token_spans", None)
    out["tokens"] = repo_toks
    out["sentences"] = repo_sents
    out["upos"] = proj_upos
    out["heads"] = proj_heads
    out["deprels"] = proj_rels
    return out


def batch_nlp_doc(
    binary: Path, texts: list[str], chunk_size: int = 250
) -> list[tuple[list[str], list[tuple[int, int]], list[str | None], list[list[int]]] | None]:
    """Run `nlp_doc` in fast paragraph-separated chunks."""
    results: list[
        tuple[list[str], list[tuple[int, int]], list[str | None], list[list[int]]] | None
    ] = []
    for start in range(0, len(texts), chunk_size):
        sub = texts[start : start + chunk_size]
        joined = "\n\n".join(sub)
        proc = subprocess.run(
            [str(binary)], input=joined.encode("utf-8"), capture_output=True, check=True
        )
        doc = json.loads(proc.stdout.decode("utf-8"))
        paras = doc["segmentation"]["paragraphs"]
        if len(paras) != len(sub):
            for t in sub:
                p1 = subprocess.run(
                    [str(binary)], input=t.encode("utf-8"), capture_output=True, check=True
                )
                d1 = json.loads(p1.stdout.decode("utf-8"))
                toks = d1["tokens"]
                if not toks:
                    results.append(None)
                    continue
                smap: dict[int, list[int]] = {}
                for idx, tk in enumerate(toks):
                    smap.setdefault(tk["sentence"], []).append(idx)
                results.append(
                    (
                        [t[tk["span"]["start_char"] : tk["span"]["end_char"]] for tk in toks],
                        [(tk["span"]["start_char"], tk["span"]["end_char"]) for tk in toks],
                        [tk.get("upos") for tk in toks],
                        [[v[0], v[-1] + 1] for _, v in sorted(smap.items())],
                    )
                )
            continue

        sents = doc["segmentation"]["sentences"]
        toks = doc["tokens"]
        sent_to_para = {si: s["paragraph"] for si, s in enumerate(sents)}
        by_para: dict[int, list[dict]] = {pi: [] for pi in range(len(paras))}
        for tk in toks:
            by_para[sent_to_para[tk["sentence"]]].append(tk)

        for pi, ptoks in sorted(by_para.items()):
            if not ptoks:
                results.append(None)
                continue
            p_base = paras[pi]["span"]["start_char"]
            smap = {}
            for local_i, tk in enumerate(ptoks):
                smap.setdefault(tk["sentence"], []).append(local_i)
            results.append(
                (
                    [joined[tk["span"]["start_char"] : tk["span"]["end_char"]] for tk in ptoks],
                    [
                        (tk["span"]["start_char"] - p_base, tk["span"]["end_char"] - p_base)
                        for tk in ptoks
                    ],
                    [tk.get("upos") for tk in ptoks],
                    [[v[0], v[-1] + 1] for _, v in sorted(smap.items())],
                )
            )
    return results


# ---------------------------------------------------------------------------
# Authentic Taiwan (`zh-TW`) vs Mainland (`zh-CN`) Chinese generators
# ---------------------------------------------------------------------------

ZH_TW_CONFIG = {
    "variant": "zh-TW",
    "prons": ["我", "你", "他", "她", "我們", "他們", "她們", "牠"],
    "obj_prons": ["它", "它們", "他", "她"],
    "dets": ["這個", "那個", "這些", "那些", "每", "各", "本"],
    "clfs": ["個", "組", "份", "項", "次", "台", "本", "張"],
    "nums": ["兩", "三", "四", "五", "十", "2", "3", "5", "8"],
    "numclfs": ["一個", "兩個", "三個", "一本", "一張", "兩組", "三組", "一套", "兩份", "一台", "一封"],
    "de": "的",
    "le": "了",
    "guo": "過",
    "zhe": "著",
    "ma": "嗎",
    "ba": ["把", "將"],
    "bei": "被",
    "shi": "是",
    "you": "有",
    "bu": "不",
    "mei": "沒有",
    "preps": ["在", "從", "透過", "根據", "按照", "用", "對", "為了"],
    "bi": "比",
    "geng": "更",
    "hen": ["很", "非常", "十分", "相當", "太"],
    "advs": ["慢慢", "仔細", "自動", "順利", "準時", "立刻", "重新", "持續", "已經", "馬上", "剛"],
    "sconjs": ["因為", "如果", "雖然", "只要", "當", "等到"],
    "corrs": ["就", "才", "都", "還是", "卻"],
    "modals": ["會", "能", "可以", "應該", "必須"],
    "cconjs_np": ["與", "和", "或"],
    "cconjs_vp": ["並且", "而且"],
    "conns_adv": ["接著", "隨後", "同時", "此外", "因此"],
    "conns_cc": ["然而", "不過", "可是"],
    "intjs": ["哇", "唉", "咦"],
    "disc_parts": ["吧", "呢", "喔", "啊"],
    "locs_post": ["裡", "上", "下", "中", "內", "外", "旁", "之前", "之後"],
    "times": ["今天", "昨天", "明天", "早上", "下午", "傍晚", "週末", "清晨", "深夜", "剛才"],
    "v_ctrl": ["決定", "開始", "希望", "計畫", "準備", "嘗試", "繼續"],
    "v_comp": ["發現", "認為", "表示", "確認", "記得", "證明", "覺得", "顯示"],
    "v_give": ["傳給", "借給", "交給", "還給", "寄給"],
    "v_dest": ["放到", "搬到", "送到", "移到"],
    "vi_list": ["出發", "抵達", "離開", "休息", "散步", "運作", "收斂", "下降", "上升", "停止", "出現", "發車"],
    "rvc_v": ["放", "搬", "寫", "排", "改", "存", "包", "鎖", "讀", "釘", "刷", "擦", "煮"],
    "rvc_c": [("好", "VERB"), ("完", "VERB"), ("走", "VERB"), ("開", "VERB"), ("清楚", "ADJ"), ("乾淨", "ADJ"), ("出來", "VERB")],
    "nn_compounds": [
        ("研究", "團隊"), ("便利", "商店"), ("行動", "電源"), ("隨身", "碟"), ("鐵", "門"),
        ("木", "橋"), ("竹", "籃"), ("花", "盆"), ("儲物", "櫃"), ("歷史", "書"),
        ("公車", "站"), ("實驗", "數據"), ("模型", "架構"), ("訓練", "樣本"), ("玻璃", "窗"),
    ],
    "adjs": ["安靜", "乾淨", "穩定", "準確", "清楚", "簡單", "複雜", "溫暖", "涼爽", "明亮", "完整", "快速", "新鮮", "方便", "輕巧", "重要", "有效", "沉重", "濕滑"],
    "genres": {
        "novel": {
            "subjs": ["木匠", "畫家", "船長", "園丁", "旅人", "鄰居", "老師傅", "作者", "阿哲", "小芸", "志明", "雅婷"],
            "objs": ["木門", "窗戶", "燈籠", "信封", "地圖", "筆記本", "行李箱", "外套", "字典", "木箱", "雨傘", "茶壺"],
            "locs": ["港口", "花園", "院子", "走廊", "書房", "車站", "巷口", "月台"],
            "vts": ["打開", "關上", "整理", "修理", "檢查", "挑選", "擦拭", "搬動", "觀察", "收好"],
            "propns": ["臺北", "臺南", "花蓮", "高雄"],
        },
        "paper": {
            "subjs": ["研究員", "工程師", "團隊", "系統", "演算法", "模型", "伺服器", "受試者"],
            "objs": ["軟體", "資料庫", "記憶體", "網路", "參數", "樣本", "訊號", "曲線", "矩陣", "向量", "誤差", "結果", "報告", "圖表", "檔案", "模組", "介面", "專案", "硬碟"],
            "locs": ["實驗室", "伺服器", "記憶體", "網路", "資料庫", "硬碟"],
            "vts": ["記錄", "測量", "分析", "比較", "更新", "下載", "儲存", "載入", "讀取", "訓練", "測試", "評估", "完成", "調整", "設計", "優化", "驗證", "列印"],
            "propns": ["臺灣", "臺北"],
        },
        "casual": {
            "subjs": ["老師", "學生", "店員", "計程車司機", "廚師", "姐姐", "弟弟", "哥哥", "朋友", "怡君", "冠宇", "小芸"],
            "objs": ["腳踏車", "便當", "影片", "滑鼠", "螢幕", "水壺", "茶杯", "麵包", "咖啡", "沙拉", "車票", "包裹", "相機", "吉他", "悠遊卡"],
            "locs": ["捷運站", "月台", "超商", "圖書館", "教室", "廚房", "公園", "書店", "市場"],
            "vts": ["購買", "準備", "分享", "收到", "寄出", "借走", "歸還", "整理", "挑選", "下載", "列印"],
            "propns": ["臺北", "臺中", "高雄"],
        },
    },
}

ZH_CN_CONFIG = {
    "variant": "zh-CN",
    "prons": ["我", "你", "他", "她", "我们", "他们", "她们", "它"],
    "obj_prons": ["它", "它们", "他", "她"],
    "dets": ["这个", "那个", "这些", "那些", "每", "各", "本", "这", "那"],
    "clfs": ["个", "组", "份", "项", "次", "台", "本", "张", "套"],
    "nums": ["两", "三", "四", "五", "十", "2", "3", "5", "8"],
    "numclfs": ["一个", "两个", "三个", "一本", "一张", "两组", "三组", "一套", "两份", "一台", "一封"],
    "de": "的",
    "le": "了",
    "guo": "过",
    "zhe": "着",
    "ma": "吗",
    "ba": ["把", "将"],
    "bei": "被",
    "shi": "是",
    "you": "有",
    "bu": "不",
    "mei": "没有",
    "preps": ["在", "从", "通过", "根据", "按照", "用", "对", "为了", "跟"],
    "bi": "比",
    "geng": "更",
    "hen": ["很", "非常", "十分", "相当", "太"],
    "advs": ["慢慢", "仔细", "自动", "顺利", "准时", "立刻", "重新", "持续", "已经", "马上", "刚"],
    "sconjs": ["因为", "如果", "虽然", "只要", "当", "等到"],
    "corrs": ["就", "才", "都", "还是", "却"],
    "modals": ["会", "能", "可以", "应该", "必须"],
    "cconjs_np": ["和", "与", "或"],
    "cconjs_vp": ["并且", "而且"],
    "conns_adv": ["接着", "随后", "同时", "此外", "因此"],
    "conns_cc": ["然而", "不过", "可是"],
    "intjs": ["哇", "唉", "咦"],
    "disc_parts": ["吧", "呢", "啊", "呀"],
    "locs_post": ["里", "上", "下", "中", "内", "外", "旁", "之前", "之后"],
    "times": ["今天", "昨天", "明天", "早上", "下午", "傍晚", "周末", "清晨", "深夜", "刚才"],
    "v_ctrl": ["决定", "开始", "希望", "计划", "准备", "尝试", "继续"],
    "v_comp": ["发现", "认为", "表示", "确认", "记得", "证明", "觉得", "显示"],
    "v_give": ["传给", "借给", "交给", "还给", "寄给"],
    "v_dest": ["放到", "搬到", "发到", "送到"],
    "vi_list": ["出发", "到达", "离开", "休息", "散步", "运行", "收敛", "下降", "上升", "停止", "出现", "发车"],
    "rvc_v": ["放", "搬", "写", "排", "改", "存", "包", "锁", "读", "修", "刷", "擦", "煮"],
    "rvc_c": [("好", "VERB"), ("完", "VERB"), ("走", "VERB"), ("开", "VERB"), ("清楚", "ADJ"), ("干净", "ADJ"), ("出来", "VERB")],
    "nn_compounds": [
        ("研究", "团队"), ("数据", "集"), ("充电", "宝"), ("显示", "器"), ("铁", "门"),
        ("木", "桥"), ("竹", "篮"), ("花", "盆"), ("快递", "驿站"), ("历史", "书"),
        ("公交", "站"), ("实验", "结果"), ("模型", "结构"), ("训练", "样本"), ("玻璃", "窗"),
    ],
    "adjs": ["安静", "干净", "稳定", "准确", "清楚", "简单", "复杂", "温暖", "凉爽", "明亮", "完整", "快速", "新鲜", "方便", "轻巧", "重要", "有效", "沉重", "显着"],
    "genres": {
        "novel": {
            "subjs": ["木匠", "画家", "船长", "园丁", "旅人", "邻居", "老师傅", "作者", "老张", "小王", "林芳", "陈静"],
            "objs": ["木门", "窗户", "灯笼", "信封", "地图", "笔记本", "行李箱", "外套", "词典", "木箱", "雨伞", "茶壶"],
            "locs": ["港口", "花园", "院子", "走廊", "书房", "车站", "街口", "站台"],
            "vts": ["打开", "关上", "整理", "修理", "检查", "挑选", "擦拭", "搬动", "观察", "收好"],
            "propns": ["北京", "上海", "杭州", "成都"],
        },
        "paper": {
            "subjs": ["研究员", "工程师", "团队", "系统", "算法", "模型", "服务器", "被试"],
            "objs": ["软件", "数据库", "内存", "网络", "参数", "样本", "信号", "曲线", "矩阵", "向量", "误差", "结果", "报告", "图表", "文件", "模块", "界面", "项目", "代码", "硬盘"],
            "locs": ["实验室", "服务器", "内存", "网络", "数据库", "硬盘", "文件夹"],
            "vts": ["记录", "测量", "分析", "比较", "更新", "下载", "保存", "加载", "读取", "训练", "测试", "评估", "完成", "调整", "设计", "调优", "部署", "打印"],
            "propns": ["北京", "上海", "深圳"],
        },
        "casual": {
            "subjs": ["老师", "学生", "店员", "出租车司机", "厨师", "姐姐", "弟弟", "哥哥", "朋友", "李明", "张伟", "小王"],
            "objs": ["自行车", "外卖", "快递", "视频", "鼠标", "屏幕", "水壶", "茶杯", "面包", "咖啡", "沙拉", "车票", "相机", "吉他", "充电宝"],
            "locs": ["地铁站", "公交站", "超市", "小区", "图书馆", "教室", "厨房", "公园", "书店"],
            "vts": ["购买", "准备", "分享", "收到", "寄出", "借走", "归还", "整理", "挑选", "下载", "打印"],
            "propns": ["北京", "杭州", "广州", "深圳"],
        },
    },
}


def build_zh_variant_sentences(
    rng: random.Random, cfg: dict, target_count: int
) -> list[dict]:
    variant = cfg["variant"]
    de = cfg["de"]
    le = cfg["le"]
    guo = cfg["guo"]
    zhe = cfg["zhe"]
    ma = cfg["ma"]
    prons = [W(p, "PRON") for p in cfg["prons"]]

    def w_noun_or_propn(s: str, propn_set: set[str]) -> Phrase:
        return W(s, "PROPN" if s in propn_set else "NOUN")

    def make_np(pool: list[str], propn_set: set[str], allow_mod: bool = True) -> Phrase:
        if allow_mod and rng.random() < 0.16:
            c1, c2 = rng.choice(cfg["nn_compounds"])
            head = assemble(W(c2, "NOUN"), [(W(c1, "NOUN"), "compound")], [])
        else:
            s = rng.choice(pool)
            if s in propn_set:
                return W(s, "PROPN")
            head = W(s, "NOUN")
        left: list[tuple[Phrase, str]] = []
        r = rng.random()
        if allow_mod and r < 0.28:
            d = rng.choice(cfg["dets"])
            if d in ("這", "那", "这", "那"):
                left.append((W(d, "DET"), "det"))
                left.append((W(rng.choice(cfg["clfs"]), "NOUN"), "clf"))
            else:
                left.append((W(d, "DET"), "det"))
        elif allow_mod and r < 0.54:
            if rng.random() < 0.5:
                left.append((W(rng.choice(cfg["numclfs"]), "NUM"), "nummod"))
            else:
                left.append((W(rng.choice(cfg["nums"]), "NUM"), "nummod"))
                left.append((W(rng.choice(cfg["clfs"]), "NOUN"), "clf"))
        elif allow_mod and r < 0.78:
            if rng.random() < 0.28:
                # Circumfixal/postpositional adnominal locative: 花園 裡 的 木箱
                lm = W(rng.choice(cfg["genres"]["novel"]["locs"]), "NOUN")
                post = W(rng.choice(cfg["locs_post"][:4]), "ADP")
                nmod_p = assemble(lm, [], [(post, "case"), (W(de, "PART"), "case")])
            else:
                owner = rng.choice(prons) if rng.random() < 0.55 else W(rng.choice(pool), "PROPN" if pool[0] in propn_set else "NOUN")
                if owner.tokens[0] in propn_set:
                    owner = W(owner.tokens[0], "PROPN")
                nmod_p = assemble(owner, [], [(W(de, "PART"), "case")])
            left.append((nmod_p, "nmod"))
        if allow_mod and rng.random() < 0.35:
            adj = W(rng.choice(cfg["adjs"]), "ADJ")
            if rng.random() < 0.65:
                left.append((assemble(adj, [], [(W(de, "PART"), "mark")]), "amod"))
            else:
                left.append((adj, "amod"))
        return assemble(head, left, [])

    items: list[dict] = []
    genre_keys = ["novel", "paper", "casual"]

    while len(items) < target_count:
        genre = genre_keys[len(items) % 3]
        gcfg = cfg["genres"][genre]
        propn_set = set(gcfg["propns"]) | {"阿哲", "小芸", "志明", "雅婷", "怡君", "冠宇", "老张", "小王", "李明", "陈静", "张伟", "林芳"}
        subjs = gcfg["subjs"]
        objs = gcfg["objs"]
        locs = gcfg["locs"]
        vts = gcfg["vts"]

        pattern = rng.randrange(21)
        subj = rng.choice(prons) if rng.random() < 0.30 else make_np(subjs, propn_set, allow_mod=(rng.random() < 0.45))
        obj = make_np(objs, propn_set)
        loc = w_noun_or_propn(rng.choice(locs + gcfg["propns"]), propn_set)

        if pattern == 0:
            # Active SVO + time + adv + aspect 了/過/著
            v = W(rng.choice(vts), "VERB")
            left = [(subj, "nsubj")]
            if rng.random() < 0.5:
                left.append((W(rng.choice(cfg["times"]), "NOUN"), "obl"))
            if rng.random() < 0.45:
                left.append((W(rng.choice(cfg["advs"]), "ADV"), "advmod"))
            right: list[tuple[Phrase, str]] = []
            asp = rng.choice([le, guo, zhe, None])
            if asp is not None:
                right.append((W(asp, "PART"), "aux"))
            right.append((obj, "obj"))
            right.append((W("。", "PUNCT"), "punct"))
            items.append(finalize_sentences([assemble(v, left, right)], genre, "zh", variant))

        elif pattern == 1:
            # 把 / 將 disposal + V-V or V-Adj resultative compound
            v = W(rng.choice(cfg["rvc_v"]), "VERB")
            rc_txt, rc_pos = rng.choice(cfg["rvc_c"])
            rvc = W(rc_txt, rc_pos)
            ba = W(rng.choice(cfg["ba"]), "ADP")
            ba_np = assemble(make_np(objs, propn_set), [(ba, "case")], [])
            left = [(subj, "nsubj")]
            if rng.random() < 0.45:
                left.append((W(rng.choice(cfg["advs"]), "ADV"), "advmod"))
            left.append((ba_np, "obl"))
            right = [(rvc, "compound"), (W(le, "PART"), "aux"), (W("。", "PUNCT"), "punct")]
            items.append(finalize_sentences([assemble(v, left, right)], genre, "zh", variant))

        elif pattern == 2:
            # 被 passive
            pat_np = make_np(objs, propn_set)
            v = W(rng.choice(vts), "VERB")
            left = [(pat_np, "nsubj")]
            if rng.random() < 0.45:
                left.append((W(rng.choice(cfg["advs"]), "ADV"), "advmod"))
            if rng.random() < 0.55:
                left.append((W(cfg["bei"], "ADP"), "aux"))
            else:
                ag_w = rng.choice(subjs)
                ag = assemble(w_noun_or_propn(ag_w, propn_set), [(W(cfg["bei"], "ADP"), "case")], [])
                left.append((ag, "obl"))
            right = [(W(le, "PART"), "aux"), (W("。", "PUNCT"), "punct")] if rng.random() < 0.6 else [(W("。", "PUNCT"), "punct")]
            items.append(finalize_sentences([assemble(v, left, right)], genre, "zh", variant))

        elif pattern == 3:
            # Prepositional / circumfixal locative phrase (在 NP 上/裡/之前, 從/从, 透過/通過)
            prep_word = rng.choice(cfg["preps"])
            if prep_word in ("在", "從", "从"):
                r_post = [(W(rng.choice(cfg["locs_post"]), "ADP"), "case")] if rng.random() < 0.7 else []
                pp = assemble(loc, [(W(prep_word, "ADP"), "case")], r_post)
            else:
                pp = assemble(make_np(objs, propn_set, allow_mod=False), [(W(prep_word, "ADP"), "case")], [])
            v = W(rng.choice(vts), "VERB")
            left = [(subj, "nsubj"), (pp, "obl")]
            if rng.random() < 0.4:
                left.append((W(rng.choice(cfg["corrs"]), "ADV"), "advmod"))
            right = [(W(le, "PART"), "aux"), (obj, "obj"), (W("。", "PUNCT"), "punct")]
            items.append(finalize_sentences([assemble(v, left, right)], genre, "zh", variant))

        elif pattern == 4:
            # Adjectival predicate or 比 comparative
            adj = W(rng.choice(cfg["adjs"]), "ADJ")
            target_np = make_np(objs, propn_set)
            left = [(target_np, "nsubj")]
            if rng.random() < 0.55:
                left.append((W(rng.choice(cfg["hen"]), "ADV"), "advmod"))
            else:
                std_np = assemble(make_np(objs, propn_set, allow_mod=False), [(W(cfg["bi"], "ADP"), "case")], [])
                left.append((std_np, "obl"))
                left.append((W(cfg["geng"], "ADV"), "advmod"))
            items.append(finalize_sentences([assemble(adj, left, [(W("。", "PUNCT"), "punct")])], genre, "zh", variant))

        elif pattern == 5:
            # Copula 是 (+ optional negation 不)
            pred = make_np(objs if genre == "paper" else subjs, propn_set)
            left = [(make_np(objs, propn_set, allow_mod=False) if genre == "paper" else subj, "nsubj")]
            if rng.random() < 0.35:
                left.append((W(cfg["bu"], "ADV"), "advmod"))
            left.append((W(cfg["shi"], "AUX"), "cop"))
            items.append(finalize_sentences([assemble(pred, left, [(W("。", "PUNCT"), "punct")])], genre, "zh", variant))

        elif pattern == 6:
            # Relative clause acl + mark 的: [VP 的 N]
            rc_v = W(rng.choice(vts), "VERB")
            rc_ag = rng.choice(prons) if rng.random() < 0.55 else w_noun_or_propn(rng.choice(subjs), propn_set)
            rc_vp = assemble(rc_v, [(rc_ag, "nsubj")], [(W(de, "PART"), "mark")])
            rel_np = assemble(W(rng.choice(objs), "NOUN"), [(rc_vp, "acl")], [])
            main_v = W(rng.choice(vts), "VERB")
            s1 = assemble(main_v, [(subj, "nsubj")], [(W(le, "PART"), "aux"), (rel_np, "obj"), (W("。", "PUNCT"), "punct")])
            items.append(finalize_sentences([s1], genre, "zh", variant))

        elif pattern == 7:
            # Subordinate advcl + main clause: 因為/因为/雖然/虽然/如果 S1 ， S2 。
            sconj = W(rng.choice(cfg["sconjs"]), "SCONJ")
            v1 = W(rng.choice(cfg["vi_list"]), "VERB")
            c1 = assemble(v1, [(sconj, "mark"), (make_np(objs, propn_set, allow_mod=False), "nsubj")], [])
            v2 = W(rng.choice(vts), "VERB")
            corr = W(rng.choice(cfg["corrs"]), "ADV")
            s1 = assemble(
                v2,
                [(c1, "advcl"), (W("，", "PUNCT"), "punct"), (subj, "nsubj"), (corr, "advmod")],
                [(obj, "obj"), (W("。", "PUNCT"), "punct")],
            )
            items.append(finalize_sentences([s1], genre, "zh", variant))

        elif pattern == 8:
            # Modal AUX + xcomp or modal + adv + V
            aux = W(rng.choice(cfg["modals"]), "AUX")
            if rng.random() < 0.5:
                vc = W(rng.choice(cfg["v_ctrl"]), "VERB")
                v2 = assemble(W(rng.choice(vts), "VERB"), [], [(obj, "obj")])
                s1 = assemble(vc, [(subj, "nsubj"), (aux, "aux")], [(v2, "xcomp"), (W("。", "PUNCT"), "punct")])
            else:
                v = W(rng.choice(vts), "VERB")
                s1 = assemble(v, [(subj, "nsubj"), (aux, "aux"), (W(rng.choice(cfg["advs"]), "ADV"), "advmod")], [(obj, "obj"), (W("。", "PUNCT"), "punct")])
            items.append(finalize_sentences([s1], genre, "zh", variant))

        elif pattern == 9:
            # Clausal complement ccomp
            vm = W(rng.choice(cfg["v_comp"]), "VERB")
            inner_pred = W(rng.choice(cfg["adjs"]), "ADJ")
            inner_cl = assemble(inner_pred, [(make_np(objs, propn_set, allow_mod=False), "nsubj"), (W(rng.choice(cfg["hen"]), "ADV"), "advmod")], [])
            s1 = assemble(vm, [(subj, "nsubj")], [(inner_cl, "ccomp"), (W("。", "PUNCT"), "punct")])
            items.append(finalize_sentences([s1], genre, "zh", variant))

        elif pattern == 10:
            # Coordination: NP 與/和 NP or VP 並且/而且 VP
            if rng.random() < 0.5:
                np1 = W(rng.choice(objs), "NOUN")
                np2 = assemble(W(rng.choice(objs), "NOUN"), [(W(rng.choice(cfg["cconjs_np"]), "CCONJ"), "cc")], [])
                coord_obj = assemble(np1, [], [(np2, "conj")])
                v = W(rng.choice(vts), "VERB")
                s1 = assemble(v, [(subj, "nsubj")], [(coord_obj, "obj"), (W("。", "PUNCT"), "punct")])
            else:
                v1 = W(rng.choice(vts), "VERB")
                v2 = assemble(W(rng.choice(vts), "VERB"), [(W(rng.choice(cfg["cconjs_vp"]), "CCONJ"), "cc")], [(make_np(objs, propn_set, allow_mod=False), "obj")])
                s1 = assemble(v1, [(subj, "nsubj")], [(make_np(objs, propn_set, allow_mod=False), "obj"), (v2, "conj"), (W("。", "PUNCT"), "punct")])
            items.append(finalize_sentences([s1], genre, "zh", variant))

        elif pattern == 11:
            # Existential 有 / 放著: LOC + 裡/里/上/下(ADP case) + 有/放著 + NP
            loc_p = assemble(W(rng.choice(locs), "NOUN"), [], [(W(rng.choice(cfg["locs_post"][:4]), "ADP"), "case")])
            if rng.random() < 0.6:
                s1 = assemble(W(cfg["you"], "VERB"), [(loc_p, "obl")], [(make_np(objs, propn_set), "obj"), (W("。", "PUNCT"), "punct")])
            else:
                s1 = assemble(W("放", "VERB"), [(loc_p, "obl")], [(W(zhe, "PART"), "aux"), (make_np(objs, propn_set), "obj"), (W("。", "PUNCT"), "punct")])
            items.append(finalize_sentences([s1], genre, "zh", variant))

        elif pattern == 12:
            # Yes-no question with 嗎/吗 discourse + ？
            v = W(rng.choice(vts), "VERB")
            q_subj = W("你", "PRON") if genre == "casual" else subj
            s1 = assemble(
                v,
                [(q_subj, "nsubj"), (W(rng.choice(cfg["times"]), "NOUN"), "obl")],
                [(obj, "obj"), (W(le, "PART"), "aux"), (W(ma, "PART"), "discourse"), (W("？", "PUNCT"), "punct")],
            )
            items.append(finalize_sentences([s1], genre, "zh", variant))

        elif pattern == 13:
            # Pivotal / causative 讓/让 + NP + V + obj
            rang = W("讓" if variant == "zh-TW" else "让", "VERB")
            pivot = w_noun_or_propn(rng.choice(subjs), propn_set)
            v2 = assemble(W(rng.choice(vts), "VERB"), [], [(make_np(objs, propn_set, allow_mod=False), "obj")])
            s1 = assemble(rang, [(subj, "nsubj")], [(pivot, "obj"), (v2, "xcomp"), (W("。", "PUNCT"), "punct")])
            items.append(finalize_sentences([s1], genre, "zh", variant))

        elif pattern == 14:
            # Postverbal locative: V + 在 + LOC + 裡/上(ADP case) OR 放到/搬到 + LOC(obl)
            if rng.random() < 0.55:
                v = W(rng.choice(["放", "停", "坐", "住", "留"]), "VERB")
                pp = assemble(W(rng.choice(locs), "NOUN"), [(W("在", "ADP"), "case")], [(W(rng.choice(cfg["locs_post"][:4]), "ADP"), "case")])
                s1 = assemble(v, [(subj, "nsubj")], [(pp, "obl"), (W("。", "PUNCT"), "punct")])
            else:
                v = W(rng.choice(cfg["v_dest"]), "VERB")
                ba_np = assemble(make_np(objs, propn_set, allow_mod=False), [(W(rng.choice(cfg["ba"]), "ADP"), "case")], [])
                dest_np = assemble(W(rng.choice(locs), "NOUN"), [], [(W(rng.choice(cfg["locs_post"][:3]), "ADP"), "case")])
                s1 = assemble(v, [(subj, "nsubj"), (ba_np, "obl")], [(dest_np, "obl"), (W(le, "PART"), "aux"), (W("。", "PUNCT"), "punct")])
            items.append(finalize_sentences([s1], genre, "zh", variant))

        elif pattern == 15:
            # Comma-joined independent clauses (`parataxis`), with or without pro-drop in S2
            v1 = W(rng.choice(vts), "VERB")
            if rng.random() < 0.5:
                s2_left: list[tuple[Phrase, str]] = [(W("，", "PUNCT"), "punct"), (rng.choice(prons), "nsubj")]
            else:
                s2_left = [(W("，", "PUNCT"), "punct"), (W(rng.choice(cfg["advs"]), "ADV"), "advmod")]
            c2 = assemble(W(rng.choice(vts), "VERB"), s2_left, [(make_np(objs, propn_set, allow_mod=False), "obj")])
            s1 = assemble(v1, [(subj, "nsubj")], [(W(le, "PART"), "aux"), (obj, "obj"), (c2, "parataxis"), (W("。", "PUNCT"), "punct")])
            items.append(finalize_sentences([s1], genre, "zh", variant))

        elif pattern == 16:
            # Serial verb construction (`VP1 + VP2` attached as `conj`, matching p11)
            v1 = W(rng.choice(vts), "VERB")
            v2 = assemble(W(rng.choice(vts), "VERB"), [], [(make_np(objs, propn_set, allow_mod=False), "obj")])
            s1 = assemble(v1, [(subj, "nsubj")], [(make_np(objs, propn_set, allow_mod=False), "obj"), (v2, "conj"), (W("。", "PUNCT"), "punct")])
            items.append(finalize_sentences([s1], genre, "zh", variant))

        elif pattern == 17:
            # Ditransitive `iobj` (`傳給/借給/還給 + iobj + obj` or `把 + obj + 傳給 + iobj`)
            vg = W(rng.choice(cfg["v_give"]), "VERB")
            rec = rng.choice(prons) if rng.random() < 0.65 else w_noun_or_propn(rng.choice(subjs), propn_set)
            if rng.random() < 0.55:
                s1 = assemble(vg, [(subj, "nsubj")], [(rec, "iobj"), (make_np(objs, propn_set), "obj"), (W("。", "PUNCT"), "punct")])
            else:
                ba_np = assemble(make_np(objs, propn_set), [(W(rng.choice(cfg["ba"]), "ADP"), "case")], [])
                s1 = assemble(vg, [(subj, "nsubj"), (ba_np, "obl")], [(rec, "iobj"), (W(le, "PART"), "aux"), (W("。", "PUNCT"), "punct")])
            items.append(finalize_sentences([s1], genre, "zh", variant))

        elif pattern == 18:
            # Interjection + modal `discourse` OR `得` degree complement (`advcl` + `mark`)
            if rng.random() < 0.55:
                intj = W(rng.choice(cfg["intjs"]), "INTJ")
                adj = W(rng.choice(cfg["adjs"]), "ADJ")
                dp = W(rng.choice(cfg["disc_parts"]), "PART")
                s1 = assemble(
                    adj,
                    [(intj, "discourse"), (W("，", "PUNCT"), "punct"), (make_np(objs, propn_set), "nsubj"), (W(rng.choice(cfg["hen"]), "ADV"), "advmod")],
                    [(dp, "discourse"), (W("！", "PUNCT"), "punct")],
                )
            else:
                v = W(rng.choice(["寫", "整理", "分析", "準備"] if variant == "zh-TW" else ["写", "整理", "分析", "准备"]), "VERB")
                deg = assemble(W(rng.choice(cfg["adjs"]), "ADJ"), [(W("得", "PART"), "mark"), (W(rng.choice(cfg["hen"]), "ADV"), "advmod")], [])
                s1 = assemble(v, [(subj, "nsubj")], [(deg, "advcl"), (W("。", "PUNCT"), "punct")])
            items.append(finalize_sentences([s1], genre, "zh", variant))

        elif pattern == 19:
            # Inline backtick / citation `X` apposition (`參數 `batch_size`` / `定理 [12]`)
            code_tok = rng.choice(["`batch_size`", "`max_len`", "`lr`", "[12]", "[3]"])
            head_n = W(rng.choice(["參數", "變數", "定理", "報告"] if variant == "zh-TW" else ["参数", "变量", "定理", "报告"]), "NOUN")
            app_np = assemble(head_n, [], [(W(code_tok, "X"), "appos")])
            v = W(rng.choice(vts), "VERB")
            s1 = assemble(v, [(subj, "nsubj")], [(app_np, "obj"), (W("。", "PUNCT"), "punct")])
            items.append(finalize_sentences([s1], genre, "zh", variant))

        else:
            # Two-sentence item with anaphoric pronoun + connector
            v1 = W(rng.choice(vts), "VERB")
            sent1 = assemble(v1, [(subj, "nsubj")], [(W(le, "PART"), "aux"), (obj, "obj"), (W("。", "PUNCT"), "punct")])
            use_adv = rng.random() < 0.5
            conn = W(rng.choice(cfg["conns_adv"]), "ADV") if use_adv else W(rng.choice(cfg["conns_cc"]), "CCONJ")
            rel_c = "advmod" if use_adv else "cc"
            v2 = W(rng.choice(cfg["vi_list"]), "VERB")
            anaph = W(rng.choice(cfg["obj_prons"]), "PRON")
            sent2 = assemble(
                v2,
                [(conn, rel_c), (W("，", "PUNCT"), "punct"), (anaph, "nsubj"), (W(rng.choice(cfg["advs"]), "ADV"), "advmod")],
                [(W("。", "PUNCT"), "punct")],
            )
            items.append(finalize_sentences([sent1, sent2], genre, "zh", variant))

    return items


# ---------------------------------------------------------------------------
# English (`en`) generator
# ---------------------------------------------------------------------------


def build_en_sentences(rng: random.Random, target_count: int) -> list[dict]:
    dets = ["the", "a", "this", "that", "every", "each", "no"]
    poss_prons = ["my", "your", "his", "her", "its", "our", "their"]
    propns = [
        ("Alice", None), ("Liam", None), ("Maya", None), ("Owen", None),
        ("Elena", None), ("Marcus", None), ("Dr.", "Lin"), ("Prof.", "Chen"),
    ]

    nouns_novel = ["sailor", "painter", "gardener", "traveler", "carpenter", "captain", "archivist", "lantern", "cabin", "bridge", "letter", "journal", "compass", "harbor", "meadow", "staircase", "blanket", "kettle"]
    nouns_paper = ["algorithm", "encoder", "parser", "dataset", "classifier", "sensor", "module", "pipeline", "buffer", "matrix", "gradient", "latency", "variance", "threshold", "baseline", "equation", "protocol", "network"]
    nouns_casual = ["barista", "neighbor", "cousin", "driver", "baker", "laptop", "umbrella", "backpack", "scooter", "sandwich", "playlist", "calendar", "printer", "battery", "charger", "sweater", "receipt", "ticket"]

    compounds = [
        ("weather", "station"), ("train", "platform"), ("coffee", "mug"), ("stone", "wall"),
        ("glass", "jar"), ("data", "stream"), ("memory", "cache"), ("test", "suite"),
        ("room", "temperature"), ("heat", "pump"), ("bus", "stop"), ("book", "shelf"),
    ]

    adjs = ["quiet", "clean", "stable", "accurate", "simple", "compact", "bright", "warm", "heavy", "narrow", "fresh", "reliable", "fast", "wooden", "modern", "clear", "empty", "useful", "new", "old", "second", "small"]
    advs = ["slowly", "carefully", "quickly", "quietly", "clearly", "directly", "smoothly", "often", "already", "always", "rarely", "easily", "once", "twice", "again"]

    vt_past = ["opened", "closed", "checked", "repaired", "packed", "carried", "stored", "measured", "recorded", "updated", "loaded", "tested", "compared", "cleaned", "sorted", "shared", "printed", "reviewed", "verified", "collected"]
    vt_base = ["open", "close", "check", "repair", "pack", "carry", "store", "measure", "record", "update", "load", "test", "compare", "clean", "sort", "share", "print", "review", "verify", "collect"]
    vt_3sg = ["opens", "closes", "checks", "repairs", "packs", "carries", "stores", "measures", "records", "updates", "loads", "tests", "compares", "cleans", "sorts", "shares", "prints", "reviews", "verifies", "collects"]
    vi_past = ["arrived", "departed", "waited", "rested", "paused", "converged", "dropped", "shifted", "stopped", "worked"]

    preps = ["in", "on", "under", "near", "behind", "across", "inside", "from", "with", "during", "before", "after", "without", "at"]
    nums = ["two", "three", "four", "five", "six", "ten", "twelve", "3", "8", "15", "40"]

    def make_time_obl(cap: bool = False) -> Phrase:
        kind = rng.randrange(4)
        if kind == 0:
            w = rng.choice(["today", "yesterday", "tonight", "tomorrow"])
            return W(w.capitalize() if cap else w, "NOUN")
        if kind == 1:
            d = "This" if cap else rng.choice(["this", "every", "each"])
            n = rng.choice(["morning", "afternoon", "evening", "week"])
            return assemble(W(n, "NOUN"), [(W(d, "DET"), "det")], [])
        if kind == 2:
            a = "Last" if cap else "last"
            n = rng.choice(["night", "week", "month", "summer"])
            return assemble(W(n, "NOUN"), [(W(a, "ADJ"), "amod")], [])
        p = "At" if cap else rng.choice(["at", "before", "after"])
        n = rng.choice(["dawn", "noon", "dusk", "midnight", "sunrise"])
        return assemble(W(n, "NOUN"), [(W(p, "ADP"), "case")], [])

    def make_en_np(pool: list[str], cap_first: bool = False, allow_pp: bool = False) -> Phrase:
        r = rng.random()
        left: list[tuple[Phrase, str]] = []
        right: list[tuple[Phrase, str]] = []

        if r < 0.15:
            first, second = rng.choice(propns)
            p = W(first, "PROPN")
            if second:
                p = assemble(p, [], [(W(second, "PROPN"), "flat")])
            return p

        use_comp = rng.random() < 0.20
        if use_comp:
            c1, c2 = rng.choice(compounds)
            head = W(c2, "NOUN")
            comp_mod = c1
        else:
            head = W(rng.choice(pool), "NOUN")
            comp_mod = None

        adj_w = rng.choice(adjs) if rng.random() < 0.42 else None

        if r < 0.60:
            d = rng.choice(dets)
            first_word = adj_w if adj_w else (comp_mod if comp_mod else head.tokens[0])
            if d == "a" and first_word[0].lower() in "aeiou":
                d = "an"
            if d in ("every", "each", "a", "an"):
                pass
            d_surf = d.capitalize() if cap_first else d
            left.append((W(d_surf, "DET"), "det"))
        elif r < 0.80:
            pr = rng.choice(poss_prons)
            left.append((W(pr.capitalize() if cap_first else pr, "PRON"), "nmod"))
        else:
            if rng.random() < 0.45:
                nm = rng.choice(["Alice", "Liam", "Maya", "Owen"])
                poss = assemble(W(nm, "PROPN"), [], [(W("'s", "PART"), "case")])
                left.append((poss, "nmod"))
            else:
                num_w = rng.choice(nums)
                num_s = num_w.capitalize() if cap_first and not num_w[0].isdigit() else num_w
                num_p = W(num_s, "NUM")
                if not cap_first and rng.random() < 0.25:
                    num_p = assemble(num_p, [(W("only", "ADV"), "advmod")], [])
                left.append((num_p, "nummod"))
                if not head.tokens[0].endswith(("s", "x", "ch", "sh", "y")):
                    head = W(head.tokens[0] + "s", "NOUN")

        if adj_w is not None:
            aw = adj_w.capitalize() if (not left and cap_first) else adj_w
            left.append((W(aw, "ADJ"), "amod"))

        if comp_mod is not None:
            cw = comp_mod.capitalize() if (not left and cap_first) else comp_mod
            left.append((W(cw, "NOUN"), "compound"))

        if allow_pp and rng.random() < 0.35:
            prep = rng.choice(["of", "in", "on", "near", "from", "for"])
            p_obj = assemble(W(rng.choice(pool), "NOUN"), [(W("the", "DET"), "det")], [])
            pp = assemble(p_obj, [(W(prep, "ADP"), "case")], [])
            right.append((pp, "nmod"))

        return assemble(head, left, right)

    items: list[dict] = []
    genres = [("novel", nouns_novel), ("paper", nouns_paper), ("casual", nouns_casual)]

    while len(items) < target_count:
        genre, pool = genres[len(items) % 3]
        pat = rng.randrange(16)

        if pat == 0:
            # Active SVO + PP obl + optional temporal obl / advmod
            subj = W(rng.choice(["He", "She", "We", "They", "It"]), "PRON") if rng.random() < 0.28 else make_en_np(pool, cap_first=True)
            is_pl = subj.tokens[subj.root_idx] in ("We", "They") or (subj.tokens[subj.root_idx].endswith("s") and subj.upos[subj.root_idx] == "NOUN")
            v = W(rng.choice(vt_past if is_pl or rng.random() < 0.65 else vt_3sg), "VERB")
            obj = make_en_np(pool, allow_pp=True)
            right: list[tuple[Phrase, str]] = [(obj, "obj")]
            if rng.random() < 0.55:
                obl = assemble(make_en_np(pool), [(W(rng.choice(preps), "ADP"), "case")], [])
                right.append((obl, "obl"))
            if rng.random() < 0.45:
                right.append((make_time_obl(cap=False), "obl"))
            elif rng.random() < 0.45:
                right.append((W(rng.choice(advs), "ADV"), "advmod"))
            right.append((W(".", "PUNCT"), "punct"))
            items.append(finalize_sentences([assemble(v, [(subj, "nsubj")], right)], genre, "en", "en"))

        elif pat == 1:
            # Copula: NP + is/was/are/were (+ not) + ADJ or NOUN (+ temporal obl)
            subj = make_en_np(pool, cap_first=True)
            is_pl = subj.tokens[subj.root_idx].endswith("s") and subj.upos[subj.root_idx] == "NOUN"
            cop = W(rng.choice(["were", "are"] if is_pl else ["was", "is"]), "AUX")
            left: list[tuple[Phrase, str]] = [(subj, "nsubj"), (cop, "cop")]
            if rng.random() < 0.30:
                left.append((W("not", "PART"), "advmod"))
            if rng.random() < 0.65:
                pred = W(rng.choice(adjs), "ADJ")
                if len(left) == 2 and rng.random() < 0.4:
                    left.append((W(rng.choice(["very", "quite", "rather", "still"]), "ADV"), "advmod"))
            else:
                pred = make_en_np(pool, allow_pp=True)
            right = []
            if rng.random() < 0.4:
                right.append((make_time_obl(cap=False), "obl"))
            right.append((W(".", "PUNCT"), "punct"))
            items.append(finalize_sentences([assemble(pred, left, right)], genre, "en", "en"))

        elif pat == 2:
            # Passive: NP + was/were + (then/carefully) + V_ed + PP obl
            subj = make_en_np(pool, cap_first=True)
            is_pl = subj.tokens[subj.root_idx].endswith("s") and subj.upos[subj.root_idx] == "NOUN"
            aux = W("were" if is_pl else "was", "AUX")
            v = W(rng.choice(["measured", "recorded", "stored", "checked", "repaired", "updated", "tested", "verified", "packed", "cleaned", "weighed"]), "VERB")
            left = [(subj, "nsubj"), (aux, "aux")]
            if rng.random() < 0.45:
                left.append((W(rng.choice(["then", "carefully", "already", "quickly"]), "ADV"), "advmod"))
            obl = assemble(make_en_np(pool), [(W(rng.choice(["by", "in", "at", "after"]), "ADP"), "case")], [])
            right = [(obl, "obl")]
            if rng.random() < 0.3:
                right.append((W(rng.choice(["once", "twice"]), "ADV"), "advmod"))
            right.append((W(".", "PUNCT"), "punct"))
            items.append(finalize_sentences([assemble(v, left, right)], genre, "en", "en"))

        elif pat == 3:
            # Clitic negation: doesn't / didn't / couldn't / wouldn't + V_base
            subj = W(rng.choice(["He", "She", "It"]), "PRON") if rng.random() < 0.45 else make_en_np(pool, cap_first=True)
            is_pl = subj.tokens[subj.root_idx].endswith("s") and subj.upos[subj.root_idx] == "NOUN"
            aux_pair = rng.choice(
                [("did", "n't"), ("could", "n't"), ("would", "n't"), ("did", "not")]
                if is_pl
                else [("does", "n't"), ("did", "n't"), ("could", "n't"), ("did", "not")]
            )
            v = W(rng.choice(vt_base), "VERB")
            right = [(make_en_np(pool), "obj")]
            if rng.random() < 0.45:
                right.append((W(rng.choice(["again", "anymore", "either"]), "ADV"), "advmod"))
            right.append((W(".", "PUNCT"), "punct"))
            s1 = assemble(v, [(subj, "nsubj"), (W(aux_pair[0], "AUX"), "aux"), (W(aux_pair[1], "PART"), "advmod")], right)
            items.append(finalize_sentences([s1], genre, "en", "en"))

        elif pat == 4:
            # Infinitival xcomp: wants/decided/tried/planned/began + to + V_base + obj
            subj = W(rng.choice(["She", "He", "We", "They"]), "PRON") if rng.random() < 0.45 else make_en_np(pool, cap_first=True)
            vc = W(rng.choice(["decided", "tried", "planned", "agreed", "started", "began", "wanted"]), "VERB")
            v2 = assemble(W(rng.choice(vt_base), "VERB"), [(W("to", "PART"), "mark")], [(make_en_np(pool), "obj")])
            s1 = assemble(vc, [(subj, "nsubj")], [(v2, "xcomp"), (W(".", "PUNCT"), "punct")])
            items.append(finalize_sentences([s1], genre, "en", "en"))

        elif pat == 5:
            # Clausal complement ccomp: that/whether S2 OR indirect wh-clause `know which N it V`
            subj = make_en_np(pool, cap_first=True)
            if rng.random() < 0.65:
                vm = W(rng.choice(["confirmed", "showed", "reported", "found", "noticed", "explained"]), "VERB")
                mark_w = W(rng.choice(["that", "whether"]), "SCONJ")
                ccomp_p = assemble(W(rng.choice(vi_past), "VERB"), [(mark_w, "mark"), (make_en_np(pool), "nsubj")], [(W(rng.choice(advs), "ADV"), "advmod")])
                s1 = assemble(vm, [(subj, "nsubj")], [(ccomp_p, "ccomp"), (W(".", "PUNCT"), "punct")])
            else:
                wh_n = assemble(W(rng.choice(pool), "NOUN"), [(W("which", "DET"), "det")], [])
                ccomp_p = assemble(W(rng.choice(vt_past), "VERB"), [(wh_n, "obj"), (W(rng.choice(["he", "she", "it", "they"]), "PRON"), "nsubj")], [])
                s1 = assemble(W("knew", "VERB") if rng.random() < 0.5 else W("checked", "VERB"), [(subj, "nsubj")], [(ccomp_p, "ccomp"), (W(".", "PUNCT"), "punct")])
            items.append(finalize_sentences([s1], genre, "en", "en"))

        elif pat == 6:
            # Initial advcl: Before/After/When/Although/Because/If S1 , S2 .
            sconj = W(rng.choice(["Before", "After", "When", "Although", "Because", "If"]), "SCONJ")
            c1 = assemble(W(rng.choice(vi_past), "VERB"), [(sconj, "mark"), (make_en_np(pool), "nsubj")], [])
            s2_subj = W(rng.choice(["he", "she", "we", "they"]), "PRON") if rng.random() < 0.45 else make_en_np(pool)
            s1 = assemble(
                W(rng.choice(vt_past), "VERB"),
                [(c1, "advcl"), (W(",", "PUNCT"), "punct"), (s2_subj, "nsubj")],
                [(make_en_np(pool), "obj"), (W(".", "PUNCT"), "punct")],
            )
            items.append(finalize_sentences([s1], genre, "en", "en"))

        elif pat == 7:
            # Final advcl: S1 because/while/before/after/unless S2 .
            sconj = W(rng.choice(["because", "while", "before", "after", "unless"]), "SCONJ")
            c2 = assemble(W(rng.choice(vi_past), "VERB"), [(sconj, "mark"), (make_en_np(pool), "nsubj")], [])
            s1 = assemble(
                W(rng.choice(vt_past), "VERB"),
                [(make_en_np(pool, cap_first=True), "nsubj")],
                [(make_en_np(pool), "obj"), (c2, "advcl"), (W(".", "PUNCT"), "punct")],
            )
            items.append(finalize_sentences([s1], genre, "en", "en"))

        elif pat == 8:
            # Relative clause acl: The N [that/which V obj] V_past PP .
            head_n = W(rng.choice(pool), "NOUN")
            rel_pron = W(rng.choice(["that", "which"]), "PRON")
            rc_v = assemble(W(rng.choice(vt_past), "VERB"), [(rel_pron, "nsubj")], [(make_en_np(pool), "obj")])
            subj_np = assemble(head_n, [(W("The", "DET"), "det")], [(rc_v, "acl")])
            obl = assemble(make_en_np(pool), [(W(rng.choice(["in", "on", "near"]), "ADP"), "case")], [])
            s1 = assemble(W(rng.choice(vi_past), "VERB"), [(subj_np, "nsubj")], [(obl, "obl"), (W(".", "PUNCT"), "punct")])
            items.append(finalize_sentences([s1], genre, "en", "en"))

        elif pat == 9:
            # Coordination: VP and/but VP OR NP and/or NP
            subj = make_en_np(pool, cap_first=True)
            if rng.random() < 0.5:
                v1 = W(rng.choice(vt_past), "VERB")
                v2 = assemble(W(rng.choice(vt_past), "VERB"), [(W(rng.choice(["and", "but"]), "CCONJ"), "cc")], [(make_en_np(pool), "obj")])
                s1 = assemble(v1, [(subj, "nsubj")], [(make_en_np(pool), "obj"), (v2, "conj"), (W(".", "PUNCT"), "punct")])
            else:
                n1 = make_en_np(pool)
                n2 = assemble(make_en_np(pool), [(W(rng.choice(["and", "or"]), "CCONJ"), "cc")], [])
                coord = assemble(n1, [], [(n2, "conj")])
                s1 = assemble(W(rng.choice(vt_past), "VERB"), [(subj, "nsubj")], [(coord, "obj"), (W(".", "PUNCT"), "punct")])
            items.append(finalize_sentences([s1], genre, "en", "en"))

        elif pat == 10:
            # Yes-no Question: Did/Will/Can + subj + V_base + obj ?
            aux_w = W(rng.choice(["Did", "Will", "Can", "Should", "Could"]), "AUX")
            subj = W(rng.choice(["you", "he", "she", "they"]), "PRON") if rng.random() < 0.55 else make_en_np(pool)
            v = W(rng.choice(vt_base), "VERB")
            s1 = assemble(v, [(aux_w, "aux"), (subj, "nsubj")], [(make_en_np(pool, allow_pp=True), "obj"), (W("?", "PUNCT"), "punct")])
            items.append(finalize_sentences([s1], genre, "en", "en"))

        elif pat == 11:
            # Ditransitive: Subj + sent/gave/handed/showed + iobj + obj .
            subj = W(rng.choice(["She", "He", "We", "They"]), "PRON")
            vd = W(rng.choice(["sent", "gave", "handed", "offered", "showed"]), "VERB")
            iobj = W(rng.choice(["me", "him", "her", "us", "them"]), "PRON") if rng.random() < 0.55 else make_en_np(pool)
            s1 = assemble(vd, [(subj, "nsubj")], [(iobj, "iobj"), (make_en_np(pool), "obj"), (W(".", "PUNCT"), "punct")])
            items.append(finalize_sentences([s1], genre, "en", "en"))

        elif pat == 12:
            # Existential expl: There + was/were + NP + PP .
            expl = W("There", "PRON")
            obl = assemble(make_en_np(pool), [(W(rng.choice(["in", "on", "under", "near"]), "ADP"), "case")], [])
            np_clean = assemble(W(rng.choice(["box", "note", "sensor", "folder", "lantern", "ticket"]), "NOUN"), [(W("a", "DET"), "det"), (W(rng.choice(["small", "wooden", "clean", "new"]), "ADJ"), "amod")], [])
            s1 = assemble(W("was", "VERB"), [(expl, "expl")], [(np_clean, "nsubj"), (obl, "obl"), (W(".", "PUNCT"), "punct")])
            items.append(finalize_sentences([s1], genre, "en", "en"))

        elif pat == 13:
            # Sentence-initial temporal obl: Tonight / Last night / At dawn + pron + V + obj .
            t_obl = make_time_obl(cap=True)
            subj = W(rng.choice(["he", "she", "we", "they"]), "PRON")
            s1 = assemble(W(rng.choice(vt_past), "VERB"), [(t_obl, "obl"), (subj, "nsubj")], [(make_en_np(pool), "obj"), (W(".", "PUNCT"), "punct")])
            items.append(finalize_sentences([s1], genre, "en", "en"))

        elif pat == 14:
            # Parenthetical / appositive NP: The module (v2) processed ... OR Maya, our neighbor, ...
            if rng.random() < 0.5:
                app = assemble(W("neighbor" if genre != "paper" else "baseline", "NOUN"), [(W(",", "PUNCT"), "punct"), (W("our", "PRON"), "nmod")], [(W(",", "PUNCT"), "punct")])
                subj = assemble(W(rng.choice(["Maya", "Liam", "Elena"]), "PROPN"), [], [(app, "appos")])
            else:
                app = assemble(W(rng.choice(["v2", "beta", "B"]), "PROPN"), [(W("(", "PUNCT"), "punct")], [(W(")", "PUNCT"), "punct")])
                subj = assemble(W(rng.choice(pool), "NOUN"), [(W("The", "DET"), "det")], [(app, "appos")])
            s1 = assemble(W(rng.choice(vt_past), "VERB"), [(subj, "nsubj")], [(make_en_np(pool), "obj"), (W(".", "PUNCT"), "punct")])
            items.append(finalize_sentences([s1], genre, "en", "en"))

        else:
            # Two-sentence item with However, / Then + degree advmod chain
            subj1 = make_en_np(pool, cap_first=True)
            s_a = assemble(W(rng.choice(vt_past), "VERB"), [(subj1, "nsubj")], [(make_en_np(pool), "obj"), (W(".", "PUNCT"), "punct")])
            adv_head = W(rng.choice(["slowly", "quickly", "smoothly", "clearly"]), "ADV")
            if rng.random() < 0.45:
                adv_head = assemble(adv_head, [(W("twice", "ADV"), "advmod"), (W("as", "ADV"), "advmod")], [])
            s_b = assemble(
                W(rng.choice(vi_past), "VERB"),
                [(W("However", "ADV"), "advmod"), (W(",", "PUNCT"), "punct"), (W("it", "PRON"), "nsubj")],
                [(adv_head, "advmod"), (W(".", "PUNCT"), "punct")],
            )
            items.append(finalize_sentences([s_a, s_b], genre, "en", "en"))

    return items


# ---------------------------------------------------------------------------
# Code-switched `mixed-TW` and `mixed-CN` generators
# ---------------------------------------------------------------------------


def build_mixed_variant_sentences(
    rng: random.Random, is_tw: bool, target_count: int
) -> list[dict]:
    variant = "mixed-TW" if is_tw else "mixed-CN"
    cfg = ZH_TW_CONFIG if is_tw else ZH_CN_CONFIG
    de = cfg["de"]
    le = cfg["le"]
    ma = cfg["ma"]

    en_tech = ["model", "dataset", "server", "parser", "script", "plugin", "module", "config", "buffer", "pipeline", "token", "branch", "commit", "schema", "cache", "router", "API", "GPU"]
    en_casual = ["app", "email", "podcast", "vlog", "menu", "ticket", "camera", "notebook", "poster", "video", "link", "file", "folder", "WiFi"]
    en_compounds = [
        ("Python", "script"), ("JSON", "file"), ("CoreML", "model"), ("API", "server"),
        ("train", "ticket"), ("coffee", "shop"), ("data", "pipeline"), ("unit", "test"),
        ("SQLite", "database"), ("memory", "buffer"), ("web", "page"), ("audio", "clip"),
    ]
    en_verbs_cs = ["check", "review", "test", "update", "debug", "export", "merge", "build", "deploy"]

    zh_subjs = (
        ["工程師", "研究員", "團隊", "學生", "老師", "作者", "小芸", "阿哲", "冠宇", "雅婷"]
        if is_tw
        else ["工程师", "研究员", "团队", "学生", "老师", "作者", "小王", "老张", "李明", "陈静"]
    )
    zh_vts = (
        ["下載", "更新", "檢查", "測試", "儲存", "載入", "讀取", "整理", "分享", "收到", "完成", "調整", "驗證", "優化"]
        if is_tw
        else ["下载", "更新", "检查", "测试", "保存", "加载", "读取", "整理", "分享", "收到", "完成", "调整", "验证", "部署"]
    )
    zh_props = (
        ["介面", "記憶體", "資料庫", "網路", "演算法", "專案", "結果", "速率"]
        if is_tw
        else ["界面", "内存", "数据库", "网络", "算法", "项目", "结果", "速度"]
    )
    zh_adjs = cfg["adjs"]
    prons = [W(p, "PRON") for p in cfg["prons"]]
    propn_names = {"小芸", "阿哲", "冠宇", "雅婷", "小王", "老张", "李明", "陈静"}

    items: list[dict] = []
    genres = ["novel", "paper", "casual"]

    def fin_mixed(sents: list[Phrase], g: str) -> dict:
        return finalize_sentences(sents, g, "mixed", variant, space_cjk_ascii=(rng.random() < 0.65))

    while len(items) < target_count:
        genre = genres[len(items) % 3]
        pat = rng.randrange(12)
        s_str = rng.choice(zh_subjs)
        subj = rng.choice(prons) if rng.random() < 0.35 else W(s_str, "PROPN" if s_str in propn_names else "NOUN")
        en_pool = en_tech if genre == "paper" else en_casual

        if pat == 0:
            # zh subj + (透過/通過 API) + zh V + 了 + det/numclf + en noun/compound 。
            v = W(rng.choice(zh_vts), "VERB")
            if rng.random() < 0.5:
                c1, c2 = rng.choice(en_compounds)
                en_head = assemble(
                    W(c2, "NOUN"),
                    [
                        (W(rng.choice(cfg["dets"][:2]), "DET"), "det"),
                        (W(c1, "PROPN" if c1[0].isupper() else "NOUN"), "compound"),
                    ],
                    [],
                )
            else:
                use_num = rng.random() < 0.5
                mod_p = W(rng.choice(cfg["numclfs"][:3]), "NUM") if use_num else W(rng.choice(cfg["dets"][:2]), "DET")
                ew = rng.choice(en_pool)
                en_head = assemble(W(ew, "PROPN" if ew.isupper() else "NOUN"), [(mod_p, "nummod" if use_num else "det")], [])
            left = [(subj, "nsubj")]
            if rng.random() < 0.45:
                via = "透過" if is_tw else "通过"
                left.append((assemble(W("API", "PROPN"), [(W(via, "ADP"), "case")], []), "obl"))
            s1 = assemble(v, left, [(W(le, "PART"), "aux"), (en_head, "obj"), (W("。", "PUNCT"), "punct")])
            items.append(fin_mixed([s1], genre))

        elif pat == 1:
            # 把 + en noun + V + 在 + en/zh loc (+ 裡/里 ADP case)
            ew = rng.choice(en_pool)
            en_obj = assemble(W(ew, "PROPN" if ew.isupper() else "NOUN"), [(W("把", "ADP"), "case"), (W(cfg["dets"][0], "DET"), "det")], [])
            post_r = [(W("裡" if is_tw else "里", "ADP"), "case")] if rng.random() < 0.6 else []
            loc = assemble(W(rng.choice(["server", "cache", "folder"]), "NOUN"), [(W("在", "ADP"), "case")], post_r)
            v = W(rng.choice(["儲存", "載入", "整理"] if is_tw else ["保存", "加载", "整理"]), "VERB")
            s1 = assemble(v, [(subj, "nsubj"), (en_obj, "obl")], [(loc, "obl"), (W("。", "PUNCT"), "punct")])
            items.append(fin_mixed([s1], genre))

        elif pat == 2:
            # Possessive: 這個/这个 <en_noun> 的 <zh_prop> 很 <zh_adj> 。
            ew = rng.choice(en_pool)
            en_owner = assemble(W(ew, "PROPN" if ew.isupper() else "NOUN"), [(W(cfg["dets"][0], "DET"), "det")], [(W(de, "PART"), "case")])
            zh_prop = assemble(W(rng.choice(zh_props), "NOUN"), [(en_owner, "nmod")], [])
            adj = W(rng.choice(zh_adjs), "ADJ")
            s1 = assemble(adj, [(zh_prop, "nsubj"), (W(rng.choice(cfg["hen"]), "ADV"), "advmod")], [(W("。", "PUNCT"), "punct")])
            items.append(fin_mixed([s1], genre))

        elif pat == 3:
            # Passive mixed: 那個/那个 <en_noun> 已經/已经 被 <zh_v> 了 。
            ew = rng.choice(en_pool)
            en_pat = assemble(W(ew, "PROPN" if ew.isupper() else "NOUN"), [(W(cfg["dets"][1], "DET"), "det")], [])
            v = W(rng.choice(zh_vts[:6]), "VERB")
            already = "已經" if is_tw else "已经"
            s1 = assemble(v, [(en_pat, "nsubj"), (W(already, "ADV"), "advmod"), (W("被", "ADP"), "aux")], [(W(le, "PART"), "aux"), (W("。", "PUNCT"), "punct")])
            items.append(fin_mixed([s1], genre))

        elif pat == 4:
            # Code-switched English verb inside Chinese modal matrix
            ev = W(rng.choice(en_verbs_cs), "VERB")
            ew = rng.choice(en_pool)
            eobj = assemble(W(ew, "PROPN" if ew.isupper() else "NOUN"), [(W(cfg["dets"][0], "DET"), "det")], [])
            aux = W(rng.choice(cfg["modals"]), "AUX")
            s1 = assemble(ev, [(subj, "nsubj"), (aux, "aux")], [(eobj, "obj"), (W("。", "PUNCT"), "punct")])
            items.append(fin_mixed([s1], genre))

        elif pat == 5:
            # Mixed coordination: 把 資料/数据 分成 train set 與/和 test set 。
            p1 = assemble(W("set", "NOUN"), [(W("train", "NOUN"), "compound")], [])
            cc_w = "與" if is_tw else "和"
            p2 = assemble(W("set", "NOUN"), [(W(cc_w, "CCONJ"), "cc"), (W("test", "NOUN"), "compound")], [])
            coord = assemble(p1, [], [(p2, "conj")])
            data_w = rng.choice(["資料", "樣本", "檔案"] if is_tw else ["数据", "样本", "文件"])
            ba_np = assemble(W(data_w, "NOUN"), [(W("把", "ADP"), "case")], [])
            v = W(rng.choice(["調整", "整理", "比較"] if is_tw else ["调整", "整理", "比较"]), "VERB")
            s1 = assemble(v, [(subj, "nsubj"), (ba_np, "obl")], [(coord, "obj"), (W("。", "PUNCT"), "punct")])
            items.append(fin_mixed([s1], genre))

        elif pat == 6:
            # Mixed question with 嗎/吗
            ew = rng.choice(["email", "ticket", "file", "link", "video"])
            en_obj = assemble(W(ew, "NOUN"), [(W(cfg["dets"][1], "DET"), "det")], [])
            v = W(rng.choice(["收到", "下載", "檢查", "打開"] if is_tw else ["收到", "下载", "检查", "打开"]), "VERB")
            s1 = assemble(v, [(W("你", "PRON"), "nsubj")], [(en_obj, "obj"), (W(le, "PART"), "aux"), (W(ma, "PART"), "discourse"), (W("？", "PUNCT"), "punct")])
            items.append(fin_mixed([s1], genre))

        elif pat == 7:
            # Subordinate mixed: 因為/因为 server 收斂/收敛 ...
            sconj = W(rng.choice(cfg["sconjs"][:3]), "SCONJ")
            v1 = W(rng.choice(cfg["vi_list"][5:9]), "VERB")
            c1 = assemble(v1, [(sconj, "mark"), (W(rng.choice(["server", "model", "network"]), "NOUN"), "nsubj")], [])
            v2 = W(rng.choice(zh_vts[:6]), "VERB")
            zh_o = W(rng.choice(zh_props), "NOUN")
            s1 = assemble(v2, [(c1, "advcl"), (W("，", "PUNCT"), "punct"), (subj, "nsubj"), (W(rng.choice(cfg["corrs"][:3]), "ADV"), "advmod")], [(zh_o, "obj"), (W("。", "PUNCT"), "punct")])
            items.append(fin_mixed([s1], genre))

        elif pat == 8:
            # Mixed comma-joined parataxis
            ew = rng.choice(en_pool)
            en_obj = assemble(W(ew, "PROPN" if ew.isupper() else "NOUN"), [(W(cfg["dets"][0], "DET"), "det")], [])
            c2 = assemble(
                W(rng.choice(cfg["v_give"]), "VERB"),
                [(W("，", "PUNCT"), "punct"), (W(rng.choice(cfg["advs"][:4]), "ADV"), "advmod")],
                [(W("我", "PRON"), "iobj"), (W(le, "PART"), "aux")],
            )
            s1 = assemble(W(rng.choice(zh_vts[:6]), "VERB"), [(subj, "nsubj")], [(W(le, "PART"), "aux"), (en_obj, "obj"), (c2, "parataxis"), (W("。", "PUNCT"), "punct")])
            items.append(fin_mixed([s1], genre))

        elif pat == 9:
            # Mixed backtick X span (`batch_size` / `max_len`)
            code_x = W(rng.choice(["`batch_size`", "`max_len`", "`top_k`", "[12]"]), "X")
            p_n = assemble(W("參數" if is_tw else "参数", "NOUN"), [], [(code_x, "appos")])
            s1 = assemble(W(rng.choice(zh_vts[:8]), "VERB"), [(subj, "nsubj")], [(p_n, "obj"), (W("。", "PUNCT"), "punct")])
            items.append(fin_mixed([s1], genre))

        elif pat == 10:
            # Mixed split classifier: 這/这 + 個/个/款 + en_noun
            d_w = "這" if is_tw else "这"
            c_w = rng.choice(["個", "組", "套"] if is_tw else ["个", "组", "款"])
            ew = rng.choice(en_pool)
            en_np = assemble(W(ew, "PROPN" if ew.isupper() else "NOUN"), [(W(d_w, "DET"), "det"), (W(c_w, "NOUN"), "clf")], [])
            s1 = assemble(W(rng.choice(zh_adjs[:8]), "ADJ"), [(en_np, "nsubj"), (W(cfg["hen"][0], "ADV"), "advmod")], [(W("。", "PUNCT"), "punct")])
            items.append(fin_mixed([s1], genre))

        else:
            # Two-sentence mixed
            ew = rng.choice(en_pool)
            en_obj = assemble(W(ew, "PROPN" if ew.isupper() else "NOUN"), [(W(cfg["numclfs"][0], "NUM"), "nummod")], [])
            s_a = assemble(W(rng.choice(zh_vts[:5]), "VERB"), [(subj, "nsubj")], [(W(le, "PART"), "aux"), (en_obj, "obj"), (W("。", "PUNCT"), "punct")])
            s_b = assemble(W(rng.choice(zh_adjs), "ADJ"), [(W("它", "PRON"), "nsubj"), (W(cfg["hen"][1], "ADV"), "advmod")], [(W("。", "PUNCT"), "punct")])
            items.append(fin_mixed([s_a, s_b], genre))

    return items


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).lower()


def load_forbidden_texts() -> set[str]:
    forbidden: set[str] = set()
    for rel in [
        "training/probe/probe_texts.jsonl",
        "training/silver/sample_dryrun.jsonl",
        "tests/data/gold_layers4_6_v1.jsonl",
    ]:
        p = ROOT / rel
        if p.is_file():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    obj = json.loads(line)
                    if "text" in obj:
                        forbidden.add(norm_text(obj["text"]))
    # Also exclude swarm/gold-eval if present as a local git branch
    proc = subprocess.run(
        ["git", "show", "swarm/gold-eval:tests/data/gold_layers4_6_v1.jsonl"],
        capture_output=True,
        check=False,
    )
    if proc.returncode == 0:
        for line in proc.stdout.decode("utf-8").splitlines():
            if line.strip():
                obj = json.loads(line)
                if "text" in obj:
                    forbidden.add(norm_text(obj["text"]))
    return forbidden


def generate_dataset(nlp_doc: Path, out_dir: Path, seed: int = 20261111) -> dict:
    rng = random.Random(seed)
    forbidden = load_forbidden_texts()

    raw_candidates = (
        build_zh_variant_sentences(rng, ZH_TW_CONFIG, 3000)
        + build_zh_variant_sentences(rng, ZH_CN_CONFIG, 3000)
        + build_en_sentences(rng, 2800)
        + build_mixed_variant_sentences(rng, True, 1400)
        + build_mixed_variant_sentences(rng, False, 1400)
    )
    rng.shuffle(raw_candidates)

    parsed_list = batch_nlp_doc(nlp_doc, [c["text"] for c in raw_candidates], chunk_size=250)

    seen_texts = set(forbidden)
    verified: list[dict] = []
    rejected_tok = 0
    rejected_check = 0
    rejected_dup = 0

    for cand, parsed in zip(raw_candidates, parsed_list):
        nt = norm_text(cand["text"])
        if nt in seen_texts:
            rejected_dup += 1
            continue
        if parsed is None:
            rejected_tok += 1
            continue
        repo_toks, repo_spans, repo_upos, repo_sents = parsed
        projected = project_to_repo_tokens(cand, repo_toks, repo_spans, repo_upos, repo_sents)
        if projected is None:
            rejected_tok += 1
            continue
        projected["id"] = f"syn_{len(verified) + 1:05d}"

        err = check_annotation(projected, repo_toks, repo_sents, repo_upos)
        if err is not None:
            rejected_check += 1
            continue

        seen_texts.add(nt)
        verified.append(projected)

    # Stratified split: 90% train, 10% dev across (genre, variant)
    buckets: dict[tuple[str, str], list[dict]] = {}
    for item in verified:
        buckets.setdefault((item["genre"], item["variant"]), []).append(item)

    train_rows: list[dict] = []
    dev_rows: list[dict] = []
    for _, group in sorted(buckets.items()):
        n_dev = max(15, len(group) // 10)
        dev_rows.extend(group[:n_dev])
        train_rows.extend(group[n_dev:])

    rng.shuffle(train_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in train_rows),
        encoding="utf-8",
    )
    (out_dir / "dev.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in dev_rows),
        encoding="utf-8",
    )
    stats = {
        "train": len(train_rows),
        "dev": len(dev_rows),
        "total_verified": len(verified),
        "rejected_tok": rejected_tok,
        "rejected_check": rejected_check,
        "rejected_dup": rejected_dup,
        "by_cell": {f"{g}/{v}": len(rows) for (g, v), rows in sorted(buckets.items())},
    }
    (out_dir / "corpus_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nlp-doc", type=Path, default=ROOT / "target/release/nlp_doc")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "training/.work/syntax")
    parser.add_argument("--seed", type=int, default=20261111)
    args = parser.parse_args()

    stats = generate_dataset(args.nlp_doc, args.out_dir, args.seed)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
