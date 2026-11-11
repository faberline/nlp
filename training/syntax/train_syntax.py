#!/usr/bin/env python3
"""Train `doc-syntax-v1` (multilingual zh-TW / zh-CN / en / mixed Layer-4 UD v2 + UPOS parser).

Architecture (100% faithful to the 5 runtime invariants of `src/doc/model/` & `src/doc/syntax.rs`):
  - Vocab-pruned `intfloat/multilingual-e5-small` (MIT licence) Unigram tokenizer (`32,768` pieces,
    `[PAD]=0, [UNK]=1, [CLS]=2, [SEP]=3`), isolating single Han characters (both Traditional and
    Simplified), English clitics, and punctuation so subword-to-token alignment is collision-free.
  - Pretrained `word_embeddings` (`[32768, 384]`), frozen pretrained `[1, 256, 384]`
    `position_embeddings[0:256] + token_type_embeddings[0]`, `embeddings.LayerNorm` (`eps=1e-5`),
    and 12 Transformer encoder layers (`d_model=384, n_heads=12, d_ff=1536`, exact `erf` GELU,
    `eps=1e-5`) with internal pad-mask bias `(valid * 10000.0 - 10000.0)`.
  - Packed `[B, 256, 421]` output heads:
      * `0..17`: 2-layer MLP UPOS head (`384 -> 192 -> 17`)
      * `17..273`: Biaffine arc logits `(A_dep / sqrt(192)) @ A_head^T` (`[B, 256, 256]`)
      * `273..421`: Factorized Biaffine relation vectors `r_dep (37) || r_head (37) || r_bi_dep (37) || r_bi_head (37)`
        scored as `r_dep[u] + r_head[v] + r_bi_dep[u] * r_bi_head[v]`.

Saves float32 weights `.npz` and pruned `tokenizer.json` to `training/.work/syntax/` (gitignored).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from tokenizers import Regex, Tokenizer, pre_tokenizers, processors

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "training" / "silver"))
from validate_merge import DEPRELS, UPOS_TAGS  # noqa: E402

UPOS_LIST = tuple(sorted(UPOS_TAGS))
DEPREL_LIST = tuple(sorted(DEPRELS))
assert len(UPOS_LIST) == 17 and UPOS_LIST[0] == "ADJ" and UPOS_LIST[16] == "X"
assert len(DEPREL_LIST) == 37 and DEPREL_LIST[0] == "acl" and DEPREL_LIST[36] == "xcomp"
UPOS_TO_ID = {u: i for i, u in enumerate(UPOS_LIST)}
DEPREL_TO_ID = {d: i for i, d in enumerate(DEPREL_LIST)}
ROOT_DEPREL_ID = DEPREL_TO_ID["root"]
assert ROOT_DEPREL_ID == 34

VOCAB_SIZE = 32768
MAX_LEN = 256
D_MODEL = 384
N_HEADS = 12
D_HEAD = D_MODEL // N_HEADS
D_FF = 1536
D_HEAD_MLP = 192
LN_EPS = 1e-5


def find_hf_snapshot(repo_id: str = "intfloat/multilingual-e5-small") -> Path:
    folder = "models--" + repo_id.replace("/", "--")
    matches = sorted(
        glob.glob(str(Path.home() / ".cache/huggingface/hub" / folder / "snapshots" / "*"))
    )
    if not matches:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(repo_id))
    return Path(matches[-1])


def is_han(ch: str) -> bool:
    cp = ord(ch)
    return (
        (0x3400 <= cp <= 0x4DBF)
        or (0x4E00 <= cp <= 0x9FFF)
        or (0xF900 <= cp <= 0xFAFF)
        or cp in (0x3005, 0x3007)
    )


def is_allowed_char(ch: str) -> bool:
    if ch == "▁" or is_han(ch):
        return True
    cp = ord(ch)
    return (
        cp < 0x0250
        or (0x0370 <= cp <= 0x03FF)
        or (0x2000 <= cp <= 0x2BFF)
        or (0x3000 <= cp <= 0x303F)
        or (0xFE30 <= cp <= 0xFFEF)
    )


def build_pruned_tokenizer_and_embedding_map(
    hf_dir: Path, out_tok_path: Path
) -> tuple[Tokenizer, list[int]]:
    """Prune `multilingual-e5-small` Unigram vocab to `32,768` covering zh-TW, zh-CN, and en."""
    raw = json.loads((hf_dir / "tokenizer.json").read_text(encoding="utf-8"))
    orig_vocab: list[list] = raw["model"]["vocab"]

    # Map new IDs 0..3 -> original XLM-R / e5-small IDs:
    # orig: 0="<s>" ([CLS]), 1="<pad>" ([PAD]), 2="</s>" ([SEP]), 3="<unk>" ([UNK])
    orig_indices: list[int] = [1, 3, 0, 2]
    new_vocab: list[list] = [
        ["[PAD]", 0.0],
        ["[UNK]", 0.0],
        ["[CLS]", 0.0],
        ["[SEP]", 0.0],
    ]

    must_single: list[tuple[int, str, float]] = []
    en_subwords: list[tuple[int, str, float]] = []
    for idx, (piece, score) in enumerate(orig_vocab):
        if idx in (0, 1, 2, 3):
            continue
        if not all(is_allowed_char(c) for c in piece):
            continue
        s = piece.lstrip("▁")
        if any(is_han(c) for c in s):
            if len(s) == 1 and not piece.startswith("▁"):
                must_single.append((idx, piece, float(score)))
        elif len(s) <= 1:
            must_single.append((idx, piece, float(score)))
        else:
            en_subwords.append((idx, piece, float(score)))

    en_subwords.sort(key=lambda x: x[2], reverse=True)
    budget = VOCAB_SIZE - len(new_vocab) - len(must_single)
    assert budget > 20000, f"expected >20k English subword budget, got {budget}"
    chosen = must_single + en_subwords[:budget]
    assert len(new_vocab) + len(chosen) == VOCAB_SIZE

    for orig_idx, piece, score in chosen:
        orig_indices.append(orig_idx)
        new_vocab.append([piece, score])

    raw["model"]["vocab"] = new_vocab
    raw["model"]["unk_id"] = 1
    raw["added_tokens"] = [
        {
            "id": 0,
            "content": "[PAD]",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        },
        {
            "id": 1,
            "content": "[UNK]",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        },
        {
            "id": 2,
            "content": "[CLS]",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        },
        {
            "id": 3,
            "content": "[SEP]",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        },
    ]

    tok = Tokenizer.from_str(json.dumps(raw))
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always"),
            pre_tokenizers.Split(
                Regex(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF\u3005\u3007]"),
                behavior="isolated",
            ),
            pre_tokenizers.Split(
                Regex(r"(?i)n['’]t|['’](?:s|ll|re|ve|m|d)"), behavior="isolated"
            ),
            pre_tokenizers.Split(
                Regex(r"[`_$€£¥%()\[\]{}<>/\\+*=—–\-;:,!?.\"“”‘’'「」『』（）《》【】、。，！？；：…]"),
                behavior="isolated",
            ),
            pre_tokenizers.Split(Regex(r"^▁+$"), behavior="removed"),
        ]
    )
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS]:0 $A:0 [SEP]:0",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
    )
    out_tok_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_tok_path))
    return tok, orig_indices


def exact_gelu(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


class EncoderLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q = nn.Linear(D_MODEL, D_MODEL)
        self.k = nn.Linear(D_MODEL, D_MODEL)
        self.v = nn.Linear(D_MODEL, D_MODEL)
        self.o = nn.Linear(D_MODEL, D_MODEL)
        self.ln1 = nn.LayerNorm(D_MODEL, eps=LN_EPS)
        self.fc1 = nn.Linear(D_MODEL, D_FF)
        self.fc2 = nn.Linear(D_FF, D_MODEL)
        self.ln2 = nn.LayerNorm(D_MODEL, eps=LN_EPS)

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.q(x).view(bsz, seq_len, N_HEADS, D_HEAD).transpose(1, 2)
        k = self.k(x).view(bsz, seq_len, N_HEADS, D_HEAD).transpose(1, 2)
        v = self.v(x).view(bsz, seq_len, N_HEADS, D_HEAD).transpose(1, 2)
        scores = torch.matmul(q / math.sqrt(D_HEAD), k.transpose(-1, -2)) + attn_bias
        probs = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(probs, v).transpose(1, 2).contiguous().view(bsz, seq_len, D_MODEL)
        x = self.ln1(x + self.o(ctx))
        ff = self.fc2(exact_gelu(self.fc1(x)))
        return self.ln2(x + ff)


class DocSyntaxModel(nn.Module):
    def __init__(self, num_layers: int = 12, dropout: float = 0.15) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.word_embeddings = nn.Embedding(VOCAB_SIZE, D_MODEL, padding_idx=0)
        self.register_buffer("pos_type_bias", torch.zeros(1, MAX_LEN, D_MODEL))
        self.emb_ln = nn.LayerNorm(D_MODEL, eps=LN_EPS)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(num_layers)])
        self.drop = nn.Dropout(dropout)

        # Heads -> packed [B, L, 421] when L == 256 (or dynamic L during training)
        self.upos_fc1 = nn.Linear(D_MODEL, D_HEAD_MLP)
        self.upos_fc2 = nn.Linear(D_HEAD_MLP, 17)

        self.arc_dep_fc = nn.Linear(D_MODEL, D_HEAD_MLP)
        self.arc_head_fc = nn.Linear(D_MODEL, D_HEAD_MLP)

        self.rel_dep_in = nn.Linear(D_MODEL, D_HEAD_MLP)
        self.rel_head_in = nn.Linear(D_MODEL, D_HEAD_MLP)
        self.r_dep_fc = nn.Linear(D_HEAD_MLP, 37)
        self.r_head_fc = nn.Linear(D_HEAD_MLP, 37)
        self.r_bi_dep_fc = nn.Linear(D_HEAD_MLP, 37)
        self.r_bi_head_fc = nn.Linear(D_HEAD_MLP, 37)

        self._init_heads()

    def _init_heads(self) -> None:
        for m in [
            self.upos_fc1,
            self.upos_fc2,
            self.arc_dep_fc,
            self.arc_head_fc,
            self.rel_dep_in,
            self.rel_head_in,
            self.r_dep_fc,
            self.r_head_fc,
            self.r_bi_dep_fc,
            self.r_bi_head_fc,
        ]:
            nn.init.xavier_uniform_(m.weight, gain=0.5)
            nn.init.zeros_(m.bias)

    def load_pretrained_e5(self, hf_dir: Path, orig_indices: list[int]) -> None:
        sd = load_file(str(hf_dir / "model.safetensors"))
        with torch.no_grad():
            idx_t = torch.tensor(orig_indices, dtype=torch.long)
            w_emb = sd["embeddings.word_embeddings.weight"][idx_t].clone()
            w_emb[0].zero_()  # pad_id = 0
            self.word_embeddings.weight.copy_(w_emb)

            pos_w = sd["embeddings.position_embeddings.weight"][:MAX_LEN]
            tt0 = sd["embeddings.token_type_embeddings.weight"][0].unsqueeze(0)
            self.pos_type_bias.copy_((pos_w + tt0).unsqueeze(0))

            self.emb_ln.weight.copy_(sd["embeddings.LayerNorm.weight"])
            self.emb_ln.bias.copy_(sd["embeddings.LayerNorm.bias"])

            for i in range(self.num_layers):
                pfx = f"encoder.layer.{i}."
                layer = self.layers[i]
                layer.q.weight.copy_(sd[pfx + "attention.self.query.weight"])
                layer.q.bias.copy_(sd[pfx + "attention.self.query.bias"])
                layer.k.weight.copy_(sd[pfx + "attention.self.key.weight"])
                layer.k.bias.copy_(sd[pfx + "attention.self.key.bias"])
                layer.v.weight.copy_(sd[pfx + "attention.self.value.weight"])
                layer.v.bias.copy_(sd[pfx + "attention.self.value.bias"])
                layer.o.weight.copy_(sd[pfx + "attention.output.dense.weight"])
                layer.o.bias.copy_(sd[pfx + "attention.output.dense.bias"])
                layer.ln1.weight.copy_(sd[pfx + "attention.output.LayerNorm.weight"])
                layer.ln1.bias.copy_(sd[pfx + "attention.output.LayerNorm.bias"])
                layer.fc1.weight.copy_(sd[pfx + "intermediate.dense.weight"])
                layer.fc1.bias.copy_(sd[pfx + "intermediate.dense.bias"])
                layer.fc2.weight.copy_(sd[pfx + "output.dense.weight"])
                layer.fc2.bias.copy_(sd[pfx + "output.dense.bias"])
                layer.ln2.weight.copy_(sd[pfx + "output.LayerNorm.weight"])
                layer.ln2.bias.copy_(sd[pfx + "output.LayerNorm.bias"])

    def encode_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.shape[1]
        valid = torch.clamp(input_ids.to(torch.float32), 0.0, 1.0)
        attn_bias = (valid * 10000.0 - 10000.0).view(input_ids.shape[0], 1, 1, seq_len)

        x = self.word_embeddings(input_ids) + self.pos_type_bias[:, :seq_len, :]
        x = self.emb_ln(x)
        x = self.drop(x)
        for layer in self.layers:
            x = layer(x, attn_bias)
        return x

    def forward_heads(
        self, h: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        h_d = self.drop(h)
        upos_logits = self.upos_fc2(exact_gelu(self.upos_fc1(h_d)))
        a_dep = exact_gelu(self.arc_dep_fc(h_d))
        a_head = exact_gelu(self.arc_head_fc(h_d))
        arc_logits = torch.matmul(a_dep / math.sqrt(D_HEAD_MLP), a_head.transpose(-1, -2))

        rh_dep = exact_gelu(self.rel_dep_in(h_d))
        rh_head = exact_gelu(self.rel_head_in(h_d))
        r_dep = self.r_dep_fc(rh_dep)
        r_head = self.r_head_fc(rh_head)
        r_bi_dep = self.r_bi_dep_fc(rh_dep)
        r_bi_head = self.r_bi_head_fc(rh_head)
        return upos_logits, arc_logits, r_dep, r_head, r_bi_dep, r_bi_head

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Full packed `[B, 256, 421]` output matching the exact runtime layout."""
        assert input_ids.shape[1] == MAX_LEN
        h = self.encode_hidden(input_ids)
        upos_logits, arc_logits, r_dep, r_head, r_bi_dep, r_bi_head = self.forward_heads(h)
        return torch.cat([upos_logits, arc_logits, r_dep, r_head, r_bi_dep, r_bi_head], dim=-1)

    def save_npz(self, path: Path) -> None:
        arrays: dict[str, np.ndarray] = {
            "num_layers": np.array([self.num_layers], dtype=np.int64),
            "word_embeddings": self.word_embeddings.weight.detach().cpu().to(torch.float32).numpy(),
            "pos_type_bias": self.pos_type_bias.detach().cpu().to(torch.float32).numpy(),
            "emb_ln_w": self.emb_ln.weight.detach().cpu().to(torch.float32).numpy(),
            "emb_ln_b": self.emb_ln.bias.detach().cpu().to(torch.float32).numpy(),
            "upos_fc1_w": self.upos_fc1.weight.detach().cpu().to(torch.float32).numpy(),
            "upos_fc1_b": self.upos_fc1.bias.detach().cpu().to(torch.float32).numpy(),
            "upos_fc2_w": self.upos_fc2.weight.detach().cpu().to(torch.float32).numpy(),
            "upos_fc2_b": self.upos_fc2.bias.detach().cpu().to(torch.float32).numpy(),
            "arc_dep_w": self.arc_dep_fc.weight.detach().cpu().to(torch.float32).numpy(),
            "arc_dep_b": self.arc_dep_fc.bias.detach().cpu().to(torch.float32).numpy(),
            "arc_head_w": self.arc_head_fc.weight.detach().cpu().to(torch.float32).numpy(),
            "arc_head_b": self.arc_head_fc.bias.detach().cpu().to(torch.float32).numpy(),
            "rel_dep_in_w": self.rel_dep_in.weight.detach().cpu().to(torch.float32).numpy(),
            "rel_dep_in_b": self.rel_dep_in.bias.detach().cpu().to(torch.float32).numpy(),
            "rel_head_in_w": self.rel_head_in.weight.detach().cpu().to(torch.float32).numpy(),
            "rel_head_in_b": self.rel_head_in.bias.detach().cpu().to(torch.float32).numpy(),
            "r_dep_w": self.r_dep_fc.weight.detach().cpu().to(torch.float32).numpy(),
            "r_dep_b": self.r_dep_fc.bias.detach().cpu().to(torch.float32).numpy(),
            "r_head_w": self.r_head_fc.weight.detach().cpu().to(torch.float32).numpy(),
            "r_head_b": self.r_head_fc.bias.detach().cpu().to(torch.float32).numpy(),
            "r_bi_dep_w": self.r_bi_dep_fc.weight.detach().cpu().to(torch.float32).numpy(),
            "r_bi_dep_b": self.r_bi_dep_fc.bias.detach().cpu().to(torch.float32).numpy(),
            "r_bi_head_w": self.r_bi_head_fc.weight.detach().cpu().to(torch.float32).numpy(),
            "r_bi_head_b": self.r_bi_head_fc.bias.detach().cpu().to(torch.float32).numpy(),
        }
        for i, layer in enumerate(self.layers):
            arrays[f"l{i}_q_w"] = layer.q.weight.detach().cpu().to(torch.float32).numpy()
            arrays[f"l{i}_q_b"] = layer.q.bias.detach().cpu().to(torch.float32).numpy()
            arrays[f"l{i}_k_w"] = layer.k.weight.detach().cpu().to(torch.float32).numpy()
            arrays[f"l{i}_k_b"] = layer.k.bias.detach().cpu().to(torch.float32).numpy()
            arrays[f"l{i}_v_w"] = layer.v.weight.detach().cpu().to(torch.float32).numpy()
            arrays[f"l{i}_v_b"] = layer.v.bias.detach().cpu().to(torch.float32).numpy()
            arrays[f"l{i}_o_w"] = layer.o.weight.detach().cpu().to(torch.float32).numpy()
            arrays[f"l{i}_o_b"] = layer.o.bias.detach().cpu().to(torch.float32).numpy()
            arrays[f"l{i}_ln1_w"] = layer.ln1.weight.detach().cpu().to(torch.float32).numpy()
            arrays[f"l{i}_ln1_b"] = layer.ln1.bias.detach().cpu().to(torch.float32).numpy()
            arrays[f"l{i}_fc1_w"] = layer.fc1.weight.detach().cpu().to(torch.float32).numpy()
            arrays[f"l{i}_fc1_b"] = layer.fc1.bias.detach().cpu().to(torch.float32).numpy()
            arrays[f"l{i}_fc2_w"] = layer.fc2.weight.detach().cpu().to(torch.float32).numpy()
            arrays[f"l{i}_fc2_b"] = layer.fc2.bias.detach().cpu().to(torch.float32).numpy()
            arrays[f"l{i}_ln2_w"] = layer.ln2.weight.detach().cpu().to(torch.float32).numpy()
            arrays[f"l{i}_ln2_b"] = layer.ln2.bias.detach().cpu().to(torch.float32).numpy()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **arrays)


# ---------------------------------------------------------------------------
# Exact single-root Chu-Liu/Edmonds matching `src/doc/syntax.rs`
# ---------------------------------------------------------------------------


def chu_liu_edmonds(arc_scores: np.ndarray) -> list[int]:
    """Exact single-root MSA over 0..=m matching `src/doc/syntax.rs`."""
    m = arc_scores.shape[0]
    if m == 0:
        return []
    if m == 1:
        return [0]

    num_nodes = m + 1
    edges: list[tuple[int, int, int, float, int]] = []  # (u, v, root_pen, score, parent_idx)
    for v_idx in range(m):
        v = v_idx + 1
        for u in range(m + 1):
            if u == v:
                continue
            raw = float(arc_scores[v_idx, u])
            if math.isnan(raw):
                raw = -3.4e38
            edges.append((u, v, -1 if u == 0 else 0, raw, 0))

    def gt(w1: tuple[int, float], w2: tuple[int, float]) -> bool:
        return w1[0] > w2[0] or (w1[0] == w2[0] and w1[1] > w2[1])

    def msa(n_nodes: int, root: int, ed_list: list[tuple[int, int, int, float, int]]) -> list[int]:
        best_in: list[int | None] = [None] * n_nodes
        for idx, (u, v, rp, sc, _) in enumerate(ed_list):
            if v == root or u == v:
                continue
            prev = best_in[v]
            if prev is None or gt((rp, sc), (ed_list[prev][2], ed_list[prev][3])):
                best_in[v] = idx

        visited = [-1] * n_nodes
        comp = [-1] * n_nodes
        in_cycle = [False] * n_nodes
        num_cycles = 0
        for start in range(n_nodes):
            cur = start
            while cur != root and comp[cur] == -1 and visited[cur] == -1:
                visited[cur] = start
                bi = best_in[cur]
                assert bi is not None
                cur = ed_list[bi][0]
            if cur != root and comp[cur] == -1 and visited[cur] == start:
                cid = num_cycles
                num_cycles += 1
                walk = cur
                while True:
                    comp[walk] = cid
                    in_cycle[walk] = True
                    bi = best_in[walk]
                    assert bi is not None
                    walk = ed_list[bi][0]
                    if walk == cur:
                        break

        if num_cycles == 0:
            return [best_in[v] for v in range(n_nodes) if v != root]  # type: ignore[misc]

        total_comps = num_cycles
        for i in range(n_nodes):
            if comp[i] == -1:
                comp[i] = total_comps
                total_comps += 1
        new_root = comp[root]

        best_pair: dict[tuple[int, int], tuple[int, int, int, float, int]] = {}
        for idx, (u, v, rp, sc, _) in enumerate(ed_list):
            cu, cv = comp[u], comp[v]
            if cu == cv or cv == new_root:
                continue
            if in_cycle[v]:
                ci = best_in[v]
                assert ci is not None
                rp_adj = rp - ed_list[ci][2]
                sc_adj = sc - ed_list[ci][3]
            else:
                rp_adj, sc_adj = rp, sc
            cand = (cu, cv, rp_adj, sc_adj, idx)
            prev_c = best_pair.get((cu, cv))
            if prev_c is None or gt((rp_adj, sc_adj), (prev_c[2], prev_c[3])):
                best_pair[(cu, cv)] = cand

        new_edges = list(best_pair.values())
        sub_chosen = msa(total_comps, new_root, new_edges)

        chosen_node = [-1 if v == root else best_in[v] for v in range(n_nodes)]  # type: ignore[misc]
        for ne_idx in sub_chosen:
            p_idx = new_edges[ne_idx][4]
            target_v = ed_list[p_idx][1]
            chosen_node[target_v] = p_idx
        return [chosen_node[v] for v in range(n_nodes) if v != root]  # type: ignore[misc]

    chosen_idxs = msa(num_nodes, 0, edges)
    return [edges[idx][0] for idx in chosen_idxs]


# ---------------------------------------------------------------------------
# Sentence-level dataset preparation (exact `nlp_doc` sentence spans)
# ---------------------------------------------------------------------------


@dataclass
class SentExample:
    item_id: str
    genre: str
    lang: str
    variant: str
    input_ids: list[int]
    subword_indices: list[int]  # length m, values in 1..len(input_ids)-1
    upos_ids: list[int] | None  # length m (None on probe where gold UPOS is omitted)
    l2_upos_ids: list[int | None]  # Layer-2 decided UPOS
    heads: list[int]  # length m, values in 0..=m (sentence-local 1-based, 0 = root)
    deprel_ids: list[int]  # length m, values in 0..37


def prepare_examples(
    jsonl_path: Path,
    nlp_doc: Path,
    tok: Tokenizer,
    texts_override: dict[str, dict] | None = None,
) -> tuple[list[SentExample], int, int]:
    rows = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    texts = [
        texts_override[r["id"]]["text"] if texts_override else r["text"]
        for r in rows
    ]

    # Run nlp_doc per item or in paragraph chunks to get exact sentence char spans
    examples: list[SentExample] = []
    unmapped_sents = 0
    collisions = 0

    chunk_size = 200
    for start in range(0, len(rows), chunk_size):
        sub_rows = rows[start : start + chunk_size]
        sub_texts = texts[start : start + chunk_size]
        joined = "\n\n".join(sub_texts)
        proc = subprocess.run(
            [str(nlp_doc)], input=joined.encode("utf-8"), capture_output=True, check=True
        )
        doc = json.loads(proc.stdout.decode("utf-8"))
        paras = doc["segmentation"]["paragraphs"]
        assert len(paras) == len(sub_rows), "unexpected multi-paragraph item in dataset"

        sents = doc["segmentation"]["sentences"]
        all_toks = doc["tokens"]
        sent_by_para: dict[int, list[tuple[int, dict]]] = {pi: [] for pi in range(len(paras))}
        for si, s_obj in enumerate(sents):
            sent_by_para[s_obj["paragraph"]].append((si, s_obj))
        toks_by_sent: dict[int, list[dict]] = {si: [] for si in range(len(sents))}
        for tk in all_toks:
            toks_by_sent[tk["sentence"]].append(tk)

        for pi, (r, txt) in enumerate(zip(sub_rows, sub_texts)):
            p_base = paras[pi]["span"]["start_char"]
            meta = texts_override[r["id"]] if texts_override else r
            genre = meta.get("genre", "unknown")
            lang = meta.get("lang", "unknown")
            variant = meta.get("variant", lang)

            para_sents = sent_by_para[pi]
            assert len(para_sents) == len(r["sentences"])

            for (si, s_obj), (tok_lo, tok_hi) in zip(para_sents, r["sentences"]):
                s_start = s_obj["span"]["start_char"] - p_base
                s_end = s_obj["span"]["end_char"] - p_base
                sent_str = txt[s_start:s_end]
                s_toks = toks_by_sent[si]
                m = tok_hi - tok_lo
                assert len(s_toks) == m

                enc = tok.encode(sent_str)
                if len(enc.ids) == 0 or len(enc.ids) > MAX_LEN:
                    unmapped_sents += 1
                    continue

                sub_idxs: list[int] = []
                ok = True
                for tk in s_toks:
                    ts = tk["span"]["start_char"] - s_obj["span"]["start_char"]
                    te = tk["span"]["end_char"] - s_obj["span"]["start_char"]
                    found = next(
                        (
                            idx
                            for idx, (os, oe) in enumerate(enc.offsets)
                            if idx >= 1 and os < oe and os < te and ts < oe
                        ),
                        None,
                    )
                    if found is None:
                        ok = False
                        break
                    sub_idxs.append(found)
                if not ok:
                    unmapped_sents += 1
                    continue
                collisions += len(sub_idxs) - len(set(sub_idxs))

                local_heads: list[int] = []
                for gh in r["heads"][tok_lo:tok_hi]:
                    if gh == 0:
                        local_heads.append(0)
                    else:
                        local_heads.append(gh - tok_lo)
                local_rels = [DEPREL_TO_ID[d] for d in r["deprels"][tok_lo:tok_hi]]
                local_upos = (
                    [UPOS_TO_ID[u] for u in r["upos"][tok_lo:tok_hi]]
                    if "upos" in r
                    else None
                )
                l2_u = [
                    UPOS_TO_ID[tk["upos"]] if tk.get("upos") in UPOS_TO_ID else None
                    for tk in s_toks
                ]
                examples.append(
                    SentExample(
                        item_id=r["id"],
                        genre=genre,
                        lang=lang,
                        variant=variant,
                        input_ids=list(enc.ids),
                        subword_indices=sub_idxs,
                        upos_ids=local_upos,
                        l2_upos_ids=l2_u,
                        heads=local_heads,
                        deprel_ids=local_rels,
                    )
                )
    return examples, unmapped_sents, collisions


def compute_batch_loss(
    model: DocSyntaxModel, batch: list[SentExample], device: torch.device
) -> torch.Tensor:
    max_s = min(MAX_LEN, max(len(ex.input_ids) for ex in batch))
    # Round up to multiple of 8 for clean tensor shapes
    seq_len = min(MAX_LEN, ((max_s + 7) // 8) * 8)
    bsz = len(batch)

    ids_np = np.zeros((bsz, seq_len), dtype=np.int64)
    for bi, ex in enumerate(batch):
        ids_np[bi, : len(ex.input_ids)] = ex.input_ids
    input_ids = torch.from_numpy(ids_np).to(device)

    h = model.encode_hidden(input_ids)
    upos_logits, arc_logits, r_dep, r_head, r_bi_dep, r_bi_head = model.forward_heads(h)

    upos_preds: list[torch.Tensor] = []
    upos_tgts: list[int] = []
    arc_losses: list[torch.Tensor] = []
    rel_preds: list[torch.Tensor] = []
    rel_tgts: list[int] = []

    for bi, ex in enumerate(batch):
        s_idx = torch.tensor(ex.subword_indices, dtype=torch.long, device=device)
        cand_s = torch.tensor([0] + ex.subword_indices, dtype=torch.long, device=device)

        if ex.upos_ids is not None:
            upos_preds.append(upos_logits[bi, s_idx])
            upos_tgts.extend(ex.upos_ids)

        # Arc logits for u in 1..=m over candidate heads [0, s_1, ..., s_m] (shape [m, m + 1])
        arc_mat = arc_logits[bi].index_select(0, s_idx).index_select(1, cand_s)
        tgt_heads = torch.tensor(ex.heads, dtype=torch.long, device=device)
        arc_losses.append(F.cross_entropy(arc_mat, tgt_heads, label_smoothing=0.02))

        # Relation logits at gold head positions s_v = cand_s[tgt_heads]
        s_v = cand_s.index_select(0, tgt_heads)
        rel_mat = (
            r_dep[bi].index_select(0, s_idx)
            + r_head[bi].index_select(0, s_v)
            + r_bi_dep[bi].index_select(0, s_idx) * r_bi_head[bi].index_select(0, s_v)
        )
        rel_preds.append(rel_mat)
        rel_tgts.extend(ex.deprel_ids)

    loss_arc = torch.stack(arc_losses).mean()
    loss_rel = F.cross_entropy(
        torch.cat(rel_preds, dim=0),
        torch.tensor(rel_tgts, dtype=torch.long, device=device),
        label_smoothing=0.02,
    )
    if upos_preds:
        loss_upos = F.cross_entropy(
            torch.cat(upos_preds, dim=0),
            torch.tensor(upos_tgts, dtype=torch.long, device=device),
            label_smoothing=0.02,
        )
    else:
        loss_upos = torch.zeros((), device=device)

    return loss_arc + loss_rel + 0.6 * loss_upos


@torch.no_grad()
def evaluate_examples(
    model: DocSyntaxModel, examples: list[SentExample], device: torch.device
) -> dict:
    model.eval()
    total_toks = 0
    uas_ok = 0
    las_ok = 0
    upos_total = 0
    upos_ok = 0
    low_conf_toks = 0

    by_variant: dict[str, list[int]] = {}  # variant -> [toks, uas, las, upos_tot, upos_ok]

    batch_size = 64
    for start in range(0, len(examples), batch_size):
        sub = examples[start : start + batch_size]
        seq_len = min(MAX_LEN, ((max(len(ex.input_ids) for ex in sub) + 7) // 8) * 8)
        ids_np = np.zeros((len(sub), seq_len), dtype=np.int64)
        for bi, ex in enumerate(sub):
            ids_np[bi, : len(ex.input_ids)] = ex.input_ids
        h = model.encode_hidden(torch.from_numpy(ids_np).to(device))
        upos_l, arc_l, r_dep, r_head, r_bi_dep, r_bi_head = model.forward_heads(h)

        upos_np = upos_l.cpu().numpy()
        arc_np = arc_l.cpu().numpy()
        r_dep_np = r_dep.cpu().numpy()
        r_head_np = r_head.cpu().numpy()
        r_bi_dep_np = r_bi_dep.cpu().numpy()
        r_bi_head_np = r_bi_head.cpu().numpy()

        for bi, ex in enumerate(sub):
            v_stat = by_variant.setdefault(ex.variant, [0, 0, 0, 0, 0])
            m = len(ex.subword_indices)
            cand = [0] + ex.subword_indices
            arc_scores = arc_np[bi][np.ix_(ex.subword_indices, cand)]  # [m, m + 1]
            pred_heads = chu_liu_edmonds(arc_scores)

            for u_zero, s_u in enumerate(ex.subword_indices):
                total_toks += 1
                v_stat[0] += 1

                # UPOS matching `src/doc/syntax.rs` (Layer-2 decided kept, else upgrade if >= 0.50)
                if ex.upos_ids is not None:
                    upos_total += 1
                    v_stat[3] += 1
                    if ex.l2_upos_ids[u_zero] is not None:
                        pred_u = ex.l2_upos_ids[u_zero]
                    else:
                        ul = upos_np[bi, s_u]
                        ul_e = np.exp(ul - np.max(ul))
                        ul_p = ul_e / np.sum(ul_e)
                        best_u = int(np.argmax(ul))
                        pred_u = best_u if float(ul_p[best_u]) >= 0.50 else -1
                    if pred_u == ex.upos_ids[u_zero]:
                        upos_ok += 1
                        v_stat[4] += 1

                pv = pred_heads[u_zero]
                row_arc = arc_scores[u_zero]
                ra_e = np.exp(row_arc - np.max(row_arc))
                p_arc = float(ra_e[pv] / np.sum(ra_e))

                if pv == 0:
                    pred_rel = ROOT_DEPREL_ID
                    p_rel = 1.0
                else:
                    s_v = ex.subword_indices[pv - 1]
                    rl = (
                        r_dep_np[bi, s_u]
                        + r_head_np[bi, s_v]
                        + r_bi_dep_np[bi, s_u] * r_bi_head_np[bi, s_v]
                    )
                    non_root_mask = np.arange(37) != ROOT_DEPREL_ID
                    nr_logits = rl[non_root_mask]
                    nr_best = int(np.argmax(nr_logits))
                    nr_e = np.exp(nr_logits - np.max(nr_logits))
                    p_rel = float(nr_e[nr_best] / np.sum(nr_e))
                    pred_rel = nr_best if nr_best < ROOT_DEPREL_ID else nr_best + 1

                milli = int(round(min(max(p_arc * p_rel, 0.0), 1.0) * 1000.0))
                if milli < 350:
                    low_conf_toks += 1
                    continue

                if pv == ex.heads[u_zero]:
                    uas_ok += 1
                    v_stat[1] += 1
                    if pred_rel == ex.deprel_ids[u_zero]:
                        las_ok += 1
                        v_stat[2] += 1

    return {
        "tokens": total_toks,
        "uas": round(uas_ok / max(1, total_toks), 4),
        "las": round(las_ok / max(1, total_toks), 4),
        "upos": round(upos_ok / max(1, upos_total), 4) if upos_total else None,
        "low_conf_rate": round(low_conf_toks / max(1, total_toks), 4),
        "by_variant": {
            k: {
                "tokens": st[0],
                "uas": round(st[1] / max(1, st[0]), 4),
                "las": round(st[2] / max(1, st[0]), 4),
                "upos": round(st[4] / max(1, st[3]), 4) if st[3] else None,
            }
            for k, st in sorted(by_variant.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=ROOT / "training/.work/syntax")
    parser.add_argument("--nlp-doc", type=Path, default=ROOT / "target/release/nlp_doc")
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--lr-enc", type=float, default=6e-5)
    parser.add_argument("--lr-head", type=float, default=8e-4)
    parser.add_argument("--seed", type=int, default=20261111)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    hf_dir = find_hf_snapshot("intfloat/multilingual-e5-small")
    tok_path = args.work_dir / "tokenizer.json"
    tok, orig_indices = build_pruned_tokenizer_and_embedding_map(hf_dir, tok_path)
    print(f"Built pruned tokenizer ({VOCAB_SIZE} pieces) from {hf_dir.name} -> {tok_path}")

    train_ex, tr_unmap, tr_coll = prepare_examples(args.work_dir / "train.jsonl", args.nlp_doc, tok)
    dev_ex, dv_unmap, dv_coll = prepare_examples(args.work_dir / "dev.jsonl", args.nlp_doc, tok)
    assert tr_unmap == 0 and dv_unmap == 0, f"unmapped sentences: train={tr_unmap}, dev={dv_unmap}"
    assert tr_coll == 0 and dv_coll == 0, f"subword collisions: train={tr_coll}, dev={dv_coll}"

    probe_texts = {
        r["id"]: r
        for r in (
            json.loads(l)
            for l in (ROOT / "training/probe/probe_texts.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()
        )
    }
    probe_ex, pr_unmap, pr_coll = prepare_examples(
        ROOT / "training/probe/annotations.jsonl",
        args.nlp_doc,
        tok,
        texts_override=probe_texts,
    )
    assert pr_unmap == 0 and pr_coll == 0, f"probe unmapped={pr_unmap}, coll={pr_coll}"
    print(
        f"Prepared sentences: train={len(train_ex)}, dev={len(dev_ex)}, "
        f"probe(read-only OOD)={len(probe_ex)} | 0 unmapped, 0 collisions"
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training {args.num_layers}-layer DocSyntaxModel on device={device} ...")

    model = DocSyntaxModel(num_layers=args.num_layers, dropout=0.15)
    model.load_pretrained_e5(hf_dir, orig_indices)
    model.to(device)

    enc_params = list(model.word_embeddings.parameters()) + list(model.emb_ln.parameters()) + list(model.layers.parameters())
    head_params = [
        p
        for n, p in model.named_parameters()
        if not n.startswith(("word_embeddings.", "emb_ln.", "layers."))
    ]

    optimizer = torch.optim.AdamW(
        [
            {"params": enc_params, "lr": args.lr_enc, "weight_decay": 0.01},
            {"params": head_params, "lr": args.lr_head, "weight_decay": 0.01},
        ]
    )
    total_steps = args.epochs * math.ceil(len(train_ex) / args.batch_size)
    warmup_steps = max(20, total_steps // 10)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        prog = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_dev_las = -1.0
    weights_path = args.work_dir / "doc_syntax_v1_weights.npz"
    metrics_log: list[dict] = []
    rng = np.random.default_rng(args.seed)

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        model.train()
        # Length-bucketed mini-batches with random local shuffle for speed & diversity
        order = rng.permutation(len(train_ex)).tolist()
        # Group into chunks of 8 * batch_size, sort by length inside chunk, slice into batches
        batches: list[list[SentExample]] = []
        pool_sz = args.batch_size * 8
        for i in range(0, len(order), pool_sz):
            sub_ord = sorted(order[i : i + pool_sz], key=lambda idx: len(train_ex[idx].input_ids))
            for j in range(0, len(sub_ord), args.batch_size):
                batches.append([train_ex[k] for k in sub_ord[j : j + args.batch_size]])
        rng.shuffle(batches)

        running_loss = 0.0
        for batch in batches:
            optimizer.zero_grad(set_to_none=True)
            loss = compute_batch_loss(model, batch, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += float(loss.item())
            # Keep pad_id = 0 embedding strictly zero
            with torch.no_grad():
                model.word_embeddings.weight[0].zero_()

        dt = time.perf_counter() - t0
        dev_m = evaluate_examples(model, dev_ex, device)
        probe_m = evaluate_examples(model, probe_ex, device)
        avg_loss = running_loss / max(1, len(batches))
        rec = {
            "epoch": epoch,
            "sec": round(dt, 2),
            "loss": round(avg_loss, 4),
            "dev": dev_m,
            "probe_ood": {
                "uas": probe_m["uas"],
                "las": probe_m["las"],
                "low_conf_rate": probe_m["low_conf_rate"],
                "by_variant": probe_m["by_variant"],
            },
        }
        metrics_log.append(rec)
        print(
            f"Epoch {epoch:02d}/{args.epochs} ({dt:.1f}s) | loss={avg_loss:.4f} | "
            f"dev UAS={dev_m['uas']:.4f} LAS={dev_m['las']:.4f} UPOS={dev_m['upos']:.4f} | "
            f"probe(OOD) UAS={probe_m['uas']:.4f} LAS={probe_m['las']:.4f} low_conf={probe_m['low_conf_rate']:.4f}",
            flush=True,
        )
        if dev_m["las"] >= best_dev_las:
            best_dev_las = dev_m["las"]
            model.save_npz(weights_path)

    (args.work_dir / "train_metrics.json").write_text(
        json.dumps(metrics_log, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Saved best float32 weights (dev LAS={best_dev_las:.4f}) -> {weights_path}")


if __name__ == "__main__":
    main()
