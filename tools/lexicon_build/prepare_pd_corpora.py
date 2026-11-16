#!/usr/bin/env python3
"""Prepare deterministic, verifiable Project Gutenberg PD JSONL corpora for C1 lexicon builds."""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

EN_PG_BOOKS: list[tuple[str, str, int, str, str]] = [
    ("en-US", "novel", 76, "Huckleberry_Finn", "d617a37aa7ae1e1a93dcde2634db2bccb86824e31e29a55230bdbf77d6872d59"),
    ("en-US", "novel", 2701, "Moby_Dick", "b5e88b71a1cda068e9ab1d94344c9d932338ba6da1a66def97a1aecd9cfcc87d"),
    ("en-US", "novel", 64317, "Great_Gatsby", "ce760ec377accd352b41bb8f64504a72d7aa18ab3afb42ded2b56cecacf29e35"),
    ("en-US", "novel", 541, "Age_of_Innocence", "aae279f1a60ac64be53f63357e708a2fdaabdb334a1c92f81fc1c3b5c848a092"),
    ("en-US", "essay", 16643, "Emerson_Essays", "bd1e8cd9feb2c331b560dbb254cde873cb3db4b5d5464996d98f6e43649ad1e1"),
    ("en-US", "essay", 205, "Thoreau_Walden", "2d9a76a2e3e8195c69430516ebd33c4d0757a53ad432ff6186b7b794e6fe99f9"),
    ("en-US", "essay", 5116, "William_James_Pragmatism", "92c8654ebc3052887740ab0ce8cd953456f63ba3f7749d31377488d807765db6"),
    ("en-US", "essay", 408, "Du_Bois_Souls_of_Black_Folk", "87f8e832484792b761a510b953d67e3835fa46a9f4231a508bfeb914a98559ca"),
    ("en-GB", "essay", 575, "Bacon_Essays", "4434aa015d325e65816eb0e694e7c573fa4b71523530e45adfeb23e7fcf63bf1"),
    ("en-GB", "essay", 5827, "Russell_Problems_of_Philosophy", "2505ea57849c64abe7962ffe56972174c9160606348e08cb7876d46b2c5b2d96"),
    ("en-GB", "essay", 130, "Chesterton_Orthodoxy", "ec525c93afddb05465bf6fa0af8f873078b8e42a2a24f9b999582994f054fdf2"),
    ("en-GB", "essay", 470, "Chesterton_Heretics", "291fb27d68c0cfca7523032a42d57a743e62a6e58faf9cc728f78ef748703829"),
]

ZH_PG_BOOKS: list[tuple[int, int, str]] = [
    (23950, 80, "ed422e10c16c19af7ca7b09ca0320012e60776b3d7cb054769b8f7d9a19aea7d"),
    (23962, 70, "bd18088ccb70d3164e3a4bc9805a6ea72095d6cc3891e4f172ed0a3d281292bb"),
]

OPENCC_TABLES: dict[str, str] = {
    "STPhrases.txt": "24d23700f4f3d1860b899aa90896dcc7ec56c535a0b3a246bbf5a7048c7bdeb8",
    "STCharacters.txt": "a0ca1601c70648cf48b33c3c6210ccbecc5c7eead4b4c3daf76587ba2c03582b",
    "TWPhrases.txt": "4d3f83634d8a02be5fc82def7e6ae418ab2ab501b100952da5ef934460813a41",
    "TWVariants.txt": "245b94eb5842957e735dd44b7e7d4ff469a3643126cc8fa511adda5281e9cb86",
    "TWVariantsPhrases.txt": "36df033675a2e9152927fa8419f0732a061cb4909c5060f25a5cfc9b08ffff06",
    "TSPhrases.txt": "1b1e60a3d50492bc088bcd4372152570b5f0ccbe2ded0b66c90b6b8eeb5f849f",
    "TSCharacters.txt": "9ff46a7d30e5765375eb13d33f2b03a34d298913caf2b120380679f33ae1642d",
    "TWPhrasesRev.txt": "46d6165c6ff17dbcb169fe5ee72c410f5a9555b64ca99f067bb2d0b30248a2f6",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_sha256(path: Path, expected: str) -> bytes:
    raw = path.read_bytes()
    got = sha256_bytes(raw)
    if got != expected:
        raise ValueError(f"SHA-256 mismatch for {path.name}: expected {expected}, got {got}")
    return raw


def strip_gutenberg_boilerplate(text: str) -> str:
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG EBOOK",
        "*** START OF THIS PROJECT GUTENBERG EBOOK",
        "*END*THE SMALL PRINT",
    ]
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG EBOOK",
        "*** END OF THIS PROJECT GUTENBERG EBOOK",
        "End of the Project Gutenberg EBook",
        "End of Project Gutenberg's",
    ]
    start_idx = 0
    for m in start_markers:
        pos = text.find(m)
        if pos != -1:
            nl = text.find("\n", pos)
            start_idx = nl + 1 if nl != -1 else pos + len(m)
            break
    end_idx = len(text)
    for m in end_markers:
        pos = text.find(m, start_idx)
        if pos != -1:
            end_idx = pos
            break
    return text[start_idx:end_idx]


def chunk_en_paragraphs(text: str, min_chars: int = 350, max_chars: int = 1400, target_count: int = 55) -> list[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    raw_paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    paras = []
    for p in raw_paras:
        p_clean = re.sub(r"\s+", " ", p).strip()
        if len(p_clean) < 40:
            continue
        if re.match(r"^(CHAPTER|Chapter|BOOK|Book|PART|Part|ACT|SCENE|I+|V+|X+)\b", p_clean) and len(p_clean) < 80:
            continue
        if p_clean.startswith("[Illustration") or p_clean.startswith("Transcriber"):
            continue
        paras.append(p_clean)

    passages = []
    buf: list[str] = []
    buf_len = 0
    for p in paras:
        if len(p) > max_chars:
            if buf and buf_len >= min_chars:
                passages.append("\n\n".join(buf))
            buf = []
            buf_len = 0
            cut = p[:max_chars]
            m = re.search(r"[.!?][\"'”’]?\s+[^.!?]*$", cut)
            if m and m.start() >= min_chars:
                passages.append(cut[: m.start() + 1].strip())
            continue
        if buf_len + len(p) + 2 > max_chars and buf_len >= min_chars:
            passages.append("\n\n".join(buf))
            buf = [p]
            buf_len = len(p)
        else:
            buf.append(p)
            buf_len += len(p) + 2
    if buf and buf_len >= min_chars:
        passages.append("\n\n".join(buf))

    if not passages:
        return []
    step = max(1, len(passages) // target_count)
    return passages[::step][:target_count]


def count_han(s: str) -> int:
    return sum(1 for c in s if "\u4e00" <= c <= "\u9fff")


def chunk_zh_text(text: str, min_chars: int = 350, max_chars: int = 950) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
    chunks = []
    cur: list[str] = []
    cur_len = 0
    for p in paras:
        if len(p) < 10 and not cur:
            continue
        if cur_len + len(p) + 1 > max_chars and cur_len >= min_chars:
            chunks.append("\n".join(cur))
            cur = [p]
            cur_len = len(p)
        else:
            cur.append(p)
            cur_len += len(p) + 1
            if cur_len >= max_chars:
                chunks.append("\n".join(cur))
                cur = []
                cur_len = 0
    if cur and cur_len >= 120:
        chunks.append("\n".join(cur))
    return [c for c in chunks if count_han(c) >= 80]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare clean Project Gutenberg PD JSONL corpora.")
    parser.add_argument("--en-raw-dir", type=Path, required=True, help="Directory with EN pg*.txt files")
    parser.add_argument("--zh-raw-dir", type=Path, required=True, help="Directory with ZH pg23950.txt, pg23962.txt")
    parser.add_argument("--opencc-dir", type=Path, required=True, help="Directory with OpenCC tables")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for JSONL corpora")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Verify OpenCC tables and import OpenCCConverter
    for fname, expected_sha in OPENCC_TABLES.items():
        verify_sha256(args.opencc_dir / fname, expected_sha)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_zh import OpenCCConverter  # noqa: E402

    conv = OpenCCConverter(args.opencc_dir)

    # 2. Build en_pg_pd_660.jsonl
    en_docs: list[dict] = []
    cell_counts: dict[tuple[str, str], int] = {}
    for lang_variant, genre, gid, title, expected_sha in EN_PG_BOOKS:
        raw_bytes = verify_sha256(args.en_raw_dir / f"pg{gid}.txt", expected_sha)
        raw_text = raw_bytes.decode("utf-8", errors="replace")
        body = strip_gutenberg_boilerplate(raw_text)
        chunks = chunk_en_paragraphs(body, target_count=55)
        for ch in chunks:
            idx = cell_counts.get((lang_variant, genre), 0)
            cell_counts[(lang_variant, genre)] = idx + 1
            en_docs.append({
                "id": f"{lang_variant}_{genre}_{idx:04d}",
                "lang_variant": lang_variant,
                "genre": genre,
                "source": f"gutenberg:{gid}:{title}",
                "text": ch,
            })

    en_out = args.out_dir / "en_pg_pd_660.jsonl"
    en_out.write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in en_docs), encoding="utf-8")

    # 3. Build zh_hant_pg_pd_150.jsonl and zh_hans_pg_projected_150.jsonl
    zh_hant_docs: list[dict] = []
    zh_hans_docs: list[dict] = []
    for gid, max_chunks, expected_sha in ZH_PG_BOOKS:
        raw_bytes = verify_sha256(args.zh_raw_dir / f"pg{gid}.txt", expected_sha)
        txt = raw_bytes.decode("utf-8", errors="replace")
        m_start = re.search(r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK .*? \*\*\*", txt)
        m_end = re.search(r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK .*? \*\*\*", txt)
        if m_start:
            txt = txt[m_start.end() :]
        if m_end:
            txt = txt[: m_end.start()]
        chunks = chunk_zh_text(txt, min_chars=350, max_chars=950)
        if len(chunks) > max_chunks:
            step = len(chunks) / max_chunks
            chunks = [chunks[int(i * step)] for i in range(max_chunks)]
        for i, c in enumerate(chunks):
            zh_hant_docs.append({"id": f"gutenberg-{gid}-{i}", "source": f"gutenberg:{gid}", "text": c})
            zh_hans_docs.append({
                "id": f"gutenberg-{gid}-{i}",
                "source": f"gutenberg:{gid}",
                "text": conv.hant_to_hans(c),
            })

    zh_hant_out = args.out_dir / "zh_hant_pg_pd_150.jsonl"
    zh_hans_out = args.out_dir / "zh_hans_pg_projected_150.jsonl"
    zh_hant_out.write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in zh_hant_docs), encoding="utf-8")
    zh_hans_out.write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in zh_hans_docs), encoding="utf-8")

    for label, p, count in [
        ("en_pg_pd_660.jsonl", en_out, len(en_docs)),
        ("zh_hant_pg_pd_150.jsonl", zh_hant_out, len(zh_hant_docs)),
        ("zh_hans_pg_projected_150.jsonl", zh_hans_out, len(zh_hans_docs)),
    ]:
        b = p.read_bytes()
        print(f"{label}: docs={count} bytes={len(b)} sha256={sha256_bytes(b)}")


if __name__ == "__main__":
    main()
