#!/usr/bin/env python3
"""Run the repo's nlp_doc binary (layers 1 and 2) over probe texts and print the tokens.

usage: dump_tokens.py <nlp_doc binary> <texts.jsonl>  (each line: {"id","genre","lang","text"})
Writes JSON Lines to stdout: id, genre, lang, text, sentences (token index ranges), tokens (strings, upos).
Only the standard library; runs a local binary, no network.
"""
import json, subprocess, sys

def main():
    binary, path = sys.argv[1], sys.argv[2]
    for line in open(path, encoding="utf-8"):
        item = json.loads(line)
        text = item["text"]
        out = subprocess.run([binary], input=text.encode("utf-8"), capture_output=True, check=True).stdout
        doc = json.loads(out)
        toks = doc["tokens"]
        sents = {}
        for i, t in enumerate(toks):
            sents.setdefault(t["sentence"], []).append(i)
        item["tokens"] = [text[t["span"]["start_char"]:t["span"]["end_char"]] for t in toks]
        item["upos"] = [t.get("upos") for t in toks]
        item["sentences"] = [[v[0], v[-1] + 1] for _, v in sorted(sents.items())]
        print(json.dumps(item, ensure_ascii=False))

main()
