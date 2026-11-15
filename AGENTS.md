## ja → zh-Hant development rules

The rules in [docs/ja-zh-development-rules.md](docs/ja-zh-development-rules.md) are mandatory for all ja → zh work. Hard limits:

- One writer for each set of files. Codex owns the engine (`crates/faberline-nlp-translate/src/**`); Antigravity owns the lexicon (`crates/faberline-nlp/lexicon/**`, `tools/lexicon_build/**`).
- Nobody reads `test` rows. Only the aggregate test number is reported.
- Ratchet: no dev phenomenon drops more than 0.5; overall and set/dev must not drop.
- The author does not accept their own PR; the other agent checks it.
- Data licences allowed: MIT, Apache-2.0, Unicode-3.0, public domain.
- Each agent merges only with the owner's permission in its own session.
