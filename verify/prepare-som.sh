#!/usr/bin/env bash
set -euo pipefail

git -C som worktree add --detach "$RUNNER_TEMP/som-baseline" HEAD
git -C som apply --check "$GITHUB_WORKSPACE/verify/som.patch"
git -C som apply "$GITHUB_WORKSPACE/verify/som.patch"
mkdir -p som/som-rs/vendor
cp -R cnl som/som-rs/vendor/faberline-nlp-cnl
cp verify/vendor-readme.md som/som-rs/vendor/faberline-nlp-cnl/README.md
cp -R verify/repo-examples/cnl-dsl som/som-rs/tests/cnl-dsl
cp verify/repo-examples/cnl_repo_examples.rs som/som-rs/tests/
git -C som diff --check
shasum -a 256 verify/som.patch cnl/Cargo.toml cnl/src/lib.rs cnl/src/som_cnl.pest \
  verify/repo-examples/cnl_repo_examples.rs verify/repo-examples/cnl-dsl/*
