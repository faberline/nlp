//! Cargo integration tests. All source text and vocabulary are synthetic.
use std::{
    fs,
    process::{Command, Output},
};

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tempfile::{TempDir, tempdir};

const BIN: &str = env!("CARGO_BIN_EXE_nlp_source_stream");

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

struct Fixture {
    dir: TempDir,
    chapters: Vec<String>,
}

impl Fixture {
    fn new(chapters: &[&str]) -> Self {
        let dir = tempdir().expect("synthetic fixture directory");
        let mut entries = Vec::new();
        for (index, text) in chapters.iter().enumerate() {
            let file = format!("chapter-{}.txt", index + 1);
            fs::write(dir.path().join(&file), text).expect("synthetic chapter");
            entries.push(
                json!({"order": index + 1, "title": format!("合成{}", index + 1),
                "file": file, "status": "ok"}),
            );
        }
        fs::write(
            dir.path().join("manifest.json"),
            serde_json::to_vec(&json!({
                "count": entries.len(), "chapters": entries
            }))
            .unwrap(),
        )
        .expect("manifest");
        fs::write(
            dir.path().join("work.json"),
            serde_json::to_vec(&json!({
                "actors": [
                    {"surface": "小甲", "canonical": "synthetic-a"},
                    {"surface": "小乙", "canonical": "synthetic-b"}
                ],
                "objects": [{"surface": "劍", "canonical": "synthetic-object"}],
                "footers": ["<end>"], "line_footers": ["<line>"]
            }))
            .unwrap(),
        )
        .expect("work vocabulary");
        Self {
            dir,
            chapters: chapters.iter().map(|text| (*text).to_owned()).collect(),
        }
    }

    fn run(&self, args: &[&str], work: bool) -> Output {
        let mut cmd = Command::new(BIN);
        cmd.arg("--manifest")
            .arg(self.dir.path().join("manifest.json"));
        if work {
            cmd.arg("--work-lexicon")
                .arg(self.dir.path().join("work.json"));
        }
        cmd.args(args).output().expect("start source stream")
    }

    fn chapter(&self, number: usize) -> &str {
        &self.chapters[number - 1]
    }
}

fn stream(output: &Output, schema: &str) -> Vec<Value> {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stdout.ends_with(b"\n"));
    let lines: Vec<&[u8]> = output
        .stdout
        .split_inclusive(|byte| *byte == b'\n')
        .collect();
    let rows: Vec<Value> = lines
        .iter()
        .map(|line| serde_json::from_slice(line).unwrap())
        .collect();
    assert_eq!(rows[0]["type"], "stream_start");
    assert_eq!(rows[0]["schema"], schema);
    let receipt = rows.last().unwrap();
    assert_eq!(receipt["type"], "stream_complete");
    assert_eq!(
        receipt["prior_records"].as_u64().unwrap() as usize,
        lines.len() - 1
    );
    let prefix_len = output.stdout.len() - lines.last().unwrap().len();
    assert_eq!(
        receipt["prior_records_sha256"],
        sha(&output.stdout[..prefix_len])
    );
    rows
}

fn selected<'a>(rows: &'a [Value], kind: &str) -> Vec<&'a Value> {
    rows.iter().filter(|row| row["type"] == kind).collect()
}

fn source_slice(text: &str, start: usize, end: usize, cs: usize, ce: usize) -> &str {
    assert!(start < end && end <= text.len() && cs < ce && ce <= text.chars().count());
    let byte_slice = text.get(start..end).expect("UTF-8 boundaries");
    let scalar_slice: String = text.chars().skip(cs).take(ce - cs).collect();
    assert_eq!(byte_slice, scalar_slice);
    assert_eq!(
        text.chars().take(cs).map(char::len_utf8).sum::<usize>(),
        start
    );
    assert_eq!(
        text.chars().take(ce).map(char::len_utf8).sum::<usize>(),
        end
    );
    byte_slice
}

fn cross_slice<'a>(fixture: &'a Fixture, position: &Value) -> &'a str {
    let chapter = position["chapter"].as_u64().unwrap() as usize;
    let text = fixture.chapter(chapter);
    assert_eq!(position["sha256"], sha(text.as_bytes()));
    source_slice(
        text,
        position["bytes"]["start"].as_u64().unwrap() as usize,
        position["bytes"]["end"].as_u64().unwrap() as usize,
        position["scalars"]["start_char"].as_u64().unwrap() as usize,
        position["scalars"]["end_char"].as_u64().unwrap() as usize,
    )
}

fn scoped_slice<'a>(text: &'a str, range: &Value) -> &'a str {
    source_slice(
        text,
        range["byte_start"].as_u64().unwrap() as usize,
        range["byte_end"].as_u64().unwrap() as usize,
        range["scalars"]["start_char"].as_u64().unwrap() as usize,
        range["scalars"]["end_char"].as_u64().unwrap() as usize,
    )
}

#[test]
fn v1_default_is_explicit_and_has_a_valid_receipt() {
    let fixture = Fixture::new(&["𠮷e\u{301}。小甲到屋內。"]);
    let output = fixture.run(&["--all"], false);
    let rows = stream(&output, "avl-source-stream-v1");
    assert!(rows[0]["work_lexicon_sha256"].is_null());
    for mention in selected(&rows, "source_mention") {
        assert_eq!(
            mention["actor_observation"]["vocabulary_origin_detail"],
            "empty_default"
        );
    }
    assert!(selected(&rows, "cross_chapter_summary").is_empty());
    assert!(selected(&rows, "written_scoped_summary").is_empty());
    assert_eq!(rows.last().unwrap()["completed_chapters"], 1);
}

#[test]
fn v2_missing_chapter_candidates_bind_both_source_snapshots() {
    let fixture = Fixture::new(&["𠮷e\u{301}。小甲到屋內。", "", "次日，他離開了。"]);
    let output = fixture.run(&["--chapters", "1,3", "--cross-chapter-candidates"], true);
    let rows = stream(&output, "avl-source-stream-v2-cross-candidates");
    let summaries = selected(&rows, "cross_chapter_summary");
    let candidates = selected(&rows, "cross_chapter_candidate");
    assert_eq!(summaries.len(), 2);
    let current = summaries.iter().find(|r| r["chapter"] == 3).unwrap();
    assert_eq!(
        current["chapter_sha256"],
        sha(fixture.chapter(3).as_bytes())
    );
    assert_eq!(
        current["observation"]["missing_chapter_ranges"],
        json!([{"start": 2, "end": 2}])
    );
    assert_eq!(current["observation"]["history_incomplete"], true);
    assert_eq!(current["observation"]["coverage"], "unknown");
    assert_eq!(
        current["candidate_records"].as_u64().unwrap() as usize,
        candidates.len()
    );
    assert_eq!(candidates.len(), 1, "synthetic written name and pronoun");
    let candidate = candidates[0];
    assert_eq!(cross_slice(&fixture, &candidate["prior_name"]), "小甲");
    assert_eq!(cross_slice(&fixture, &candidate["current_pronoun"]), "他");
    for field in [
        "identity_status",
        "role_status",
        "association_status",
        "occurrence_status",
        "narrative_scope_status",
    ] {
        assert_eq!(candidate[field], "unknown");
    }
    let prior = summaries.iter().find(|r| r["chapter"] == 1).unwrap();
    let name = prior["observation"]["names"]
        .as_array()
        .unwrap()
        .first()
        .unwrap();
    assert_eq!(candidate["prior_name"]["bytes"], name["bytes"]);
    assert_eq!(candidate["prior_name"]["scalars"], name["scalars"]);
    let pronoun = &current["observation"]["pronouns"][0]["source"];
    assert_eq!(candidate["current_pronoun"]["bytes"], pronoun["bytes"]);
    assert_eq!(candidate["current_pronoun"]["scalars"], pronoun["scalars"]);
    assert_eq!(rows.last().unwrap()["cross_chapter_summaries"], 2);
    assert_eq!(rows.last().unwrap()["cross_chapter_candidates"], 1);
}

#[test]
fn v3_scoped_quotes_and_transfer_keep_gaps_and_source_offsets() {
    let text = "𠮷e\u{301}。小甲說道：「前文。」<line>隱藏「假」。\n乙「甲『乙」後文<end>小甲把劍交給小乙。";
    let fixture = Fixture::new(&[text]);
    let output = fixture.run(
        &[
            "--all",
            "--scoped-written-observations",
            "--written-quotes",
            "--written-transfer-cues",
            "--written-transfer-role-candidates",
            "--written-transfer-role-alternatives",
        ],
        true,
    );
    let rows = stream(&output, "avl-source-stream-v3-scoped-written");
    assert_eq!(rows[0]["quote_transfer_scope"], "admitted_segments");
    assert_eq!(rows[0]["cross_segment_associations"], "not_examined");
    let summaries = selected(&rows, "written_scoped_summary");
    assert_eq!(summaries.len(), 1);
    let summary = summaries[0];
    assert_eq!(summary["sha256"], sha(text.as_bytes()));
    assert_eq!(summary["segments"].as_array().unwrap().len(), 2);
    assert_eq!(summary["transfer_cues"], 0);
    assert_eq!(summary["transfer_role_candidates"], 0);
    assert_eq!(summary["transfer_role_alternatives"], 0);
    assert!(selected(&rows, "written_scoped_transfer_cue").is_empty());
    let quote_segments = selected(&rows, "written_scoped_quote_segment");
    assert_eq!(quote_segments.len(), 2);
    for segment in &quote_segments {
        assert_eq!(segment["sha256"], sha(text.as_bytes()));
        assert_eq!(segment["index_scope"], "segment");
        assert_eq!(segment["boundary_scope"], "hard_break");
    }
    assert_eq!(
        scoped_slice(
            text,
            &quote_segments[0]["report"]["quotations"][0]["source"]
        ),
        "「前文。」"
    );
    let diagnostics = quote_segments[1]["report"]["quote_scan_diagnostics"]
        .as_array()
        .unwrap();
    assert_eq!(diagnostics.len(), 1);
    assert_eq!(diagnostics[0]["reason"], "crossed_delimiter");
    assert_eq!(diagnostics[0]["status"], "unexamined");
    assert_eq!(scoped_slice(text, &diagnostics[0]["source"]), "」後文");
    assert_eq!(rows.last().unwrap()["quote_degraded_chapters"], 1);
    assert_eq!(rows.last().unwrap()["scoped_written_summaries"], 1);
}

#[test]
fn failure_emits_no_success_receipt() {
    let fixture = Fixture::new(&["合成文字。"]);
    let manifest = fixture.dir.path().join("manifest.json");
    let mut value: Value = serde_json::from_slice(&fs::read(&manifest).unwrap()).unwrap();
    value["chapters"][0]["status"] = json!("not_ready");
    fs::write(&manifest, serde_json::to_vec(&value).unwrap()).unwrap();
    let output = fixture.run(&["--all", "--cross-chapter-candidates"], true);
    assert!(!output.status.success());
    for line in output
        .stdout
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
    {
        let row: Value = serde_json::from_slice(line).unwrap();
        assert_ne!(row["type"], "stream_complete");
    }
}

#[test]
fn chapter_path_cannot_escape_the_manifest_directory() {
    let fixture = Fixture::new(&["合成文字。"]);
    let outside = tempdir().unwrap();
    fs::write(outside.path().join("other.txt"), "其他合成文字。").unwrap();
    let name = outside.path().file_name().unwrap().to_str().unwrap();
    let manifest = fixture.dir.path().join("manifest.json");
    let mut value: Value = serde_json::from_slice(&fs::read(&manifest).unwrap()).unwrap();
    value["chapters"][0]["file"] = json!(format!("../{name}/other.txt"));
    fs::write(&manifest, serde_json::to_vec(&value).unwrap()).unwrap();
    let output = fixture.run(&["--all"], true);
    assert!(!output.status.success());
    for line in output
        .stdout
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
    {
        let row: Value = serde_json::from_slice(line).unwrap();
        assert_ne!(row["type"], "stream_complete");
    }
}

#[cfg(unix)]
#[test]
fn chapter_symlink_cannot_escape_the_manifest_directory() {
    let fixture = Fixture::new(&["合成文字。"]);
    let outside = tempdir().unwrap();
    fs::write(outside.path().join("other.txt"), "其他合成文字。").unwrap();
    std::os::unix::fs::symlink(
        outside.path().join("other.txt"),
        fixture.dir.path().join("link.txt"),
    )
    .unwrap();
    let manifest = fixture.dir.path().join("manifest.json");
    let mut value: Value = serde_json::from_slice(&fs::read(&manifest).unwrap()).unwrap();
    value["chapters"][0]["file"] = json!("link.txt");
    fs::write(&manifest, serde_json::to_vec(&value).unwrap()).unwrap();
    let output = fixture.run(&["--all"], true);
    assert!(!output.status.success());
    for line in output
        .stdout
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
    {
        let row: Value = serde_json::from_slice(line).unwrap();
        assert_ne!(row["type"], "stream_complete");
    }
}
