use std::io::Write;
use std::process::{Command, Stdio};

#[test]
fn all_commands_offer_help_and_reject_invalid_options() {
    for command in [
        "doc",
        "novel-batch",
        "source-stream",
        "narrative-doc",
        "narrative-batch",
        "translate",
        "translate-batch",
    ] {
        let help = Command::new(env!("CARGO_BIN_EXE_nlp"))
            .args([command, "--help"])
            .output()
            .unwrap();
        assert!(help.status.success(), "{command}");
        assert!(
            String::from_utf8_lossy(&help.stdout).contains(&format!("Usage: nlp {command}")),
            "{command}"
        );
        let error = Command::new(env!("CARGO_BIN_EXE_nlp"))
            .args([command, "--invalid-option"])
            .output()
            .unwrap();
        assert_eq!(error.status.code(), Some(2), "{command}");
        assert!(
            String::from_utf8_lossy(&error.stderr).contains(&format!("Usage: nlp {command}")),
            "{command}"
        );
    }
}

#[test]
fn translate_accepts_stdin() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_nlp"))
        .args([
            "translate",
            "--target",
            "zh-Hant",
            "--source-hint",
            "ja",
            "--format",
            "text",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all("これはテストです。".as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success());
    assert!(!output.stdout.is_empty());
}
