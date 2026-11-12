//! Dev automation for fairway: `cargo xtask <task>`.
//!
//! Std-only on purpose, so it builds instantly and works offline.

use std::env;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::time::Instant;

const HELP: &str = "\
Usage: cargo xtask <task>

Tasks:
  dev-check            Everything a push must pass: formatting, clippy and
                       tests on every feature set, rustdoc, then coverage.
  coverage [--html]    Run the tests under cargo-llvm-cov and fail below the
                       thresholds; --html also writes <target>/llvm-cov/html.
";

/// Feature combinations that change which code is compiled.
const FEATURE_SETS: &[&[&str]] = &[
    &[],
    &["--no-default-features"],
    &["--features", "portable-only"],
    &["--no-default-features", "--features", "portable-only"],
];

/// Coverage floor (percent). Raise these as coverage improves; never lower
/// them to make a push pass. They are below 100 only because some branches
/// depend on the CPU (e.g. SSE4.1-without-AVX2 dispatch, skipped tests on
/// CPUs without AVX2).
const MIN_LINES: &str = "99";
const MIN_REGIONS: &str = "99";
const MIN_FUNCTIONS: &str = "100";

type Result<T = ()> = std::result::Result<T, String>;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    let started = Instant::now();
    let result = match args.as_slice() {
        ["dev-check"] => dev_check(),
        ["coverage"] => coverage(false),
        ["coverage", "--html"] => coverage(true),
        [] | ["help" | "--help" | "-h"] => {
            print!("{HELP}");
            return ExitCode::SUCCESS;
        }
        _ => Err(format!("unknown arguments: {}\n\n{HELP}", args.join(" "))),
    };
    match result {
        Ok(()) => {
            eprintln!("\nxtask: OK ({:.1?})", started.elapsed());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("\nxtask: FAILED: {e}");
            ExitCode::FAILURE
        }
    }
}

fn dev_check() -> Result {
    cargo(&["fmt", "--all", "--check"], &[])?;
    for features in FEATURE_SETS {
        cargo(
            &[
                &["clippy", "--workspace", "--all-targets"],
                *features,
                &["--", "-D", "warnings"],
            ]
            .concat(),
            &[],
        )?;
    }
    for features in FEATURE_SETS {
        cargo(
            &[&["test", "--package", "fairway-hash"], *features].concat(),
            &[],
        )?;
    }
    cargo(
        &["doc", "--package", "fairway-hash", "--no-deps"],
        &[("RUSTDOCFLAGS", "-D warnings")],
    )?;
    coverage(false)
}

fn coverage(html: bool) -> Result {
    if !succeeds(&["llvm-cov", "--version"]) {
        return Err("cargo-llvm-cov is missing; install it with \
                    `rustup component add llvm-tools && cargo install cargo-llvm-cov`"
            .into());
    }
    let mut args = vec![
        "llvm-cov",
        "--package",
        "fairway-hash",
        "--fail-under-lines",
        MIN_LINES,
        "--fail-under-regions",
        MIN_REGIONS,
        "--fail-under-functions",
        MIN_FUNCTIONS,
    ];
    args.push(if html {
        "--html"
    } else {
        "--show-missing-lines"
    });
    cargo(&args, &[])?;
    if html {
        let target_dir = env::var("CARGO_TARGET_DIR").unwrap_or_else(|_| "target".into());
        eprintln!("coverage report: {target_dir}/llvm-cov/html/index.html");
    }
    Ok(())
}

/// Runs `cargo <args>` from the workspace root with the toolchain that runs
/// this xtask, failing on a non-zero exit.
fn cargo(args: &[&str], envs: &[(&str, &str)]) -> Result {
    eprintln!("\n$ cargo {}", args.join(" "));
    let status = cargo_command(args)
        .envs(envs.iter().copied())
        .status()
        .map_err(|e| format!("cannot run cargo: {e}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("`cargo {}` failed ({status})", args.join(" ")))
    }
}

fn succeeds(args: &[&str]) -> bool {
    cargo_command(args)
        .output()
        .is_ok_and(|o| o.status.success())
}

fn cargo_command(args: &[&str]) -> Command {
    let mut cmd = Command::new(env::var_os("CARGO").unwrap_or_else(|| "cargo".into()));
    cmd.args(args).current_dir(workspace_root());
    cmd
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask lives in the workspace root")
        .to_path_buf()
}
