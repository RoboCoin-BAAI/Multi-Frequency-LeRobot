#!/usr/bin/env python
"""Batch convert Jianzhi MCAP trees while preserving the source layout."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONVERTER_SCRIPT = Path("format_convert/jianzhi/convert_jianzhi_mcap.py")
COMPLETED_STATUSES = {"done", "skipped_by_converter", "skipped_existing"}


def discover_mcaps(source: Path) -> list[Path]:
    files: list[Path] = []
    if source.is_file() and source.suffix.lower() in {".mcap", ".macp"}:
        return [source]
    if source.is_dir():
        files.extend(source.rglob("*.mcap"))
        files.extend(source.rglob("*.macp"))
    return sorted(dict.fromkeys(files), key=lambda path: str(path))


def _relative_stem_path(source: Path, mcap: Path) -> Path:
    source_root = source.resolve().parent if source.is_file() else source.resolve()
    relative = mcap.resolve().relative_to(source_root)
    return relative.with_suffix("")


def output_dir_for(source: Path, out: Path, mcap: Path) -> Path:
    return out / _relative_stem_path(source, mcap)


def status_and_log_paths(source: Path, out: Path, mcap: Path) -> tuple[Path, Path]:
    relative_stem = _relative_stem_path(source, mcap)
    return (
        out / "_batch_status" / relative_stem.with_suffix(".json"),
        out / "_batch_logs" / relative_stem.with_suffix(".log"),
    )


def build_convert_command(mcap: Path, out_dir: Path, *, video_mode: str) -> list[str]:
    return [
        sys.executable,
        str(CONVERTER_SCRIPT),
        str(mcap),
        "--out",
        str(out_dir),
        "--force",
        "--video-mode",
        video_mode,
    ]


def is_completed_status(status_path: Path) -> bool:
    if not status_path.exists():
        return False
    try:
        status = json.loads(status_path.read_text()).get("status")
    except (OSError, json.JSONDecodeError):
        return False
    return status in COMPLETED_STATUSES


def status_from_return(returncode: int, stdout: str, stderr: str) -> str:
    if returncode != 0:
        return "failed"
    combined = f"{stdout}\n{stderr}"
    if "[skipped]" in combined:
        return "skipped_by_converter"
    return "done"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _write_log(path: Path, payload: dict[str, Any], stdout: str, stderr: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"status: {payload['status']}",
        f"returncode: {payload['returncode']}",
        f"source: {payload['source']}",
        f"output: {payload['output']}",
        f"duration_s: {payload['duration_s']:.3f}",
        f"command: {' '.join(payload['command'])}",
        "",
        "STDOUT",
        stdout.rstrip(),
        "",
        "STDERR",
        stderr.rstrip(),
        "",
    ]
    path.write_text("\n".join(lines))


def run_one(
    source: Path,
    out: Path,
    mcap: Path,
    *,
    video_mode: str,
    resume: bool,
) -> dict[str, Any]:
    out_dir = output_dir_for(source, out, mcap)
    status_path, log_path = status_and_log_paths(source, out, mcap)
    command = build_convert_command(mcap, out_dir, video_mode=video_mode)

    if resume and is_completed_status(status_path):
        payload = {
            "status": "skipped_existing",
            "source": str(mcap),
            "output": str(out_dir),
            "status_path": str(status_path),
            "log_path": str(log_path),
            "returncode": 0,
            "duration_s": 0.0,
            "command": command,
        }
        print(f"[skip-existing] {mcap}")
        return payload

    started = time.monotonic()
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"[start] {mcap}")
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    duration_s = time.monotonic() - started
    status = status_from_return(completed.returncode, completed.stdout, completed.stderr)
    payload = {
        "status": status,
        "source": str(mcap),
        "output": str(out_dir),
        "status_path": str(status_path),
        "log_path": str(log_path),
        "returncode": completed.returncode,
        "duration_s": duration_s,
        "command": command,
    }
    _write_json(status_path, payload)
    _write_log(log_path, payload, completed.stdout, completed.stderr)
    print(f"[{status}] {mcap} ({duration_s:.1f}s)")
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="source directory or a single .mcap/.macp file")
    parser.add_argument("--out", type=Path, required=True, help="batch output root")
    parser.add_argument("--workers", type=int, default=2, help="parallel converter subprocesses")
    parser.add_argument("--video-mode", choices=("remux", "decode"), default="remux")
    parser.add_argument("--limit", type=int, default=None, help="convert only first N files")
    parser.add_argument("--dry-run", action="store_true", help="print planned input/output pairs without converting")
    parser.add_argument("--no-resume", action="store_true", help="rerun items even if batch status says completed")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.workers < 1:
        print("[error] --workers must be >= 1")
        return 1

    mcaps = discover_mcaps(args.source)
    if args.limit is not None:
        mcaps = mcaps[: args.limit]
    if not mcaps:
        print(f"[error] no .mcap/.macp files found under {args.source}")
        return 1

    print(f"[batch] {len(mcaps)} file(s), workers={args.workers}, video_mode={args.video_mode}")
    if args.dry_run:
        for mcap in mcaps:
            print(f"[plan] {mcap} -> {output_dir_for(args.source, args.out, mcap)}")
        return 0

    counts = {"done": 0, "failed": 0, "skipped_by_converter": 0, "skipped_existing": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                run_one,
                args.source,
                args.out,
                mcap,
                video_mode=args.video_mode,
                resume=not args.no_resume,
            )
            for mcap in mcaps
        ]
        for future in as_completed(futures):
            result = future.result()
            counts[result["status"]] = counts.get(result["status"], 0) + 1

    print(
        "[summary] "
        f"done={counts.get('done', 0)} "
        f"skipped_by_converter={counts.get('skipped_by_converter', 0)} "
        f"skipped_existing={counts.get('skipped_existing', 0)} "
        f"failed={counts.get('failed', 0)}"
    )
    return 1 if counts.get("failed", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
