import json
import sys
from pathlib import Path

from format_convert.jianzhi.batch_convert import (
    build_convert_command,
    discover_mcaps,
    is_completed_status,
    output_dir_for,
    status_and_log_paths,
    status_from_return,
)


def test_discover_mcaps_finds_mcap_and_macp_in_sorted_order(tmp_path: Path):
    (tmp_path / "b").mkdir()
    (tmp_path / "a").mkdir()
    second = tmp_path / "b" / "two.macp"
    first = tmp_path / "a" / "one.mcap"
    ignored = tmp_path / "a" / "notes.txt"
    second.write_bytes(b"")
    first.write_bytes(b"")
    ignored.write_text("ignore")

    assert discover_mcaps(tmp_path) == [first, second]


def test_output_dir_preserves_relative_structure_without_mcap_suffix(tmp_path: Path):
    source = tmp_path / "source"
    out = tmp_path / "out"
    mcap = source / "task_a" / "session_001" / "recording.mcap"

    assert output_dir_for(source, out, mcap) == out / "task_a" / "session_001" / "recording"


def test_output_dir_for_single_file_source_uses_file_stem(tmp_path: Path):
    mcap = tmp_path / "recording.mcap"
    out = tmp_path / "out"
    mcap.write_bytes(b"")

    assert output_dir_for(mcap, out, mcap) == out / "recording"


def test_status_and_log_paths_preserve_relative_structure(tmp_path: Path):
    source = tmp_path / "source"
    out = tmp_path / "out"
    mcap = source / "task_a" / "session_001" / "recording.macp"

    status_path, log_path = status_and_log_paths(source, out, mcap)

    assert status_path == out / "_batch_status" / "task_a" / "session_001" / "recording.json"
    assert log_path == out / "_batch_logs" / "task_a" / "session_001" / "recording.log"


def test_build_convert_command_targets_one_mcap_and_forces_item_output(tmp_path: Path):
    mcap = tmp_path / "input.mcap"
    out_dir = tmp_path / "out" / "input"

    command = build_convert_command(mcap, out_dir, video_mode="remux")

    assert command == [
        sys.executable,
        str(Path("format_convert/jianzhi/convert_jianzhi_mcap.py")),
        str(mcap),
        "--out",
        str(out_dir),
        "--force",
        "--video-mode",
        "remux",
    ]


def test_status_from_return_marks_converter_skip():
    status = status_from_return(0, "[skipped] 1 episode(s) with inconsistent camera frame counts", "")

    assert status == "skipped_by_converter"


def test_completed_status_accepts_done_and_converter_skip(tmp_path: Path):
    status_path = tmp_path / "status.json"
    status_path.write_text(json.dumps({"status": "skipped_by_converter"}))

    assert is_completed_status(status_path)

    status_path.write_text(json.dumps({"status": "done"}))
    assert is_completed_status(status_path)

    status_path.write_text(json.dumps({"status": "failed"}))
    assert not is_completed_status(status_path)
