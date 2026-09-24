import subprocess

from format_convert.jianzhi.convert_jianzhi_mcap import remux_h264_annexb_to_mp4


def test_remux_h264_annexb_to_mp4_pipes_packets_to_ffmpeg(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, input, check, stdout, stderr):
        calls.append(
            {
                "cmd": cmd,
                "input": input,
                "check": check,
                "stdout": stdout,
                "stderr": stderr,
            }
        )
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    output = tmp_path / "episode_000000.mp4"
    packets = [b"\x00\x00\x00\x01gabc", b"\x00\x00\x00\x01Adef"]

    remux_h264_annexb_to_mp4(packets, output, fps=30)

    assert len(calls) == 1
    assert calls[0]["input"] == b"".join(packets)
    assert calls[0]["check"] is True
    assert calls[0]["cmd"][:8] == [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "h264",
        "-r",
    ]
    assert "30" in calls[0]["cmd"]
    assert calls[0]["cmd"][-1] == str(output)
