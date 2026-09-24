#!/usr/bin/env python
"""Convert Jianzhi MCAP recordings into a Multi-Frequency LeRobot dataset.

Each input MCAP becomes one episode. Camera calibration messages are exported
to ``<out>/calibrations`` in a Kalibr-like YAML layout, preserving the original
Jianzhi ``T_b_c`` 7-vector alongside a derived 4x4 matrix.
"""

from __future__ import annotations

import argparse
import base64
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from format_convert.field_naming import (
    HAND_FEATURES,
    HAND_LANDMARK_NAMES,
    MOTION_FEATURES,
    image_feature,
)

if TYPE_CHECKING:
    from mf_lerobot import MultiFrequencyLeRobotDataset


MASTER_FPS = 30
CAMERA_COUNT = 6
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "jianzhi"
DEFAULT_OUT = PROJECT_ROOT / "data" / "jianzhi_lerobot"

ANNOTATION_TOPIC = "/robot0/annotation_v2"
TIME_RANGE_TOPIC = "/robot0/time_range_validity"
IMU_TOPIC = "/robot0/sensor/imu"
EEF_POSE_TOPIC = "/robot0/vio/eef_pose"
RELATIVE_EEF_POSE_TOPIC = "/robot0/vio/relative_eef_pose"
HAND_LEFT_TOPIC = "/robot0/handtracking/left"
HAND_RIGHT_TOPIC = "/robot0/handtracking/right"
ROBOT_INFO_TOPIC = "/robot0/sim/robot_info"


def camera_compressed_topic(index: int) -> str:
    return f"/robot0/sensor/camera{index}/compressed"


def camera_info_topic(index: int) -> str:
    return f"/robot0/sensor/camera{index}/camera_info"


CAMERA_TOPICS = tuple(camera_compressed_topic(i) for i in range(CAMERA_COUNT))
CAMERA_INFO_TOPICS = tuple(camera_info_topic(i) for i in range(CAMERA_COUNT))
JIANZHI_CAMERA_FEATURES = {
    0: image_feature("head_left_outer1"),
    1: image_feature("head_left_outer0"),
    2: image_feature("head_left"),
    3: image_feature("head_right"),
    4: image_feature("head_right_outer0"),
    5: image_feature("head_right_outer1"),
}
JIANZHI_MASTER_FEATURE = image_feature("head_left")
CALIBRATION_KEYS = {
    idx: feature.removeprefix("observation.images.")
    for idx, feature in JIANZHI_CAMERA_FEATURES.items()
}
DECODE_TOPICS = (
    ANNOTATION_TOPIC,
    TIME_RANGE_TOPIC,
    IMU_TOPIC,
    EEF_POSE_TOPIC,
    RELATIVE_EEF_POSE_TOPIC,
    HAND_LEFT_TOPIC,
    HAND_RIGHT_TOPIC,
    ROBOT_INFO_TOPIC,
    *CAMERA_INFO_TOPICS,
    *CAMERA_TOPICS,
)


def _as_float_list(values: Any) -> list[float]:
    return [float(v) for v in values]


def _matrix_from_flat(values: Any, cols: int) -> list[list[float]]:
    flat = _as_float_list(values)
    return [flat[i:i + cols] for i in range(0, len(flat), cols)]


def transform_vec7_to_matrix(transform: list[float] | tuple[float, ...]) -> list[list[float]]:
    """Convert ``[x, y, z, qx, qy, qz, qw]`` into a 4x4 transform matrix."""
    if len(transform) != 7:
        raise ValueError(f"expected 7 values for xyz+xyzw transform, got {len(transform)}")
    tx, ty, tz, qx, qy, qz, qw = [float(v) for v in transform]
    norm = float(np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw))
    if norm == 0.0:
        raise ValueError("quaternion norm is zero")
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm

    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    return [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy), tx],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx), ty],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy), tz],
        [0.0, 0.0, 0.0, 1.0],
    ]


def camera_calibration_to_kalibr(cam_key: str, topic: str, decoded: dict[str, Any]) -> dict[str, Any]:
    """Map a Foxglove CameraCalibration message to a Kalibr-like camera entry."""
    k = _as_float_list(decoded["K"])
    result = {
        "camera_model": "pinhole",
        "distortion_model": str(decoded.get("distortion_model", "")),
        "distortion_coeffs": _as_float_list(decoded.get("D", [])),
        "intrinsics": [k[0], k[4], k[2], k[5]],
        "resolution": [int(decoded["width"]), int(decoded["height"])],
        "rostopic": topic.replace("/camera_info", "/compressed"),
        "frame_id": decoded.get("frame_id", ""),
        "K": _matrix_from_flat(decoded.get("K", []), 3),
        "R": _matrix_from_flat(decoded.get("R", []), 3),
        "P": _matrix_from_flat(decoded.get("P", []), 4),
    }
    if "T_b_c" in decoded:
        t_b_c = _as_float_list(decoded["T_b_c"])
        result["T_b_c"] = t_b_c
        result["T_b_c_matrix"] = transform_vec7_to_matrix(t_b_c)
    return result


class DynamicProtobufDecoder:
    """Decode protobuf MCAP messages from in-file descriptor sets."""

    def __init__(self) -> None:
        from google.protobuf import descriptor_pool

        self._pool = descriptor_pool.DescriptorPool()
        self._loaded_schema_ids: set[int] = set()

    def decode(self, schema: Any, data: bytes) -> dict[str, Any] | None:
        if schema is None or schema.encoding != "protobuf":
            return None

        from google.protobuf import descriptor_pb2, json_format, message_factory

        if schema.id not in self._loaded_schema_ids:
            descriptor_set = descriptor_pb2.FileDescriptorSet()
            descriptor_set.ParseFromString(schema.data)
            pending = list(descriptor_set.file)
            while pending:
                next_pending = []
                progressed = False
                for file_descriptor in pending:
                    try:
                        self._pool.Add(file_descriptor)
                        progressed = True
                    except Exception:
                        next_pending.append(file_descriptor)
                if not progressed:
                    break
                pending = next_pending
            self._loaded_schema_ids.add(schema.id)

        descriptor = self._pool.FindMessageTypeByName(schema.name)
        message_class = message_factory.GetMessageClass(descriptor)
        message = message_class()
        message.ParseFromString(data)
        return json_format.MessageToDict(message, preserving_proto_field_name=True)


def header_timestamp_s(decoded: dict[str, Any], fallback_ns: int) -> float:
    header = decoded.get("header") or {}
    timestamp = header.get("timestamp")
    if timestamp is None:
        return fallback_ns / 1e9
    return int(timestamp) / 1e9


def pose_to_vec7(decoded: dict[str, Any]) -> np.ndarray:
    pose = decoded.get("pose") or {}
    pos = pose.get("position") or {}
    quat = pose.get("orientation") or {}
    return np.array(
        [
            float(pos.get("x", 0.0)),
            float(pos.get("y", 0.0)),
            float(pos.get("z", 0.0)),
            float(quat.get("x", 0.0)),
            float(quat.get("y", 0.0)),
            float(quat.get("z", 0.0)),
            float(quat.get("w", 1.0)),
        ],
        dtype=np.float32,
    )


def imu_to_vec6(decoded: dict[str, Any]) -> np.ndarray:
    gyro = decoded.get("angular_velocity") or {}
    accel = decoded.get("linear_acceleration") or {}
    return np.array(
        [
            float(accel.get("x", 0.0)),
            float(accel.get("y", 0.0)),
            float(accel.get("z", 0.0)),
            float(gyro.get("x", 0.0)),
            float(gyro.get("y", 0.0)),
            float(gyro.get("z", 0.0)),
        ],
        dtype=np.float32,
    )


def hand_mano_to_vec(decoded: dict[str, Any]) -> np.ndarray:
    mano = decoded.get("mano_params") or {}
    values: list[float] = []
    for key in ("global_transl", "global_orient", "hand_pose", "betas"):
        values.extend(float(v) for v in mano.get(key, []))
    return np.asarray(values, dtype=np.float32)


def hand_points_to_vec(decoded: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    values: list[float] = []
    names: list[str] = []
    for i, bone in enumerate(decoded.get("bone_data") or []):
        landmark_name = HAND_LANDMARK_NAMES[i] if i < len(HAND_LANDMARK_NAMES) else f"bone{i}"
        position = ((bone.get("to_global") or {}).get("position") or {})
        for axis in ("x", "y", "z"):
            values.append(float(position.get(axis, 0.0)))
            names.append(f"{landmark_name}_{axis}")
    return np.asarray(values, dtype=np.float32), names


def extract_task_name(annotation: dict[str, Any] | None, mcap_path: Path) -> str:
    if annotation:
        for key in ("bold_mark", "task_description"):
            value = annotation.get(key)
            if value:
                return str(value).strip()
        segments = annotation.get("segments_info") or []
        if segments and segments[0].get("fine_label"):
            return str(segments[0]["fine_label"]).strip()
    return mcap_path.stem


def extract_valid_window(time_range: dict[str, Any] | None) -> tuple[float | None, float | None]:
    if not time_range:
        return None, None
    valid_ranges = [r for r in time_range.get("ranges", []) if r.get("is_valid")]
    if not valid_ranges:
        return None, None
    return (
        min(float(r.get("start_time_s", 0.0)) for r in valid_ranges),
        max(float(r.get("end_time_s", 0.0)) for r in valid_ranges),
    )


def in_window(timestamp_rel: float, window: tuple[float | None, float | None]) -> bool:
    start, end = window
    if start is not None and timestamp_rel < start:
        return False
    if end is not None and timestamp_rel > end:
        return False
    return True


def read_mcap(path: Path) -> dict[str, Any]:
    from mcap.reader import make_reader

    decoder = DynamicProtobufDecoder()
    records: dict[str, Any] = {
        "source_path": str(path),
        "calibrations": {},
        "annotation": None,
        "time_range": None,
        "streams": defaultdict(list),
        "camera_packets": defaultdict(list),
    }
    with path.open("rb") as stream:
        reader = make_reader(stream)
        for schema, channel, message in reader.iter_messages(topics=DECODE_TOPICS):
            topic = channel.topic
            decoded = decoder.decode(schema, message.data)
            if decoded is None:
                continue

            if topic in CAMERA_INFO_TOPICS:
                cam_idx = CAMERA_INFO_TOPICS.index(topic)
                cal_key = CALIBRATION_KEYS[cam_idx]
                records["calibrations"][cal_key] = camera_calibration_to_kalibr(
                    cal_key, topic, decoded
                )
                records["calibrations"][cal_key]["raw_camera_id"] = f"camera{cam_idx}"
            elif topic == ANNOTATION_TOPIC:
                records["annotation"] = decoded
            elif topic == TIME_RANGE_TOPIC:
                records["time_range"] = decoded
            elif topic == IMU_TOPIC:
                records["streams"][MOTION_FEATURES["imu"]].append(
                    (header_timestamp_s(decoded, message.log_time), imu_to_vec6(decoded))
                )
            elif topic == EEF_POSE_TOPIC:
                records["streams"][MOTION_FEATURES["head_pose"]].append(
                    (header_timestamp_s(decoded, message.log_time), pose_to_vec7(decoded))
                )
            elif topic == RELATIVE_EEF_POSE_TOPIC:
                records["streams"][MOTION_FEATURES["relative_head_pose"]].append(
                    (header_timestamp_s(decoded, message.log_time), pose_to_vec7(decoded))
                )
            elif topic == HAND_LEFT_TOPIC:
                timestamp = header_timestamp_s(decoded, message.log_time)
                mano = hand_mano_to_vec(decoded)
                points, point_names = hand_points_to_vec(decoded)
                if mano.size:
                    records["streams"][HAND_FEATURES["left_mano"]].append(
                        (timestamp, mano)
                    )
                if points.size:
                    records["streams"][HAND_FEATURES["left_points"]].append(
                        (timestamp, points)
                    )
                    records.setdefault("hand_point_names", {})["left"] = point_names
            elif topic == HAND_RIGHT_TOPIC:
                timestamp = header_timestamp_s(decoded, message.log_time)
                mano = hand_mano_to_vec(decoded)
                points, point_names = hand_points_to_vec(decoded)
                if mano.size:
                    records["streams"][HAND_FEATURES["right_mano"]].append(
                        (timestamp, mano)
                    )
                if points.size:
                    records["streams"][HAND_FEATURES["right_points"]].append(
                        (timestamp, points)
                    )
                    records.setdefault("hand_point_names", {})["right"] = point_names
            elif topic in CAMERA_TOPICS:
                cam_idx = CAMERA_TOPICS.index(topic)
                payload = decoded.get("data")
                if payload:
                    records["camera_packets"][JIANZHI_CAMERA_FEATURES[cam_idx]].append(
                        (header_timestamp_s(decoded, message.log_time), base64.b64decode(payload))
                    )
    return records


def stream_rate(samples: list[tuple[float, Any]]) -> float:
    if len(samples) < 2:
        return 0.0
    duration = samples[-1][0] - samples[0][0]
    return round(len(samples) / duration, 2) if duration > 0 else 0.0


def feature_specs(records_by_episode: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    for idx in range(CAMERA_COUNT):
        key = JIANZHI_CAMERA_FEATURES[idx]
        calibration_key = CALIBRATION_KEYS[idx]
        for records in records_by_episode:
            cal = records["calibrations"].get(calibration_key)
            if cal and records["camera_packets"].get(key):
                width, height = cal["resolution"]
                specs[key] = {
                    "dtype": "video",
                    "shape": (height, width, 3),
                    "names": ["h", "w", "c"],
                    "fps": MASTER_FPS,
                    "tolerance_s": 0.005,
                }
                break

    candidates = {
        MOTION_FEATURES["imu"]: (6, ["ax", "ay", "az", "gx", "gy", "gz"], (-0.033, 0.0)),
        MOTION_FEATURES["head_pose"]: (7, ["x", "y", "z", "qx", "qy", "qz", "qw"], None),
        MOTION_FEATURES["relative_head_pose"]: (7, ["x", "y", "z", "qx", "qy", "qz", "qw"], None),
        HAND_FEATURES["left_mano"]: (157, None, None),
        HAND_FEATURES["right_mano"]: (157, None, None),
        HAND_FEATURES["left_points"]: (63, None, None),
        HAND_FEATURES["right_points"]: (63, None, None),
    }
    for key, (fallback_dim, names, window) in candidates.items():
        for records in records_by_episode:
            samples = records["streams"].get(key, [])
            if samples:
                dim = int(np.asarray(samples[0][1]).shape[0]) or fallback_dim
                rate = stream_rate(samples)
                feature_names = names
                if key == HAND_FEATURES["left_points"]:
                    feature_names = records.get("hand_point_names", {}).get("left")
                elif key == HAND_FEATURES["right_points"]:
                    feature_names = records.get("hand_point_names", {}).get("right")
                specs[key] = {
                    "dtype": "float32",
                    "shape": (dim,),
                    "names": feature_names,
                    "fps": rate or MASTER_FPS,
                    "tolerance_s": max(0.005, 3.0 / max(rate, 1.0)),
                }
                if window:
                    specs[key]["window"] = window
                break
    return specs


def write_calibrations(out: Path, records_by_episode: list[dict[str, Any]]) -> None:
    calibration_dir = out / "calibrations"
    calibration_dir.mkdir(parents=True, exist_ok=True)
    merged: dict[str, Any] = {}
    for records in records_by_episode:
        for key, value in records["calibrations"].items():
            merged.setdefault(key, value)
    (calibration_dir / "camera-camchain.yaml").write_text(
        yaml.safe_dump(merged, sort_keys=True, allow_unicode=True),
        encoding="utf-8",
    )
    (calibration_dir / "jianzhi-camera-calibration.json").write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_annotations(out: Path, records_by_episode: list[dict[str, Any]]) -> Path:
    meta_dir = out / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    out_path = meta_dir / "annotations.json"
    payload = {
        "format": "jianzhi_annotations_v1",
        "episodes": [
            {
                "episode_index": i,
                "source_path": records.get("source_path"),
                "annotation": records.get("annotation"),
                "time_range": records.get("time_range"),
            }
            for i, records in enumerate(records_by_episode)
        ],
    }
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return out_path


def master_timestamps(records: dict[str, Any], specs: dict[str, dict[str, Any]]) -> list[float]:
    if JIANZHI_MASTER_FEATURE in specs and records["camera_packets"].get(JIANZHI_MASTER_FEATURE):
        return [t for t, _ in records["camera_packets"][JIANZHI_MASTER_FEATURE]]
    for idx in range(CAMERA_COUNT):
        key = JIANZHI_CAMERA_FEATURES[idx]
        if key in specs and records["camera_packets"].get(key):
            return [t for t, _ in records["camera_packets"][key]]
    if records["streams"].get(MOTION_FEATURES["head_pose"]):
        return [t for t, _ in records["streams"][MOTION_FEATURES["head_pose"]]]
    raise ValueError("episode has no camera or state stream for master clock")


def validate_camera_frame_counts(records: dict[str, Any]) -> tuple[bool, str]:
    counts = {}
    for idx in range(CAMERA_COUNT):
        key = JIANZHI_CAMERA_FEATURES[idx]
        samples = records["camera_packets"].get(key, [])
        if not samples:
            return False, f"missing camera stream: {key}"
        counts[key] = len(samples)
    unique_counts = set(counts.values())
    if len(unique_counts) != 1:
        return False, f"camera frame count mismatch: {counts}"
    return True, ""


def nearest_indices(source_ts: list[float], target_ts: list[float]) -> np.ndarray:
    src = np.asarray(source_ts, dtype=np.float64)
    tgt = np.asarray(target_ts, dtype=np.float64)
    idx = np.searchsorted(src, tgt)
    idx = np.clip(idx, 0, len(src) - 1)
    prev = np.clip(idx - 1, 0, len(src) - 1)
    use_prev = np.abs(src[prev] - tgt) < np.abs(src[idx] - tgt)
    return np.where(use_prev, prev, idx)


def iter_h264_frames(packets: list[bytes]):
    import av

    codec = av.CodecContext.create("h264", "r")
    for raw in packets:
        packet = av.Packet(raw)
        for frame in codec.decode(packet):
            yield frame.to_ndarray(format="rgb24")
    for frame in codec.decode(None):
        yield frame.to_ndarray(format="rgb24")


def remux_h264_annexb_to_mp4(packets: list[bytes], output_path: Path, fps: float) -> None:
    """Write Annex B H264 packets to an MP4 container without decoding frames."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "h264",
        "-r",
        str(float(fps)).rstrip("0").rstrip("."),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "copy",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    subprocess.run(
        cmd,
        input=b"".join(packets),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def write_episode(
    ds: MultiFrequencyLeRobotDataset,
    mcap_path: Path,
    records: dict[str, Any],
    specs: dict[str, dict[str, Any]],
    video_mode: str,
) -> bool:
    cameras_ok, reason = validate_camera_frame_counts(records)
    if not cameras_ok:
        print(f"[skip] {mcap_path.name}: {reason}")
        return False

    master_abs_all = master_timestamps(records, specs)
    t0 = master_abs_all[0]
    valid_window = (None, None) if video_mode == "remux" else extract_valid_window(records.get("time_range"))
    master_abs = [t for t in master_abs_all if in_window(t - t0, valid_window)]
    master_rel = [t - t0 for t in master_abs]
    task_name = extract_task_name(records.get("annotation"), mcap_path)

    for t in master_rel:
        ds.add_frame("task", task_name, float(t))

    for key, samples in records["streams"].items():
        if key not in specs:
            continue
        for t, value in samples:
            rel = t - t0
            if in_window(rel, valid_window):
                ds.add_frame(key, value, float(rel))

    for key, samples in records["camera_packets"].items():
        if key not in specs or not samples:
            continue
        ts = [t for t, _ in samples]
        packets = [p for _, p in samples]
        vf = ds._features[key]
        if video_mode == "remux":
            video_path = vf.add_external_video(master_rel)
            selected_packets = packets[: len(master_rel)]
            remux_h264_annexb_to_mp4(
                selected_packets,
                video_path,
                fps=float(specs[key].get("fps", MASTER_FPS) or MASTER_FPS),
            )
            continue

        decoded = 0
        last_frame = None
        written = np.zeros(len(master_rel), dtype=bool)
        for source_idx, frame in enumerate(iter_h264_frames(packets)):
            if source_idx >= len(master_rel):
                break
            decoded += 1
            last_frame = frame
            vf.add(frame, float(master_rel[source_idx]))
            written[source_idx] = True
        if decoded == 0:
            print(f"[warn] {mcap_path.name}: {key} decoded no frames")
            continue
        if not written.all():
            for target_idx in np.where(~written)[0]:
                vf.add(last_frame, float(master_rel[target_idx]))

    missing = {
        key: ds._features.pop(key)
        for key in list(ds._features)
        if key in specs and key not in records["streams"] and key not in records["camera_packets"]
    }
    ds.save_episode()
    for key, feature in missing.items():
        feature.next_episode()
        ds._features[key] = feature
    return True


def cleanup_images(root: Path, ep_idx: int, video_keys: list[str]) -> None:
    for key in video_keys:
        image_dir = root / "images" / key / f"episode_{ep_idx:06d}"
        if image_dir.is_dir():
            shutil.rmtree(image_dir)
        try:
            image_dir.parent.rmdir()
        except OSError:
            pass
    try:
        (root / "images").rmdir()
    except OSError:
        pass


def require_full_conversion_dependencies(video_mode: str) -> None:
    missing = []
    for module, package in (
        ("torch", "torch"),
        ("lerobot", "lerobot==0.3.3"),
    ):
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    if video_mode == "decode":
        try:
            __import__("av")
        except ImportError:
            missing.append("av")
    if video_mode == "remux" and shutil.which("ffmpeg") is None:
        missing.append("ffmpeg")
    if missing:
        packages = " ".join(missing)
        raise RuntimeError(
            "full dataset conversion requires missing dependencies: "
            f"{', '.join(missing)}. Install them, for example: "
            f"python -m pip install {packages}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, help="MCAP files or directories")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help=f"default input dir: {DEFAULT_SOURCE}")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"output dataset dir: {DEFAULT_OUT}")
    parser.add_argument("--force", action="store_true", help="remove output dir before conversion")
    parser.add_argument("--keep-images", action="store_true", help="keep intermediate PNG frames")
    parser.add_argument("--max-episodes", type=int, default=None, help="convert only first N files")
    parser.add_argument(
        "--video-mode",
        choices=("remux", "decode"),
        default="remux",
        help="remux H264 packets directly or decode frames through the PNG fallback (default: %(default)s)",
    )
    parser.add_argument(
        "--calibrations-only",
        action="store_true",
        help="only export camera calibration files; does not require lerobot/torch",
    )
    return parser.parse_args(argv)


def resolve_inputs(args: argparse.Namespace) -> list[Path]:
    roots = args.inputs or [args.source]
    files: list[Path] = []
    for root in roots:
        if root.is_dir():
            files.extend(sorted(root.rglob("*.mcap")))
            files.extend(sorted(root.rglob("*.macp")))
        elif root.is_file() and root.suffix.lower() in {".mcap", ".macp"}:
            files.append(root)
    files = sorted(dict.fromkeys(files))
    if args.max_episodes:
        files = files[: args.max_episodes]
    return files


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    mcap_files = resolve_inputs(args)
    if not mcap_files:
        print(f"[error] no .mcap/.macp files found under {args.source}")
        return 1
    if not args.calibrations_only:
        try:
            require_full_conversion_dependencies(args.video_mode)
        except RuntimeError as exc:
            print(f"[error] {exc}")
            return 1
    if args.out.exists():
        if not args.force:
            print(f"[error] output exists: {args.out} (use --force)")
            return 1
        shutil.rmtree(args.out)

    print(f"[input] {len(mcap_files)} MCAP file(s)")
    records_by_episode = []
    for path in mcap_files:
        print(f"[read] {path}")
        records_by_episode.append(read_mcap(path))

    specs = feature_specs(records_by_episode)
    if not specs:
        print("[error] no convertible streams found")
        return 1

    if args.calibrations_only:
        args.out.mkdir(parents=True, exist_ok=True)
        write_calibrations(args.out, records_by_episode)
        annotations_path = write_annotations(args.out, records_by_episode)
        print(f"[calib] {args.out / 'calibrations' / 'camera-camchain.yaml'}")
        print(f"[annotation] {annotations_path}")
        return 0

    from mf_lerobot import MultiFrequencyLeRobotDataset

    ds = MultiFrequencyLeRobotDataset.create(
        repo_id=args.out.name,
        fps=MASTER_FPS,
        features=specs,
        root=args.out,
        use_videos=True,
        master_feature=JIANZHI_MASTER_FEATURE,
    )
    write_calibrations(args.out, records_by_episode)
    annotations_path = write_annotations(args.out, records_by_episode)

    video_keys = [key for key, spec in specs.items() if spec.get("dtype") == "video"]
    skipped = 0
    for ep_idx, (path, records) in enumerate(zip(mcap_files, records_by_episode)):
        print(f"[episode {ep_idx}] {path.name}")
        wrote = write_episode(ds, path, records, specs, args.video_mode)
        if not wrote:
            skipped += 1
            continue
        if args.video_mode == "decode" and not args.keep_images:
            cleanup_images(args.out, ep_idx, video_keys)

    print(f"[done] {ds.meta.total_episodes} episodes, {ds.meta.total_frames} frames -> {args.out}")
    if skipped:
        print(f"[skipped] {skipped} episode(s) with inconsistent camera frame counts")
    print(f"[calib] {args.out / 'calibrations' / 'camera-camchain.yaml'}")
    print(f"[annotation] {annotations_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
