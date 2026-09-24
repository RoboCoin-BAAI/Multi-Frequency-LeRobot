import math

import pytest

from format_convert.jianzhi.convert_jianzhi_mcap import (
    camera_calibration_to_kalibr,
    feature_specs,
    hand_mano_to_vec,
    hand_points_to_vec,
    transform_vec7_to_matrix,
    master_timestamps,
    validate_camera_frame_counts,
    write_annotations,
)


def test_transform_vec7_to_matrix_uses_xyz_xyzw_order():
    matrix = transform_vec7_to_matrix([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0])

    assert matrix == [
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 2.0],
        [0.0, 0.0, 1.0, 3.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def test_transform_vec7_to_matrix_normalizes_quaternion():
    matrix = transform_vec7_to_matrix([0.0, 0.0, 0.0, 0.0, 0.0, math.sqrt(2), math.sqrt(2)])

    assert matrix[0][0] == pytest.approx(0.0, abs=1e-12)
    assert matrix[0][1] == pytest.approx(-1.0, abs=1e-12)
    assert matrix[1][0] == pytest.approx(1.0, abs=1e-12)
    assert matrix[1][1] == pytest.approx(0.0, abs=1e-12)


def test_camera_calibration_to_kalibr_preserves_intrinsics_and_extrinsics():
    decoded = {
        "width": 1600,
        "height": 1300,
        "distortion_model": "ds",
        "D": [514.0, 515.0, 789.0, 638.0, -0.001, 0.57],
        "K": [514.0, 0.0, 789.0, 0.0, 515.0, 638.0, 0.0, 0.0, 1.0],
        "R": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "P": [514.0, 0.0, 789.0, 0.0, 0.0, 515.0, 638.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        "frame_id": "camera0_optical_frame_update2body",
        "T_b_c": [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0],
    }

    result = camera_calibration_to_kalibr(
        "cam0", "/robot0/sensor/camera0/camera_info", decoded
    )

    assert result["camera_model"] == "pinhole"
    assert result["distortion_model"] == "ds"
    assert result["intrinsics"] == [514.0, 515.0, 789.0, 638.0]
    assert result["distortion_coeffs"] == decoded["D"]
    assert result["resolution"] == [1600, 1300]
    assert result["rostopic"] == "/robot0/sensor/camera0/compressed"
    assert result["frame_id"] == "camera0_optical_frame_update2body"
    assert result["T_b_c"] == decoded["T_b_c"]
    assert result["T_b_c_matrix"][0] == [1.0, 0.0, 0.0, 1.0]


def test_write_annotations_exports_one_json_record_per_mcap(tmp_path):
    records = [
        {
            "source_path": "/data/a.mcap",
            "annotation": {
                "bold_mark": "assemble tube",
                "segments_info": [
                    {
                        "fine_label": "assemble",
                        "end_time_s": 3.0,
                        "sub_segments_info": [{"fine_label": "pick", "end_frame": 30}],
                    }
                ],
            },
            "time_range": {"ranges": [{"start_time_s": 0.0, "end_time_s": 3.0, "is_valid": True}]},
        },
        {"source_path": "/data/b.mcap", "annotation": None, "time_range": None},
    ]

    out_path = write_annotations(tmp_path, records)

    assert out_path == tmp_path / "meta" / "annotations.json"
    exported = __import__("json").loads(out_path.read_text(encoding="utf-8"))
    assert exported["format"] == "jianzhi_annotations_v1"
    assert exported["episodes"][0]["source_path"] == "/data/a.mcap"
    assert exported["episodes"][0]["annotation"]["bold_mark"] == "assemble tube"
    assert exported["episodes"][0]["annotation"]["segments_info"][0]["sub_segments_info"][0]["fine_label"] == "pick"
    assert exported["episodes"][1]["annotation"] is None


def test_hand_mano_to_vec_exports_157_mano_parameters():
    decoded = {
        "mano_params": {
            "global_transl": [1, 2, 3],
            "global_orient": list(range(9)),
            "hand_pose": list(range(135)),
            "betas": list(range(10)),
        }
    }

    vec = hand_mano_to_vec(decoded)

    assert vec.shape == (157,)
    assert vec[:3].tolist() == [1.0, 2.0, 3.0]


def test_hand_points_to_vec_exports_21_xyz_points():
    decoded = {
        "bone_data": [
            {
                "bone_name": f"Bone{i}",
                "to_global": {"position": {"x": i, "y": i + 0.1, "z": i + 0.2}},
            }
            for i in range(21)
        ]
    }

    vec, names = hand_points_to_vec(decoded)

    assert vec.shape == (63,)
    assert names[:6] == ["wrist_x", "wrist_y", "wrist_z", "thumbCMC_x", "thumbCMC_y", "thumbCMC_z"]
    assert names[-3:] == ["pinkyTip_x", "pinkyTip_y", "pinkyTip_z"]
    assert vec[:6].tolist() == pytest.approx([0.0, 0.1, 0.2, 1.0, 1.1, 1.2])


def test_feature_specs_use_canonical_head_camera_names():
    records = {
        "calibrations": {
            "head_left_outer1": {"resolution": [1600, 1300]},
            "head_right_outer1": {"resolution": [1600, 1300]},
        },
        "camera_packets": {
            "observation.images.head_left_outer1": [(0.0, b"packet")],
            "observation.images.head_right_outer1": [(0.0, b"packet")],
        },
        "streams": {},
    }

    specs = feature_specs([records])

    assert "observation.images.head_left_outer1" in specs
    assert "observation.images.head_right_outer1" in specs
    assert "observation.images.camera0" not in specs
    assert specs["observation.images.head_left_outer1"]["shape"] == (1300, 1600, 3)


def test_jianzhi_master_feature_is_head_left():
    from format_convert.jianzhi.convert_jianzhi_mcap import JIANZHI_MASTER_FEATURE

    assert JIANZHI_MASTER_FEATURE == "observation.images.head_left"


def test_master_timestamps_prefer_head_left():
    records = {
        "camera_packets": {
            "observation.images.head_left_outer1": [(1.0, b"a"), (2.0, b"b")],
            "observation.images.head_left": [(10.0, b"a"), (20.0, b"b")],
        },
        "streams": {},
    }
    specs = {
        "observation.images.head_left_outer1": {"dtype": "video"},
        "observation.images.head_left": {"dtype": "video"},
    }

    assert master_timestamps(records, specs) == [10.0, 20.0]


def test_validate_camera_frame_counts_rejects_missing_camera_frame():
    records = {
        "camera_packets": {
            "observation.images.head_left_outer1": [(0.0, b"a"), (1.0, b"b")],
            "observation.images.head_left_outer0": [(0.0, b"a"), (1.0, b"b")],
            "observation.images.head_left": [(0.0, b"a"), (1.0, b"b")],
            "observation.images.head_right": [(0.0, b"a"), (1.0, b"b")],
            "observation.images.head_right_outer0": [(0.0, b"a"), (1.0, b"b")],
            "observation.images.head_right_outer1": [(0.0, b"a")],
        },
    }

    ok, reason = validate_camera_frame_counts(records)

    assert ok is False
    assert "frame count" in reason


def test_validate_camera_frame_counts_accepts_equal_camera_counts():
    records = {
        "camera_packets": {
            "observation.images.head_left_outer1": [(0.001, b"a"), (1.001, b"b")],
            "observation.images.head_left_outer0": [(0.001, b"a"), (1.001, b"b")],
            "observation.images.head_left": [(0.0, b"a"), (1.0, b"b")],
            "observation.images.head_right": [(0.001, b"a"), (1.001, b"b")],
            "observation.images.head_right_outer0": [(0.001, b"a"), (1.001, b"b")],
            "observation.images.head_right_outer1": [(0.001, b"a"), (1.001, b"b")],
        },
    }

    ok, reason = validate_camera_frame_counts(records)

    assert ok is True
    assert reason == ""


def test_feature_specs_use_head_pose_names():
    records = {
        "calibrations": {},
        "camera_packets": {},
        "streams": {
            "observation.head_pose": [(0.0, __import__("numpy").zeros(7, dtype="float32"))],
            "observation.relative_head_pose": [(0.0, __import__("numpy").zeros(7, dtype="float32"))],
        },
    }

    specs = feature_specs([records])

    assert "observation.head_pose" in specs
    assert "observation.relative_head_pose" in specs
    assert "observation.state" not in specs
    assert "observation.relative_eef_pose" not in specs
