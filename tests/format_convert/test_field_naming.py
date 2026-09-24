from format_convert.field_naming import (
    CAMERA_OPTICAL_FRAME_CONVENTION,
    CANONICAL_HEAD_RING_CAMERA_NAMES,
    HAND_JOINT_NAMES,
    HAND_LANDMARK_NAMES,
    HAND_FEATURES,
    MOTION_FEATURES,
    WRIST_CAMERA_FEATURES,
    image_feature,
)


def test_head_ring_camera_name_vocabulary_is_canonical():
    assert CANONICAL_HEAD_RING_CAMERA_NAMES == {
        "head_left_outer1",
        "head_left_outer0",
        "head_left",
        "head_right",
        "head_right_outer0",
        "head_right_outer1",
    }
    assert image_feature("head_left_outer1") == "observation.images.head_left_outer1"


def test_camera_optical_frame_convention_is_opencv():
    assert CAMERA_OPTICAL_FRAME_CONVENTION == {
        "convention": "opencv_optical",
        "image_u": "right",
        "image_v": "down",
        "x": "right",
        "y": "down",
        "z": "forward",
    }


def test_wrist_and_hand_feature_names_are_canonical():
    assert WRIST_CAMERA_FEATURES == {
        "left": "observation.images.wrist_left",
        "right": "observation.images.wrist_right",
    }
    assert HAND_FEATURES["left_mano"] == "observation.hand_left_mano"
    assert HAND_FEATURES["left_points"] == "observation.hand_left_points"
    assert HAND_FEATURES["left_joints"] == "observation.hand_left_joints"
    assert HAND_FEATURES["right_mano"] == "observation.hand_right_mano"
    assert HAND_FEATURES["right_points"] == "observation.hand_right_points"
    assert HAND_FEATURES["right_joints"] == "observation.hand_right_joints"
    assert HAND_LANDMARK_NAMES == [
        "wrist",
        "thumbCMC", "thumbMCP", "thumbIP", "thumbTip",
        "indexMCP", "indexPIP", "indexDIP", "indexTip",
        "middleMCP", "middlePIP", "middleDIP", "middleTip",
        "ringMCP", "ringPIP", "ringDIP", "ringTip",
        "pinkyMCP", "pinkyPIP", "pinkyDIP", "pinkyTip",
    ]
    assert HAND_JOINT_NAMES == [
        "ThumbMCPSpread", "ThumbMCPStretch", "ThumbPIPStretch", "ThumbDIPStretch",
        "IndexSpread", "IndexMCPStretch", "IndexPIPStretch", "IndexDIPStretch",
        "MiddleSpread", "MiddleMCPStretch", "MiddlePIPStretch", "MiddleDIPStretch",
        "RingSpread", "RingMCPStretch", "RingPIPStretch", "RingDIPStretch",
        "PinkySpread", "PinkyMCPStretch", "PinkyPIPStretch", "PinkyDIPStretch",
    ]


def test_head_pose_feature_names_are_canonical():
    assert MOTION_FEATURES["head_pose"] == "observation.head_pose"
    assert MOTION_FEATURES["relative_head_pose"] == "observation.relative_head_pose"
