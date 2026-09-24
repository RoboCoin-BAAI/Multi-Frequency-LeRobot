"""Canonical dataset feature names shared by format converters.

This module defines the allowed output names. Dataset-specific raw stream IDs
must be mapped to these names inside each converter or its config.
"""

from __future__ import annotations


IMAGE_PREFIX = "observation.images"

CAMERA_OPTICAL_FRAME_CONVENTION = {
    "convention": "opencv_optical",
    "image_u": "right",
    "image_v": "down",
    "x": "right",
    "y": "down",
    "z": "forward",
}

CANONICAL_HEAD_RING_CAMERA_NAMES = {
    "head_left_outer1",
    "head_left_outer0",
    "head_left",
    "head_right",
    "head_right_outer0",
    "head_right_outer1",
}


def image_feature(camera_name: str) -> str:
    return f"{IMAGE_PREFIX}.{camera_name}"


WRIST_CAMERA_FEATURES = {
    "left": image_feature("wrist_left"),
    "right": image_feature("wrist_right"),
}

HAND_FEATURES = {
    "left_mano": "observation.hand_left_mano",
    "right_mano": "observation.hand_right_mano",
    "left_points": "observation.hand_left_points",
    "right_points": "observation.hand_right_points",
    "left_joints": "observation.hand_left_joints",
    "right_joints": "observation.hand_right_joints",
}

HAND_LANDMARK_NAMES = [
    "wrist",
    "thumbCMC", "thumbMCP", "thumbIP", "thumbTip",
    "indexMCP", "indexPIP", "indexDIP", "indexTip",
    "middleMCP", "middlePIP", "middleDIP", "middleTip",
    "ringMCP", "ringPIP", "ringDIP", "ringTip",
    "pinkyMCP", "pinkyPIP", "pinkyDIP", "pinkyTip",
]

HAND_JOINT_NAMES = [
    "ThumbMCPSpread", "ThumbMCPStretch", "ThumbPIPStretch", "ThumbDIPStretch",
    "IndexSpread", "IndexMCPStretch", "IndexPIPStretch", "IndexDIPStretch",
    "MiddleSpread", "MiddleMCPStretch", "MiddlePIPStretch", "MiddleDIPStretch",
    "RingSpread", "RingMCPStretch", "RingPIPStretch", "RingDIPStretch",
    "PinkySpread", "PinkyMCPStretch", "PinkyPIPStretch", "PinkyDIPStretch",
]

MOTION_FEATURES = {
    "imu": "observation.imu",
    "state": "observation.state",
    "head_pose": "observation.head_pose",
    "relative_head_pose": "observation.relative_head_pose",
    "action": "action",
}
