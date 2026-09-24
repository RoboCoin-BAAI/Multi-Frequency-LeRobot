# Dataset Field Naming Rules


## General Rules

- Image features use `observation.images.<camera_name>`.
- Motion and sensor features use `observation.<name>`.
- Control targets use `action`.
- Converter-specific raw identifiers do not appear in final feature keys.
- Calibration and annotation metadata should refer to the same canonical names as
  the dataset features.

## Head Ring Cameras

Head-ring output names are canonical camera positions, not raw source IDs.
Converters must map each source's raw identifiers to the appropriate canonical
name in that converter or its config.

Camera image and optical-frame axes follow the OpenCV optical convention:

```text
image u: right
image v: down

camera optical frame:
  +x: right
  +y: down
  +z: forward along the optical axis
```

Allowed six-position head-ring names:

- `observation.images.head_left_outer1`
- `observation.images.head_left_outer0`
- `observation.images.head_left`
- `observation.images.head_right`
- `observation.images.head_right_outer0`
- `observation.images.head_right_outer1`

Do not publish final features as `camera0`, `camera1`, etc. Those raw IDs may be
stored in calibration metadata as provenance fields such as `raw_camera_id`.

## Wrist Cameras

Use:

- `observation.images.wrist_left`
- `observation.images.wrist_right`

If a wrist has multiple cameras, extend under the same side prefix, for example
`observation.images.wrist_left_inner` and `observation.images.wrist_left_outer`.

## Hand Features

Use MANO and 21-point features separately:

- `observation.hand_left_mano`: MANO parameters, 157 dimensions.
- `observation.hand_right_mano`: MANO parameters, 157 dimensions.
- `observation.hand_left_points`: 21 global xyz points flattened to 63 dimensions.
- `observation.hand_right_points`: 21 global xyz points flattened to 63 dimensions.
- `observation.hand_left_joints`: MANUS-style joint angles, 20 dimensions.
- `observation.hand_right_joints`: MANUS-style joint angles, 20 dimensions.

Hand point features use this 21-landmark order:

```text
wrist,
thumbCMC, thumbMCP, thumbIP, thumbTip,
indexMCP, indexPIP, indexDIP, indexTip,
middleMCP, middlePIP, middleDIP, middleTip,
ringMCP, ringPIP, ringDIP, ringTip,
pinkyMCP, pinkyPIP, pinkyDIP, pinkyTip
```

Flatten point coordinates in point-major xyz order:

```text
wrist_x, wrist_y, wrist_z, thumbCMC_x, thumbCMC_y, thumbCMC_z, ...
```

MANUS-style joint features are split by hand. Because the feature key already
contains `left` or `right`, per-dimension names do not include a side prefix:

```text
ThumbMCPSpread, ThumbMCPStretch, ThumbPIPStretch, ThumbDIPStretch,
IndexSpread, IndexMCPStretch, IndexPIPStretch, IndexDIPStretch,
MiddleSpread, MiddleMCPStretch, MiddlePIPStretch, MiddleDIPStretch,
RingSpread, RingMCPStretch, RingPIPStretch, RingDIPStretch,
PinkySpread, PinkyMCPStretch, PinkyPIPStretch, PinkyDIPStretch
```

## Motion Features

Use:

- `observation.imu`: IMU `[ax, ay, az, gx, gy, gz]`.
- `observation.head_pose`: head-mounted device pose in the dataset's main world frame.
- `observation.relative_head_pose`: relative head-mounted device pose when present.
- `observation.state`: generic robot state when the source is not a head-mounted device.
- `action`: action or target command.

When a source has multiple IMUs or robot states, add explicit position or device
qualifiers after the base name, such as `observation.imu_head` or
`observation.state_left_arm`.
