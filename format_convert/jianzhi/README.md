# Jianzhi Converter

This converter follows the global naming rules in
`docs/dataset_field_naming_rules.md`.

The Jianzhi MCAP files observed in this workspace expose six head-ring camera
topics as `/robot0/sensor/camera{0..5}/compressed`. For this source only, the
converter maps those raw IDs to canonical output features as:

| Raw id | Output feature |
| --- | --- |
| `camera0` | `observation.images.head_left_outer1` |
| `camera1` | `observation.images.head_left_outer0` |
| `camera2` | `observation.images.head_left` |
| `camera3` | `observation.images.head_right` |
| `camera4` | `observation.images.head_right_outer0` |
| `camera5` | `observation.images.head_right_outer1` |

Other data sources should define their own raw-to-canonical mapping instead of
assuming their `camera0` means the same physical camera.

## Frame Count Policy

Full conversion requires all six head-ring camera streams to have identical raw
frame counts. If any camera is missing or the counts differ, the converter skips
that MCAP episode instead of nearest-neighbor filling, trimming, or duplicating
frames.

High-rate non-camera streams such as IMU may have different sample counts and
are kept on their own timestamps.

## Batch Conversion

For the 1000h NAS tree, run:

```bash
python format_convert/jianzhi/batch_convert.py \
  --source /mnt/nas/synnas/jianzhi1000h/genrobot_BAAI_ego_1000h_细标_0918-dt-h5mdjncv \
  --out /mnt/nas/synnas/jianzhi1000h/jianzhi-lerobot \
  --workers 2
```

The batch script preserves the source relative layout. For example:

```text
<source>/task/session/file.mcap
-> <out>/task/session/file/
```

Each MCAP is converted by an independent subprocess. Batch logs are written to
`<out>/_batch_logs/.../*.log`, and resumable status files are written to
`<out>/_batch_status/.../*.json`. By default, items marked `done` or
`skipped_by_converter` are not rerun.

Use `--dry-run --limit N` to inspect the planned paths before launching a large
job. Start with `--workers 2` on NAS storage; increase only after checking that
the storage and ffmpeg remux throughput still have headroom.
