# Camera-Trap Direction & Activity Pipeline

A Python tool that processes camera-trap image sequences and produces, for each
movement sequence, the **activity class** (on foot, mountain bike, dog, horse,
wildlife, empty) and the **direction of movement** (e.g. in/out or up/down).
The result is written to an Excel file, one row per sequence.

The pipeline:

1. Reads images from a folder and extracts capture times from the EXIF headers.
2. Groups images into movement sequences (a new sequence starts after a time gap).
3. Detects objects with **MegaDetector V6** (via PytorchWildlife). **YOLOv8n** refines animal detections into dog / horse / wildlife and acts as a fallback if MegaDetector is unavailable.
4. Classifies the activity of each sequence from per-frame detection counts.
5. Estimates the direction with a **virtual grid** (a tripwire grid across the
   frame) and a centroid-trajectory fallback.
6. Writes `results/<site>_sequences.xlsx` (and `.csv`).

---

## 1. Installation

```bash
git clone <your-repo-url>
cd <your-repo>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

> **PyTorch:** `PytorchWildlife` and `ultralytics` both depend on PyTorch. If
> the automatic install does not match your system, install the correct CPU or
> CUDA build from <https://pytorch.org> first, then run the line above again.
> The pipeline runs on CPU; a GPU only makes detection faster.

---

## 2. Add your images

Put your camera-trap JPEGs into a folder under `data/`, for example:

```
data/
└── my_site/
    ├── IMG_0001.JPG
    ├── IMG_0002.JPG
    └── ...
```

Sub-folders are not scanned automatically — list each folder in the config.
Only `.jpg` / `.jpeg` files are read. Capture times come from the EXIF header
(if a file has no EXIF time, the file-modification time is used).

---

## 3. Configure

Copy the example config and edit it:

```bash
cp config.example.json config.json
```

```jsonc
{
  "sites": {
    "my_site": {                              // any name you like
      "image_folders": ["data/my_site"],      // one or more folders
      "calibration": {
        "axis": "x",        // "x" = subjects move LEFT/RIGHT in the image,
                            // "y" = subjects move UP/DOWN in the image
        "in_sign": "+",     // see "How to set the direction" below
        "in_label": "in",   // label written for one direction (e.g. "in" or "up")
        "out_label": "out"  // label for the opposite direction (e.g. "out" or "down")
      }
    }
  },

  "output_dir": "results",
  "daytime_hours": null,          // e.g. [6, 20] to keep only 06:00-19:59; null = all
  "sequence_gap_seconds": 120,    // gap that starts a new sequence
  "detect_conf": 0.20,            // detection confidence threshold
  "count_conf": 0.40,             // confidence threshold for counting persons/vehicles
  "grid_cols": 5,                 // virtual-grid columns along the path
  "grid_rows": 3,
  "min_move_px": 15.0,            // minimum centroid movement for the fallback
  "speed_slow_px_s": 80,          // speed-hint thresholds (pixels/second)
  "speed_fast_px_s": 300
}
```

### How to set the direction (calibration)

Open one image where you know which way the subject is going.

- If subjects move **left/right** in the image, use `"axis": "x"`.
  If the direction you want to call `in_label` moves **towards the right**
  (increasing x), set `"in_sign": "+"`; if it moves **towards the left**, set
  `"in_sign": "-"`.
- If subjects move **up/down** in the image, use `"axis": "y"`.
  `"in_sign": "+"` means the `in_label` direction moves **towards the bottom**
  (increasing y); `"-"` means **towards the top**.

The labels are free text — use `"in"`/`"out"`, `"up"`/`"down"`,
`"north"`/`"south"`, whatever fits your site. If the result comes out mirrored,
just flip `in_sign` (or swap the two labels).

---

## 4. Run

```bash
python run_pipeline.py                 # all sites in config.json
python run_pipeline.py --site my_site  # only one site
```

Detection results are cached as `results/<site>_cache.pkl`, so a second run
skips detection. **Delete that file to re-detect** (e.g. after adding images).

---

## 5. Output

`results/<site>_sequences.xlsx` (and `.csv`), one row per sequence:

| Column | Meaning |
|---|---|
| `timestamp` | capture time of the first image in the sequence |
| `location_id` | the site name from the config |
| `sequence_id` | running index |
| `primary_activity` | `on_foot` / `mtb` / `dog` / `horse` / `wildlife` / `empty` |
| `n_total_persons`, `n_mtb`, `n_on_foot`, `n_dogs`, `n_horses`, `n_other_wildlife` | peak per-frame counts |
| `n_empty_frames` | frames with no detection |
| `direction` | `in_label` / `out_label` / `undefined` |
| `speed_px_s` | pixel speed estimate |
| `activity_hint` | coarse speed-based hint |
| `confidence_flag` | `ok` if the subject was seen in ≥ 2 frames, else `low` |
| `n_images`, `n_points` | images in the sequence; frames with a detected subject |

A sequence is `undefined` when no person is detected, or when neither the grid
nor the trajectory fallback finds a clear movement.

---

## Project layout

```
.
├── run_pipeline.py          # command-line entry point
├── config.example.json      # copy to config.json and edit
├── requirements.txt
├── pathusage/
│   ├── config.py            # config loading + defaults
│   ├── io_utils.py          # EXIF, image listing, sequence grouping
│   ├── detection.py         # MegaDetector V6 + YOLOv8n fallback
│   ├── classification.py    # activity classification
│   ├── direction.py         # virtual grid + trajectory fallback
│   └── pipeline.py          # orchestration + Excel/CSV output
├── data/                    # your images go here (not tracked by git)
└── results/                 # output Excel/CSV go here (not tracked by git)
```

## Notes

- Works with any camera resolution — the grid adapts to each image's width.
- `data/` and `results/` are git-ignored; the tool ships with no data.
- Detection is the slow part; on CPU expect roughly a second or more per image.
- The repository contains the reusable production workflow. The validation notebooks used for the thesis evaluation are not included, as they depend on non-public ground-truth files.

## License

MIT — see [LICENSE](LICENSE).
