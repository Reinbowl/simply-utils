# [simply-utils](https://pypi.org/project/simply-utils/)

Simple utility library for file management, image processing, and object detection workflows.

```bash
pip install simply-utils
```
```python
import simply
```

---

## Table of Contents

### `general`
| Function | Description |
|---|---|
| [`mkdir`](#mkdir) | Join path parts and create directories |
| [`get_files`](#get_files) | Find files by extension in a directory or wildcard path |
| [`read_data`](#read_data) | Read a text file into a list of lines |
| [`write_data`](#write_data) | Write a flat or nested list to a text file |
| [`consolidate_files`](#consolidate_files) | Consolidate files from multiple sources into one directory |
| [`load_image`](#load_image) | Load an image as RGB, handling PNG transparency |
| [`load_image_resized`](#load_image_resized) | Load and resize an image with aspect, pad, pad_only, or stretch modes |

### `detection`
| Function | Description |
|---|---|
| [`read_label`](#read_label) | Parse a YOLO-format label file into structured detections |
| [`write_label`](#write_label) | Write structured detections to a label file |
| [`draw_bboxes`](#draw_bboxes) | Draw bounding boxes onto an image |
| [`crop_bboxes`](#crop_bboxes) | Crop bounding boxes from an image, returns crops and relative labels |
| [`viz_bboxes`](#viz_bboxes) | Draw bounding boxes onto an image and save as JPEG |

---

## general

### `mkdir`

Join path parts and create all directories. If the path looks like a file path (has a suffix),
the parent directory is created instead.

```python
simply.mkdir("outputs", "images")       # creates outputs/images
simply.mkdir("/a/b", "c/d")             # creates /a/b/c/d
simply.mkdir("outputs/result.jpg")      # creates outputs/
```

---

### `get_files`

Find files by extension inside a directory, a single file path, or a wildcard path.
Defaults to `{".jpg", ".jpeg", ".png", ".bmp"}` when `exts` is not provided.

```python
# Search a directory recursively
files = simply.get_files("data/images", recursive=True)

# Filter by extension — string or list, with or without dot
files = simply.get_files("data", recursive=True, exts=".txt")
files = simply.get_files("data", recursive=True, exts=["jpg", ".png"])

# Single file
files = simply.get_files("image.jpg")

# Wildcard path — files in each matched folder only
files = simply.get_files("data/202601*/images")

# Wildcard path — also search subfolders of each matched directory
files = simply.get_files("data/202601*/images", recursive=True)
```

---

### `read_data`

Read a text file and return a list of stripped, non-empty lines.

```python
lines = simply.read_data("labels.txt")
# ["line one", "line two", "line three"]
```

---

### `write_data`

Write a flat or nested list to a text file.

```python
# Flat list — one item per line
simply.write_data("out.txt", ["a", "b", "c"])
# a
# b
# c

# Flat list — joined into a single line
simply.write_data("out.txt", ["a", "b", "c"], sep=",")
# a,b,c

# Nested list — one inner list per line, joined by sep
simply.write_data("out.txt", [["a", "b"], ["c", "d"]], sep=",")
# a,b
# c,d

# Nested list — default space separator
simply.write_data("out.txt", [["a", "b"], ["c", "d"]])
# a b
# c d
```

---

### `consolidate_files`

Consolidate files from one or more source directories into a single destination.
Source symlinks are always resolved before transfer.

```python
# Flat consolidation via symlink (default)
simply.consolidate_files(
    src="data/batch01",
    dst_dir="data/consolidated",
    recursive=True,
)

# Multiple sources, mirror structure, copy mode, auto-suffix on conflict
simply.consolidate_files(
    src=["data/batch01", "data/batch02"],
    dst_dir="data/consolidated",
    recursive=True,
    mode="copy",               # "symlink" (default) | "copy"
    structure="mirror",        # "flat" (default) | "mirror"
    on_conflict="auto_suffix", # "skip" (default) | "overwrite" | "auto_suffix"
)

# Wildcard source
simply.consolidate_files(
    src="data/202601*/images",
    dst_dir="data/consolidated",
    recursive=True,
    exts=[".jpg", ".png"],
)
```

---

### `load_image`

Load an image as RGB. Handles PNG transparency by compositing onto a background color.

```python
img = simply.load_image("photo.jpg")

# Custom background for transparent PNGs
img = simply.load_image("photo.png", bg_color=(0, 0, 0))

# Also accepts a PIL Image
from PIL import Image
pil_img = Image.open("photo.png")
img = simply.load_image(pil_img)
```

---

### `load_image_resized`

Load and resize an image using one of four strategies. Internally calls `load_image`
so transparency is always handled. All resizing uses the Lanczos filter.

```python
# "aspect" — scale longest side to max_len, preserve aspect ratio (default)
img = simply.load_image_resized("photo.jpg", 640)

# "pad" — resize then pad to square (YOLO letterbox)
img = simply.load_image_resized("photo.jpg", 640, mode="pad")
img = simply.load_image_resized("photo.jpg", 640, mode="pad", pad_color=(114, 114, 114))

# "pad" + downscale_only — skip resize if fits, still pad to square
img = simply.load_image_resized("photo.jpg", 640, mode="pad", downscale_only=True)

# "pad_only" — no resize, always pad to max_len x max_len
img = simply.load_image_resized("photo.jpg", 640, mode="pad_only")

# "stretch" — hard resize to max_len x max_len
img = simply.load_image_resized("photo.jpg", 640, mode="stretch")

# downscale_only — no-op if image already fits within max_len
img = simply.load_image_resized("photo.jpg", 640, downscale_only=True)
```

| `mode` | Resize | Output size | `downscale_only` effect |
|---|---|---|---|
| `"aspect"` | Longest side → max_len | Variable | No-op if longest side ≤ max_len |
| `"pad"` | Longest side → max_len | max_len × max_len | No resize, pad to longest side² |
| `"pad_only"` | None | max_len × max_len | Ignored |
| `"stretch"` | Both sides → max_len | max_len × max_len | No-op if both sides ≤ max_len |

---

## detection

### `read_label`

Parse a detection label `.txt` file into a list of detections. Supports two formats:
- `"norm"` — YOLO normalised: `class_id cx cy w h` (values in [0.0, 1.0])
- `"pixel"` — absolute pixel corners: `class_name x1 y1 x2 y2`

Confidence is optional, detected automatically by field count (5 = no conf, 6 = conf present).

```python
CLASS_MAP = ["car", "person", "bike"]

# Read pixel format (default)
detections = simply.read_label("image.txt")
# [["car", 100.0, 200.0, 400.0, 600.0], ...]

# Read norm format, convert to pixel
detections = simply.read_label(
    "image.txt",
    fmt_in="norm",
    fmt_out="pixel",
    class_map=CLASS_MAP,
    image_path="image.jpg",
)

# Skip malformed lines instead of raising
detections = simply.read_label("image.txt", skip_malformed=True)
```

---

### `write_label`

Write a list of detections to a label `.txt` file. Mirrors `read_label` — accepts
detections in `fmt_in` format and writes in `fmt_out` format.

```python
# Write pixel format (default)
simply.write_label("image.txt", detections)

# Convert pixel → norm on write
simply.write_label(
    "image.txt",
    detections,
    fmt_in="pixel",
    fmt_out="norm",
    class_map=CLASS_MAP,
    image_path="image.jpg",
)
```

---

### `draw_bboxes`

Draw bounding boxes onto an image and return the annotated copy. The original is not modified.
Colors are deterministically auto-assigned per class name, or overridden via `class_colors`.

```python
from PIL import Image

img = Image.open("image.jpg")
annotated = simply.draw_bboxes(img, detections)
annotated.save("annotated.jpg")

# Filter to specific classes
annotated = simply.draw_bboxes(img, detections, class_filter=["car", "person"])

# Custom colors
annotated = simply.draw_bboxes(
    img,
    detections,
    class_colors={"car": (255, 100, 100), "person": (100, 255, 100)},
)

# Draw from norm format directly
annotated = simply.draw_bboxes(
    "image.jpg",
    detections,
    fmt_in="norm",
    class_map=CLASS_MAP,
)
```

---

### `crop_bboxes`

Crop bounding boxes from an image and return paired crops and labels.
Returned labels are relative to each crop's top-left corner `(0, 0)`.
Confidence is forwarded to the returned label if present in the input detection.

```python
# Basic crop — all detections
crops, labels = simply.crop_bboxes("image.jpg", detections)

# With padding and class filter
crops, labels = simply.crop_bboxes(
    "image.jpg",
    detections,
    class_filter=["car", "person"],
    pad=0.1,  # expand bbox by 10% on each side, clipped to image bounds
)

# Iterate crops and labels together
for crop, label in zip(crops, labels):
    print(label)  # [class_name, x1, y1, x2, y2] relative to crop

# Full example — read labels then crop
detections = simply.read_label(
    "image.txt",
    fmt_in="norm",
    fmt_out="pixel",
    class_map=CLASS_MAP,
    image_path="image.jpg",
)
crops, labels = simply.crop_bboxes("image.jpg", detections, class_filter=["car"], pad=0.1)
```

---

### `viz_bboxes`

Draw bounding boxes onto an image and save as JPEG. Accepts detections directly —
use `read_label` beforehand to load from a label file.
Output suffix is always `.jpg` regardless of input format.

```python
# Basic usage — pixel format detections (default)
detections = simply.read_label("image.txt")
simply.viz_bboxes("image.jpg", detections, "output/image_viz")

# Norm format labels — read and convert first
detections = simply.read_label(
    "image.txt",
    fmt_in="norm",
    fmt_out="pixel",
    class_map=CLASS_MAP,
    image_path="image.jpg",
)
simply.viz_bboxes("image.jpg", detections, "output/image_viz")

# Filter classes and custom colors
simply.viz_bboxes(
    "image.jpg",
    detections,
    "output/image_viz",
    class_filter=["car"],
    class_colors={"car": (255, 180, 0)},
)
```

---

## Package Structure

```
simply/
    general/
        file_utils.py     # get_files, read_data, write_data, consolidate_files
        image_utils.py    # load_image, load_image_resized
    detection/
        label_utils.py    # read_label, write_label
        bbox_utils.py     # draw_bboxes, viz_bboxes
        crop_utils.py     # crop_bboxes
        fonts/
            DejaVuSans.ttf  # place font here for anti-aliased label text
```

## License

MIT
