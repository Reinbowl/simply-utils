from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from simply.detection.label_utils import _convert
from simply.general.file_utils import mkdir

FONT_PATH = Path(__file__).parent / "fonts" / "DejaVuSans.ttf"


def _auto_color(class_name: str) -> tuple[int, int, int]:
    """Deterministic bright RGB color from class name via hash, clamped to [100, 255]."""
    h = hash(class_name) & 0xFFFFFF
    r = ((h >> 16) & 0xFF) % 156 + 100
    g = ((h >> 8) & 0xFF) % 156 + 100
    b = (h & 0xFF) % 156 + 100
    return r, g, b


def _scale_thickness(image: Image.Image) -> int:
    """Scale border thickness relative to image size."""
    return max(1, int(min(image.width, image.height) * 0.003))


def _load_image(image: str | Path | Image.Image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    return Image.open(Path(image)).convert("RGB")


def _load_font(thickness: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    size = max(12, thickness * 6)
    if FONT_PATH.exists():
        return ImageFont.truetype(str(FONT_PATH), size=size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def draw_bboxes(
    image: str | Path | Image.Image,
    detections: list[list],
    *,
    fmt_in: str = "pixel",
    class_map: list[str] | None = None,
    class_colors: dict[str, tuple[int, int, int]] | None = None,
    class_filter: list[str] | None = None,
) -> Image.Image:
    """Draw bounding boxes onto an image and return the annotated copy.

    Draws each detection as a colored rectangle with a label showing the class
    name and confidence (if present). Colors are auto-assigned per class name
    deterministically, or overridden via `class_colors`. The original image is
    not modified.

    Args:
        image: File path (str or Path) or a PIL Image object.
        detections: List of detections in `fmt_in` format. Each detection is a list:
                    [class_name, x1, y1, x2, y2] or [class_name, conf, x1, y1, x2, y2] for pixel,
                    [class_id, cx, cy, w, h] or [class_id, conf, cx, cy, w, h] for norm.
        fmt_in: Format of the detections — "pixel" (default) or "norm".
        class_map: List of class names where index = class_id.
                   Required when fmt_in="norm".
        class_colors: Optional dict mapping class name to RGB color tuple.
                      Falls back to auto-assigned colors for unmapped classes.
        class_filter: If provided, only draw detections whose class name is in this list.

    Returns:
        Annotated RGB PIL Image (copy of the input).

    Raises:
        ValueError: If fmt_in is invalid or class_map is missing when fmt_in="norm".

    Example:
        >>> annotated = draw_bboxes("image.jpg", detections)
        >>> annotated = draw_bboxes(
        ...     "image.jpg",
        ...     detections,
        ...     fmt_in="norm",
        ...     class_map=["car", "person"],
        ...     class_filter=["car"],
        ... )
    """
    if fmt_in not in ("pixel", "norm"):
        raise ValueError(f"Invalid fmt_in '{fmt_in}', expected 'pixel' or 'norm'")
    if fmt_in == "norm" and class_map is None:
        raise ValueError("class_map is required when fmt_in='norm'")

    img = _load_image(image)
    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)
    thickness = _scale_thickness(annotated)
    font = _load_font(thickness)

    converted = detections
    if fmt_in == "norm":
        converted = [
            _convert(d, "norm", "pixel", class_map, annotated.width, annotated.height)
            for d in detections
        ]

    for detection in converted:
        class_name = str(detection[0])
        if class_filter is not None and class_name not in class_filter:
            continue

        has_conf = len(detection) == 6
        offset = 2 if has_conf else 1
        conf = float(detection[1]) if has_conf else None
        x1, y1, x2, y2 = (float(v) for v in detection[offset : offset + 4])

        color = (class_colors or {}).get(class_name) or _auto_color(class_name)

        draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)

        label = f"{class_name} {conf:.2f}" if conf is not None else class_name
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        pad = 3
        draw.rectangle(
            [x1, y1 - text_h - pad * 2, x1 + text_w + pad * 2, y1],
            fill=color,
        )
        draw.text((x1 + pad, y1 - text_h - pad), label, fill=(0, 0, 0), font=font)

    return annotated


def viz_bboxes(
    image: str | Path | Image.Image,
    detections: list[list],
    output_path: str | Path,
    *,
    fmt_in: str = "pixel",
    class_map: list[str] | None = None,
    class_colors: dict[str, tuple[int, int, int]] | None = None,
    class_filter: list[str] | None = None,
) -> None:
    """Draw bounding boxes onto an image and save the result.

    Accepts detections directly — use `read_label` beforehand to load from a
    label file. Output is always saved as a JPEG regardless of input image format.

    Args:
        image: File path (str or Path) or a PIL Image object.
        detections: List of detections in `fmt_in` format.
        output_path: Path to save the annotated image. Suffix is forced to .jpg.
        fmt_in: Format of the detections — "pixel" (default) or "norm".
        class_map: List of class names where index = class_id.
                   Required when fmt_in="norm".
        class_colors: Optional dict mapping class name to RGB color tuple.
        class_filter: If provided, only draw detections whose class name is in this list.

    Raises:
        ValueError: If fmt_in is invalid or class_map is missing when fmt_in="norm".

    Example:
        >>> detections = simply.read_label("image.txt")
        >>> viz_bboxes("image.jpg", detections, "output/image_viz")

        >>> detections = simply.read_label(
        ...     "image.txt",
        ...     fmt_in="norm",
        ...     fmt_out="pixel",
        ...     class_map=CLASS_MAP,
        ...     image_path="image.jpg",
        ... )
        >>> viz_bboxes("image.jpg", detections, "output/image_viz", class_filter=["car"])
    """
    annotated = draw_bboxes(
        image,
        detections,
        fmt_in=fmt_in,
        class_map=class_map,
        class_colors=class_colors,
        class_filter=class_filter,
    )

    dst = Path(output_path).with_suffix(".jpg")
    mkdir(dst)
    annotated.save(dst, format="JPEG", quality=95)
