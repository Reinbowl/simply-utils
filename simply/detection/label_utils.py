from pathlib import Path


def _requires_conversion_args(
    class_map: list[str] | None,
    image_path: str | Path | None,
    direction: str,
) -> tuple[int, int]:
    if class_map is None:
        raise ValueError(f"class_map is required for {direction} conversion")
    if image_path is None:
        raise ValueError(f"image_path is required for {direction} conversion")
    from PIL import Image
    with Image.open(Path(image_path)) as img:
        return img.width, img.height


def _norm_to_pixel(
    detection: list,
    class_map: list[str],
    image_w: int,
    image_h: int,
) -> list:
    class_id = int(detection[0])
    if class_id >= len(class_map):
        raise ValueError(
            f"class_id {class_id} exceeds class_map length {len(class_map)}"
        )
    class_name = class_map[class_id]

    has_conf = len(detection) == 6
    offset = 2 if has_conf else 1
    conf = detection[1] if has_conf else None

    cx, cy, w, h = (float(v) for v in detection[offset : offset + 4])
    x1 = (cx - w / 2) * image_w
    y1 = (cy - h / 2) * image_h
    x2 = (cx + w / 2) * image_w
    y2 = (cy + h / 2) * image_h

    result: list = [class_name]
    if has_conf:
        result.append(conf)
    result.extend([x1, y1, x2, y2])
    return result


def _pixel_to_norm(
    detection: list,
    class_map: list[str],
    image_w: int,
    image_h: int,
) -> list:
    class_name = str(detection[0])
    if class_name not in class_map:
        raise ValueError(f"class_name '{class_name}' not found in class_map")
    class_id = class_map.index(class_name)

    has_conf = len(detection) == 6
    offset = 2 if has_conf else 1
    conf = detection[1] if has_conf else None

    x1, y1, x2, y2 = (float(v) for v in detection[offset : offset + 4])
    cx = ((x1 + x2) / 2) / image_w
    cy = ((y1 + y2) / 2) / image_h
    w = (x2 - x1) / image_w
    h = (y2 - y1) / image_h

    result: list = [class_id]
    if has_conf:
        result.append(conf)
    result.extend([cx, cy, w, h])
    return result


def _parse_line(line: str, fmt_in: str) -> list:
    parts = line.strip().split()
    if len(parts) not in (5, 6):
        raise ValueError(f"Malformed detection line: '{line}'")

    if fmt_in == "norm":
        first = int(parts[0])
    else:
        first = parts[0]

    rest = [float(p) for p in parts[1:]]
    return [first, *rest]


def _convert(
    detection: list,
    fmt_in: str,
    fmt_out: str,
    class_map: list[str] | None,
    image_w: int | None,
    image_h: int | None,
) -> list:
    if fmt_in == fmt_out:
        return detection
    if fmt_in == "norm" and fmt_out == "pixel":
        return _norm_to_pixel(detection, class_map, image_w, image_h)  # type: ignore[arg-type]
    return _pixel_to_norm(detection, class_map, image_w, image_h)  # type: ignore[arg-type]


def _serialize(detection: list, fmt_out: str) -> str:
    has_conf = len(detection) == 6
    offset = 2 if has_conf else 1
    conf_str = f" {float(detection[1]):.6f}" if has_conf else ""

    if fmt_out == "norm":
        coords = " ".join(f"{v:.6f}" for v in detection[offset : offset + 4])
        return f"{detection[0]}{conf_str} {coords}"
    else:
        coords = " ".join(f"{float(v):.2f}" for v in detection[offset : offset + 4])
        return f"{detection[0]}{conf_str} {coords}"


def read_label(
    file_path: str | Path,
    *,
    fmt_in: str = "pixel",
    fmt_out: str = "pixel",
    class_map: list[str] | None = None,
    image_path: str | Path | None = None,
    skip_malformed: bool = False,
) -> list[list]:
    """Read a detection label file and return a list of detections.

    Each line in the file represents one detection. Supports two formats:
    - "norm": normalised YOLO format — class_id cx cy w h (values in [0.0, 1.0])
    - "pixel": absolute pixel format — class_name x1 y1 x2 y2

    Confidence score is optional. Detected automatically by field count:
    5 fields = no confidence, 6 fields = confidence present (after class).

    Args:
        file_path: Path to the label .txt file.
        fmt_in: Format of the label file — "pixel" (default) or "norm".
        fmt_out: Desired output format — "pixel" (default) or "norm".
        class_map: List of class names where index = class_id.
                   Required when converting between formats.
        image_path: Path to the corresponding image file. Used to infer
                    image dimensions. Required when converting between formats.
        skip_malformed: If True, silently skip malformed lines. If False, raises ValueError.

    Returns:
        List of detections. Each detection is a list:
        [class_name, x1, y1, x2, y2] or [class_name, conf, x1, y1, x2, y2] for pixel fmt,
        [class_id, cx, cy, w, h] or [class_id, conf, cx, cy, w, h] for norm fmt.
        Returns an empty list if the file has no detections.

    Raises:
        ValueError: On invalid fmt values, missing conversion args, malformed lines
                    (when skip_malformed=False), or class_id/name not found in class_map.

    Example:
        >>> detections = read_label("image.txt")
        >>> detections = read_label(
        ...     "image.txt",
        ...     fmt_in="norm",
        ...     fmt_out="pixel",
        ...     class_map=["car", "person"],
        ...     image_path="image.jpg",
        ... )
    """
    if fmt_in not in ("norm", "pixel"):
        raise ValueError(f"Invalid fmt_in '{fmt_in}', expected 'norm' or 'pixel'")
    if fmt_out not in ("norm", "pixel"):
        raise ValueError(f"Invalid fmt_out '{fmt_out}', expected 'norm' or 'pixel'")

    image_w, image_h = None, None
    if fmt_in != fmt_out:
        image_w, image_h = _requires_conversion_args(class_map, image_path, f"{fmt_in}→{fmt_out}")

    lines = Path(file_path).read_text(encoding="utf-8").splitlines()
    detections: list[list] = []

    for line in lines:
        if not line.strip():
            continue
        try:
            parsed = _parse_line(line, fmt_in)
            converted = _convert(parsed, fmt_in, fmt_out, class_map, image_w, image_h)
            detections.append(converted)
        except ValueError:
            if skip_malformed:
                continue
            raise

    return detections


def write_label(
    file_path: str | Path,
    detections: list[list],
    *,
    fmt_in: str = "pixel",
    fmt_out: str = "pixel",
    class_map: list[str] | None = None,
    image_path: str | Path | None = None,
) -> None:
    """Write a list of detections to a label .txt file.

    Each detection is written as one line. Mirrors the input/output format
    convention of `read_label` — accepts detections in `fmt_in` format and
    writes them in `fmt_out` format, converting if necessary.

    Args:
        file_path: Path to the output label .txt file.
        detections: List of detections, each a list in `fmt_in` format.
        fmt_in: Format of the incoming detections — "pixel" (default) or "norm".
        fmt_out: Format to write on disk — "pixel" (default) or "norm".
        class_map: List of class names where index = class_id.
                   Required when converting between formats.
        image_path: Path to the corresponding image file. Used to infer
                    image dimensions. Required when converting between formats.

    Raises:
        ValueError: On invalid fmt values or missing conversion args.

    Example:
        >>> write_label("image.txt", detections)
        >>> write_label(
        ...     "image.txt",
        ...     detections,
        ...     fmt_in="pixel",
        ...     fmt_out="norm",
        ...     class_map=["car", "person"],
        ...     image_path="image.jpg",
        ... )
    """
    if fmt_in not in ("norm", "pixel"):
        raise ValueError(f"Invalid fmt_in '{fmt_in}', expected 'norm' or 'pixel'")
    if fmt_out not in ("norm", "pixel"):
        raise ValueError(f"Invalid fmt_out '{fmt_out}', expected 'norm' or 'pixel'")

    image_w, image_h = None, None
    if fmt_in != fmt_out:
        image_w, image_h = _requires_conversion_args(class_map, image_path, f"{fmt_in}→{fmt_out}")

    lines: list[str] = []
    for detection in detections:
        converted = _convert(detection, fmt_in, fmt_out, class_map, image_w, image_h)
        lines.append(_serialize(converted, fmt_out))

    Path(file_path).write_text(
        "\n".join(lines) + "\n" if lines else "", encoding="utf-8"
    )
