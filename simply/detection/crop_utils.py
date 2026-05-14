from pathlib import Path

from PIL import Image

from simply.detection.label_utils import _convert


def _load_image(image: str | Path | Image.Image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    return Image.open(Path(image)).convert("RGB")


def _expand_bbox(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    pad: float,
    img_w: int,
    img_h: int,
) -> tuple[float, float, float, float]:
    """Expand bbox by pad fraction of its own dimensions, clipped to image bounds."""
    w = x2 - x1
    h = y2 - y1
    x1 = max(0.0, x1 - w * pad)
    y1 = max(0.0, y1 - h * pad)
    x2 = min(float(img_w), x2 + w * pad)
    y2 = min(float(img_h), y2 + h * pad)
    return x1, y1, x2, y2


def crop_bboxes(
    image: str | Path | Image.Image,
    detections: list[list],
    *,
    fmt_in: str = "pixel",
    fmt_out: str = "pixel",
    class_map: list[str] | None = None,
    class_filter: list[str | int] | None = None,
    pad: float = 0.0,
) -> tuple[list[Image.Image], list[list]]:
    """Crop bounding boxes from an image and return crops with their adjusted labels.

    Each detection is cropped from the image with an optional padding expansion.
    Returned labels are relative to each crop's top-left corner (0, 0).
    Padding is clipped to image bounds.

    Args:
        image: File path (str or Path) or a PIL Image object.
        detections: List of detections in `fmt_in` format.
        fmt_in: Format of the input detections — "pixel" (default) or "norm".
        fmt_out: Format of the returned labels — "pixel" (default) or "norm".
        class_map: List of class names where index = class_id.
                   Required when converting between formats.
        class_filter: If provided, only crop detections matching these classes.
                      Use class names (str) when fmt_in="pixel",
                      class ids (int) when fmt_in="norm".
        pad: Fractional padding to expand each bbox before cropping.
             e.g. 0.1 expands by 10% of the bbox width/height on each side.
             Expansion is clipped to image bounds. Defaults to 0.0.

    Returns:
        A tuple of (crops, labels) where:
        - crops: list of PIL Image objects, one per matched detection.
        - labels: list of single detections in fmt_out format, relative to
                  each crop's top-left corner. Pairs with crops by index.

    Raises:
        ValueError: If fmt values are invalid, or class_map is missing
                    when format conversion is required.

    Example:
        >>> detections = simply.read_label(
        ...     "image.txt", fmt_in="norm", fmt_out="pixel",
        ...     class_map=CLASS_MAP, image_path="image.jpg",
        ... )
        >>> crops, labels = simply.crop_bboxes("image.jpg", detections)

        >>> # With padding and class filter
        >>> crops, labels = simply.crop_bboxes(
        ...     "image.jpg", detections, class_filter=["car", "person"], pad=0.1
        ... )

        >>> # Iterate crops and labels together
        >>> for crop, label in zip(crops, labels):
        ...     print(label)  # [class_name, x1, y1, x2, y2]
    """
    if fmt_in not in ("pixel", "norm"):
        raise ValueError(f"Invalid fmt_in '{fmt_in}', expected 'pixel' or 'norm'")
    if fmt_out not in ("pixel", "norm"):
        raise ValueError(f"Invalid fmt_out '{fmt_out}', expected 'pixel' or 'norm'")

    img = _load_image(image)
    img_w, img_h = img.size

    # Validate class_map upfront if format conversion needed
    if fmt_in != fmt_out and class_map is None:
        raise ValueError(
            f"class_map is required for {fmt_in}→{fmt_out} conversion"
        )

    crops: list[Image.Image] = []
    labels: list[list] = []

    for detection in detections:
        # Convert to pixel for cropping
        if fmt_in == "norm":
            pixel_det = _convert(detection, "norm", "pixel", class_map, img_w, img_h)
        else:
            pixel_det = detection

        class_val = pixel_det[0]
        has_conf = len(pixel_det) == 6
        offset = 2 if has_conf else 1

        # Apply class filter
        if class_filter is not None:
            filter_val = detection[0]  # use original fmt_in value for filtering
            if filter_val not in class_filter:
                continue

        orig_x1, orig_y1, orig_x2, orig_y2 = (
            float(v) for v in pixel_det[offset : offset + 4]
        )

        # Expand bbox with padding, clipped to image bounds
        crop_x1, crop_y1, crop_x2, crop_y2 = _expand_bbox(
            orig_x1, orig_y1, orig_x2, orig_y2, pad, img_w, img_h
        )

        crop = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))

        # Original tight bbox relative to padded crop top-left
        rel_x1 = orig_x1 - crop_x1
        rel_y1 = orig_y1 - crop_y1
        rel_x2 = orig_x2 - crop_x1
        rel_y2 = orig_y2 - crop_y1

        # Build output detection in pixel fmt first
        out_det: list = [class_val]
        if has_conf:
            out_det.append(pixel_det[1])
        out_det.extend([rel_x1, rel_y1, rel_x2, rel_y2])

        # Convert output label to fmt_out if needed
        if fmt_out == "norm":
            crop_w, crop_h = crop.size
            out_det = _convert(out_det, "pixel", "norm", class_map, crop_w, crop_h)

        crops.append(crop)
        labels.append(out_det)

    return crops, labels
