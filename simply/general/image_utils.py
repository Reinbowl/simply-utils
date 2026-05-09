from pathlib import Path

from PIL import Image


def load_image(
    image: str | Path | Image.Image,
    *,
    bg_color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Load an image and convert it to RGB.

    Handles PNG transparency by compositing onto a solid background color
    before converting to RGB. Accepts a file path or an existing PIL Image.

    Args:
        image: File path (str or Path) or a PIL Image object.
        bg_color: RGB background color used when flattening transparent images.
                  Defaults to white (255, 255, 255).

    Returns:
        RGB PIL Image.

    Example:
        >>> img = load_image("photo.png")
        >>> img = load_image("photo.png", bg_color=(0, 0, 0))
    """
    if isinstance(image, Image.Image):
        img = image
    else:
        img = Image.open(Path(image))

    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        background = Image.new("RGB", img.size, bg_color)
        converted = img.convert("RGBA")
        background.paste(converted, mask=converted.split()[3])
        return background

    return img.convert("RGB")


def _compute_aspect_size(w: int, h: int, max_len: int) -> tuple[int, int]:
    """Scale w, h so longest side equals max_len, preserving aspect ratio."""
    scale = max_len / max(w, h)
    return round(w * scale), round(h * scale)


def _pad_to_square(
    img: Image.Image,
    target_size: int,
    pad_color: tuple[int, int, int],
) -> Image.Image:
    """Pad image to target_size x target_size, centering the image."""
    canvas = Image.new("RGB", (target_size, target_size), pad_color)
    x = (target_size - img.width) // 2
    y = (target_size - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def load_image_resized(
    image: str | Path | Image.Image,
    max_len: int,
    *,
    mode: str = "aspect",
    pad_color: tuple[int, int, int] = (0, 0, 0),
    downscale_only: bool = False,
) -> Image.Image:
    """Load and resize an image using one of four resize strategies.

    Internally calls `load_image`, so transparency is always handled.
    All resizing uses the Lanczos filter for both upscaling and downscaling.

    Args:
        image: File path (str or Path) or a PIL Image object.
        max_len: Target size for the longest side (or both sides for "stretch" and "pad_only").
        mode: Resize strategy:
            - "aspect"   — scale longest side to max_len, preserve aspect ratio.
            - "pad"      — same as "aspect", then pad shorter side to make a square
                           (equivalent to YOLO letterboxing when downscale_only=False).
            - "pad_only" — no resize, pad to max_len x max_len. Ignores downscale_only.
            - "stretch"  — hard resize to max_len x max_len, ignoring aspect ratio.
        pad_color: RGB color used for padding. Defaults to black (0, 0, 0).
        downscale_only: If True, skip resizing when the image already fits within max_len.
            For "aspect": no-op if longest side <= max_len.
            For "pad": no resize, but still pads to a square based on longest side.
            For "stretch": no-op if both sides <= max_len.
            Ignored for "pad_only".

    Returns:
        Resized (and optionally padded) RGB PIL Image.

    Raises:
        ValueError: If `mode` is not one of the accepted values.

    Example:
        >>> img = load_image_resized("photo.jpg", 640)
        >>> img = load_image_resized("photo.jpg", 640, mode="pad")
        >>> img = load_image_resized("photo.jpg", 640, mode="pad", downscale_only=True)
        >>> img = load_image_resized("photo.jpg", 640, mode="pad_only", pad_color=(114, 114, 114))
        >>> img = load_image_resized("photo.jpg", 640, mode="stretch")
    """
    if mode not in ("aspect", "pad", "pad_only", "stretch"):
        raise ValueError(f"Invalid mode '{mode}', expected 'aspect', 'pad', 'pad_only' or 'stretch'")

    img = load_image(image)
    w, h = img.size

    if mode == "aspect":
        if downscale_only and max(w, h) <= max_len:
            return img
        new_w, new_h = _compute_aspect_size(w, h, max_len)
        return img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    if mode == "pad":
        if downscale_only and max(w, h) <= max_len:
            return _pad_to_square(img, max(w, h), pad_color)
        new_w, new_h = _compute_aspect_size(w, h, max_len)
        resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        return _pad_to_square(resized, max_len, pad_color)

    if mode == "pad_only":
        return _pad_to_square(img, max_len, pad_color)

    # stretch
    if downscale_only and w <= max_len and h <= max_len:
        return img
    return img.resize((max_len, max_len), Image.Resampling.LANCZOS)
