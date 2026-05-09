import shutil
from pathlib import Path

DEFAULT_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def get_files(file_path: str, *, recursive: bool = False, exts: list[str] | str | None = None) -> list[Path]:
    """Return a list of file paths matching the given extensions.

    Searches a directory, a single file, or a wildcard path for files whose
    suffix matches `exts`. Defaults to common image extensions if `exts` is
    not provided.

    Args:
        file_path: Path to a file, directory, or wildcard pattern (e.g. "data/202601*/images").
        recursive: If True, search all subdirectories. Required when `file_path` contains wildcards.
        exts: File extension(s) to match. Accepts a single string (e.g. ".jpg" or "jpg"),
              a list of strings, or None to use DEFAULT_EXTS.

    Returns:
        List of matching Path objects.

    Raises:
        ValueError: If a wildcard path is used with recursive=False, or if the path does not exist.

    Example:
        >>> get_files("data/images", recursive=True, exts=[".jpg", ".png"])
        >>> get_files("data/202601*/images", recursive=True)
        >>> get_files("image.jpg")
    """
    _exts_input = [exts] if isinstance(exts, str) else exts
    _exts: set[str] = {f".{e.lstrip('.')}" for e in _exts_input} if _exts_input else DEFAULT_EXTS
    has_wildcard = any(c in file_path for c in ("*", "?", "["))

    if has_wildcard and not recursive:
        raise ValueError(
            f"Wildcard paths require recursive=True, got recursive=False for: '{file_path}'"
        )

    root = Path(file_path)

    if not has_wildcard and root.is_file():
        return [root] if root.suffix.lower() in _exts else []

    if has_wildcard:
        parts = root.parts
        root_parts, wildcard_parts = [], []
        in_wildcard = False
        for part in parts:
            if any(c in part for c in ("*", "?", "[")) or in_wildcard:
                in_wildcard = True
                wildcard_parts.append(part)
            else:
                root_parts.append(part)

        root = Path(*root_parts) if root_parts else Path(".")
        pattern = str(Path(*wildcard_parts) / "**" / "*") if wildcard_parts else "**/*"

        if not root.exists():
            raise ValueError(f"Base path does not exist: '{root}'")

        return [
            p for p in root.glob(pattern)
            if p.is_file() and p.suffix.lower() in _exts
        ]

    if not root.exists():
        raise ValueError(f"File path does not exist: '{file_path}'")

    pattern = "**/*" if recursive else "*"
    return [p for p in root.glob(pattern) if p.is_file() and p.suffix.lower() in _exts]


def read_data(file_path: str | Path) -> list[str]:
    """Read a text file and return a list of non-empty stripped lines.

    Args:
        file_path: Path to the text file.

    Returns:
        List of stripped, non-empty lines.

    Example:
        >>> lines = read_data("labels.txt")
    """
    return [
        stripped
        for line in Path(file_path).read_text(encoding="utf-8").splitlines()
        if (stripped := line.strip())
    ]


def write_data(
    file_path: str | Path,
    items: list,
    *,
    sep: str | None = None,
) -> None:
    """Write a list of items to a text file.

    For a flat list, items are joined with `sep` into a single line, or written
    one per line if `sep` is None. For a nested list, each inner list becomes one
    line with items joined by `sep` (defaults to space if `sep` is None).

    Args:
        file_path: Path to the output text file.
        items: Flat list of items, or list of lists for multi-line output.
        sep: Separator used to join items. Defaults to newline for flat lists
             and space for nested lists when None.

    Example:
        >>> write_data("out.txt", ["a", "b", "c"])
        >>> write_data("out.txt", ["a", "b", "c"], sep=",")
        >>> write_data("out.txt", [["a", "b"], ["c", "d"]], sep=",")
    """
    def to_line(item: object) -> str:
        if isinstance(item, list):
            return (sep if sep is not None else " ").join(str(v) for v in item)
        return str(item)

    is_nested = items and isinstance(items[0], list)

    if is_nested:
        content = "\n".join(to_line(item) for item in items)
    else:
        content = sep.join(str(v) for v in items) if sep is not None else "\n".join(str(v) for v in items)

    Path(file_path).write_text(content + "\n", encoding="utf-8")


def _get_src_root(src_str: str) -> Path:
    """Return the non-wildcard prefix of a src path as the mirror root."""
    has_wildcard = any(c in src_str for c in ("*", "?", "["))
    if not has_wildcard:
        return Path(src_str).resolve()
    parts = Path(src_str).parts
    root_parts = []
    for part in parts:
        if any(c in part for c in ("*", "?", "[")):
            break
        root_parts.append(part)
    return Path(*root_parts).resolve() if root_parts else Path(".").resolve()


def _next_available(dst: Path) -> Path:
    """Find next available path by appending incrementing integer suffix."""
    counter = 1
    candidate = dst.with_stem(f"{dst.stem}_{counter}")
    while candidate.exists():
        counter += 1
        candidate = dst.with_stem(f"{dst.stem}_{counter}")
    return candidate


def _transfer(src: Path, dst: Path, *, mode: str, on_conflict: str) -> Path | None:
    """Handle transfer of a single resolved file to dst, respecting conflict rules."""
    if dst.exists() or dst.is_symlink():
        if on_conflict == "skip":
            return None
        elif on_conflict == "overwrite":
            dst.unlink()
        elif on_conflict == "auto_suffix":
            dst = _next_available(dst)

    dst.parent.mkdir(parents=True, exist_ok=True)

    if mode == "symlink":
        dst.symlink_to(src)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Invalid mode '{mode}', expected 'symlink' or 'copy'")

    return dst


def consolidate_files(
    src: str | Path | list[str | Path],
    dst_dir: str | Path,
    *,
    recursive: bool = False,
    exts: list[str] | str | None = None,
    mode: str = "symlink",
    structure: str = "flat",
    on_conflict: str = "skip",
) -> list[Path]:
    """Consolidate files from one or more source directories into a destination directory.

    Internally uses `get_files` for discovery. Supports symlink or copy transfer,
    flat or mirrored directory structure, and configurable conflict resolution.
    Source symlinks are always resolved before transfer.

    Args:
        src: A single source path, wildcard path, or list of paths.
        dst_dir: Destination directory. Created if it does not exist.
        recursive: Passed to `get_files` for subdirectory traversal.
        exts: File extensions to filter. Passed to `get_files`.
        mode: Transfer method — "symlink" (default) or "copy".
        structure: Output layout — "flat" (all files in dst_dir) or "mirror"
                   (preserves relative structure, each src dir as its own root).
        on_conflict: Conflict resolution — "skip" (default), "overwrite", or "auto_suffix".

    Returns:
        List of destination Path objects that were successfully written.

    Raises:
        ValueError: If `structure` or `on_conflict` values are invalid.

    Example:
        >>> consolidate_files(
        ...     src=["data/batch01", "data/batch02"],
        ...     dst_dir="data/consolidated",
        ...     recursive=True,
        ...     mode="symlink",
        ...     structure="mirror",
        ...     on_conflict="auto_suffix",
        ... )
    """
    if structure not in ("flat", "mirror"):
        raise ValueError(f"Invalid structure '{structure}', expected 'flat' or 'mirror'")
    if on_conflict not in ("skip", "overwrite", "auto_suffix"):
        raise ValueError(f"Invalid on_conflict '{on_conflict}', expected 'skip', 'overwrite' or 'auto_suffix'")

    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    sources: list[str | Path] = src if isinstance(src, list) else [src]
    created: list[Path] = []

    for src_entry in sources:
        src_str = str(src_entry)
        files = get_files(src_str, recursive=recursive, exts=exts)
        src_root = _get_src_root(src_str) if structure == "mirror" else None

        for file in files:
            resolved = file.resolve()

            if structure == "mirror" and src_root is not None:
                try:
                    rel = resolved.relative_to(src_root)
                except ValueError:
                    rel = file.relative_to(src_root) if file.is_relative_to(src_root) else Path(file.name)
                dst = dst_dir / src_root.name / rel
            else:
                dst = dst_dir / resolved.name

            result = _transfer(resolved, dst, mode=mode, on_conflict=on_conflict)
            if result is not None:
                created.append(result)

    return created
