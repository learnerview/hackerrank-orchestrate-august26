"""Media file path resolution.

Single responsibility: resolve the relative media paths stored in
images.csv / voice_notes.csv against the dataset directory into absolute
paths. Nothing else in the media layer needs to know how paths are stored.
"""

from __future__ import annotations

from pathlib import Path

from .interfaces import MediaAnalysisError


def resolve_media_path(dataset_dir: Path, relative_path: str) -> Path:
    """Resolve a media-relative path (e.g. ``media/images/img_001.jpg``)
    against the dataset directory and verify the file exists."""
    candidate = Path(dataset_dir) / relative_path
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise MediaAnalysisError(f"media file not found: {resolved}")
    return resolved
