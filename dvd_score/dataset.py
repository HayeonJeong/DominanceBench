"""Prompt table loading and generated-image lookup."""

import csv
import os
import re
from typing import Dict, List, Optional, Tuple


IMAGE_EXTENSIONS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp")
GALLERY_FOLDER_NAME: str = "0_gallery"


class PromptTable:
    """Reads a prompt CSV with columns case_num, content_type, content, nm_word, prompt."""

    def __init__(self, csv_path: str) -> None:
        # Load every row of the prompt CSV into memory.
        self.csv_path = csv_path
        self.rows = self._load_rows()

    def _load_rows(self) -> List[Dict[str, str]]:
        # Read rows as dictionaries keyed by the CSV header.
        with open(self.csv_path, "r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            return list(reader)

    @staticmethod
    def parse_case_num(row: Dict[str, str]) -> Optional[int]:
        """Return case_num as int, accepting '12' or '12.0'; None if missing or malformed."""
        raw_value = str(row.get("case_num", "")).strip()
        if raw_value.isdigit():
            return int(raw_value)
        if raw_value.replace(".", "", 1).isdigit():
            return int(float(raw_value))
        return None

    @staticmethod
    def default_content_type(case_num: int) -> str:
        """Fallback concept type by case range, used only when the CSV has no content_type column."""
        if 1 <= case_num <= 101:
            return "artist"
        elif 102 <= case_num <= 166:
            return "landmark"
        elif 202 <= case_num <= 236:
            return "character"
        else:
            return "artist"

    def resolve_content_type(self, row: Dict[str, str], case_num: int) -> str:
        """Use the content_type column when present, otherwise fall back to the case range."""
        raw_value = str(row.get("content_type", "")).strip()
        if raw_value:
            return raw_value
        return self.default_content_type(case_num)


class ImageLocator:
    """Finds generated images laid out as <gen_dir>/caseNNN_<anything>/<image files>."""

    def __init__(self, gen_dir: str, allow_gallery_fallback: bool) -> None:
        # Store the root folder of generated images and the fallback option.
        self.gen_dir = gen_dir
        self.allow_gallery_fallback = allow_gallery_fallback

    def find_case_dir(self, case_num: int) -> Optional[str]:
        """Return the first folder (sorted) whose name starts with caseNNN_, or None."""
        prefix = f"case{case_num:03d}_"
        candidates: List[str] = []
        for name in os.listdir(self.gen_dir):
            full_path = os.path.join(self.gen_dir, name)
            if os.path.isdir(full_path) and name.startswith(prefix):
                candidates.append(name)
        if not candidates:
            return None
        candidates.sort()
        return os.path.join(self.gen_dir, candidates[0])

    @staticmethod
    def collect_images_in_dir(case_dir: str) -> List[str]:
        """Return all image files directly inside case_dir, sorted by name."""
        image_paths: List[str] = []
        for name in sorted(os.listdir(case_dir)):
            full_path = os.path.join(case_dir, name)
            if os.path.isfile(full_path) and name.lower().endswith(IMAGE_EXTENSIONS):
                image_paths.append(full_path)
        return image_paths

    def collect_gallery_images(self, case_num: int) -> List[str]:
        """Return images for this case from <gen_dir>/0_gallery (files named caseNNN_*)."""
        gallery_dir = os.path.join(self.gen_dir, GALLERY_FOLDER_NAME)
        if not os.path.isdir(gallery_dir):
            return []
        pattern = re.compile(rf"^case0*{case_num}_.*")
        image_paths: List[str] = []
        for name in sorted(os.listdir(gallery_dir)):
            if pattern.match(name) and name.lower().endswith(IMAGE_EXTENSIONS):
                image_paths.append(os.path.join(gallery_dir, name))
        return image_paths

    def collect_images(self, case_dir: str, case_num: int) -> List[str]:
        """Collect images from the case folder, using the gallery fallback only if enabled and empty."""
        image_paths = self.collect_images_in_dir(case_dir)
        if not image_paths and self.allow_gallery_fallback:
            image_paths = self.collect_gallery_images(case_num)
        return image_paths
