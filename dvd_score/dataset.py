"""Prompt table loading and generated-image lookup."""

import csv
import os
from typing import Dict, List, Optional, Tuple


IMAGE_EXTENSIONS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp")


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
    def resolve_content_type(row: Dict[str, str], case_num: int) -> str:
        """Return the content_type column value; raise if it is missing so no prompt is scored with the wrong questions."""
        raw_value = str(row.get("content_type", "")).strip()
        if not raw_value:
            raise ValueError(f"case {case_num}: the prompt CSV must have a non-empty content_type column")
        return raw_value


class ImageLocator:
    """Finds generated images laid out as <gen_dir>/caseNNN_<anything>/<image files>."""

    def __init__(self, gen_dir: str) -> None:
        # Store the root folder of generated images.
        self.gen_dir = gen_dir

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
    def collect_images(case_dir: str) -> List[str]:
        """Return all image files directly inside case_dir, sorted by name."""
        image_paths: List[str] = []
        for name in sorted(os.listdir(case_dir)):
            full_path = os.path.join(case_dir, name)
            if os.path.isfile(full_path) and name.lower().endswith(IMAGE_EXTENSIONS):
                image_paths.append(full_path)
        return image_paths
