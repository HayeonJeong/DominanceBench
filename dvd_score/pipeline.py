"""End-to-end scoring loop: prompt table -> images -> per-image DvD CSV."""

import csv
import os
from typing import Dict, List, Optional

from tqdm import tqdm

from dvd_score.config import ScoreConfig
from dvd_score.dataset import ImageLocator, PromptTable
from dvd_score.judge import QwenYesNoJudge
from dvd_score.scorer import DvDScorer, ImageScore


OUTPUT_FIELDNAMES: List[str] = [
    "case_num",
    "content_type",
    "content",
    "nm_word",
    "prompt",
    "image_path",
    "image_name",
    "content_answers",
    "nm_answers",
    "content_score",
    "nm_score",
    "dvd_score",
    "dvd_mean_case",
]


class ScoringPipeline:
    """Scores every image of every prompt in the table and writes one CSV row per image."""

    def __init__(self, config: ScoreConfig) -> None:
        # Load the prompt table, the image locator, and the Qwen2-VL judge.
        self.config = config
        self.prompt_table = PromptTable(config.meta_csv)
        self.image_locator = ImageLocator(gen_dir=config.gen_dir)
        device = config.torch_device()
        print(f"[INFO] Loading {config.model_name} on {device}")
        judge = QwenYesNoJudge(
            model_name=config.model_name,
            device=device,
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
        )
        self.scorer = DvDScorer(judge=judge)
        self.missing_cases: List[int] = []

    def _is_in_case_range(self, case_num: int) -> bool:
        # Apply the optional --case_start / --case_end filters (inclusive).
        if self.config.case_start is not None and case_num < self.config.case_start:
            return False
        if self.config.case_end is not None and case_num > self.config.case_end:
            return False
        return True

    def _images_for_case(self, case_num: int) -> Optional[List[str]]:
        # Return the images to score for this case, or None if the case folder is missing or empty.
        case_dir = self.image_locator.find_case_dir(case_num)
        if case_dir is None:
            self.missing_cases.append(case_num)
            return None

        image_paths = self.image_locator.collect_images(case_dir)
        if not image_paths:
            print(f"[WARN] No images for case{case_num:03d} in {case_dir}")
            return None

        if self.config.max_images_per_case > 0:
            image_paths = image_paths[: self.config.max_images_per_case]
        return image_paths

    @staticmethod
    def _build_record(
        case_num: int,
        content_type: str,
        row: Dict[str, str],
        image_path: str,
        image_score: ImageScore,
    ) -> Dict[str, object]:
        # Convert one image score into an output CSV row.
        return {
            "case_num": case_num,
            "content_type": content_type,
            "content": str(row.get("content", "")).strip(),
            "nm_word": str(row.get("nm_word", "")).strip(),
            "prompt": str(row.get("prompt", "")).strip(),
            "image_path": image_path,
            "image_name": os.path.basename(image_path),
            "content_answers": str(image_score.content_answers),
            "nm_answers": str(image_score.object_answers),
            "content_score": int(image_score.content_score),
            "nm_score": int(image_score.object_score),
            "dvd_score": round(float(image_score.dvd_score), 4),
        }

    @staticmethod
    def _attach_case_mean(case_records: List[Dict[str, object]]) -> None:
        # Write the mean DvD over this case's images into every row of the case.
        if not case_records:
            return
        total = 0.0
        for record in case_records:
            total += float(record["dvd_score"])
        mean_dvd = round(total / len(case_records), 4)
        for record in case_records:
            record["dvd_mean_case"] = mean_dvd

    def _score_case(self, case_num: int, row: Dict[str, str], image_paths: List[str]) -> List[Dict[str, object]]:
        # Score all images of one prompt and attach the per-prompt mean.
        content = str(row.get("content", "")).strip()
        object_word = str(row.get("nm_word", "")).strip()
        content_type = PromptTable.resolve_content_type(row=row, case_num=case_num)

        case_records: List[Dict[str, object]] = []
        for image_path in image_paths:
            image_score = self.scorer.score_image(
                image_path=image_path,
                content=content,
                content_type=content_type,
                object_word=object_word,
            )
            record = self._build_record(
                case_num=case_num,
                content_type=content_type,
                row=row,
                image_path=image_path,
                image_score=image_score,
            )
            case_records.append(record)

        self._attach_case_mean(case_records)
        return case_records

    def _write_csv(self, records: List[Dict[str, object]]) -> None:
        # Write all rows to the output CSV, creating the parent folder if needed.
        output_dir = os.path.dirname(os.path.abspath(self.config.output_csv))
        os.makedirs(output_dir, exist_ok=True)
        with open(self.config.output_csv, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_FIELDNAMES)
            writer.writeheader()
            writer.writerows(records)

    def run(self) -> None:
        """Score all prompts in the table and save the per-image CSV."""
        all_records: List[Dict[str, object]] = []
        for row in tqdm(self.prompt_table.rows, desc="Evaluating"):
            case_num = PromptTable.parse_case_num(row)
            if case_num is None:
                continue
            if not self._is_in_case_range(case_num):
                continue
            image_paths = self._images_for_case(case_num)
            if image_paths is None:
                continue
            case_records = self._score_case(case_num=case_num, row=row, image_paths=image_paths)
            all_records.extend(case_records)

        self._write_csv(all_records)
        print(f"\n[OK] Saved CSV: {self.config.output_csv}")
        print(f"[OK] Rows: {len(all_records)}")
        if self.missing_cases:
            print(f"[WARN] Missing case folders: {len(self.missing_cases)} (e.g., {self.missing_cases[:10]})")
