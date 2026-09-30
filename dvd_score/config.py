"""Command-line configuration for DvD scoring."""

import argparse
from dataclasses import dataclass
from typing import List, Optional

import torch

from dvd_score.judge import DEFAULT_MAX_PIXELS, DEFAULT_MIN_PIXELS, DEFAULT_MODEL_NAME


@dataclass
class ScoreConfig:
    """All settings for one scoring run."""

    meta_csv: str
    gen_dir: str
    output_csv: str
    device: int
    model_name: str
    min_pixels: int
    max_pixels: int
    case_start: Optional[int]
    case_end: Optional[int]
    max_images_per_case: int
    allow_gallery_fallback: bool

    def torch_device(self) -> str:
        """Return 'cuda:<index>' when a GPU index is given and CUDA exists, otherwise 'cpu'."""
        if torch.cuda.is_available() and self.device >= 0:
            return f"cuda:{self.device}"
        return "cpu"


class ScoreArgumentParser:
    """Builds the argparse parser and converts parsed arguments into a ScoreConfig."""

    def __init__(self) -> None:
        # Create the parser and register every option.
        self.parser = argparse.ArgumentParser(
            description="Score generated images with the DvD Score (Qwen2-VL-2B-Instruct judge) and save a CSV."
        )
        self._add_arguments()

    def _add_arguments(self) -> None:
        # Register input/output paths, judge settings, and case filters.
        self.parser.add_argument("--meta_csv", type=str, required=True, help="Prompt CSV, e.g. data/dominancebench_300.csv")
        self.parser.add_argument("--gen_dir", type=str, required=True, help="Folder with caseNNN_*/ image subfolders")
        self.parser.add_argument("--output_csv", type=str, required=True, help="Where to write per-image scores")
        self.parser.add_argument("--device", type=int, default=0, help="CUDA device index (-1 for CPU)")
        self.parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
        self.parser.add_argument("--min_pixels", type=int, default=DEFAULT_MIN_PIXELS)
        self.parser.add_argument("--max_pixels", type=int, default=DEFAULT_MAX_PIXELS)
        self.parser.add_argument("--case_start", type=int, default=None)
        self.parser.add_argument("--case_end", type=int, default=None)
        self.parser.add_argument("--max_images_per_case", type=int, default=0, help="0 means use all images")
        self.parser.add_argument(
            "--allow_gallery_fallback",
            action="store_true",
            help="If a case folder has no images, use <gen_dir>/0_gallery/caseNNN_* images instead.",
        )

    def parse(self, argv: Optional[List[str]] = None) -> ScoreConfig:
        """Parse command-line arguments into a ScoreConfig."""
        args = self.parser.parse_args(argv)
        return ScoreConfig(
            meta_csv=args.meta_csv,
            gen_dir=args.gen_dir,
            output_csv=args.output_csv,
            device=args.device,
            model_name=args.model_name,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            case_start=args.case_start,
            case_end=args.case_end,
            max_images_per_case=args.max_images_per_case,
            allow_gallery_fallback=args.allow_gallery_fallback,
        )
