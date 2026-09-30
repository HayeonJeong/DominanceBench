"""Entry point: score generated images with the DvD Score.

Example:
    python score_images.py \
        --meta_csv data/dominancebench_300.csv \
        --gen_dir outputs/sd14_dominancebench \
        --output_csv results/sd14_dominancebench_dvd.csv
"""

from dvd_score.config import ScoreArgumentParser
from dvd_score.pipeline import ScoringPipeline


def main() -> None:
    """Parse arguments and run the scoring pipeline."""
    config = ScoreArgumentParser().parse()
    pipeline = ScoringPipeline(config=config)
    pipeline.run()


if __name__ == "__main__":
    main()
