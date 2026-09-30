"""DvD Score computation for a single generated image."""

from dataclasses import dataclass
from typing import List

from dvd_score.judge import QwenYesNoJudge
from dvd_score.questions import NUM_QUESTIONS_PER_SET, build_content_questions, build_object_questions


@dataclass
class ImageScore:
    """Per-image answers and scores.

    content_score: number of Yes answers to the five dominant-concept questions (0-5).
    object_score: number of Yes answers to the five object questions (0-5).
    dvd_score: content_score * (5 - object_score) / 25 * 100, in [0, 100].
    """

    content_answers: List[int]
    object_answers: List[int]
    content_score: int
    object_score: int
    dvd_score: float


class DvDScorer:
    """Scores how strongly the dominant concept appears while the object disappears."""

    def __init__(self, judge: QwenYesNoJudge) -> None:
        # Keep a reference to the loaded yes/no judge.
        self.judge = judge

    @staticmethod
    def compute_dvd(content_score: int, object_score: int) -> float:
        """DvD = c * (5 - n) / 25 * 100. High when the concept is present and the object is absent."""
        max_score = NUM_QUESTIONS_PER_SET
        return content_score * (max_score - object_score) / float(max_score * max_score) * 100.0

    def _answer_questions(
        self,
        image_path: str,
        questions: List[str],
        content_type: str,
        is_object_question: bool,
    ) -> List[int]:
        # Ask each question independently and collect the 0/1 answers in order.
        answers: List[int] = []
        for question in questions:
            answer = self.judge.ask(
                image_path=image_path,
                question=question,
                content_type=content_type,
                is_object_question=is_object_question,
            )
            answers.append(answer)
        return answers

    def score_image(self, image_path: str, content: str, content_type: str, object_word: str) -> ImageScore:
        """Ask the five content and five object questions and return the DvD Score of one image."""
        content_questions = build_content_questions(content=content, content_type=content_type)
        object_questions = build_object_questions(object_word=object_word)

        content_answers = self._answer_questions(
            image_path=image_path,
            questions=content_questions,
            content_type=content_type,
            is_object_question=False,
        )
        object_answers = self._answer_questions(
            image_path=image_path,
            questions=object_questions,
            content_type=content_type,
            is_object_question=True,
        )

        content_score = sum(content_answers)
        object_score = sum(object_answers)
        dvd_value = self.compute_dvd(content_score=content_score, object_score=object_score)

        return ImageScore(
            content_answers=content_answers,
            object_answers=object_answers,
            content_score=content_score,
            object_score=object_score,
            dvd_score=dvd_value,
        )
