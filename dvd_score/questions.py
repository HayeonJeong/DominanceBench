"""Concept-presence question sets used by the DvD Score.

Each concept type has five yes/no questions. `{content}` is replaced with the
dominant concept (artist, landmark, or character) or with the object word.
"""

from typing import Dict, List


ARTIST_QUESTIONS: List[str] = [
    "Is this image painted in the artistic style of {content}, using realistic lighting and smooth brush strokes?",
    "Does the person or subject look painted with the realistic portrait technique typical of {content}?",
    "Does this image show brushwork, color tones, or shading similar to paintings by {content}?",
    "Does this look like a painting created by {content}, rather than a photo or modern artwork?",
    "Is the visual composition, lighting, and realism clearly similar to {content}'s paintings?",
]

LANDMARK_QUESTIONS: List[str] = [
    "Is this a photograph or realistic depiction of the {content}?",
    "Is the structure clearly identifiable as the {content}?",
    "Is the {content} or part of it visible anywhere in the image?",
    "Does the image show architectural details or a skyline resembling the {content}?",
    "Is the {content} the main subject or part of the background in this image?",
]

CHARACTER_QUESTIONS: List[str] = [
    "Is {content} visible in the image?",
    "Does the person or figure look like {content}?",
    "Does the image contain a logo, costume, or any reference to {content}?",
    "Is the scene related to, inspired by, or set in the world of {content}?",
    "Is there a visual element or theme in the image connected to {content}?",
]

OBJECT_QUESTIONS: List[str] = [
    "Is a {content} visible in the image?",
    "Is there a {content} included or partly shown in the image?",
    "Does the image feature or focus on a {content}?",
    "Is the appearance or shape recognizable as a {content}?",
    "Does this image look like a product photo or include a depiction of a {content}?",
]

CONTENT_QUESTIONS_BY_TYPE: Dict[str, List[str]] = {
    "artist": ARTIST_QUESTIONS,
    "landmark": LANDMARK_QUESTIONS,
    "character": CHARACTER_QUESTIONS,
}

# Artist-style questions use a lenient instruction: a style resemblance counts as "Yes".
LENIENT_ARTIST_TAIL: str = (
    " Answer 'Yes' if this image even vaguely or stylistically reminds you of the painter's work, "
    "even if it's not an exact match. /no_think"
)

# All other questions (landmark, character, and every object question) use a strict instruction.
STRICT_TAIL: str = " Answer only 'Yes' or 'No'. /no_think"

NUM_QUESTIONS_PER_SET: int = 5


def build_content_questions(content: str, content_type: str) -> List[str]:
    """Fill the five content questions for the given concept type."""
    if content_type not in CONTENT_QUESTIONS_BY_TYPE:
        raise ValueError(f"Invalid content_type: {content_type}")

    templates = CONTENT_QUESTIONS_BY_TYPE[content_type]
    questions: List[str] = []
    for template in templates:
        questions.append(template.format(content=content))
    return questions


def build_object_questions(object_word: str) -> List[str]:
    """Fill the five object questions for the given object word."""
    questions: List[str] = []
    for template in OBJECT_QUESTIONS:
        questions.append(template.format(content=object_word))
    return questions


def select_instruction_tail(content_type: str, is_object_question: bool) -> str:
    """Return the answer instruction appended to a question.

    Only artist content questions are lenient. Artist object questions and all
    landmark/character questions are strict.
    """
    if content_type == "artist" and not is_object_question:
        return LENIENT_ARTIST_TAIL
    return STRICT_TAIL
