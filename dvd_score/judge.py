"""Qwen2-VL yes/no judge used to answer concept-presence questions."""

from typing import Any, Dict, List, Tuple

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from dvd_score.questions import select_instruction_tail


DEFAULT_MODEL_NAME: str = "Qwen/Qwen2-VL-2B-Instruct"
DEFAULT_MIN_PIXELS: int = 256 * 28 * 28
DEFAULT_MAX_PIXELS: int = 1280 * 28 * 28
MAX_NEW_TOKENS: int = 5


class QwenYesNoJudge:
    """Loads Qwen2-VL once and answers one yes/no question per call (Yes -> 1, No -> 0)."""

    def __init__(
        self,
        model_name: str,
        device: str,
        min_pixels: int,
        max_pixels: int,
    ) -> None:
        # Store settings and load the model and processor immediately.
        self.model_name = model_name
        self.device = device
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.model, self.processor = self._load_model_and_processor()

    def _load_model_and_processor(self) -> Tuple[Qwen2VLForConditionalGeneration, Any]:
        # Load the judge in its native dtype on GPU, or float32 on CPU.
        if torch.cuda.is_available():
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map="auto",
            )
        else:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
            )

        processor = AutoProcessor.from_pretrained(
            self.model_name,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        return model, processor

    def _build_messages(self, image_path: str, question_with_tail: str) -> List[Dict[str, Any]]:
        # Build a single-turn chat message holding one image and one question.
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": question_with_tail},
                ],
            }
        ]

    def ask(self, image_path: str, question: str, content_type: str, is_object_question: bool) -> int:
        """Ask one question about one image with greedy decoding and return 1 for Yes, 0 otherwise."""
        tail = select_instruction_tail(content_type=content_type, is_object_question=is_object_question)
        messages = self._build_messages(image_path=image_path, question_with_tail=f"{question}{tail}")

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)

        prompt_length = len(inputs.input_ids[0])
        generated_tokens = outputs[0, prompt_length:]
        decoded_answer = self.processor.decode(generated_tokens, skip_special_tokens=True).strip().lower()

        if decoded_answer.startswith("yes"):
            return 1
        return 0
