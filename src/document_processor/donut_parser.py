from __future__ import annotations

import json
import os
from typing import Any, Dict

from PIL import Image


class DonutDocumentParser:
    """
    Donut image-to-JSON parser for ID cards.
    Uses DocVQA fine-tune with optional INT8 dynamic quantization on CPU.
    """

    def __init__(self, model_id: str = "naver-clova-ix/donut-base-finetuned-docvqa") -> None:
        self.model_id = model_id
        self.processor = None
        self.model = None
        self.device = "cpu"
        self._init_model()

    def _init_model(self) -> None:
        try:
            import torch
            from transformers import DonutProcessor, VisionEncoderDecoderModel

            self.processor = DonutProcessor.from_pretrained(self.model_id)
            model = VisionEncoderDecoderModel.from_pretrained(self.model_id)
            if os.getenv("DONUT_INT8", "1") != "0":
                model = torch.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8
                )
            model = model.to("cpu")
            model.eval()
            self.model = model
        except Exception:
            self.processor = None
            self.model = None
            self.device = "cpu"

    def parse_to_json(self, image_path: str) -> Dict[str, Any]:
        if self.processor is None or self.model is None:
            return {"warning": "donut_model_unavailable", "raw": ""}

        try:
            import torch

            image = Image.open(image_path).convert("RGB")
            pixel_values = self.processor(image, return_tensors="pt").pixel_values

            task_prompt = "<s_docvqa><s_question>Extract all key-value fields from ID document.</s_question><s_answer>"
            decoder_input_ids = self.processor.tokenizer(
                task_prompt, add_special_tokens=False, return_tensors="pt"
            ).input_ids

            with torch.no_grad():
                outputs = self.model.generate(
                    pixel_values,
                    decoder_input_ids=decoder_input_ids,
                    max_length=int(self.model.decoder.config.max_position_embeddings),
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True,
                    bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                    return_dict_in_generate=True,
                )

            seq = self.processor.batch_decode(outputs.sequences)[0]
            seq = seq.replace(self.processor.tokenizer.eos_token, "").replace(
                self.processor.tokenizer.pad_token, ""
            )
            seq = self.processor.token2json(seq)
            if isinstance(seq, dict):
                return seq
            if isinstance(seq, str):
                try:
                    return json.loads(seq)
                except json.JSONDecodeError:
                    return {"raw": seq}
            return {"raw": str(seq)}
        except Exception as exc:
            return {"warning": "donut_inference_failed", "error": str(exc)}
