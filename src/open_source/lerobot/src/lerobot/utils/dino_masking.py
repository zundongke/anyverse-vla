"""Shared Revo2 offline/online Grounding DINO rectangular blackout masking."""

from pathlib import Path
import threading

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


class DinoMasker:
    def __init__(self, model_path: Path, prompt: str, threshold: float, device: str, max_boxes: int, *, short_edge: int | None = None):
        if not 0 <= threshold <= 1:
            raise ValueError("DINO threshold must be between 0 and 1")
        if max_boxes < 1:
            raise ValueError("DINO max_boxes must be positive")
        self.lock = threading.Lock()
        self.device = device
        self.prompt = prompt
        self.threshold = threshold
        self.max_boxes = max_boxes
        self.processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
        if short_edge is not None:
            if short_edge < 32:
                raise ValueError("DINO short_edge must be at least 32")
            self.processor.image_processor.size = {
                **self.processor.image_processor.size, "shortest_edge": short_edge,
            }
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            str(model_path), local_files_only=True
        ).to(device).eval()

    @torch.inference_mode()
    def process(self, images: list[Image.Image]) -> tuple[list[Image.Image], np.ndarray]:
        if not images:
            return [], np.full((0, self.max_boxes, 4), -1.0, dtype=np.float32)
        with self.lock:
            return self._process(images)

    def _process(self, images: list[Image.Image]) -> tuple[list[Image.Image], np.ndarray]:
        inputs = self.processor(
            images=images, text=[self.prompt] * len(images), return_tensors="pt", padding=True
        ).to(self.device)
        outputs = self.model(**inputs)
        scores = outputs.logits.sigmoid().max(-1).values.detach().cpu().numpy()
        boxes = outputs.pred_boxes.detach().cpu().numpy()
        result_images: list[Image.Image] = []
        result_boxes = np.full((len(images), self.max_boxes, 4), -1.0, dtype=np.float32)
        for i, image in enumerate(images):
            width, height = image.size
            keep = np.flatnonzero(scores[i] >= self.threshold)
            keep = keep[np.argsort(scores[i][keep])[::-1]][: self.max_boxes]
            draw_image = image.copy()
            draw = ImageDraw.Draw(draw_image)
            for box_i, query_i in enumerate(keep):
                cx, cy, bw, bh = boxes[i, query_i]
                x1 = float(np.clip((cx - bw / 2) * width, 0, width - 1))
                y1 = float(np.clip((cy - bh / 2) * height, 0, height - 1))
                x2 = float(np.clip((cx + bw / 2) * width, 0, width - 1))
                y2 = float(np.clip((cy + bh / 2) * height, 0, height - 1))
                result_boxes[i, box_i] = [x1 / width, y1 / height, x2 / width, y2 / height]
                draw.rectangle((x1, y1, x2, y2), fill=(0, 0, 0))
            result_images.append(draw_image)
        return result_images, result_boxes

