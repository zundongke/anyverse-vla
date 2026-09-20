"""Apply SAM2 contour blackout to RGB images using normalized DINO boxes."""
import numpy as np
import torch
from PIL import Image


class SamBoxMasker:
    def __init__(self, config, checkpoint, device):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self.predictor = SAM2ImagePredictor(build_sam2(config, str(checkpoint), device=device))

    @torch.inference_mode()
    def process(self, images, boxes):
        result = []
        for image, normalized in zip(images, boxes):
            rgb = np.asarray(image.convert('RGB')).copy()
            valid = normalized[np.all(normalized >= 0, axis=1)]
            if len(valid):
                h, w = rgb.shape[:2]
                self.predictor.set_image(rgb)
                masks, _, _ = self.predictor.predict(
                    box=valid * np.array([w, h, w, h]), multimask_output=False)
                masks = np.asarray(masks)
                if masks.ndim == 4:
                    masks = masks[:, 0]
                if masks.ndim != 3 or masks.shape[1:] != (h, w):
                    raise ValueError(f'Unexpected SAM2 mask shape: {masks.shape}')
                rgb[np.any(masks.astype(bool), axis=0)] = 0
            result.append(Image.fromarray(rgb))
        return result, boxes
