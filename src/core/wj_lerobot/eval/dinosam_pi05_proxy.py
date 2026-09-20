#!/usr/bin/env python3
"""HTTP proxy that applies the offline DINO+SAM cam_high masking before PI05."""

import argparse
import io
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
import requests
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image


class DinoSamMasker:
    def __init__(self, args):
        if str(args.sam2_root) not in sys.path:
            sys.path.insert(0, str(args.sam2_root))
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.device = args.device
        self.lock = threading.Lock()
        self.processor = AutoProcessor.from_pretrained(str(args.dino_model), local_files_only=True)
        self.dino = AutoModelForZeroShotObjectDetection.from_pretrained(
            str(args.dino_model), local_files_only=True
        ).to(self.device).eval()
        self.sam = SAM2ImagePredictor(
            build_sam2(str(args.sam2_config), str(args.sam2_checkpoint), device=self.device)
        )
        self.prompt = args.prompt
        self.threshold = args.box_threshold

    def mask_jpeg(self, payload: bytes) -> bytes:
        image_bgr = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError("cam_high is not a decodable image")
        with self.lock, torch.inference_mode():
            height, width = image_bgr.shape[:2]
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            model_inputs = self.processor(
                images=Image.fromarray(image_rgb), text=self.prompt, return_tensors="pt"
            ).to(self.device)
            outputs = self.dino(**model_inputs)
            scores = outputs.logits.sigmoid()[0].max(-1).values
            boxes = outputs.pred_boxes[0][scores > self.threshold]
            if len(boxes):
                scale = torch.tensor([width, height, width, height], device=self.device)
                cx, cy, bw, bh = (boxes * scale).unbind(-1)
                xyxy = torch.stack(
                    (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), dim=-1
                )
                self.sam.set_image(image_bgr)
                masks, _, _ = self.sam.predict(
                    box=xyxy.detach().cpu().numpy(), multimask_output=False
                )
                masks = np.asarray(masks)
                if masks.ndim == 4:
                    masks = masks[:, 0]
                if masks.ndim != 3:
                    raise RuntimeError(f"unexpected SAM2 mask shape: {masks.shape}")
                image_bgr[np.any(masks, axis=0)] = 0
        ok, encoded = cv2.imencode(
            ".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95]
        )
        if not ok:
            raise RuntimeError("failed to encode masked cam_high")
        return encoded.tobytes()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6013)
    parser.add_argument("--upstream", default="http://127.0.0.1:6011/v1/predict")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sam2-root", type=Path, default=Path("/tmp/sam2_kitt"))
    parser.add_argument(
        "--dino-model",
        type=Path,
        default=Path("/mnt/dataset/qichen_zhang/test_runs/Anyverse-VLA/models/grounding-dino-tiny"),
    )
    parser.add_argument(
        "--sam2-checkpoint",
        type=Path,
        default=Path("/mnt/dataset/qichen_zhang/test_runs/Anyverse-VLA/models/sam2_hiera_tiny.pt"),
    )
    parser.add_argument(
        "--sam2-config", type=Path, default=Path("/tmp/sam2_kitt/sam2/sam2_hiera_t.yaml")
    )
    parser.add_argument("--prompt", default="robot arm . robot gripper .")
    parser.add_argument("--box-threshold", type=float, default=0.30)
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    return parser.parse_args()


def create_app(args):
    masker = DinoSamMasker(args)
    app = FastAPI(title="DINO+SAM to PI05 proxy")

    @app.get("/health")
    def health():
        return {"status": "ok", "upstream": args.upstream}

    @app.post("/v1/predict")
    async def predict(
        cam_high: list[UploadFile] = File(...),
        wrist_left: list[UploadFile] = File(...),
        wrist_right: list[UploadFile] = File(...),
        state: str = Form(...),
        task: str | None = Form(None),
        clip_running_status: str | None = Form(None),
        include_action_chunk: str | None = Form(None),
        chunk_max_steps: str | None = Form(None),
        flow_noise_seed: str | None = Form(None),
        save_request: str | None = Form(None),
        client_request_id: str | None = Form(None),
        observation_stamp_ns: str | None = Form(None),
        client_request_started_unix_ns: str | None = Form(None),
        client_received_monotonic_ns: str | None = Form(None),
    ):
        try:
            masked = [masker.mask_jpeg(await image.read()) for image in cam_high]
            files = []
            files.extend(
                ("cam_high", (image.filename or "cam_high.jpg", payload, image.content_type or "image/jpeg"))
                for image, payload in zip(cam_high, masked)
            )
            for field, uploads in (("wrist_left", wrist_left), ("wrist_right", wrist_right)):
                for image in uploads:
                    files.append(
                        (field, (image.filename or f"{field}.jpg", await image.read(), image.content_type or "image/jpeg"))
                    )
            data = {"state": state}
            for key, value in {
                "task": task, "clip_running_status": clip_running_status,
                "include_action_chunk": include_action_chunk, "chunk_max_steps": chunk_max_steps,
                "flow_noise_seed": flow_noise_seed, "save_request": save_request,
                "client_request_id": client_request_id, "observation_stamp_ns": observation_stamp_ns,
                "client_request_started_unix_ns": client_request_started_unix_ns,
                "client_received_monotonic_ns": client_received_monotonic_ns,
            }.items():
                if value is not None:
                    data[key] = value
            upstream = requests.post(args.upstream, files=files, data=data, timeout=args.timeout_sec)
            content_type = upstream.headers.get("content-type", "application/json")
            return Response(content=upstream.content, status_code=upstream.status_code, media_type=content_type)
        except requests.RequestException as exc:
            raise HTTPException(status_code=502, detail=f"PI05 upstream unavailable: {exc}") from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"DINO+SAM preprocessing failed: {exc}") from exc

    return app


if __name__ == "__main__":
    args = parse_args()
    import uvicorn
    uvicorn.run(create_app(args), host=args.host, port=args.port)
