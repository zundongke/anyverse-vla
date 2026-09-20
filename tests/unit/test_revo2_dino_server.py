from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image, ImageOps
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from lerobot.utils.dino_masking import DinoMasker

SERVER_PATH = Path(__file__).resolve().parents[2] / 'src/core/wj_lerobot/eval/pi05_remote_server.py'
spec = importlib.util.spec_from_file_location('revo2_dino_server', SERVER_PATH)
server = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = server
spec.loader.exec_module(server)


class Inputs(dict):
    def to(self, device):
        return self


class FakeProcessor:
    def __call__(self, images, **kwargs):
        return Inputs(count=len(images))


class FakeDetector:
    def __call__(self, count):
        return SimpleNamespace(
            logits=torch.tensor([[[6.], [-6.], [4.]]]).repeat(count, 1, 1),
            pred_boxes=torch.tensor([[[.5, .5, .5, .5], [.1, .1, .2, .2], [.9, .9, .1, .1]]]).repeat(count, 1, 1),
        )


@pytest.fixture
def masker(monkeypatch):
    # Load predictable detection outputs without a network/model download.
    import lerobot.utils.dino_masking as masking
    monkeypatch.setattr(masking.AutoProcessor, 'from_pretrained', lambda *a, **k: FakeProcessor())
    detector = FakeDetector()
    detector.to = lambda device: detector
    detector.eval = lambda: detector
    monkeypatch.setattr(masking.AutoModelForZeroShotObjectDetection, 'from_pretrained', lambda *a, **k: detector)
    return DinoMasker(Path('/unused'), 'robot arm . robot gripper .', .3, 'cpu', 1)


@pytest.fixture(autouse=True)
def light_imports(monkeypatch):
    monkeypatch.setattr(server, '_lazy_imports', lambda: dict(
        torch=torch, Image=Image, FastAPI=FastAPI, File=File, Form=Form,
        UploadFile=UploadFile, JSONResponse=JSONResponse,
    ))


def png(image):
    stream = io.BytesIO()
    image.save(stream, format='PNG')
    return stream.getvalue()


def test_mask_and_padding_match_offline_pixels(masker):
    image = Image.new('RGB', (80, 40), (255, 255, 255))
    masked, boxes = masker.process([image])
    assert masked[0].getpixel((40, 20)) == (0, 0, 0)
    assert masked[0].getpixel((0, 0)) == (255, 255, 255)
    assert image.getpixel((40, 20)) == (255, 255, 255)
    assert masked[0].getpixel((72, 36)) == (255, 255, 255)  # Lower-ranked detection excluded.
    np.testing.assert_allclose(boxes[0, 0], [.25, .25, .75, .75])
    expected = np.asarray(ImageOps.pad(masked[0], (32, 32), color=(0, 0, 0)), dtype=np.float32) / 255
    actual = server._load_image_tensor(png(image), image_key='cam_high', image_size=32, masker=masker, pad=True)
    np.testing.assert_array_equal(actual.permute(1, 2, 0).numpy(), expected)


def test_no_detection_preserves_image(masker):
    masker.threshold = 1.
    image = Image.new('RGB', (80, 40), 'white')
    masked, boxes = masker.process([image])
    np.testing.assert_array_equal(np.asarray(masked[0]), np.asarray(image))
    assert (boxes == -1).all()


def test_http_single_camera_masks_all_history_frames(masker):
    class Service:
        args = server.ServerArgs(image_size=32, device='cpu')
        head_image_key = 'observation.images.top_head'
        left_image_key = right_image_key = None
        expected_state_dim = 18
        pad_input_images = True
        dino_masker = masker
        sample = None

        def predict(self, sample, **kwargs):
            self.sample = sample
            return {'action': [0.] * 30, 'action_chunk': [[0.] * 30]}

    service = Service()
    client = TestClient(server.build_app(service))
    image = png(Image.new('RGB', (80, 40), 'white'))
    response = client.post('/v1/predict', data={'state': json.dumps([0.] * 18), 'save_request': 'false'},
                           files=[('cam_high', ('a.png', image, 'image/png')), ('cam_high', ('b.png', image, 'image/png'))])
    assert response.status_code == 200, response.text
    images = service.sample['observation.images.top_head']
    assert images.shape == (1, 2, 3, 32, 32)
    assert torch.count_nonzero(images[:, :, :, 16, 16]) == 0
    assert len(response.json()['action']) == 30
    service.left_image_key = 'observation.images.wrist_left'
    service.right_image_key = 'observation.images.wrist_right'
    missing_wrists = client.post('/v1/predict', data={'state': json.dumps([0.] * 18), 'save_request': 'false'},
                                 files=[('cam_high', ('a.png', image, 'image/png'))])
    assert missing_wrists.status_code == 400


def test_camera_key_resolution():
    assert server._resolve_policy_image_keys(['observation.images.top_head']) == ('observation.images.top_head', None, None)
    keys = ['observation.images.cam_high', 'observation.images.wrist_left', 'observation.images.wrist_right']
    assert server._resolve_policy_image_keys(keys) == tuple(keys)


def test_server_output_restores_request_base_without_accumulation():
    import threading
    from lerobot.policies.pi05.revo2_relative import convert_revo2_relative

    base_pose = torch.tensor([2., 3., 4., 0., 1., 0., -1., 0., 0.])
    state = torch.cat((base_pose, base_pose))
    target_pose = base_pose.clone()
    target_pose[:3] += torch.tensor([1., 2., 3.])
    target = torch.cat((target_pose, target_pose, torch.arange(12))).reshape(1, 1, 30).repeat(1, 3, 1)
    _, relative_action = convert_revo2_relative(state[None], target)

    class Policy:
        config = SimpleNamespace(action_space='revo2_eef_pose')

        def predict_action_chunk_with_status_task_and_aux(self, model_in, **kwargs):
            return relative_action, torch.zeros(1, 3), torch.zeros(1, 2), None, None

        def get_last_inference_timing_ms(self):
            return {}

    service = server.PI05RemoteService.__new__(server.PI05RemoteService)
    service.args = server.ServerArgs(device='cpu')
    service._lock = threading.Lock()
    service.policy = Policy()
    service.action_target_mode = 'relative_pose'
    service.preprocessor = lambda sample: {'observation.state': convert_revo2_relative(sample['observation.state'][None])[0]}
    service.postprocessor = lambda action: action
    service.amp_dtype = torch.bfloat16
    service.task_id_to_label = []
    response = service.predict({'observation.state': state})
    torch.testing.assert_close(torch.tensor(response['action_chunk']), target[0])
    torch.testing.assert_close(torch.tensor(response['action']), target[0, 0])
    assert response['action_pose_frame'] == 'absolute'
    assert response['postprocessed']['action_pose_frame'] == 'relative_pose'
    torch.testing.assert_close(torch.tensor(response['postprocessed']['chunk']), relative_action)

    # A second request must use its own base rather than a cached previous pose.
    next_state = state.clone()
    next_state[0] += 5.
    next_response = service.predict({'observation.state': next_state})
    expected = target[0].clone()
    expected[:, 0] += 5.
    torch.testing.assert_close(torch.tensor(next_response['action_chunk']), expected)


def test_registered_preprocessor_accepts_observation_only():
    from lerobot.policies.pi05.processor_pi05 import Revo2RelativePoseProcessorStep
    from lerobot.processor.converters import batch_to_transition
    from lerobot.processor.core import TransitionKey

    pose = torch.tensor([2., 3., 4., 0., 1., 0., -1., 0., 0.])
    state = torch.cat((pose, pose))[None]
    transition = batch_to_transition({'observation.state': state})
    processed = Revo2RelativePoseProcessorStep()(transition)
    identity = torch.tensor([0., 0., 0., 1., 0., 0., 0., 1., 0.]).repeat(2)[None]
    torch.testing.assert_close(processed[TransitionKey.OBSERVATION]['observation.state'], identity)
    torch.testing.assert_close(transition[TransitionKey.OBSERVATION]['observation.state'], state)
    assert processed.get(TransitionKey.ACTION) is None
