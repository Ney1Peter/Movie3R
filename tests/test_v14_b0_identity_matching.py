import numpy as np
import torch

from dust3r.model import ARCroco3DStereo
from versions.v14.probe_b0_identity_matching import evaluate_matching


def human(root, detection_index):
    root = np.asarray(root, dtype=np.float64)
    torso = np.eye(3, dtype=np.float64)
    joints = root[None] + np.asarray(
        [[0.0, 0.0, 0.0], [0.0, -0.5, 0.0], [0.2, -0.3, 0.0]],
        dtype=np.float64,
    )
    return {
        "root": root,
        "torso": torso,
        "joints": joints,
        "detection_index": detection_index,
    }


def test_coarse_rotation_makes_root_identity_assignment_recoverable():
    pre = {
        "person0": human([-1.0, 0.0, 0.0], 0),
        "person1": human([1.0, 0.0, 0.0], 1),
    }
    post = {
        "person0": human([1.0, 0.0, 0.0], 0),
        "person1": human([-1.0, 0.0, 0.0], 1),
    }
    cache = {"humans": [pre, post]}
    direct = evaluate_matching(cache, np.eye(4))["matchers"]["root"]

    boundary = np.eye(4)
    boundary[:3, :3] = np.diag([-1.0, 1.0, -1.0])
    aligned = evaluate_matching(cache, boundary)["matchers"]["root"]

    assert direct["correct_count"] == 0
    assert aligned["correct_count"] == 2
    assert aligned["all_correct"]


def test_prompt_history_restarts_when_human_count_changes():
    model = ARCroco3DStereo.__new__(ARCroco3DStereo)
    previous = torch.ones(1, 3, 4)
    current = torch.full((1, 2, 4), 2.0)
    keep_previous = torch.zeros(1, 1, 1)

    blended = model._blend_v8_prompt_history(
        previous, current, update_mask=keep_previous
    )

    assert blended.shape == current.shape
    assert torch.equal(blended, torch.zeros_like(current))
