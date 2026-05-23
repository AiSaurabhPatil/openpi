import dataclasses

import einops
import numpy as np

from openpi import transforms


def make_ffw_bg2_example() -> dict:
    """Creates a random input example for the FFW BG2 policy (Isaac Sim format)."""
    return {
        "head_camera": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "left_wrist_camera": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "right_wrist_camera": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "left_arm_joints": np.random.rand(7).astype(np.float32),
        "right_arm_joints": np.random.rand(7).astype(np.float32),
        "left_gripper": np.float32(np.random.rand()),
        "right_gripper": np.float32(np.random.rand()),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    """Normalize image to uint8 [H, W, C] format."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        # (C, H, W) -> (H, W, C)
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class FFWBG2Inputs(transforms.DataTransformFn):
    """Maps raw FFW BG2 observations from Isaac Sim to the pi0 standard format.

    Expected Isaac Sim input dict keys:
      - head_camera: ndarray [H, W, 3] uint8
      - left_wrist_camera: ndarray [H, W, 3] uint8
      - right_wrist_camera: ndarray [H, W, 3] uint8
      - left_arm_joints: ndarray [7] float32
      - right_arm_joints: ndarray [7] float32
      - left_gripper: float32 (normalized [0, 1])
      - right_gripper: float32 (normalized [0, 1])
      - prompt (optional): str
    During LeRobot training, the config repacks `observation.state` to `state` and
    `action` to `actions`; this transform accepts that format too.

    Output (pi0 standard dict):
      - image: {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
      - image_mask: same keys, bool masks
      - state: ndarray [16] (7 + 1 + 7 + 1)
      - prompt: str
    """

    def __call__(self, data: dict) -> dict:
        # Parse images to uint8 [H, W, C]
        head_image = _parse_image(data["head_camera"])
        left_wrist_image = _parse_image(data["left_wrist_camera"])
        right_wrist_image = _parse_image(data["right_wrist_camera"])

        if "state" in data:
            state = np.asarray(data["state"]).ravel().astype(np.float32)
        else:
            # State: left arm (7) + left gripper (1) + right arm (7) + right gripper (1) = 16
            left_gripper = np.asarray(data["left_gripper"]).ravel()
            right_gripper = np.asarray(data["right_gripper"]).ravel()
            state = np.concatenate([
                np.asarray(data["left_arm_joints"]).ravel(),
                left_gripper,
                np.asarray(data["right_arm_joints"]).ravel(),
                right_gripper,
            ]).astype(np.float32)

        images = {
            "base_0_rgb": head_image,
            "left_wrist_0_rgb": left_wrist_image,
            "right_wrist_0_rgb": right_wrist_image,
        }
        image_masks = dict.fromkeys(images, np.True_)

        inputs = {
            "state": state,
            "image": images,
            "image_mask": image_masks,
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                inputs["prompt"] = data["prompt"].decode("utf-8")
            else:
                inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class FFWBG2Outputs(transforms.DataTransformFn):
    """Extracts the first 16 action dims (7+1+7+1) from the full 32-dim model output."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :16])}

