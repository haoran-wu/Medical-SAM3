"""
SAM3 inference module for medical image segmentation.
Supports both box prompt and text prompt inference.
"""

import sys
import gc
from contextlib import nullcontext
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

# Resolve paths from this file so the scripts work regardless of cwd.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAM3_ROOT = PROJECT_ROOT.parent / "sam3"
sys.path.insert(0, str(SAM3_ROOT))

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model.box_ops import box_xywh_to_cxcywh


def normalize_bbox(bbox_xywh, img_w, img_h):
    """Normalize XYWH bounding boxes to [0, 1] coordinates."""
    if isinstance(bbox_xywh, list):
        assert len(bbox_xywh) == 4, "bbox_xywh list must have 4 elements."
        normalized_bbox = bbox_xywh.copy()
        normalized_bbox[0] /= img_w
        normalized_bbox[1] /= img_h
        normalized_bbox[2] /= img_w
        normalized_bbox[3] /= img_h
        return normalized_bbox

    normalized_bbox = bbox_xywh.clone()
    assert normalized_bbox.size(-1) == 4, "bbox_xywh tensor must end with 4 values."
    normalized_bbox[..., 0] /= img_w
    normalized_bbox[..., 1] /= img_h
    normalized_bbox[..., 2] /= img_w
    normalized_bbox[..., 3] /= img_h
    return normalized_bbox


class SAM3Model:
    """Wrapper for SAM3 model inference."""

    def __init__(
        self,
        confidence_threshold: float = 0.1,
        device: Optional[str] = None,
        checkpoint_path: Optional[str] = None
    ):
        """
        Initialize SAM3 model.

        Args:
            confidence_threshold: Minimum confidence for detections
            device: Device to run on ('cuda', 'mps', or 'cpu').
                    If None, auto-detect the best available device.
            checkpoint_path: Path to custom checkpoint file (optional).
                            If None, loads default SAM3 from HuggingFace.
        """
        self.device = device or self._select_device()
        self.confidence_threshold = confidence_threshold
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.processor = None

    @staticmethod
    def _select_device() -> str:
        """Pick the best available local device."""
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load_model(self):
        """Load SAM3 model (lazy loading)."""
        if self.model is not None:
            return

        if not SAM3_ROOT.exists():
            raise FileNotFoundError(
                f"SAM3 source not found at {SAM3_ROOT}. "
                "Clone https://github.com/facebookresearch/sam3.git next to Medical-SAM3 "
                "and install it with `pip install -e ../sam3`."
            )

        if self.checkpoint_path:
            print(f"Loading SAM3 model from checkpoint: {self.checkpoint_path}")
        else:
            print("Loading SAM3 model from HuggingFace...")
        print(f"Using device: {self.device}")

        # Enable CUDA-specific optimizations when available.
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Use bfloat16 autocast on CUDA only. CPU/MPS fall back to the default dtype.
        autocast_ctx = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.device == "cuda"
            else nullcontext()
        )
        autocast_ctx.__enter__()

        # Load model
        bpe_path = SAM3_ROOT / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"

        if self.checkpoint_path:
            # For custom checkpoints (e.g., MedSAM3), we need to handle
            # different checkpoint formats. First build the model without
            # loading weights, then load the custom checkpoint.
            self.model = build_sam3_image_model(
                bpe_path=str(bpe_path),
                checkpoint_path=None,  # Don't load checkpoint here
                load_from_HF=False     # Don't load from HF
            )
            # Load custom checkpoint with flexible format handling
            self._load_custom_checkpoint(self.checkpoint_path)
        else:
            # Use default HuggingFace loading
            try:
                self.model = build_sam3_image_model(
                    bpe_path=str(bpe_path),
                    checkpoint_path=None,
                    load_from_HF=True
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to load default SAM3 weights from HuggingFace. "
                    "If you want the fine-tuned Medical-SAM3 model, rerun with "
                    "`--checkpoint /path/to/your_checkpoint.pt`. "
                    "If you want the base SAM3 model, make sure this machine can "
                    "download `facebook/sam3` once or that the weights are already cached locally."
                ) from exc

        if hasattr(self.model, "to"):
            self.model = self.model.to(self.device)

        self.processor = Sam3Processor(
            self.model,
            confidence_threshold=self.confidence_threshold
        )

        print("SAM3 model loaded successfully!")

    def _load_custom_checkpoint(self, checkpoint_path: str):
        """
        Load custom checkpoint with flexible format handling.

        Handles both:
        - SAM3 format: keys with 'detector.' prefix
        - MedSAM3 format: keys without 'detector.' prefix
        """
        print(f"Loading custom checkpoint: {checkpoint_path}")

        ckpt = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )

        # Extract model state dict
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt

        # Check if keys have 'detector.' prefix
        sample_key = list(state_dict.keys())[0] if state_dict else ""
        has_detector_prefix = "detector." in sample_key

        if has_detector_prefix:
            # SAM3 format: strip 'detector.' prefix
            clean_state_dict = {
                k.replace("detector.", ""): v
                for k, v in state_dict.items()
                if "detector" in k
            }
        else:
            # MedSAM3 format: keys already match model structure
            clean_state_dict = state_dict

        # Load state dict
        missing_keys, unexpected_keys = self.model.load_state_dict(
            clean_state_dict, strict=False, assign=True
        )

        # Release checkpoint references as early as possible to reduce peak memory.
        del ckpt
        del state_dict
        del clean_state_dict
        gc.collect()

        if missing_keys:
            print(f"  Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"  Unexpected keys: {len(unexpected_keys)}")

    def encode_image(self, image: np.ndarray) -> dict:
        """
        Encode an image for inference.

        Args:
            image: RGB image as numpy array (H, W, 3)

        Returns:
            Inference state dictionary
        """
        self.load_model()

        # Convert to PIL Image
        pil_image = Image.fromarray(image)

        # Encode image
        inference_state = self.processor.set_image(pil_image)

        return inference_state

    def predict_box(
        self,
        inference_state: dict,
        bbox: Tuple[int, int, int, int],
        img_size: Tuple[int, int]
    ) -> Optional[np.ndarray]:
        """
        Run inference with bounding box prompt.

        Args:
            inference_state: Image encoding state
            bbox: Bounding box as (x_min, y_min, x_max, y_max) in pixels
            img_size: Image size as (height, width)

        Returns:
            Binary prediction mask or None if no prediction
        """
        return self.predict_boxes(inference_state, [bbox], img_size)

    def predict_boxes(
        self,
        inference_state: dict,
        bboxes: List[Tuple[int, int, int, int]],
        img_size: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        """
        Run inference with multiple bounding box prompts applied to the same state.
        """
        self.processor.reset_all_prompts(inference_state)

        img_h, img_w = img_size
        state = inference_state
        used_any = False

        for x_min, y_min, x_max, y_max in bboxes:
            width = x_max - x_min
            height = y_max - y_min
            box_xywh = torch.tensor([x_min, y_min, width, height], dtype=torch.float32).view(1, 4)
            box_cxcywh = box_xywh_to_cxcywh(box_xywh)
            norm_box = normalize_bbox(box_cxcywh, img_w, img_h).flatten().tolist()
            state = self.processor.add_geometric_prompt(
                state=state,
                box=norm_box,
                label=True,
            )
            used_any = True

        if used_any and state["masks"] is not None and len(state["masks"]) > 0:
            best_idx = torch.argmax(state["scores"]).item()
            pred_mask = state["masks"][best_idx].cpu().numpy() > 0
            return pred_mask.astype(np.uint8)

        return None

    def predict_text(
        self,
        inference_state: dict,
        text_prompt: str
    ) -> Optional[np.ndarray]:
        """
        Run inference with text prompt.

        Args:
            inference_state: Image encoding state
            text_prompt: Natural language description

        Returns:
            Binary prediction mask or None if no prediction
        """
        self.processor.reset_all_prompts(inference_state)

        # Run text-prompted inference
        text_state = self.processor.set_text_prompt(
            state=inference_state,
            prompt=text_prompt
        )

        if text_state["masks"] is not None and len(text_state["masks"]) > 0:
            best_idx = torch.argmax(text_state["scores"]).item()
            pred_mask = text_state["masks"][best_idx].cpu().numpy() > 0
            return pred_mask.astype(np.uint8)

        return None

    def predict_box_text(
        self,
        inference_state: dict,
        bbox: Tuple[int, int, int, int],
        text_prompt: str,
        img_size: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        """
        Run inference with a joint box + text prompt.

        This uses the same encoded image state, first attaching the text prompt
        and then the geometric prompt without resetting between them.
        """
        return self.predict_boxes_text(inference_state, [bbox], text_prompt, img_size)

    def predict_boxes_text(
        self,
        inference_state: dict,
        bboxes: List[Tuple[int, int, int, int]],
        text_prompt: str,
        img_size: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        """
        Run inference with a joint text prompt and multiple box prompts.
        """
        self.processor.reset_all_prompts(inference_state)

        img_h, img_w = img_size
        state = self.processor.set_text_prompt(
            state=inference_state,
            prompt=text_prompt,
        )

        used_any = False
        for x_min, y_min, x_max, y_max in bboxes:
            width = x_max - x_min
            height = y_max - y_min
            box_xywh = torch.tensor([x_min, y_min, width, height], dtype=torch.float32).view(1, 4)
            box_cxcywh = box_xywh_to_cxcywh(box_xywh)
            norm_box = normalize_bbox(box_cxcywh, img_w, img_h).flatten().tolist()
            state = self.processor.add_geometric_prompt(
                state=state,
                box=norm_box,
                label=True,
            )
            used_any = True

        if used_any and state["masks"] is not None and len(state["masks"]) > 0:
            best_idx = torch.argmax(state["scores"]).item()
            pred_mask = state["masks"][best_idx].cpu().numpy() > 0
            return pred_mask.astype(np.uint8)

        return None

    def get_confidence(self, state: dict) -> float:
        """Get confidence score from state."""
        if state["scores"] is not None and len(state["scores"]) > 0:
            best_idx = torch.argmax(state["scores"]).item()
            return state["scores"][best_idx].item()
        return 0.0


def generate_bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Generate bounding box from binary mask.

    Args:
        mask: Binary mask (H, W)

    Returns:
        Bounding box as (x_min, y_min, x_max, y_max) or None if mask is empty
    """
    if not mask.any():
        return None

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    y_indices = np.where(rows)[0]
    x_indices = np.where(cols)[0]

    if len(y_indices) == 0 or len(x_indices) == 0:
        return None

    y_min, y_max = y_indices[0], y_indices[-1]
    x_min, x_max = x_indices[0], x_indices[-1]

    return (x_min, y_min, x_max, y_max)


def resize_mask(mask: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """
    Resize a binary mask to target shape.

    Args:
        mask: Binary mask
        target_shape: (height, width)

    Returns:
        Resized mask
    """
    mask = np.squeeze(mask)

    # Convert to PIL Image for resizing
    mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
    mask_resized = mask_img.resize((target_shape[1], target_shape[0]), Image.NEAREST)

    return (np.array(mask_resized) > 127).astype(np.uint8)


if __name__ == "__main__":
    # Test SAM3 inference
    print("Testing SAM3 inference...")
    print("=" * 60)

    # Import dataset loaders
    from dataset_loaders import load_dataset

    # Initialize model
    sam3 = SAM3Model(confidence_threshold=0.1)

    # Test on one sample from each dataset
    for dataset_name in ["CHASE_DB1", "CVC-ClinicDB"]:
        print(f"\n{dataset_name}:")

        samples = list(load_dataset(dataset_name, max_samples=1))
        if not samples:
            print("  No samples found")
            continue

        sample = samples[0]
        print(f"  Sample ID: {sample.sample_id}")
        print(f"  Image shape: {sample.image.shape}")

        # Encode image
        inference_state = sam3.encode_image(sample.image)

        # Generate bbox from GT
        bbox = generate_bbox_from_mask(sample.gt_mask)
        if bbox is None:
            print("  No bbox generated from GT")
            continue

        print(f"  GT BBox: {bbox}")

        # Box prompt inference
        img_size = sample.gt_mask.shape
        pred_mask_box = sam3.predict_box(inference_state, bbox, img_size)

        if pred_mask_box is not None:
            # Resize if needed
            if pred_mask_box.shape != img_size:
                pred_mask_box = resize_mask(pred_mask_box, img_size)
            print(f"  Box prediction shape: {pred_mask_box.shape}")
            print(f"  Box prediction sum: {pred_mask_box.sum()}")
        else:
            print("  Box prediction: None")

        # Text prompt inference
        pred_mask_text = sam3.predict_text(inference_state, sample.text_prompt)

        if pred_mask_text is not None:
            if pred_mask_text.shape != img_size:
                pred_mask_text = resize_mask(pred_mask_text, img_size)
            print(f"  Text prediction shape: {pred_mask_text.shape}")
            print(f"  Text prediction sum: {pred_mask_text.sum()}")
        else:
            print("  Text prediction: None")

    print("\n" + "=" * 60)
    print("SAM3 inference test complete!")
