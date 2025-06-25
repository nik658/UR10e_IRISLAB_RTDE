import dataclasses
import pathlib
import numpy as np
from typing import override

# Assuming these imports from the OpenPI framework
from src.openpi.policies import transforms
from src.openpi.policies import _model
from src.openpi.policies import _transforms
from src.openpi.training import DataConfig, DataConfigFactory, AssetsConfig
from src.openpi.training import ModelTransformFactory
from src.openpi.training import TrainConfig, weight_loaders
from src.openpi.models import pi0


def _parse_image(image_data):
    """
    Parse image data to uint8 (H,W,C) format.
    LeRobot automatically stores as float32 (C,H,W), so we need to convert.
    This gets skipped for policy inference.
    """
    if isinstance(image_data, np.ndarray):
        if image_data.dtype != np.uint8:
            # Convert from float32 to uint8 if needed
            if image_data.max() <= 1.0:
                image_data = (image_data * 255).astype(np.uint8)
            else:
                image_data = image_data.astype(np.uint8)
        
        # Convert from (C,H,W) to (H,W,C) if needed
        if len(image_data.shape) == 3 and image_data.shape[0] in [1, 3, 4]:
            image_data = np.transpose(image_data, (1, 2, 0))
    
    return image_data


def _parse_depth(depth_data):
    """
    Parse depth data to appropriate format.
    Depth is typically single channel, so we handle it separately.
    """
    if isinstance(depth_data, np.ndarray):
        # Ensure depth is in (H,W) or (H,W,1) format
        if len(depth_data.shape) == 3:
            if depth_data.shape[0] == 1:  # (1,H,W) -> (H,W)
                depth_data = depth_data.squeeze(0)
            elif depth_data.shape[2] == 1:  # Already (H,W,1)
                pass
            else:  # (C,H,W) -> (H,W,C) then take first channel
                depth_data = np.transpose(depth_data, (1, 2, 0))[:, :, 0]
        
        # Add channel dimension if needed (H,W) -> (H,W,1)
        if len(depth_data.shape) == 2:
            depth_data = np.expand_dims(depth_data, axis=2)
    
    return depth_data


@dataclasses.dataclass(frozen=True)
class UR10eInputs(transforms.DataTransformFn):
    """
    Input transform for UR10e robot with single camera providing RGB and depth.
    Maps UR10e environment data to model input format.
    """
    
    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        mask_padding = self.model_type == _model.ModelType.PI0

        # Concatenate joints and gripper into state vector
        # UR10e has 6 joint positions + 1 gripper state = 7 dimensions
        state = np.concatenate([data["joints"], data["gripper"]])
        # Pad to expected input dimensionality of the model
        state = transforms.pad_to_dim(state, self.action_dim)

        # Parse RGB and depth images from the single base camera
        base_rgb = _parse_image(data["base_rgb"])
        base_depth = _parse_depth(data["base_depth"])
        
        # Convert depth to 3-channel format to match RGB expectations
        # Repeat depth channel 3 times: (H,W,1) -> (H,W,3)
        base_depth_3ch = np.repeat(base_depth, 3, axis=2)

        # Create inputs dictionary
        # Since we only have one camera, we'll use it for base and left_wrist slots
        # and fill the right_wrist slot with zeros
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_rgb,
                "left_wrist_0_rgb": base_depth_3ch,  # Use depth as "wrist" input
                "right_wrist_0_rgb": np.zeros_like(base_rgb),  # Unused slot
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,  # We have depth data
                "right_wrist_0_rgb": np.False_ if mask_padding else np.True_,
            },
        }

        # Add actions if present in data (for training)
        if "actions" in data:
            # UR5 produces 7D actions (6 DoF + 1 gripper)
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions

        # Add language instruction if present
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class UR10eOutputs(transforms.DataTransformFn):
    """
    Output transform for UR10e robot.
    Maps model outputs back to robot action format.
    """

    def __call__(self, data: dict) -> dict:
        # UR10e has 7 action dimensions (6 DoF + gripper)
        # Return only the first 7 dimensions from model output
        return {"actions": np.asarray(data["actions"][:, :7])}


@dataclasses.dataclass(frozen=True)
class LeRobotUR10eDataConfig(DataConfigFactory):
    """
    Data configuration for UR10e dataset from LeRobot.
    Defines how to process raw UR10e data for training.
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Transform to repack/rename keys from LeRobot dataset format
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        # Map LeRobot dataset keys to our expected keys
                        "base_rgb": "image",           # Main RGB camera
                        "base_depth": "depth_image",   # Depth from same camera
                        "joints": "joints",            # Joint positions
                        "gripper": "gripper",          # Gripper state
                        "prompt": "prompt",            # Task instruction
                    }
                )
            ]
        )

        # Apply our custom input/output transforms
        data_transforms = _transforms.Group(
            inputs=[UR10eInputs(
                action_dim=model_config.action_dim, 
                model_type=model_config.model_type
            )],
            outputs=[UR10eOutputs()],
        )

        # Convert absolute actions to delta actions
        # Apply delta transformation to the first 6 dimensions (joints)
        # Keep gripper action (7th dimension) as absolute
        delta_action_mask = _transforms.make_bool_mask(6, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        # Model transforms (tokenization, etc.) - framework handles this
        model_transforms = ModelTransformFactory()(model_config)

        # Return complete data configuration
        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


# Training configuration for UR5 with single RGB-D camera
UR5_TRAIN_CONFIG = TrainConfig(
    name="pi0_ur5_rgbd",
    model=pi0.Pi0Config(),
    data=LeRobotUR5DataConfig(
        repo_id="your_username/ur5_rgbd_dataset",  # Replace with your dataset
        # Reload normalization stats from base model for better transfer learning
        assets=AssetsConfig(
            assets_dir="s3://openpi-assets/checkpoints/pi0_base/assets",
            asset_id="ur5e_rgbd",
        ),
        base_config=DataConfig(
            # Load task instructions from 'task' field in LeRobot dataset
            prompt_from_task=True,
        ),
    ),
    # Load pre-trained pi0 base model weights
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "s3://openpi-assets/checkpoints/pi0_base/params"
    ),
    num_train_steps=30_000,
)


# Alternative configuration for LoRA fine-tuning (more memory efficient)
UR5_LORA_TRAIN_CONFIG = TrainConfig(
    name="pi0_ur5_rgbd_lora",
    model=pi0.Pi0Config(
        # Enable LoRA for parameter-efficient fine-tuning
        use_lora=True,
        lora_rank=32,
        lora_alpha=64,
    ),
    data=LeRobotUR5DataConfig(
        repo_id="your_username/ur5_rgbd_dataset",
        assets=AssetsConfig(
            assets_dir="s3://openpi-assets/checkpoints/pi0_base/assets",
            asset_id="ur5e_rgbd",
        ),
        base_config=DataConfig(
            prompt_from_task=True,
        ),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "s3://openpi-assets/checkpoints/pi0_base/params"
    ),
    num_train_steps=15_000,  # Fewer steps needed for LoRA
)


# Example usage and data format expectations
def example_data_format():
    """
    Example of the expected data format for your UR10e dataset.
    This shows what your LeRobot dataset should contain.
    """
    return {
        "base_rgb": np.zeros((480, 640, 3), dtype=np.uint8),    # RGB image
        "base_depth": np.zeros((480, 640, 1), dtype=np.float32), # Depth image
        "joints": np.zeros(6, dtype=np.float32),                 # 6 joint positions
        "gripper": np.array([0.0], dtype=np.float32),           # Gripper state
        "actions": np.zeros(7, dtype=np.float32),               # 6 DoF + gripper
        "prompt": "Pick up the red block",                       # Task instruction
    }


if __name__ == "__main__":
    # Example of how to use the configuration
    print("UR10e Single Camera Configuration Ready!")
    print(f"Training config name: {UR10e_TRAIN_CONFIG.name}")
    
    # Show expected data format
    example_data = example_data_format()
    print("\nExpected data format:")
    for key, value in example_data.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
        else:
            print(f"  {key}: {type(value)} - '{value}'")