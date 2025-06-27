from pathlib import Path
from typing import Optional, List, Union, Dict
import pickle

import sys
import cv2
import numpy as np
import mujoco
import mujoco.viewer
import tqdm
import tyro
import tempfile
import os

# Add the local dex-retargeting source to Python path to use local changes instead of pip-installed version
local_dex_retargeting_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(local_dex_retargeting_path))

from dex_retargeting.retargeting_config import RetargetingConfig

# Convert webp
# ffmpeg -i teaser.mp4 -vcodec libwebp -lossless 1 -loop 0 -preset default  -an -vsync 0 teaser.webp


def render_by_mujoco(
    meta_data: Dict,
    data: List[Union[List[float], np.ndarray]],
    output_video_path: Optional[str] = None,
    headless: Optional[bool] = False,
):
    """Render robot hand motion using MuJoCo"""
    
    # Config is loaded only to find the urdf path and robot name
    config_path = meta_data["config_path"]
    
    # Handle different config structures for loading
    with open(config_path, 'r') as f:
        import yaml
        yaml_config = yaml.safe_load(f)
    
    # Extract config based on structure
    if "retargeting" in yaml_config:
        config_dict = yaml_config["retargeting"]
    elif "left" in yaml_config:
        # For vector configs, use right hand by default
        config_dict = yaml_config.get("right", yaml_config["left"])
    else:
        config_dict = yaml_config
    
    config = RetargetingConfig.from_dict(config_dict)

    # Get URDF path from config
    urdf_path = Path(config.urdf_path)
    robot_name = ""
    
    if not urdf_path.is_absolute():
        # Determine robot type and use appropriate assets directory
        if "unitree" in str(urdf_path) or "dex3" in str(urdf_path):
            # Use the unitree_hand folder from the main assets directory
            assets_dir = Path(__file__).absolute().parent.parent.parent.parent / "assets" / "unitree_hand"
            robot_name = "unitree"
            # For unitree, just use the filename from the urdf_path
            urdf_path = assets_dir / urdf_path.name
        elif "allegro" in str(urdf_path):
            # Use the allegro_hand folder from dex-retargeting assets
            assets_dir = Path(__file__).absolute().parent.parent.parent / "assets" / "allegro_hand"
            robot_name = "allegro"
            urdf_path = assets_dir / urdf_path.name
        else:
            # Default to dex-retargeting assets
            assets_dir = Path(__file__).absolute().parent.parent.parent / "assets"
            urdf_path = assets_dir / urdf_path
    else:
        # Extract robot name from path for later use
        if "unitree" in str(urdf_path) or "dex3" in str(urdf_path):
            robot_name = "unitree"
        elif "allegro" in str(urdf_path):
            robot_name = "allegro"
    
    print(f"Loading URDF from: {urdf_path}")
    print(f"Robot type detected: {robot_name}")
    
    # Try to load the URDF with MuJoCo
    try:
        # First try to load URDF directly (MuJoCo 2.3+ supports URDF)
        model = mujoco.MjModel.from_xml_path(str(urdf_path))
        print(f"Successfully loaded URDF: {urdf_path}")
    except Exception as e:
        print(f"Failed to load URDF directly: {e}")
        
        # Try alternative URDF files in the same directory if the specified one fails
        urdf_dir = urdf_path.parent
        alternative_urdfs = []
        
        if robot_name == "unitree":
            # Look for unitree URDF alternatives
            alternative_urdfs = [
                urdf_dir / "unitree_dex3_right.urdf",
                urdf_dir / "unitree_dex3_left.urdf"
            ]
        elif robot_name == "allegro":
            # Look for allegro URDF alternatives  
            allegro_assets = Path(__file__).absolute().parent.parent.parent / "assets" / "allegro_hand"
            alternative_urdfs = list(allegro_assets.glob("*.urdf"))
        
        model = None
        for alt_urdf in alternative_urdfs:
            if alt_urdf.exists():
                try:
                    print(f"Trying alternative URDF: {alt_urdf}")
                    model = mujoco.MjModel.from_xml_path(str(alt_urdf))
                    print(f"Successfully loaded alternative URDF: {alt_urdf}")
                    break
                except Exception as alt_e:
                    print(f"Failed to load {alt_urdf}: {alt_e}")
                    continue
        
        if model is None:
            raise Exception(f"Could not load any URDF for {robot_name} robot. Please check URDF files in {urdf_dir}")

    # Create MuJoCo data
    data_mj = mujoco.MjData(model)

    # Get joint names and create mapping
    joint_names = []
    joint_ids = []
    for i in range(model.njnt):
        if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE:  # Skip free joints
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joint_name:
                joint_names.append(joint_name)
                joint_ids.append(i)

    # Create mapping between retargeting joint names and MuJoCo joint indices
    retargeting_joint_names = meta_data["joint_names"]
    retargeting_to_mujoco = []
    
    for ret_name in retargeting_joint_names:
        if ret_name in joint_names:
            idx = joint_names.index(ret_name)
            retargeting_to_mujoco.append(joint_ids[idx])
        else:
            print(f"Warning: Joint {ret_name} not found in MuJoCo model")
            retargeting_to_mujoco.append(-1)  # Invalid index

    retargeting_to_mujoco = np.array(retargeting_to_mujoco)
    
    print(f"Found {len(joint_names)} joints in MuJoCo model")
    print(f"Retargeting joint names: {retargeting_joint_names}")
    print(f"MuJoCo joint names: {joint_names}")

    # Set up rendering
    if not headless:
        # Launch MuJoCo viewer with performance optimizations
        with mujoco.viewer.launch_passive(model, data_mj) as viewer:
            # Set camera for better view
            viewer.cam.azimuth = 45
            viewer.cam.elevation = -20
            viewer.cam.distance = 0.8
            viewer.cam.lookat[:] = [0, 0, 0.1]
            
            # Video recorder setup if needed
            if output_video_path:
                Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)
                # Set up offline rendering for video
                renderer = mujoco.Renderer(model, height=480, width=640)
                writer = cv2.VideoWriter(
                    output_video_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    30.0,
                    (640, 480)
                )
                
                print(f"Recording video to: {output_video_path}")
                
                for qpos in tqdm.tqdm(data, desc="Rendering frames"):
                    # Set joint positions with better error handling
                    qpos_array = np.array(qpos)
                    for ret_idx, mj_joint_id in enumerate(retargeting_to_mujoco):
                        if mj_joint_id >= 0 and ret_idx < len(qpos_array):
                            try:
                                data_mj.qpos[mj_joint_id] = qpos_array[ret_idx]
                            except IndexError:
                                print(f"Warning: Cannot set joint {ret_idx} (MuJoCo ID {mj_joint_id})")
                    
                    # Forward kinematics
                    mujoco.mj_forward(model, data_mj)
                    
                    # Render frame
                    renderer.update_scene(data_mj)
                    rgb = renderer.render()
                    
                    # Convert RGB to BGR for OpenCV
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    writer.write(bgr)
                
                writer.release()
                print(f"Video saved to: {output_video_path}")
            else:
                # Interactive viewing
                print("Interactive mode: Use mouse to rotate view. Press Tab to show/hide help.")
                print("Playing back motion data...")
                
                for qpos in tqdm.tqdm(data, desc="Playing motion"):
                    # Set joint positions with better error handling
                    qpos_array = np.array(qpos)
                    for ret_idx, mj_joint_id in enumerate(retargeting_to_mujoco):
                        if mj_joint_id >= 0 and ret_idx < len(qpos_array):
                            try:
                                data_mj.qpos[mj_joint_id] = qpos_array[ret_idx]
                            except IndexError:
                                print(f"Warning: Cannot set joint {ret_idx} (MuJoCo ID {mj_joint_id})")
                    
                    # Forward kinematics
                    mujoco.mj_forward(model, data_mj)
                    
                    # Update viewer
                    viewer.sync()
                    
                    # Control playback speed
                    import time
                    time.sleep(1.0 / 30.0)  # 30 FPS playback
    else:
        # Headless rendering for video output only
        if output_video_path:
            Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)
            renderer = mujoco.Renderer(model, height=480, width=640)
            writer = cv2.VideoWriter(
                output_video_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                30.0,
                (640, 480)
            )
            
            print(f"Headless rendering to: {output_video_path}")
            
            for qpos in tqdm.tqdm(data, desc="Rendering frames"):
                # Set joint positions with better error handling
                qpos_array = np.array(qpos)
                for ret_idx, mj_joint_id in enumerate(retargeting_to_mujoco):
                    if mj_joint_id >= 0 and ret_idx < len(qpos_array):
                        try:
                            data_mj.qpos[mj_joint_id] = qpos_array[ret_idx]
                        except IndexError:
                            print(f"Warning: Cannot set joint {ret_idx} (MuJoCo ID {mj_joint_id})")
                
                # Forward kinematics
                mujoco.mj_forward(model, data_mj)
                
                # Render frame
                renderer.update_scene(data_mj)
                rgb = renderer.render()
                
                # Convert RGB to BGR for OpenCV
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                writer.write(bgr)
            
            writer.release()
            print(f"Video saved to: {output_video_path}")
        else:
            print("Headless mode requires output_video_path to be specified")


def main(
    pickle_path: str,
    output_video_path: Optional[str] = None,
    headless: bool = False,
):
    """
    Loads the preserved robot pose data and renders it either on screen or as an mp4 video.

    Args:
        pickle_path: Path to the .pickle file, created by `detect_from_video.py`.
        output_video_path: Path where the output video in .mp4 format would be saved.
            By default, it is set to None, implying no video will be saved.
        headless: Set to visualize the rendering on the screen by opening the viewer window.
    """
    robot_dir = (
        Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    )
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))

    pickle_data = np.load(pickle_path, allow_pickle=True)
    meta_data, data = pickle_data["meta_data"], pickle_data["data"]

    render_by_mujoco(meta_data, data, output_video_path, headless)


if __name__ == "__main__":
    tyro.cli(main)
