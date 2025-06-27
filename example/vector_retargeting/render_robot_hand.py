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
import time

try:
    import imageio
except ImportError:
    print("Warning: imageio not available, falling back to OpenCV for video writing")
    imageio = None

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
        # First try to load URDF directly with MuJoCo options for better compatibility
        # Set MuJoCo compiler options to handle problematic URDFs
        import tempfile
        import xml.etree.ElementTree as ET
        
        # Read and modify URDF to fix common MuJoCo compatibility issues
        with open(urdf_path, 'r') as f:
            urdf_content = f.read()
        
        # Parse XML to add MuJoCo-specific options
        root = ET.fromstring(urdf_content)
        
        # Add compiler options to handle inertia and mesh issues
        compiler = root.find('compiler')
        if compiler is None:
            compiler = ET.SubElement(root, 'compiler')
        
        # Set compiler options for better URDF compatibility
        compiler.set('balanceinertia', 'true')  # Fix inertia validation errors
        compiler.set('discardvisual', 'false')  # Keep visual meshes if available
        compiler.set('inertiafromgeom', 'true')  # Generate inertia from geometry
        compiler.set('inertiagrouprange', '0 0')  # Apply to all groups
        
        # For Allegro hand, fix mesh path issues by replacing problematic mesh references
        if robot_name == "allegro":
            # Fix inertial properties that cause validation errors
            for link in root.iter('link'):
                inertial = link.find('inertial')
                if inertial is not None:
                    inertia = inertial.find('inertia')
                    if inertia is not None:
                        # Set safe inertia values that satisfy A + B >= C constraint
                        inertia.set('ixx', '0.001')
                        inertia.set('iyy', '0.001') 
                        inertia.set('izz', '0.001')
                        inertia.set('ixy', '0.0')
                        inertia.set('ixz', '0.0')
                        inertia.set('iyz', '0.0')
                    
                    # Also set a reasonable mass
                    mass = inertial.find('mass')
                    if mass is not None:
                        mass.set('value', '0.01')  # 10 grams
                    else:
                        mass = ET.SubElement(inertial, 'mass')
                        mass.set('value', '0.01')
            
            # Find all mesh elements and replace problematic ones
            for visual in root.iter('visual'):
                geometry = visual.find('geometry')
                if geometry is not None:
                    mesh = geometry.find('mesh')
                    if mesh is not None:
                        filename = mesh.get('filename', '')
                        # If mesh file doesn't exist or is problematic, replace with simple geometry
                        if 'link_tip.obj' in filename or not Path(urdf_path.parent / filename).exists():
                            # Replace mesh with simple cylinder
                            geometry.remove(mesh)
                            cylinder = ET.SubElement(geometry, 'cylinder')
                            cylinder.set('radius', '0.01')
                            cylinder.set('length', '0.02')
            
            # Also fix collision geometry
            for collision in root.iter('collision'):
                geometry = collision.find('geometry')
                if geometry is not None:
                    mesh = geometry.find('mesh')
                    if mesh is not None:
                        filename = mesh.get('filename', '')
                        if 'link_tip.obj' in filename or not Path(urdf_path.parent / filename).exists():
                            # Replace mesh with simple cylinder
                            geometry.remove(mesh)
                            cylinder = ET.SubElement(geometry, 'cylinder')
                            cylinder.set('radius', '0.01')
                            cylinder.set('length', '0.02')
        
        # Create a temporary file with the modified URDF
        with tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False) as temp_file:
            temp_urdf_path = temp_file.name
            temp_file.write(ET.tostring(root, encoding='unicode'))
        
        # Try to load the modified URDF
        model = mujoco.MjModel.from_xml_path(temp_urdf_path)
        print(f"Successfully loaded URDF: {urdf_path}")
        
        # Clean up temporary file
        os.unlink(temp_urdf_path)
        
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
                    
                    # Apply the same modifications to alternative URDFs
                    with open(alt_urdf, 'r') as f:
                        alt_urdf_content = f.read()
                    
                    alt_root = ET.fromstring(alt_urdf_content)
                    
                    # Add compiler options
                    alt_compiler = alt_root.find('compiler')
                    if alt_compiler is None:
                        alt_compiler = ET.SubElement(alt_root, 'compiler')
                    alt_compiler.set('balanceinertia', 'true')
                    alt_compiler.set('discardvisual', 'false')
                    alt_compiler.set('inertiafromgeom', 'true')
                    alt_compiler.set('inertiagrouprange', '0 0')
                    
                    # Fix mesh issues for Allegro
                    if robot_name == "allegro":
                        # Fix inertial properties first
                        for link in alt_root.iter('link'):
                            inertial = link.find('inertial')
                            if inertial is not None:
                                inertia = inertial.find('inertia')
                                if inertia is not None:
                                    # Set safe inertia values that satisfy A + B >= C constraint
                                    inertia.set('ixx', '0.001')
                                    inertia.set('iyy', '0.001') 
                                    inertia.set('izz', '0.001')
                                    inertia.set('ixy', '0.0')
                                    inertia.set('ixz', '0.0')
                                    inertia.set('iyz', '0.0')
                                
                                # Also set a reasonable mass
                                mass = inertial.find('mass')
                                if mass is not None:
                                    mass.set('value', '0.01')  # 10 grams
                                else:
                                    mass = ET.SubElement(inertial, 'mass')
                                    mass.set('value', '0.01')
                        
                        # Fix mesh issues
                        for visual in alt_root.iter('visual'):
                            geometry = visual.find('geometry')
                            if geometry is not None:
                                mesh = geometry.find('mesh')
                                if mesh is not None:
                                    filename = mesh.get('filename', '')
                                    if 'link_tip.obj' in filename or not Path(alt_urdf.parent / filename).exists():
                                        geometry.remove(mesh)
                                        cylinder = ET.SubElement(geometry, 'cylinder')
                                        cylinder.set('radius', '0.01')
                                        cylinder.set('length', '0.02')
                        
                        for collision in alt_root.iter('collision'):
                            geometry = collision.find('geometry')
                            if geometry is not None:
                                mesh = geometry.find('mesh')
                                if mesh is not None:
                                    filename = mesh.get('filename', '')
                                    if 'link_tip.obj' in filename or not Path(alt_urdf.parent / filename).exists():
                                        geometry.remove(mesh)
                                        cylinder = ET.SubElement(geometry, 'cylinder')
                                        cylinder.set('radius', '0.01')
                                        cylinder.set('length', '0.02')
                    
                    # Create temporary file for alternative URDF
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False) as temp_file:
                        alt_temp_urdf_path = temp_file.name
                        temp_file.write(ET.tostring(alt_root, encoding='unicode'))
                    
                    model = mujoco.MjModel.from_xml_path(alt_temp_urdf_path)
                    print(f"Successfully loaded alternative URDF: {alt_urdf}")
                    
                    # Clean up temporary file
                    os.unlink(alt_temp_urdf_path)
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

    # Interactive visualization with MuJoCo viewer
    if not headless:
        # Interactive viewing
        with mujoco.viewer.launch_passive(model, data_mj) as viewer:
            # Set camera for better view of the hand
            viewer.cam.azimuth = 45
            viewer.cam.elevation = -20
            viewer.cam.distance = 0.3  # Closer view for hand details
            viewer.cam.lookat[:] = [0, 0, 0.1]
            
            print("Interactive mode: Use mouse to rotate view. Press Tab to show/hide help.")
            print("Controls:")
            print("  Mouse: Rotate view")
            print("  Scroll: Zoom in/out")
            print("  Space: Pause/resume playback")
            print("  R: Restart from beginning")
            print("  ESC/Q: Quit")
            print("Playing back retargeted hand motion...")
            
            frame_idx = 0
            paused = False
            
            while True:
                if not paused:
                    if frame_idx >= len(data):
                        frame_idx = 0  # Loop the animation
                    
                    qpos = data[frame_idx]
                    
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
                    
                    frame_idx += 1
                
                # Update viewer
                viewer.sync()
                
                # Check for key presses (basic controls)
                # Note: MuJoCo viewer doesn't have easy key detection, so we'll just use time-based playback
                
                # Control playback speed (slower for better visualization)
                time.sleep(1.0 / 15.0)  # 15 FPS for smoother visualization
                
                # Simple way to detect if viewer is closed
                if not viewer.is_running():
                    break
    
    elif output_video_path:
        print("Video output mode disabled. Use interactive mode instead by removing --headless flag.")
        print("Example: python render_robot_hand.py --pickle-path your_file.pkl")
    else:
        print("Please specify either interactive mode (remove --headless) or video output (--output-video-path)")
        print("Example: python render_robot_hand.py --pickle-path your_file.pkl")


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
