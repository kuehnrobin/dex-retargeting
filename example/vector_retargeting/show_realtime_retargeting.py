import multiprocessing
import time
from pathlib import Path
from queue import Empty
from typing import Optional

import cv2
import numpy as np
import mujoco
import mujoco.viewer
import tyro
from loguru import logger

import sys
from pathlib import Path

# Add the local dex-retargeting source to Python path to use local changes instead of pip-installed version
local_dex_retargeting_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(local_dex_retargeting_path))

from dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from single_hand_detector import SingleHandDetector


def start_retargeting(queue: multiprocessing.Queue, robot_dir: str, config_path: str):
    # Set URDF directory to the project assets folder for Unitree hand
    unitree_assets_dir = Path(__file__).absolute().parent.parent.parent.parent / "assets"
    RetargetingConfig.set_default_urdf_dir(str(unitree_assets_dir))
    logger.info(f"Start retargeting with config {config_path}")
    retargeting = RetargetingConfig.load_from_file(config_path).build()

    hand_type = "Right" if "right" in config_path.lower() else "Left"
    detector = SingleHandDetector(hand_type=hand_type, selfie=False)

    # Load retargeting config
    config = RetargetingConfig.load_from_file(config_path)

    # Convert URDF to XML for MuJoCo
    urdf_path = Path(config.urdf_path)
    if not urdf_path.is_absolute():
        urdf_path = unitree_assets_dir / urdf_path
    
    # For MuJoCo, we need to load the URDF directly or convert it to XML
    # Let's try to find an XML version first, or use the URDF directly if MuJoCo supports it
    xml_path = str(urdf_path).replace('.urdf', '.xml')
    
    try:
        # Try to load XML if it exists
        if Path(xml_path).exists():
            model = mujoco.MjModel.from_xml_path(xml_path)
        else:
            # Try to load URDF directly (MuJoCo 2.3+ supports URDF)
            model = mujoco.MjModel.from_xml_path(str(urdf_path))
    except Exception as e:
        logger.error(f"Failed to load model from {urdf_path}: {e}")
        logger.info("Trying to create a simple model for visualization...")
        
        # Create a simple MuJoCo XML for the hand if URDF loading fails
        simple_xml = f"""
        <mujoco>
            <worldbody>
                <body name="hand_base" pos="0 0 0">
                    <geom name="palm" type="box" size="0.05 0.08 0.02" rgba="0.8 0.8 0.8 1"/>
                    <body name="thumb" pos="0.03 0.05 0">
                        <geom name="thumb_geom" type="capsule" size="0.01 0.03" rgba="1 0.5 0.5 1"/>
                        <joint name="right_hand_thumb_0_joint" type="hinge" axis="0 0 1" range="-1.57 1.57"/>
                        <joint name="right_hand_thumb_1_joint" type="hinge" axis="1 0 0" range="-1.57 1.57"/>
                        <joint name="right_hand_thumb_2_joint" type="hinge" axis="1 0 0" range="0 1.57"/>
                    </body>
                    <body name="index" pos="0.03 -0.02 0">
                        <geom name="index_geom" type="capsule" size="0.01 0.04" rgba="0.5 1 0.5 1"/>
                        <joint name="right_hand_index_0_joint" type="hinge" axis="0 0 1" range="-1.57 0"/>
                        <joint name="right_hand_index_1_joint" type="hinge" axis="1 0 0" range="-1.57 0"/>
                    </body>
                    <body name="middle" pos="0.01 -0.02 0">
                        <geom name="middle_geom" type="capsule" size="0.01 0.04" rgba="0.5 0.5 1 1"/>
                        <joint name="right_hand_middle_0_joint" type="hinge" axis="0 0 1" range="-1.57 0"/>
                        <joint name="right_hand_middle_1_joint" type="hinge" axis="1 0 0" range="-1.57 0"/>
                    </body>
                </body>
            </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(simple_xml)

    # Create MuJoCo data
    data = mujoco.MjData(model)

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
    retargeting_joint_names = retargeting.joint_names
    retargeting_to_mujoco = []
    
    for ret_name in retargeting_joint_names:
        if ret_name in joint_names:
            mj_idx = joint_names.index(ret_name)
            retargeting_to_mujoco.append(mj_idx)
        else:
            logger.warning(f"Joint {ret_name} not found in MuJoCo model")
            retargeting_to_mujoco.append(0)  # Default to first joint

    retargeting_to_mujoco = np.array(retargeting_to_mujoco)

    # Launch MuJoCo viewer with performance optimizations
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Set camera for better view
        viewer.cam.distance = 0.5
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -30
        
        # Performance optimization: Skip frames and reduce processing frequency
        frame_skip = 2  # Process every 2nd frame
        frame_count = 0
        last_retarget_time = time.time()
        retarget_frequency = 30  # Hz - reduce retargeting frequency for better performance
        
        # Pre-allocate arrays for better performance
        qpos_cache = np.zeros(len(retargeting_to_mujoco))

        while True:
            try:
                bgr = queue.get(timeout=5)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            except Empty:
                logger.error(
                    "Fail to fetch image from camera in 5 secs. Please check your web camera device."
                )
                return

            frame_count += 1
            
            # Skip frames for better performance
            if frame_count % frame_skip != 0:
                continue

            _, joint_pos, keypoint_2d, _ = detector.detect(rgb)
            
            # Only draw skeleton every few frames to save CPU
            if frame_count % (frame_skip * 2) == 0:
                bgr = detector.draw_skeleton_on_image(bgr, keypoint_2d, style="default")
                cv2.imshow("realtime_retargeting_demo", bgr)
                
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # Limit retargeting frequency for performance
            current_time = time.time()
            if current_time - last_retarget_time < (1.0 / retarget_frequency):
                continue
            last_retarget_time = current_time

            if joint_pos is None:
                if frame_count % 30 == 0:  # Only log warning occasionally
                    logger.warning(f"{hand_type} hand is not detected.")
            else:
                retargeting_type = retargeting.optimizer.retargeting_type
                indices = retargeting.optimizer.target_link_human_indices
                
                # Process retargeting based on type
                if retargeting_type == "POSITION":
                    ref_value = joint_pos[indices, :]
                else:
                    # Vector or DexPilot retargeting
                    origin_indices = indices[0, :]
                    task_indices = indices[1, :]
                    ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
                
                # Get joint angles from retargeting
                qpos = retargeting.retarget(ref_value)
                
                # Apply joint angles to MuJoCo model (only for available joints)
                # Use pre-allocated cache for better performance
                qpos_cache[:] = 0  # Reset cache
                for i, (ret_idx, mj_idx) in enumerate(zip(range(len(qpos)), retargeting_to_mujoco)):
                    if ret_idx < len(qpos) and mj_idx < len(data.qpos):
                        qpos_cache[i] = qpos[ret_idx]
                        data.qpos[mj_idx] = qpos[ret_idx]

                # Step simulation and sync viewer
                mujoco.mj_step(model, data)
                viewer.sync()


def produce_frame(queue: multiprocessing.Queue, camera_path: Optional[str] = None):
    if camera_path is None:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(camera_path)
    
    # Optimize camera settings for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Reduce resolution for better FPS
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)            # Set consistent FPS
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)      # Minimize buffer to reduce latency

    frame_count = 0
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue
            
        frame_count += 1
        
        # Skip every other frame to reduce processing load
        if frame_count % 2 == 0:
            continue
            
        # Only put frame in queue if it's not full (non-blocking)
        try:
            queue.put_nowait(image)
        except:
            # Queue is full, skip this frame
            pass
            
        time.sleep(1 / 60.0)  # Reduce camera polling frequency


def main(
    robot_name: RobotName = RobotName.dex3,  # Default to Unitree Dex3 hand
    retargeting_type: RetargetingType = RetargetingType.vector,  # Default to vector retargeting
    hand_type: HandType = HandType.right,  # Default to right hand
    camera_path: Optional[str] = None,
):
    """
    Detects the human hand pose from a video and translates the human pose trajectory into a robot pose trajectory.
    This example demonstrates retargeting with the Unitree Dex3 hand using MuJoCo for visualization.

    PERFORMANCE TIPS:
    - Use lower camera resolution (640x480) for better FPS
    - Reduce frame skip and retarget frequency if too slow
    - Close other applications to free up CPU/GPU resources
    - Use vector retargeting instead of dexpilot for better performance

    Args:
        robot_name: The identifier for the robot. Defaults to Unitree Dex3 hand.
        retargeting_type: The type of retargeting, each type corresponds to a different retargeting algorithm.
        hand_type: Specifies which hand is being tracked, either left or right.
            Please note that retargeting is specific to the same type of hand: a left robot hand can only be retargeted
            to another left robot hand, and the same applies for the right hand.
        camera_path: the device path to feed to opencv to open the web camera. It will use 0 by default.
    """
    # For Unitree hand, use the custom config path structure
    unitree_assets_dir = Path(__file__).absolute().parent.parent.parent.parent / "assets"
    
    if robot_name == RobotName.dex3:
        # Use our custom Unitree configs
        hand_type_str = hand_type.name  # 'left' or 'right'
        if retargeting_type == RetargetingType.dexpilot:
            config_path = unitree_assets_dir / "unitree_hand" / f"unitree_dex3_{hand_type_str}_dexpilot.yml"
        else:
            config_path = unitree_assets_dir / "unitree_hand" / "unitree_dex3.yml"
    else:
        # Use default config path for other robots
        config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
    
    robot_dir = unitree_assets_dir

    # Use smaller queue for lower latency and better performance
    queue = multiprocessing.Queue(maxsize=2)  # Reduced from 1000 to 2 for better responsiveness
    producer_process = multiprocessing.Process(
        target=produce_frame, args=(queue, camera_path)
    )
    consumer_process = multiprocessing.Process(
        target=start_retargeting, args=(queue, str(robot_dir), str(config_path))
    )

    producer_process.start()
    consumer_process.start()

    producer_process.join()
    consumer_process.join()
    time.sleep(5)

    print("done")


if __name__ == "__main__":
    tyro.cli(main)
