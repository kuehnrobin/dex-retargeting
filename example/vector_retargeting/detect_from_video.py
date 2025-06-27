import pickle
from pathlib import Path
import sys

import cv2
import tqdm
import tyro
import yaml

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
from dex_retargeting.seq_retarget import SeqRetargeting
from single_hand_detector import SingleHandDetector


def retarget_video(
    retargeting: SeqRetargeting, video_path: str, output_path: str, config_path: str
):
    cap = cv2.VideoCapture(video_path)

    data = []

    if not cap.isOpened():
        print("Error: Could not open video file.")
    else:
        detector = SingleHandDetector(hand_type="Right", selfie=False)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        with tqdm.tqdm(total=length) as pbar:
            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    break

                rgb = frame[..., ::-1]
                num_box, joint_pos, keypoint_2d, mediapipe_wrist_rot = detector.detect(
                    rgb
                )
                if num_box == 0:
                    continue

                retargeting_type = retargeting.optimizer.retargeting_type
                indices = retargeting.optimizer.target_link_human_indices
                if retargeting_type == "POSITION":
                    indices = indices
                    ref_value = joint_pos[indices, :]
                else:
                    origin_indices = indices[0, :]
                    task_indices = indices[1, :]
                    ref_value = (
                        joint_pos[task_indices, :] - joint_pos[origin_indices, :]
                    )
                qpos = retargeting.retarget(ref_value)
                data.append(qpos)
                pbar.update(1)

        meta_data = dict(
            config_path=config_path,
            dof=len(retargeting.optimizer.robot.dof_joint_names),
            joint_names=retargeting.optimizer.robot.dof_joint_names,
        )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as f:
            pickle.dump(dict(data=data, meta_data=meta_data), f)

        retargeting.verbose()
        cap.release()
        cv2.destroyAllWindows()


def main(
    robot_name: RobotName,
    video_path: str,
    output_path: str,
    retargeting_type: RetargetingType,
    hand_type: HandType,
):
    """
    Detects the human hand pose from a video and translates the human pose trajectory into a robot pose trajectory.

    Args:
        robot_name: The identifier for the robot. This should match one of the default supported robots.
        video_path: The file path for the input video in .mp4 format.
        output_path: The file path for the output data in .pickle format.
        retargeting_type: The type of retargeting, each type corresponds to a different retargeting algorithm.
        hand_type: Specifies which hand is being tracked, either left or right.
            Please note that retargeting is specific to the same type of hand: a left robot hand can only be retargeted
            to another left robot hand, and the same applies for the right hand.
    """

    # Handle different asset directories based on robot type
    if robot_name == RobotName.dex3:
        # For Unitree Dex3, use our custom assets directory
        robot_dir = Path(__file__).absolute().parent.parent.parent.parent / "assets"
        # Construct config path manually for Unitree
        hand_str = hand_type.name.lower()  # 'left' or 'right'
        if retargeting_type == RetargetingType.dexpilot:
            config_path = robot_dir / "unitree_hand" / f"unitree_dex3_{hand_str}_dexpilot.yml"
        else:
            config_path = robot_dir / "unitree_hand" / "unitree_dex3.yml"
    else:
        # For other robots (Allegro, etc.), use dex-retargeting assets directory
        robot_dir = Path(__file__).absolute().parent.parent.parent / "assets"
        config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
    
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    
    # Load the retargeting configuration
    if robot_name == RobotName.dex3:
        # For Unitree, we need to handle the different config structures
        import yaml
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        # Extract the appropriate config section
        if retargeting_type == RetargetingType.vector:
            # Vector config has left/right sections
            hand_key = hand_type.name.lower()
            if hand_key in yaml_config:
                config_dict = yaml_config[hand_key]
            else:
                raise ValueError(f"Hand type {hand_key} not found in config")
        else:
            # DexPilot config structure
            if "retargeting" in yaml_config:
                config_dict = yaml_config["retargeting"]
            else:
                config_dict = yaml_config
        
        retargeting = RetargetingConfig.from_dict(config_dict).build()
    else:
        # Use standard loading for other robots
        retargeting = RetargetingConfig.load_from_file(config_path).build()
    
    retarget_video(retargeting, video_path, output_path, str(config_path))


if __name__ == "__main__":
    tyro.cli(main)
