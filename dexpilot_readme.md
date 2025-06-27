# Generate the robot joint pose trajectory from our pre-recorded video

    python example/vector_retargeting/detect_from_video.py --robot-name dex3 --video-path example/vector_retargeting/data/human_hand_video.mp4 --retargeting-type dexpilot --hand-type right --output-path example/vector_retargeting/data/unitree_joints.pkl

# Utilize the pickle file to produce a video of the robot

    python example/vector_retargeting/render_robot_hand.py \
  --pickle-path example/vector_retargeting/data/unitree_joints.pkl \
  --output-video-path example/vector_retargeting/data/unitree.mp^C\
  --headless