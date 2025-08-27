# vgr_sdk

Robot-side SDK for **vision-guided manipulation**.  
Subscribes to your Vision SDK’s UDP JSON, computes world poses with a simple px→mm mapping (fixed Z),
plans approach/grasp/place, and drives your robot & gripper. Version 1 focuses on **calibration-less**
setup (fixed plane, global scale), fast integration, and **recordable poses** from ROS.

## Quick install

```bash
python -m pip install -U pip
pip install -e .
# ROS Python deps are installed via apt from your ROS distro
