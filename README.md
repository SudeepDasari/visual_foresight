# Visual Foresight
Code for reproducing experiments in [Visual Foresight: Model-Based Deep Reinforcement Learning for Vision-Based Robotic Control](https://sites.google.com/view/visualforesight)

On a high level, Visual Model Predictive Control (visual-MPC) leverages an action-conditioned video prediction modelto enable robots to perform various tasks with only raw-pixel input. This codebase provides an implmentation of the core algorithm and planning cost, as well as the data collection and experiment framework. Additionally, we provide Dockerfiles for our simulator environments, and instructions to reproduce our experiments on your own Sawyer robot.

Crucially, this codebase does NOT implement video prediction model training, or meta-classifier model training. If you're only interested in training models, please refer to [Stochastic Adversarial Video Prediction](https://github.com/alexlee-gk/video_prediction) or [Few-Shot Goal Inference for Visuomotor Learning and Planning](https://github.com/anxie/meta_classifier).

# Installation
## General Dependencies
Since this project is deployed in sim (w/ mujoco_py 1.5) and the robot (ROS kinetic), all code is written to be compatible with Python 2.7 and Python 3.5. 
<>

## Sim
<>

## Robot
### Hardware Setup
All experiments are conducted on a [Sawyer robot](https://www.rethinkrobotics.com/sawyer/) with an attached [WSG-50 gripper](https://www.weiss-robotics.com/en/produkte/gripping-systems/performance-line-en/wsg-50-en/). The robot is filmed from two orthogonal viewing angles using [consumer webcams](https://www.amazon.com/Logitech-Widescreen-Calling-Recording-Desktop/dp/B006JH8T3S). Refer to the paper for further details.

### Software Setup
Robot code heavily uses ROS. Assuming you use our same hardware, you will need to install the following:
* [ROS and Intera 5.2](http://sdk.rethinkrobotics.com/intera/Workstation_Setup)
* [wsg 50 ROS node](http://wiki.ros.org/wsg50)
* Any ROS node to access webcams. We use a [modified version of video_stream_opencv](https://github.com/SudeepDasari/video_stream_opencv)

Once you've installed the dependencies:
1. Clone our repository into your ROS workspace's src folder. Then run `catkin_make` to rebuild your workspace.
2. Clone and install the [video_prediction](INSERT LINK) code-base.
3. (optional) Add the following lines right after the EOF in your `intera.sh`. While hacky, modifying your Python path is a quick and dirty way to get this code running while allowing you to modify it.
```
export PYTHONPATH='$PYTHONPATH:<PATH TO workspace src>/visual_foresight:<PATH TO video_prediction-1>'
```
# Experiment Reproduction
## Data Collection
<>
## Running Benchmarks
<>
# Pretrained Models
<>
# Citation
If you find this useful, consider citing:
```
BIBTEX GOES HERE
```
