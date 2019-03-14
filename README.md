# Visual Foresight
Code for reproducing experiments in [Visual Foresight: Model-Based Deep Reinforcement Learning for Vision-Based Robotic Control](https://sites.google.com/view/visualforesight)

On a high level, Visual Model Predictive Control (visual-MPC) leverages an action-conditioned video prediction model (trained from unsupervised interaction) to enable robots to perform various tasks with only raw-pixel input. This codebase provides an implmentation of: unsupervised data collection, our benchmarking framework, the various planning costs, and - of course - the visual-MPC controller! Additionally, we provide: instructions to reproduce our experiments, Dockerfiles for our simulator environments, and documentation on our Sawyer robot setup.

Crucially, this codebase does NOT implement video prediction model training, or meta-classifier model training. If you're only interested in training models, please refer to [Stochastic Adversarial Video Prediction](https://github.com/alexlee-gk/video_prediction) and/or [Few-Shot Goal Inference for Visuomotor Learning and Planning](https://github.com/anxie/meta_classifier).

# Installation
## General Dependencies
Since this project is deployed in sim and on a robot, all code is written to be compatible with Python 2.7 and Python 3.5. 

## Sim
### Manual Installation
Our simulator requires **Python 3.5.2** and [MuJoCo 1.5](https://www.roboti.us/index.html) to run successfully. We strongly recommend using a virtual environment (such as Anaconda) for this project. After you setup Python and MuJoCo, installation directions are as follows:
```
# install video prediction code
git clone https://github.com/febert/video_prediction-1.git && cd video_prediction-1 && git checkout dev && python setup.py develop && cd ..
# install meta-classifier code
git clone https://github.com/anxie/meta_classifier.git
#install visual-MPC
git clone https://github.com/SudeepDasari/visual_foresight.git && cd visual_foresight
pip install -r requirements.txt
python setup.py develop
```
### Docker Installation
Docker allows a cleaner way to get started with our code. Since we heavily use the GPU, you will have to install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) and all related dependencies. After that run:
```
git clone https://github.com/SudeepDasari/visual_foresight.git && cd docker && cp ~/.mujoco/mjkey.txt ./
nvidia-docker build -t foresight/sim:latest .
```
Now to create a new bash in this environment run: `nvidia-docker run -it foresight/sim: bash`

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
2. Clone and install the [video_prediction](https://github.com/febert/visual_mpc/tree/dev) code-base. 
3. Clone and install the [meta-classifier](https://github.com/anxie/meta_classifier) code-base
4. Remember to install our python packages by running `sudo python setup.py develop` in EVERY project workspace
5. Start up required ROS nodes:
```
# in a new intera terminal
roslaunch foresight_rospkg start_cameras.launch   # run cameras

# in a new intera terminal
roslaunch foresight_rospkg start_gripper.launch   # start gripper node

# in a new intera terminal
roscd foresight_rospkg/launch
rosrun foresight_rospkg send_urdf_fragment.py     # (optional) stop after Sawyer recognizes the gripper
./start_impedance 
```

# Experiment Reproduction
In sim, data collection and benchmarks are started by running `python visual_mpc/sim/run.py`. The correct configuration file must be supplied, for each experiment/data collection run. Similarly, `rosrun foresight_rospkg run_robot.py` is the primary entry point for the robot experiments/data-collection.

## Data Collection
By default data is saved in the same directory as the corresponding python config file. Rollouts are saved as a series of pickled dictionaries and JPEG images, or as compressed TFRecords. 
### Robot
Use `run_robot` to start random data collection on the Sawyer.
* For hard object collection: `rosrun foresight_rospkg run_robot.py <robot name/id> data_collection/sawyer/hard_object_data/hparams.py -r`
* For deformable object collection: `rosrun foresight_rospkg run_robot.py <robot name/id> data_collection/sawyer/towel_data/hparams.py -r`
### Sim
Use `visual_mpc/sim/run.py` to start random data collection in our custom MuJoCo cartgripper environment
* To collect data with l-block objects and autograsp (x, y, z, wrist rotation, grasp reflex) action space run: `python visual_mpc/sim/run.py data_collection/sim/grasp_reflex_lblocks/hparams.py --nworkers <num_threads>`
### Convert to TFRecords
While the raw (pkl/jpeg file) data format is convenient to work with, it is far less efficient for model training. Thus, we offer a utility in `visual_mpc/utils/file_2_record.py` which converts data from our raw format to compressed TFRecords.

## Running Benchmarks
Again pass in the python config file to the corresponding entry point. This time add a `--benchmark` flag!

### Robot
* For Registration Experiments: `rosrun foresight_rospkg run_robot.py <robot name/id> experiments/sawyer/registration_experiments/hparams.py --benchmark`
* For Mixed Object Experiments (one model which handles both deformable and rigid objects)
  - Rigid: `rosrun foresight_rospkg run_robot.py <robot name/id> experiments/sawyer/mixed_objects/hparams_deformable_objects.py --benchmark`
  - Deformable: `rosrun foresight_rospkg run_robot.py <robot name/id> experiments/sawyer/mixed_objects/hparams_hardobjects.py --benchmark`
* **Meta-classifier experiments are coming soon**
### Sim
**Coming soon!**
# Pretrained Models
**Coming soon**
# Citation
If you find this useful, consider citing:
```
@article{visualforesight,
  title={Visual Foresight: Model-Based Deep Reinforcement Learning for Vision-Based Robotic Control},
  author={Ebert, Frederik and Finn, Chelsea and Dasari, Sudeep and Xie, Annie and Lee, Alex and Levine, Sergey},
  journal={arXiv preprint arXiv:1812.00568},
  year={2018}
}
```
