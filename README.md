# Static vs dynamic object segmentation on LiDAR data for USV applications, a machine learning approach
This repo contains the code for the segmentation of pointcloud data, captured by an unmanned surface vessel. The pointcloud is segmented into static and dynamic points. This information can be used for path planning and for the cleaning of the static occupancy grid map. The method uses a spherical projection to convert the 3D pointclouds to 2D 360 degree images. Combined with temporal information from previous pointclouds the projected pointclouds are fed through a convolutional neural network. The output is a binary mask of the dynamic points. An example of the output is shown below.

<img src="pics/MLSegResults2.gif" width="800">

### Table of Contents
1. [Introduction]
2. [Workflow]
3. [Installation]
4. [How to use]
5. [training on pc]
6. [inferring on Xavier AGX]


## Workflow
The workflow is as follows:
On a PC
- Pointcloud data is gathered through simulation in Copeliasim
- All points are labeled as static or dynamic
- The labeled data is restructered into Kitti format
- Residual images are generated for every pointcloud
- The neural network model is trained on the labeled data
- The model is tested on other data
- output can be visualized
- The model is converted to ONNX format

On Xavier
- An engine is generated from the ONNX file for the Xavier
- Run inference on precorded rosbags or in real-time
- Output is stored in label files


## Installation

### Installing on pc
The code is tested on Ubuntu 18.04 with cuda 10.2. You may encounter some missing packages when running the code. Try to install these manually.

-Clone git repo
-install cuda 10.2

For inferring using Ros:
-install ros melodic http://wiki.ros.org/melodic/Installation/Ubuntu
-Setup catkin workspace http://wiki.ros.org/catkin/Tutorials/create_a_workspace
-Create ROS package "ml_segmentation"
-Copy ml_segmentation folder to gitlab package folder
-```catkin_make```
-```. ~/catkin_ws/devel/setup.bash```
To use python3 with ROS Melodic:
-```sudo apt-get install python3-pip python3-yaml```
-```sudo pip3 install rospkg catkin_pkg```
-```pip3 install numpy```
-```pip3 install torch```
-```pip3 install matplotlib


For training and inferring without ros:
Then create the anaconda environment with: 
```conda env create -f ml_segmentation_cuda_10_2.yml --name ml_segmentation``` then activate the environment with ```conda activate ml_segmentation```.


### Installing on Xavier AGX
Installation on the Xavier can be done with the created image of the Xavier


## How to use 

### Training on pc without ROS
* [Prepare training data]
	- On the C-disk there is some simulated and real data
	- In ml-based-lidar-segmentation/data/sequences make folders 00, 01, .. for all rosbag recordings
	- Copy all rosbags to these folders
	For each sequence folder do the following:
	- Generate points and label with ```cd ml-based-lidar-segmentation/utils``` ```python3 pclabel2.py path/to/rosbag```
	- Convert to KITTI file format ```python3 to_KITTI_format.py /path/to/pointclouds
	- Change file paths in config/data_preparing.yaml and choose how many residual images you want to use
	- Generate residual images with ml-based-lidar-segmentation/utils ```python3 genresidual.py``` 
	
* [Training]

	- ```cd ml-base-lidar-segmentation/mos_SalsaNext/train/tasks/semantic```
	- Set the training, validation and test set to the correct sequences in the salsanext_gv_mos.yml file
	- Start the training with
	- ```./train.py```
	- The best performing neural network model is saved in the output folder.
	- For Pytorch compatibility issues the model has to be changed slightly before inference is possible
	- ```python module_remover.py path/to/model/directory```
	
	
	(optional) export model to onnx with ```model_converter.py``` to be able to do fast inferrence on Xavier later.
* [Inferring]
	- With the trained model it is possible to do inference on other data with:
	- ```./infer.py -m path/to/model/directory```
	- Predictions are stored in data/infer_output/sequences...

* [Visualising]

	- To visualize the output ```python visualize_mos.py```

### Inferring with ROS
 - 	-in the sim_bag.launch or real_bag.launch (depends on type of bag, simulated or real data) file change the 		 rosbag path to make it point to the rosbag you want to use
	-in the ros_inferring.yaml change the paths of the pre-trained neural network model and of the output
	-now you should be able to run the segmentation algorithm with roslaunch ml_segmentation bag.launch
	-you can visualise the output with rviz

### Inferring on Xavier AGX
After setting up the Xavier with the created image the Xavier is ready to go. But for additional performance it is possible to create an inferrence engine with NVIDIA TensorRT.
This can be done by running the ```engine.py``` file. This speeds up the run time by about 200%

After this you can change the config file and set engine to True. This makes sure that the TensorRT engine is used instead of PyTorch.
 


 

