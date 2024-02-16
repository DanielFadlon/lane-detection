
# Lane Detection System for Dashcam Videos - by Computer Vision techniques (without CNNs)
This project implements a lane detection system for vehicle dashcam videos using computer vision techniques. 
It focuses on detecting lane lines, lane changes, and offers additional support for enhancements such as nighttime visibility improvement, vehicle detection, and curve detection.

# Main Features
### Lane Line Detection: Identifies and highlights lane lines in dashcam footage.
### Lane Change Detection: Identifies when a lane change takes place and indicates the direction of the change.

# Enhancements
### Nighttime Visibility Enhancement: Improves video visibility during nighttime for better lane detection.
### Vehicle Detection: Detects vehicles within the dashcam's field of view and provides caution alerts.
### Curve Detection: Identifies curves in the road ahead and alerts accordingly.

## Customizing for Different Videos
The performance of the lane detection system, including enhancements for nighttime visibility, vehicle detection, and curve detection, relies heavily on specific hyperparameters that are aligned with the characteristics of the input video. These hyperparameters include thresholds for edge detection, blur levels, region of interest specifications, and more.


# Set Up Environment
### Create the Environment: Run the following command to create a Conda environment based on the env.yml file:

```bash
conda env create -f environment.yml
```

This command reads the environment.yml file, creates a new Conda environment with the name specified in the file, and installs all the required packages.

### Activate the Environment
Once the environment is created, activate it using the following command:

```bash
conda activate environment_name
```
### Deactivating the Environment
When you're finished working on the project, you can deactivate the environment by running:

```bash
conda deactivate
```

# Usage
The system allows processing of dashcam videos with different modes based on the video type. The current implementation supports:

- Daytime lane detection
- Nighttime visibility enhancement
- Vehicle detection
- Curve detection
To choose the video processing mode, adjust the chose_video_type variable in the main file:

```python
chose_video_type = VideoType.LANE_CHANGE  # Options: LANE_CHANGE, NIGHT_TIME, DETECT_VEHICLES, DETECT_CURVES
```
