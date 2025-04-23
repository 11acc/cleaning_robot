
## Cleaning Robot

This repository contains code which define ROS nodes that enable a robot to complete a cleaning challenge.

The repository is structed as a ROS package and should be treated as one. Therefore the repository needs to be cloned inside a catkin workspace such that:

```
~/catkin_ws/
├── build/                # Holds temporary build files (created by catkin_make)
│   └── ...
├── devel/                # Holds built packages and setup scripts (created by catkin_make)
│   └── ...
└── src/                  # Holds different created ROS packages
    ├── cleaning_robot/   # Our repository
    │   ├── launch/
    │   │   └── ...       # Launch files for each script
    │   ├── src/
    │   │   ├── complete_bot.py                  # Orchestrator script
    │   │   ├── line_follower_state_machine.py   # Line following functionality
    │   │   ├── red_peg_grabber.py               # Peg identification and manipulation
    │   │   └── yellow_follower.py               # Deploy zone navigation
    │   ├── CMakeLists.txt                       # How to build and link the code and dependencies
    │   └── package.xml                          # Package's metadata and dependencies for ROS
    └── ...
```

Once the repository is setup correctly, in order to execute any given script the system requires at least two terminal sessions:
1. Terminal 1: SSH into the robot to establish the gripper connection
2. Terminal 2: Execute your scripts from your local machine

### Terminal 1: Robot Gripper Setup

#### 1. SSH into the robot
```bash
$ ssh husarion@agamemnon
# pass: husarion
```

#### 2. Setting up ROS Serial for the gripper
```bash
# If ROS Serial isn't installed
$ cd ~/husarion_ws/src/
$ git clone https://github.com/ros-drivers/rosserial.git

# Compile the workspace
$ cd ~/husarion_ws/
$ catkin_make
$ catkin_install
```

#### 3. Connect to the Arduino
```bash
# Identify the Arduino port
$ ls /dev/ttyUSB*

# Run the serial node, leave this terminal running
$ rosrun rosserial_python serial_node.py /dev/ttyUSB1
```

#### 4. (Optional) Gripper Manual Controls
```bash
# Move the gripper to a specific angle (0-170)
$ rostopic pub /servo std_msgs/UInt16 "data: [angle]"

# Monitor the servo load
$ rostopic echo /servoLoad
```

### Terminal 2: Script Execution

#### 1. Configure ROS environment
```bash
# Setup ROS environment
$ source /opt/ros/noetic/setup.bash

# Connect to the robot's ROS Master
$ export ROS_MASTER_URI=http://[ROBOT_IP]:11311

# Set your computer's IP for response routing
$ export ROS_IP=[COMPUTER_IP]

# To find your IP address:
$ ifconfig

# Verify the connection by viewing the ROS topics
$ rostopic list
```

#### 2. Prepare and run your scripts
```bash
# Make sure you're in the root of the workspace
$ cd ~/catkin_ws/

# Source the workspace and compile (first time or after big changes)
$ source devel/setup.bash
$ catkin_make

# Launch a script
$ roslaunch cleaning_robot red_peg_grabber.launch
```

#### 3. Update cleaning_robot repository
```bash
# Pull the latest changes
$ cd ~/catkin_ws/src/cleaning_robot/
$ git pull
```

### Troubleshooting
- If the gripper isn't responding, make sure the Arduino is connected to the correct USB port
- If `rostopic list` doesn't show any topics, check network connectivity between your machine and the robot
- Make sure both ROS_MASTER_URI and ROS_IP are correctly set to the respective IP addresses
