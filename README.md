This code is developed for UR10e in IRIS LAB for RTDE Control with the gripper

Ensure the robot is in remote access mode before running the script and and PC is connectede via LAN and is in the network same as the robot
In the code developed
ROBOT: 192.168.1.102
HOST: 192.168.1.101

Ensure Kinect is also connected to the PC, freenect- python wrapper is used in this case to connect the PC with kinect and obtain RBG and depth image
Some of the scripts use joystick, which is done using pygame
