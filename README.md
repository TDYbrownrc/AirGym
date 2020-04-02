# AirGym

This repository is a simple OpenAI Gym interface for Microsoft AirSim (https://github.com/microsoft/AirSim). 

Install to your Python3 environment with

`git clone https://github.com/TDYbrownrc/AirGym.git`

`cd air_gym`

`pip install .`

This interface supports 2 drone control types: discrete positional control and continuous velocity control. These are initialization arguments passed into the OpenAI gym initialization script.

Create a gym environment like this:

`import gym`

`import air_gym`

`env = gym.make('air_gym:airsim-drone-v0', ip_address = 'IP_ADDRESS_OF_AIRSIM_INSTANCE', control_type = {'discrete', 'continuous'}, step_length = DISTANCE_OF_DISCRETE_MOTION_OR_LENGTH_OF_TIME_FOR_VELOCITY_MOTION, image_shape = (w, h, 3), goal = [GOAL_POINT])`
