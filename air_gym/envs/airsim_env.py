import numpy as np
import airsim

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.spaces import Tuple, Box, Discrete, MultiDiscrete

from collections import OrderedDict

class AirSimEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, image_shape):
        self.observation_space = spaces.Box(low=0, high=255, shape=image_shape, dtype=np.uint8)
        self._seed()


        self.viewer = None
        self.steps = 0
        self.no_episode=0
        self.reward_sum=0

    def __del__(self):
        raise NotImplementedError()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _compute_reward(self):
        raise NotImplementedError()
    
    def step(self, action):
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()

    def render(self, mode='human'):
        img = self._get_obs()
        if mode=='human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen
        elif mode == 'rgb_array':
            return img


class AirSimDroneEnv(AirSimEnv):
    
    def __init__(self, ip_address, control_type, step_length, image_shape, goal):
        super().__init__(image_shape)

        self.step_length = step_length
        self.control_type = control_type
        self.image_shape = image_shape
        self.goal = airsim.Vector3r(goal[0],goal[1],goal[2])

        if self.control_type is 'discrete':
            self.action_space = spaces.Discrete(7)
        if self.control_type is 'continuous':
            self.action_space = spaces.Box(low=-5, high=5, shape=(3,))
        else:
            print("Must choose a control type {'discrete','continuous'}. Defaulting to discrete.")
            self.action_space = spaces.Discrete(7)

        self.state = {"position":np.zeros(3), "collision":False}

        self.drone = airsim.MultirotorClient(ip = ip_address)
        self._setup_flight()

        self.image_request = airsim.ImageRequest('front_center',airsim.ImageType.Scene, False, False)

    def __del__(self):
        self.drone.reset()

    def _setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)
        self.drone.moveToPositionAsync(0,0,-2,2).join()

    def _get_obs(self):
        response = self.drone.simGetImages([self.image_request])
        image = np.reshape(np.fromstring(response[0].image_data_uint8, dtype=np.uint8),self.image_shape)
        _drone_state = self.drone.getMultirotorState()
        position = _drone_state.kinematics_estimated.position.to_numpy_array()
        collision = self.drone.simGetCollisionInfo().has_collided
        
        self.state["position"] = position
        self.state["collision"] = collision

        return image

    def _compute_reward(self):
        pos = self.state["position"]
        current_pos = airsim.Vector3r(pos[0],pos[1],pos[2])
        if current_pos == self.goal:
            done = True
            reward = 10
            return reward, done
        elif self.state["collision"] == True:
            done = True
        else:
            done = False

        dist = current_pos.distance_to(self.goal)
        if dist > 30:
            reward = 0
        else:
            reward = (30-dist)*0.1

        return reward, done

    def _do_action(self, action):
        if self.control_type is 'discrete':
            new_position = self.actions_to_op(action)
            if new_position[2] > -1:
                new_position[2] = -1
            if new_position[2] < - 40:
                new_position[2] = -40
            self.drone.moveToPositionAsync(float(new_position[0]),float(new_position[1]),float(new_position[2]), 3).join()
        else:
            self.drone.moveByVelocityAsync(float(action[0]),float(action[1]),float(action[2]), self.step_length).join()

    def noop(self):
        new_position = self.state["position"]
        return new_position

    def forward(self):
        new_position = self.state["position"]
        new_position[0] += self.step_length
        return new_position

    def backward(self):
        new_position = self.state["position"]
        new_position[0] -= self.step_length
        return new_position
        
    def right(self):
        new_position = self.state["position"]
        new_position[1] += self.step_length
        return new_position

    def left(self):
        new_position = self.state["position"]
        new_position[1] -= self.step_length
        return new_position

    def up(self):
        new_position = self.state["position"]
        new_position[2] += self.step_length
        return new_position

    def down(self):
        new_position = self.state["position"]
        new_position[2] -= self.step_length
        return new_position

    def step(self, action):
        self._do_action(action)
        obs = self._get_obs()
        reward, done = self._compute_reward()
        
        return obs, reward, done, self.state

    def reset(self):
        self._setup_flight()
        self._get_obs()

    def actions_to_op(self, action):
        switcher = {
            0:self.noop,
            1:self.forward,
            2:self.backward,
            3:self.right,
            4:self.left,
            5:self.up,
            6:self.down
        }

        func = switcher.get(action, lambda: "Invalid Action!")
        return func()
