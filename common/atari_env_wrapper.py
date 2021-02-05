import numpy as np
import gym
from common.frame_processor import FrameProcessor
import random

class Atari(object):
    def __init__(self, envName, no_op_steps=10, agent_history_length=4):
        self.env = gym.make(envName)
        self.process_frame = FrameProcessor(obs_shape=self.env.observation_space.shape)
        self.state = None
        self.last_lives = 0
        self.no_op_steps = no_op_steps
        self.agent_history_length = agent_history_length

    def reset(self, sess, evaluation=False):
        """
        Args:
            sess: A Tensorflow session object
            evaluation: A boolean saying whether the agent is evaluating or training
        Resets the environment and stacks four frames ontop of each other to 
        create the first state
        """
        frame = self.env.reset()
        self.last_lives = 0
        terminal_life_lost = True 

        for _ in range(random.randint(1, self.no_op_steps)):
            frame, _, _, _ = self.env.step(0) 
        self.curr_frame = frame
        processed_frame = self.process_frame(sess, frame)   
        self.state = np.repeat(processed_frame, self.agent_history_length, axis=2)
        return terminal_life_lost

    def step(self, sess, action):
        """
        Args:
            sess: A Tensorflow session object
            action: Integer, action the agent performs
        Performs an action and observes the reward and terminal state from the environment
        """
        new_frame, reward, terminal, info = self.env.step(action)  
        self.curr_frame = new_frame
        if info['ale.lives'] < self.last_lives:
            terminal_life_lost = True
        else:
            terminal_life_lost = terminal
        self.last_lives = info['ale.lives']
        
        processed_new_frame = self.process_frame(sess, new_frame)   
        new_state = np.append(self.state[:, :, 1:], processed_new_frame, axis=2) 
        self.state = new_state
        
        return processed_new_frame, reward, terminal, terminal_life_lost, new_frame
