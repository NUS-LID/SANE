import random
import numpy as np 
import time

class ReplayMemory(object):
    def __init__(self, size=1000000, frame_height=84, frame_width=84, 
                 agent_history_length=4, batch_size=32):
        """
        Args:
            size: Integer, Number of stored transitions
            frame_height: Integer, Height of a frame of an Atari game
            frame_width: Integer, Width of a frame of an Atari game
            agent_history_length: Integer, Number of frames stacked together to create a state
            batch_size: Integer, Number if transitions returned in a minibatch
        """
        self.size = size
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.agent_history_length = agent_history_length
        self.batch_size = batch_size
        self.count = 0
        self.current = 0
        
        self.actions = np.empty(self.size, dtype=np.int32)
        self.rewards = np.empty(self.size, dtype=np.float32)
        self.frames = np.empty((self.size, self.frame_height, self.frame_width), dtype=np.uint8)
        self.terminal_flags = np.empty(self.size, dtype=np.bool)
        
        self.states = np.empty((self.batch_size, self.agent_history_length, 
                                self.frame_height, self.frame_width), dtype=np.uint8)
        self.new_states = np.empty((self.batch_size, self.agent_history_length, 
                                    self.frame_height, self.frame_width), dtype=np.uint8)
        self.indices = np.empty(self.batch_size, dtype=np.int32)
        
    def add_experience(self, action, frame, reward, terminal):
        """
        Args:
            action: An integer between 0 and env.action_space.n - 1 determining the action the agent perfomed
            frame: A (84, 84, 1) frame of an Atari game in grayscale
            reward: A float determining the reward the agend received for performing an action
            terminal: A bool stating whether the episode terminated
        """
        if frame.shape != (self.frame_height, self.frame_width):
            raise ValueError('Dimension of frame is wrong!')
        self.actions[self.current] = action
        self.frames[self.current, ...] = frame
        self.rewards[self.current] = reward
        self.terminal_flags[self.current] = terminal
        self.count = max(self.count, self.current+1)
        self.current = (self.current + 1) % self.size
             
    def _get_state(self, index):
        """
        Args :
            index : Index of the state to be retrieved in the replay buffer
        Returns 
            The state at Index index
        """
        if self.count is 0:
            raise ValueError("The replay memory is empty!")
        if index < self.agent_history_length - 1:
            raise ValueError("Index must be min 3")
        return self.frames[index-self.agent_history_length+1:index+1, ...]
        
    def _get_valid_indices(self):
        """
        Gets a set of randomly sampled valid indices for training
        """
        for i in range(self.batch_size):
            while True:
                index = random.randint(self.agent_history_length, self.count - 1)
                if index < self.agent_history_length:
                    continue
                if index >= self.current and index - self.agent_history_length <= self.current:
                    continue
                if self.terminal_flags[index - self.agent_history_length:index].any():
                    continue
                break
            self.indices[i] = index
            
    def get_minibatch(self):
        """
        Returns a minibatch of self.batch_size = 32 transitions
        """
        if self.count < self.agent_history_length:
            raise ValueError('Not enough memories to get a minibatch')
        
        self._get_valid_indices()
            
        for i, idx in enumerate(self.indices):
            self.states[i] = self._get_state(idx - 1)
            self.new_states[i] = self._get_state(idx)
        
        return np.transpose(self.states, axes=(0, 2, 3, 1)), self.actions[self.indices], self.rewards[self.indices], np.transpose(self.new_states, axes=(0, 2, 3, 1)), self.terminal_flags[self.indices]

    def save_buffer_to_disk(self, checkpoint_name) :
        """
        Args:
            checkpoint_name: String, name of the buffer file to be saved to the disk
        """

        start_time = time.time()
        np.save('{}_actions_.npy'.format(checkpoint_name), self.actions)
        np.save('{}_rewards_.npy'.format(checkpoint_name), self.rewards)
        np.save('{}_frames_.npy'.format(checkpoint_name), self.frames)
        np.save('{}_terminal_flags_.npy'.format(checkpoint_name), self.terminal_flags)
        print('Took {} seconds to save buffer'.format(time.time() - start_time))
        with open('{}_loadstats.dat'.format(checkpoint_name), 'a') as save_stats_file:
            print('Took {} seconds to load buffer'.format(time.time() - start_time), file=save_stats_file)

    def load_buffer_from_disk(self, checkpoint_name) :
        """
        Args:
            checkpoint_name: String, name of the buffer file to be loaded from the disk
        """
        start_time = time.time()
        self.actions = np.load('{}_actions_.npy'.format(checkpoint_name))
        self.rewards = np.load('{}_rewards_.npy'.format(checkpoint_name))
        self.frames = np.load('{}_frames_.npy'.format(checkpoint_name))
        self.terminal_flags = np.load('{}_terminal_flags_.npy'.format(checkpoint_name))
        print('Took {} seconds to load buffer'.format(time.time() - start_time))
        with open('{}_loadstats.dat'.format(checkpoint_name), 'a') as load_stats_file:
            print('Took {} seconds to load buffer'.format(time.time() - start_time), file=load_stats_file)
