import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers
from common.utils import noisy_layer, add_fc_layer, generate_noisy_latent_layers

class DQN(object):
    def __init__(self, scope, n_actions, switch_init, hidden=512, learning_rate=0.00001,
                 noisy = False, layer_wise_variance=False, single_param=False, state_dependent=False, noise_latent_size=16,
                 frame_height=84, frame_width=84, agent_history_length=4, add_q_values_perturb_module = False, lambda1=0.0001, lambda2=0.01, add_var_loss = False):
        """
        Args:
            n_actions: Integer, number of possible actions
            hidden: Integer, Number of neurons hidden layer of the main network. 
            learning_rate: Float, Learning rate for the Adam optimizer
            frame_height: Integer, Height of a frame of an Atari game
            frame_width: Integer, Width of a frame of an Atari game
            agent_history_length: Integer, Number of frames stacked together to create a state
        """
        self.n_actions = n_actions
        self.hidden = hidden
        self.learning_rate = learning_rate
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.agent_history_length = agent_history_length
        if add_q_values_perturb_module :
            self.q_val_input = tf.placeholder(shape=[None, self.n_actions], dtype=tf.float32)

        self.input = tf.placeholder(shape=[None, self.frame_height, 
                                           self.frame_width, self.agent_history_length], 
                                    dtype=tf.float32)
        self.cond_variable = tf.placeholder(shape = [1], dtype=tf.bool)
        self.inputscaled = self.input/255 
        self.noisySingle = single_param

        with tf.variable_scope ('shared_network') :
            self.conv1 = tf.layers.conv2d(
                inputs=self.inputscaled, filters=32, kernel_size=[8, 8], strides=4,
                padding="valid", activation=tf.nn.relu, use_bias=False, name='conv1')
            self.conv2 = tf.layers.conv2d(
                inputs=self.conv1, filters=64, kernel_size=[4, 4], strides=2, 
                padding="valid", activation=tf.nn.relu, use_bias=False, name='conv2')
            self.conv3 = tf.layers.conv2d(
                inputs=self.conv2, filters=64, kernel_size=[3, 3], strides=1, 
                padding="valid", activation=tf.nn.relu, use_bias=False, name='conv3')

        with tf.variable_scope('act_q') :
            self.q_valuestream_hidden = tf.layers.flatten(self.conv3)
        if state_dependent and (layer_wise_variance or single_param) :
            with tf.variable_scope('perturb_module') :
                perturb_input = self.q_valuestream_hidden
                if add_q_values_perturb_module :
                    perturb_input = tf.concat([perturb_input, self.q_val_input], axis=1)
                self.noise_latent, self.common_variance = generate_noisy_latent_layers(perturb_input, 
                                                                           noisy=noisy,
                                                                           noise_latent_size=noise_latent_size,
                                                                           layer_wise_variance=layer_wise_variance)
        elif state_dependent and noisy: 
            self.noise_latent = tf.layers.dense(self.q_valuestream_hidden,
                                units=noise_latent_size,
                                name="noise_head",
                                activation=tf.nn.relu)
            self.common_variance = tf.constant(0.0, shape = [1,1])
        elif self.noisySingle: 
            self.noise_latent, self.common_variance = None, tf.get_variable("common_noisy/w_sigma", [1], initializer=tf.constant_initializer(0.01551282917))
        else :
           self.noise_latent, self.common_variance = None, tf.constant(0.0, shape = [1,1])
        
        with tf.variable_scope('act_q') as variable_scope :
            self.q_valuestream = add_fc_layer(input=self.q_valuestream_hidden, 
                                            hidden_units=self.hidden, 
                                            noise_head=self.noise_latent, 
                                            common_variance=self.common_variance, 
                                            switch_init = switch_init,
                                            activation_fn=tf.nn.relu, 
                                            noisy = noisy,
                                            layer_wise_variance=layer_wise_variance,
                                            single_param=single_param, 
                                            state_dependent=state_dependent,
                                            cond_variable=self.cond_variable,
                                            name="q_value_hidden")
            self.q_values = add_fc_layer(input=self.q_valuestream, 
                                        hidden_units=self.n_actions, 
                                        noise_head=self.noise_latent, 
                                        common_variance=self.common_variance,
                                        switch_init = switch_init,
                                        noisy = noisy,
                                        layer_wise_variance=layer_wise_variance,
                                        single_param=single_param, 
                                        state_dependent=state_dependent,
                                        cond_variable=self.cond_variable,
                                        name="q_value")
            self.best_action = tf.argmax(self.q_values, 1)

        
        self.action = tf.placeholder(shape=[None], dtype=tf.int32)
        self.target_q = tf.placeholder(shape=[None], dtype=tf.float32)
        self.Q = tf.reduce_sum(tf.multiply(self.q_values, tf.one_hot(self.action, self.n_actions, dtype=tf.float32)), axis=1)     
        self.loss_q_act = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.target_q, predictions=self.Q)) 
        
        if add_var_loss : 
            self.var_for_l2_loss = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.var_for_l2_loss])
            self.variance_loss = tf.reduce_mean((self.common_variance**2)/2 - tf.math.log(tf.abs(self.common_variance)))          
            print ('V_LOSSSS', self.variance_loss, self.var_for_l2_loss, lambda2)
            self.kl_loss = lambda1*self.l2_loss + lambda2*self.variance_loss
            self.loss_q_act = self.loss_q_act + self.kl_loss

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1.5 * 0.0001)
        self.update = self.optimizer.minimize(self.loss_q_act)
