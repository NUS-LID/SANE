import os
import gym
import tensorflow as tf
import numpy as np
import argparse
from tqdm import tqdm

from common.atari_env_wrapper import Atari
from common.exploration_scheduler import  ExplorationExploitationScheduler
from common.frame_processor import FrameProcessor
from common.utils import clip_reward, update_min_max_frame

from DQN.replay import ReplayMemory
from DQN.network import DQN

PARSER = argparse.ArgumentParser(description=None)
PARSER.add_argument('-t', '--train', default=False, type=bool, help='Train : True, Evaluate : False')
PARSER.add_argument('-env', '--environment', default='EnduroDeterministic-v4', type=str, help='Atari Environment to run')
PARSER.add_argument('-max_ep_len', '--MAX_EPISODE_LENGTH', default=100000, type=int, help='Maximum episode length')
PARSER.add_argument('-save_freq', '--SAVE_FREQUENCY', default=1000000, type=int, help='Number of frames the agent sees between evaluations')
PARSER.add_argument('-netw_freq', '--NETW_UPDATE_FREQ', default=10000, type=int, help='Number of chosen actions between updating the target network.')
PARSER.add_argument('-g', '--DISCOUNT_FACTOR', default=0.99, type=float, help='gamma in the Bellman equation')
PARSER.add_argument('-start_learn', '--REPLAY_MEMORY_START_SIZE', default=50000, type=int, help='Number of completely random actions before agent starts learning')
PARSER.add_argument('-max_step', '--MAX_FRAMES', default=25000000, type=int, help='Total number of frames the agent sees')
PARSER.add_argument('-mem_size', '--MEMORY_SIZE', default=1000000, type=int, help='Number of transitions stored in the replay memory')
PARSER.add_argument('-num_noop', '--NO_OP_STEPS', default=30, type=int, help='Number of NOOP or FIRE actions at the beginning of an episode')
PARSER.add_argument('-upd_freq', '--UPDATE_FREQ', default=4, type=int, help='Number of actions a gradient descent step is performed')
PARSER.add_argument('-hidden', '--HIDDEN', default=512, type=int, help='Number of neurons in the final layer')
PARSER.add_argument('-lr', '--LEARNING_RATE', default=0.0000625, type=float, help='Learning rate')
PARSER.add_argument('-bs', '--BS', default=32, type=int, help='Batch size')
PARSER.add_argument('-path', '--PATH', default="output/", type=str, help='Gifs and checkpoints will be saved here')
PARSER.add_argument('-sum', '--SUMMARIES', default="summaries", type=str, help='logdir for tensorboard')
PARSER.add_argument('-id', '--RUNID', default='run_1', type=str, help='run id inside logdir')

PARSER.add_argument('-n', '--NOISY', default=False, type=bool, help='If Parameter noise included')
PARSER.add_argument('-state_dependent', '--STATE_DEPENDENT', default=False, type=bool, help='If variance is state dependent : Default all parameters independent variance')
PARSER.add_argument('-single', '--SINGLE', default=False, type=bool, help='Change to single variance for all noises: Works only with state dependent noise')
PARSER.add_argument('-layer', '--LAYERWISE', default=False, type=bool, help='Change to layer wise common variance: Works only with state dependent noise')
PARSER.add_argument('-nls', '--NOISY_LATENT_SIZE', default=16, type=int, help='Latent size for noise')

PARSER.add_argument('-no_eps', '--NO_EPSILON_GREEDY', default=False, type=bool, help='Disable epsilon greedy')
PARSER.add_argument('-eps_sched', '--EPS_SCHED', default=False, type=bool, help='Greedier epsilon schedule')

PARSER.add_argument('-switch_noise_init', '--SWITCH_NOISY_LAYER_INIT', default=1, type=int, help='Switch Noisy layer init to variance scaling initializer')
PARSER.add_argument('-d', '--DEVICE', default='gpu', type=str, help='GPU/CPU')
PARSER.add_argument('-d_id', '--DEVICE_ID', default=-1, type=int, help='GPU id to use')

PARSER.add_argument('-lm', '--LOAD_MODEL', default=False, type=bool, help='Enable to load existing model')
PARSER.add_argument('-lbs', '--LOAD_BUFFER_SIZE', default=50000, type=int, help='Enable to load existing model')

PARSER.add_argument('-exp_stop', '--EXP_STOP', default=np.inf, type=int, help='Stop parameter noise')
PARSER.add_argument('-greedy', '--GREEDY', default=False, type=bool, help='Greedy Agent')
PARSER.add_argument('-save_buffer', '--SAVE_BUFFER', default=False, type=bool, help='Set flag to True to enable saving replay_buffer')
PARSER.add_argument('-load_buffer', '--LOAD_BUFFER', default=False, type=bool, help='Set flag to True to enable loading replay_buffer')


PARSER.add_argument('-add_q_values_to_perturb', '--ADD_Q_VALUES_TO_PERTURB', default=False, type=bool, help='Set flag to True to add Q values as input to perturbation module')

PARSER.add_argument('-use_defaults', '--USE_DEFAULTS', default=False, type=bool, help='Set flag to True run with default settings (Do not add other flags except for train/evaluate and the environment to be trained')
PARSER.add_argument('-eps_greedy', '--EPS_GREEDY_AGENT', default=False, type=bool, help='Set flag to True to train/evaluate an epsilon greedy agent')
PARSER.add_argument('-noisy_net', '--NOISY_NET_AGENT', default=False, type=bool, help='Set flag to True to train/evaluate a NoisyNet agent')
PARSER.add_argument('-sane', '--SANE_AGENT', default=False, type=bool, help='Set flag to True to train/evaluate a SANE agent')
PARSER.add_argument('-q_sane', '--Q_SANE_AGENT', default=False, type=bool, help='Set flag to True to train/evaluate a Q-SANE agent')
PARSER.add_argument('-eval_steps', '--EVAL_STEPS', default=500000, type=int, help='Number of interactions to evaluate an agent')

PARSER.add_argument('-lambda1', '--LAMBDA1', default=0.0001, type=float, help='Number of interactions to evaluate an agent')
PARSER.add_argument('-lambda2', '--LAMBDA2', default=0.01, type=float, help='Number of interactions to evaluate an agent')
PARSER.add_argument('-add_var_loss', '--ADD_VAR_LOSS', default=False, type=bool, help='Number of interactions to evaluate an agent')



ARGS = PARSER.parse_args()

EPS_G_SCHEDULE = ARGS.EPS_SCHED
TRAIN = ARGS.train
ENV_NAME = ARGS.environment

MAX_EPISODE_LENGTH = ARGS.MAX_EPISODE_LENGTH       
SAVE_FREQUENCY = ARGS.SAVE_FREQUENCY          
NETW_UPDATE_FREQ = ARGS.NETW_UPDATE_FREQ         

DISCOUNT_FACTOR =ARGS.DISCOUNT_FACTOR           
REPLAY_MEMORY_START_SIZE = ARGS.REPLAY_MEMORY_START_SIZE 

MAX_FRAMES = ARGS.MAX_FRAMES            
MEMORY_SIZE = ARGS.MEMORY_SIZE            
NO_OP_STEPS = ARGS.NO_OP_STEPS                 

UPDATE_FREQ = ARGS.UPDATE_FREQ                  
HIDDEN = ARGS.HIDDEN                    

LEARNING_RATE = ARGS.LEARNING_RATE          

BS = ARGS.BS                          
PATH = ARGS.RUNID + "_" + ARGS.PATH              
SUMMARIES = ARGS.SUMMARIES         
RUNID = ARGS.RUNID

NOISY=ARGS.NOISY
STATE_DEPENDENT=ARGS.STATE_DEPENDENT
SINGLE_VARIANCE=ARGS.SINGLE
LAYERWISE_VARIANCE = ARGS.LAYERWISE
NOISY_LATENT_SIZE = ARGS.NOISY_LATENT_SIZE
NO_EPS = ARGS.NO_EPSILON_GREEDY
SWITCH_NOISY_LAYER_INIT = ARGS.SWITCH_NOISY_LAYER_INIT
DEVICE = ARGS.DEVICE
LAMBDA1 = ARGS.LAMBDA1
LAMBDA2 = ARGS.LAMBDA2
ADD_VAR_LOSS = ARGS.ADD_VAR_LOSS

if ARGS.DEVICE_ID == -1 : 
    DEVICE_ID =0
else :
    os.environ["CUDA_VISIBLE_DEVICES"] = str(ARGS.DEVICE_ID)
    DEVICE_ID =0

os.makedirs(PATH, exist_ok=True)
os.makedirs(os.path.join(SUMMARIES, RUNID), exist_ok=True)
SUMM_WRITER = tf.summary.FileWriter(os.path.join(SUMMARIES, RUNID))

LOAD_MODEL = ARGS.LOAD_MODEL
LOAD_BUFFER_SIZE = ARGS.LOAD_BUFFER_SIZE
EXP_STOP = ARGS.EXP_STOP
GREEDY = ARGS.GREEDY

SAVE_BUFFER = ARGS.SAVE_BUFFER
LOAD_BUFFER = ARGS.LOAD_BUFFER
ADD_Q_VALUES_TO_PERTURB = ARGS.ADD_Q_VALUES_TO_PERTURB

if ARGS.USE_DEFAULTS :
    if ARGS.EPS_GREEDY_AGENT :
        SWITCH_NOISY_LAYER_INIT = 2
    elif ARGS.NOISY_NET_AGENT :
        SWITCH_NOISY_LAYER_INIT = 3
        NOISY = True
        NO_EPS = True
    elif ARGS.SANE_AGENT or ARGS.Q_SANE_AGENT :
        SWITCH_NOISY_LAYER_INIT = 2
        NOISY = True
        STATE_DEPENDENT = True
        SINGLE_VARIANCE = True
        NO_EPS = True
        NOISY_LATENT_SIZE = 256
        if ARGS.Q_SANE_AGENT :
            ADD_Q_VALUES_TO_PERTURB = True

def learn(session, replay_memory, main_dqn, target_dqn, batch_size, gamma, frame_number, is_double_q = False):
    """
    Args:
        session: A tensorflow sesson object
        replay_memory: A ReplayMemory object
        main_dqn: A DQN object
        target_dqn: A DQN object
        batch_size: Integer, Batch size
        gamma: Float, discount factor for the Bellman equation
        frame_number: Frame Number to check whether to stop parameter space noise 
    Returns:
        loss: The loss of the minibatch
    """
    states, actions, rewards, new_states, terminal_flags = replay_memory.get_minibatch()    
    if frame_number - NETW_UPDATE_FREQ > EXP_STOP :
        target_cond_variable = False
    else :
        target_cond_variable = True
    
    if frame_number > EXP_STOP :
        main_cond_variable = False
    else :
        main_cond_variable = True

    if ADD_Q_VALUES_TO_PERTURB :
        dict_inp = {
        target_dqn.q_val_input : np.zeros([batch_size,target_dqn.n_actions]),
        main_dqn.q_val_input : np.zeros([batch_size,main_dqn.n_actions]),
        target_dqn.input : new_states,
        main_dqn.input : states,
        target_dqn.cond_variable:[target_cond_variable],
        main_dqn.cond_variable:[main_cond_variable]}
        main_q_for_perturb, target_q_for_perturb = session.run([main_dqn.q_values, target_dqn.q_values], feed_dict=dict_inp)
        inp_pert_dict = {main_dqn.q_val_input : main_q_for_perturb, target_dqn.q_val_input: target_q_for_perturb}
    else :
        inp_pert_dict = {}

    q_vals, arg_q_max = session.run([target_dqn.q_values, target_dqn.best_action], feed_dict={**{target_dqn.input:new_states, target_dqn.cond_variable:[target_cond_variable]}, **inp_pert_dict})
    double_q = q_vals[range(batch_size), arg_q_max]
    target_q = rewards + (gamma*double_q * (1-terminal_flags))

    loss_grad_list = session.run([main_dqn.loss_q_act, main_dqn.update], 
                          feed_dict={**{main_dqn.input:states, 
                                     main_dqn.target_q:target_q, 
                                     main_dqn.cond_variable:[main_cond_variable],
                                     main_dqn.action:actions}, **inp_pert_dict})

    return loss_grad_list[0]

class TargetNetworkUpdater(object):
    def __init__(self, main_dqn_vars, target_dqn_vars):
        self.main_dqn_vars = main_dqn_vars
        self.target_dqn_vars = target_dqn_vars

    def _update_target_vars(self):
        update_ops = []
        for i, var in enumerate(self.main_dqn_vars):
            copy_op = self.target_dqn_vars[i].assign(var.value())
            update_ops.append(copy_op)
        return update_ops
            
    def __call__(self, sess):
        """
        Args:
            sess: A Tensorflow session object
        Assigns the values of the parameters of the main network to the 
        parameters of the target network
        """
        update_ops = self._update_target_vars()
        for copy_op in update_ops:
            sess.run(copy_op)



tf.reset_default_graph()
atari = Atari(ENV_NAME, NO_OP_STEPS)
print("The environment has the following {} actions: {}".format(atari.env.action_space.n, 
                                                                atari.env.unwrapped.get_action_meanings()))
# main DQN and target DQN networks:
with tf.device("/{}:{}".format(DEVICE,DEVICE_ID)) :
    with tf.variable_scope('mainDQN'):
        MAIN_DQN = DQN(scope='mainDQN',
                    n_actions=atari.env.action_space.n, 
                    hidden=HIDDEN, 
                    learning_rate=LEARNING_RATE, 
                    switch_init=SWITCH_NOISY_LAYER_INIT,
                    noisy = NOISY, 
                    layer_wise_variance=LAYERWISE_VARIANCE, 
                    single_param=SINGLE_VARIANCE, 
                    state_dependent=STATE_DEPENDENT, 
                    noise_latent_size=NOISY_LATENT_SIZE, 
                    add_q_values_perturb_module = ADD_Q_VALUES_TO_PERTURB, lambda1=LAMBDA1, lambda2=LAMBDA2, add_var_loss=ADD_VAR_LOSS)

    with tf.variable_scope('targetDQN'):
        TARGET_DQN = DQN(scope='targetDQN',
                    n_actions=atari.env.action_space.n, 
                    hidden=HIDDEN, 
                    switch_init=SWITCH_NOISY_LAYER_INIT,
                    noisy = NOISY, 
                    layer_wise_variance=LAYERWISE_VARIANCE, 
                    single_param=SINGLE_VARIANCE, 
                    state_dependent=STATE_DEPENDENT, 
                    noise_latent_size=NOISY_LATENT_SIZE,
                    add_q_values_perturb_module = ADD_Q_VALUES_TO_PERTURB)   


    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=50) 
    best_model_saver = tf.train.Saver(max_to_keep=50)

    MAIN_DQN_VARS = tf.trainable_variables(scope='mainDQN')
    TARGET_DQN_VARS = tf.trainable_variables(scope='targetDQN')

    # Scalar summaries for tensorboard: loss, average reward and evaluation score
    with tf.name_scope('Performance'):
        LOSS_PH = tf.placeholder(tf.float32, shape=None, name='loss_summary')
        LOSS_SUMMARY = tf.summary.scalar('loss', LOSS_PH)

        REWARD_PH = tf.placeholder(tf.float32, shape=None, name='reward_summary')
        REWARD_SUMMARY = tf.summary.scalar('reward', REWARD_PH)

        AVG_SIGMA_PH = tf.placeholder(tf.float32, shape=None, name='avg_sigma_summary')
        AVG_SIGMA_SUMMARY = tf.summary.scalar('avg_sigma', AVG_SIGMA_PH)
        MIN_SIGMA_PH = tf.placeholder(tf.float32, shape=None, name='min_sigma_summary')
        MIN_SIGMA_SUMMARY = tf.summary.scalar('min_sigma', MIN_SIGMA_PH)
        MAX_SIGMA_PH = tf.placeholder(tf.float32, shape=None, name='max_sigma_summary')
        MAX_SIGMA_SUMMARY = tf.summary.scalar('max_sigma', MAX_SIGMA_PH)
        ACTION_DIFF_PH = tf.placeholder(tf.float32, shape=None, name='diff_action_summary')
        ACTION_DIFF_SUMMARY = tf.summary.scalar('diff_action', ACTION_DIFF_PH)


    PERFORMANCE_SUMMARIES = tf.summary.merge([LOSS_SUMMARY, REWARD_SUMMARY, AVG_SIGMA_SUMMARY, MIN_SIGMA_SUMMARY, MAX_SIGMA_SUMMARY, ACTION_DIFF_SUMMARY])

    # Histogramm summaries for tensorboard: parameters
    with tf.name_scope('Parameters'):
        ALL_PARAM_SUMMARIES = []
        for var in MAIN_DQN_VARS:
            with tf.name_scope('mainDQN/'):
                MAIN_DQN_KERNEL = tf.summary.histogram(str(var.name).replace("/", "_"), tf.reshape(var, shape=[-1]))
            ALL_PARAM_SUMMARIES.extend([MAIN_DQN_KERNEL])
    PARAM_SUMMARIES = tf.summary.merge(ALL_PARAM_SUMMARIES)

def train(session, my_replay_memory, explore_exploit_sched, update_networks, episode_start, frame_number_start):
    """
    Trains a DQN agent
    Args:
        session: A tensorflow sesson object
        replay_memory: A ReplayMemory object
        explore_exploit_sched: An ExplorationExploitationScheduler object, Determines an action according to an epsilon greedy/noisy strategy
        update_networks: A TargetNetworkUpdater object, updates the target network periodically
        episode_start: Integer, episodes elapsed before this function is called
        frame_number_start: Integer, frames elapsed before this function is called
    """
    with session as sess:
        frame_number = frame_number_start
        rewards = []
        diff_action_list = []
        loss_list = []
        episodes = episode_start
        max_sigmas_to_keep = 10
        while frame_number < MAX_FRAMES:
            epoch_frame = 0
            while epoch_frame < SAVE_FREQUENCY:
                terminal_life_lost = atari.reset(sess)
                episode_reward_sum = 0
                diff_action = 0.0
                episode_step = 0
                small_sigmas = []
                large_sigmas = []
                min_sigma_frames = []
                min_sigma_states = []
                max_sigma_frames = []
                max_sigma_states = []
                all_sigmas = []
                for _ in range(MAX_EPISODE_LENGTH):
                    curr_frame = atari.curr_frame
                    currr_state = atari.state
                    if NOISY :
                        feed_dict = {}
                        if ADD_Q_VALUES_TO_PERTURB :
                            feed_dict[explore_exploit_sched.DQN.q_val_input] = np.zeros([1, explore_exploit_sched.DQN.n_actions])
                        action_no_noise, _, q_values = explore_exploit_sched.get_action(sess, frame_number, atari.state, no_noise=True, other_args=feed_dict)

                    if ADD_Q_VALUES_TO_PERTURB and NOISY :
                        feed_dict  = {explore_exploit_sched.DQN.q_val_input : q_values}
                    else :
                        feed_dict = {}
                    action, sigma, _ = explore_exploit_sched.get_action(sess, frame_number, atari.state, other_args=feed_dict)
                    episode_step +=1
                    min_sigma_frames, max_sigma_frames, small_sigmas, large_sigmas, max_sigma_states, min_sigma_states = update_min_max_frame(all_sigmas, sigma, small_sigmas, max_sigmas_to_keep, min_sigma_frames, curr_frame, min_sigma_states, currr_state, large_sigmas, max_sigma_frames, max_sigma_states)
                    if NOISY and action != action_no_noise:
                        diff_action +=1
                    processed_new_frame, reward, terminal, terminal_life_lost, new_frame = atari.step(sess, action)  
                    frame_number += 1
                    epoch_frame += 1
                    episode_reward_sum += reward
                    clipped_reward = clip_reward(reward)
                    
                    my_replay_memory.add_experience(action=action, 
                                                    frame=processed_new_frame[:, :, 0],
                                                    reward=clipped_reward, 
                                                    terminal=terminal_life_lost)
                    
                    if terminal and (episodes+1) %10 == 0:
                        small_sigmas = []
                        large_sigmas = []
                        min_sigma_frames = []
                        max_sigma_frames = []
                        max_sigma_states = []
                        min_sigma_states = []

                    if frame_number % UPDATE_FREQ == 0 and frame_number > REPLAY_MEMORY_START_SIZE:
                        loss_q = learn(sess, my_replay_memory, MAIN_DQN, TARGET_DQN, BS, DISCOUNT_FACTOR,frame_number)
                        loss_list.append(loss_q)

                    if frame_number % NETW_UPDATE_FREQ == 0 and frame_number > REPLAY_MEMORY_START_SIZE:
                        update_networks(sess) 
                    
                    if terminal:
                        terminal = False
                        small_sigmas = []
                        large_sigmas = []
                        min_sigma_frames = []
                        max_sigma_frames = []
                        min_sigma_states = []
                        break

                rewards.append(episode_reward_sum)
                diff_action_list.append(float(diff_action)/episode_step)
                episodes +=1
                if len(rewards) > 100 :
                    rewards.pop(0)
                
                if episodes % 10 == 0:
                    print("LOSS : {} ".format(np.mean(loss_list)))
                    if frame_number > REPLAY_MEMORY_START_SIZE:
                        summ = sess.run(PERFORMANCE_SUMMARIES, 
                                        feed_dict={LOSS_PH:np.mean(loss_list), 
                                                   REWARD_PH:np.mean(rewards[-100:]),
                                                   ACTION_DIFF_PH : np.mean(diff_action_list),
                                                   AVG_SIGMA_PH: np.mean(all_sigmas),
                                                   MIN_SIGMA_PH: np.min(all_sigmas),
                                                   MAX_SIGMA_PH:np.max(all_sigmas)})
                        if NOISY :
                            print("Effective exploration : {} ".format(np.mean(diff_action_list)))
                        diff_action_list = []
                        SUMM_WRITER.add_summary(summ, frame_number)
                        loss_list = []

                    # Histogramm summaries for tensorboard
                    summ_param = sess.run(PARAM_SUMMARIES)
                    SUMM_WRITER.add_summary(summ_param, frame_number)
                    print("Episodes {}, Frames {}, Mean Reward {} Sigma : Average {} , MIN {} , MAX {}, Exploration {} ".format(episodes, 
                                                                                                                                frame_number, 
                                                                                                                                np.mean(rewards[-100:]), 
                                                                                                                                np.mean(np.abs(all_sigmas)), 
                                                                                                                                np.min(all_sigmas), 
                                                                                                                                np.max(all_sigmas), 
                                                                                                                                explore_exploit_sched.eps))
                    with open(PATH + 'rewards_'+ RUNID+ '.dat', 'a') as reward_file:
                        print(episodes, frame_number, 
                              np.mean(rewards[-100:]),np.mean(np.abs(all_sigmas)), np.min(all_sigmas), np.max(all_sigmas), explore_exploit_sched.eps, file=reward_file)
            
            terminal = True
            
            #Save the network parameters and buffer
            saver.save(sess, PATH+'/my_model', global_step=frame_number)
            print("Saved model at frame {}".format(frame_number) )
            if SAVE_BUFFER :
                my_replay_memory.save_buffer_to_disk(PATH+'/buffer_item_')
'''
Helper functions
get_info_from_log() : Gets information about the episodes elapsed, frames elapsed and score from the log file
load_model() : Loads a model from the mentioned checkpoint
populateExperienceReplayBuffer() : Loads the replay buffer with transitions from the loaded model
'''
def get_info_from_log() :
    log_file = open(PATH + 'rewards_'+ RUNID+ '.dat', 'r')
    lineList = log_file.readlines()
    log_file.close()
    last_log = lineList[len(lineList)-1]
    last_episode, last_frame, last_score, _, _, _, _ = last_log.split(' ')
    return int(last_episode) + 10, int(last_frame) + 1, float(last_score)


def load_model(sess, name):
    imported_graph = tf.train.import_meta_graph(tf.train.latest_checkpoint(PATH)+'.meta')
    imported_graph.restore(sess,name)
    episode_start, frame_number_start, score = get_info_from_log()
    print('Loaded model successfully : ' ,name)
    print('Episode_start {} frame_start {} score {}'.format(episode_start, frame_number_start, score))    
    return episode_start, frame_number_start

def populateExperienceReplayBuffer(session, my_replay_memory, explore_exploit_sched, buffer_size):
    buffer_frame_number = 0
    pbar = tqdm(total=buffer_size)
    while buffer_frame_number < buffer_size :
        terminal_life_lost = atari.reset(session)
        for _ in range(MAX_EPISODE_LENGTH):
            if NOISY :
                feed_dict = {}
                if ADD_Q_VALUES_TO_PERTURB :
                    feed_dict[explore_exploit_sched.DQN.q_val_input] = np.zeros([1, explore_exploit_sched.DQN.n_actions])
                _, _, q_values = explore_exploit_sched.get_action(session, buffer_frame_number, atari.state, no_noise=True, other_args=feed_dict)

            if ADD_Q_VALUES_TO_PERTURB and NOISY :
                feed_dict  = {explore_exploit_sched.DQN.q_val_input : q_values}
            else :
                feed_dict = {}
            action, _, _ = explore_exploit_sched.get_action(session, buffer_frame_number, atari.state, other_args=feed_dict)
            processed_new_frame, reward, terminal, terminal_life_lost, _ = atari.step(session, action)  
            buffer_frame_number +=1
            pbar.update(1)
            clipped_reward = clip_reward(reward)
            my_replay_memory.add_experience(action=action, 
                                        frame=processed_new_frame[:, :, 0],
                                        reward=clipped_reward, 
                                        terminal=terminal_life_lost)
            if terminal:
                terminal = False
                break
    pbar.close()
    return my_replay_memory

#Train an agent

if TRAIN:
    frame_number_start = 0
    episode_start = 0
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    session = tf.Session(config=config)
    
    my_replay_memory = ReplayMemory(size=MEMORY_SIZE, batch_size=BS) 
    update_networks = TargetNetworkUpdater(MAIN_DQN_VARS, TARGET_DQN_VARS)
    if NO_EPS :
        explore_exploit_sched = ExplorationExploitationScheduler(
        MAIN_DQN, atari.env.action_space.n, 
        replay_memory_start_size=REPLAY_MEMORY_START_SIZE, 
        max_frames=MAX_FRAMES, eps_initial=0.0, eps_final=0.0, eps_final_frame=0.0, cutoff_frame=EXP_STOP,
        )
    elif GREEDY : 
        explore_exploit_sched = ExplorationExploitationScheduler(
            MAIN_DQN, atari.env.action_space.n, 
            replay_memory_start_size=REPLAY_MEMORY_START_SIZE, 
            max_frames=MAX_FRAMES, eps_initial=0.0, eps_final=0.0, eps_final_frame=0.0)
    elif EPS_G_SCHEDULE:
        explore_exploit_sched = ExplorationExploitationScheduler(
        MAIN_DQN, atari.env.action_space.n,
        replay_memory_start_size=REPLAY_MEMORY_START_SIZE,
        max_frames=MAX_FRAMES, eps_final=0.03)
    else :
        explore_exploit_sched = ExplorationExploitationScheduler(
        MAIN_DQN, atari.env.action_space.n, 
        replay_memory_start_size=REPLAY_MEMORY_START_SIZE,eps_initial=1.0, eps_final=0.1, eps_final_frame=0.1, 
        max_frames=MAX_FRAMES)

    if LOAD_BUFFER :
        my_replay_memory.load_buffer_from_disk(PATH+'/buffer_item_')

    if LOAD_MODEL :
        tf.reset_default_graph()
        episode_start, frame_number_start = load_model(session, tf.train.latest_checkpoint(PATH))
        if not LOAD_BUFFER : #Buffer already loaded, load model now
            my_replay_memory = populateExperienceReplayBuffer(session, my_replay_memory, explore_exploit_sched, LOAD_BUFFER_SIZE)

        print("Experience Replay Memory Loaded : # of examples loaded :{}".format(my_replay_memory.count))
    else :
        session.run(init)
    train(session, my_replay_memory, explore_exploit_sched, update_networks, episode_start, frame_number_start)

#  Evaluate a model
if not TRAIN:
    if NO_EPS :
        explore_exploit_sched = ExplorationExploitationScheduler(
        MAIN_DQN, atari.env.action_space.n, 
        replay_memory_start_size=REPLAY_MEMORY_START_SIZE, 
        max_frames=MAX_FRAMES, eps_initial=0.0, eps_final=0.0, eps_final_frame=0.0
        )
    elif GREEDY : 
        explore_exploit_sched = ExplorationExploitationScheduler(
            MAIN_DQN, atari.env.action_space.n, 
            replay_memory_start_size=REPLAY_MEMORY_START_SIZE, 
            max_frames=MAX_FRAMES, eps_initial=0.0, eps_final=0.0, eps_final_frame=0.0)
    else :
        explore_exploit_sched = ExplorationExploitationScheduler(
            MAIN_DQN, atari.env.action_space.n, 
            replay_memory_start_size=REPLAY_MEMORY_START_SIZE, 
            max_frames=MAX_FRAMES)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    dir_path = RUNID + '_output/'
    with tf.Session(config=config) as sess:
        saver = tf.train.import_meta_graph(tf.train.latest_checkpoint(dir_path)+'.meta')
        saver.restore(sess,tf.train.latest_checkpoint(dir_path))
        print('Loaded model at ', tf.train.latest_checkpoint(dir_path))
        steps = 0
        eval_rewards=[]
        while steps < ARGS.EVAL_STEPS :
            terminal_life_lost = atari.reset(sess, evaluation = True)
            episode_reward_sum = 0
            while True:
                if ADD_Q_VALUES_TO_PERTURB and NOISY :
                    feed_dict = {}
                    feed_dict[explore_exploit_sched.DQN.q_val_input] = np.zeros([1, explore_exploit_sched.DQN.n_actions])
                    action_no_noise, _, q_values = explore_exploit_sched.get_action(sess, 0, atari.state, no_noise=True, other_args=feed_dict)
                    feed_dict  = {explore_exploit_sched.DQN.q_val_input : q_values}
                else :
                    feed_dict = {}
                action, _, _ = explore_exploit_sched.get_action(sess, 0, atari.state, other_args=feed_dict)

                processed_new_frame, reward, terminal, terminal_life_lost, new_frame = atari.step(sess, action)
                steps +=1
                episode_reward_sum += reward
                if terminal == True:
                    break

            eval_rewards.append(episode_reward_sum)
            
            if eval_rewards :
                eval_score = np.mean(eval_rewards)
                first_episode_return = eval_rewards[0]
            else :
                first_episode_return = eval_score = episode_reward_sum
        if ARGS.EPS_GREEDY_AGENT :
            print('Average Score  : {}'.format(eval_score))
        else :
            print('Average Score with noise injection : {}'.format(eval_score))

        if not ARGS.EPS_GREEDY_AGENT :
            steps = 0
            eval_rewards=[]
            while steps < ARGS.EVAL_STEPS :
                terminal_life_lost = atari.reset(sess, evaluation = True)
                episode_reward_sum = 0
                while True:
                    if ADD_Q_VALUES_TO_PERTURB and NOISY :
                        feed_dict = {}
                        feed_dict[explore_exploit_sched.DQN.q_val_input] = np.zeros([1, explore_exploit_sched.DQN.n_actions])
                        action, _, _ = explore_exploit_sched.get_action(sess, 0, atari.state, no_noise=True, other_args=feed_dict)
                    else :
                        action, _, _ = explore_exploit_sched.get_action(sess, 0, atari.state, no_noise=True)

                    processed_new_frame, reward, terminal, terminal_life_lost, new_frame = atari.step(sess, action)
                    steps +=1
                    episode_reward_sum += reward
                    if terminal == True:
                     break

                eval_rewards.append(episode_reward_sum)
            
                if eval_rewards :
                    eval_score = np.mean(eval_rewards)
                    first_episode_return = eval_rewards[0]
                else :
                    first_episode_return = eval_score = episode_reward_sum
            print('Average Score without noise injection : {}'.format(eval_score))