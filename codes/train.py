import torch
import numpy as np
from TD3 import TD3
from utils import ReplayBuffer
from env import Environ
import os
import matplotlib.pyplot as plt
import sys
import argparse

parser = argparse.ArgumentParser(description='PyTorch Density Function Training')

# Checkpoints
parser.add_argument('--env_name', default='', type=str)
parser.add_argument('--gpu_id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--gamma', type=float, default=1.0, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--maxstep', type=int, default=50, help='Number of steps.')
parser.add_argument('--maxepisode', type=int, default=1000, help='Number of episodes.')
parser.add_argument('--noise', type=float, default=0.1, help='noise')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
print(state)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

# Save the log
class Logger(object):
    def __init__(self, file_name="Default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, "w+")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def train():
    ######### Hyperparameters #########
    env_name = args.env_name #"mdenvironment_beta0p1_newreward"
    log_interval = 1           # print avg reward after interval
    random_seed = 0
    gamma = 0.99                # discount for future rewards
    batch_size = 100            # num of transitions sampled from replay buffer
    lr = 0.001
    exploration_noise = args.noise #0.1
    polyak = 0.995              # target policy update parameter (1-tau)
    policy_noise = 0.2          # target policy smoothing noise
    noise_clip = 0.5
    policy_delay = 2            # delayed policy updates parameter
    max_episodes = args.maxepisode        # max num of episodes
    max_timesteps = args.maxstep      # max timesteps in one episode
    directory = "./preTrained/{}".format(env_name) # save trained models
    if not os.path.exists(directory):
        os.mkdir(directory)
    filename = "TD3_{}_{}".format(env_name, random_seed)

    with open(os.path.join(directory, "Config.txt"), 'w+') as f:
        for (k, v) in args._get_kwargs():
            f.write(k + ' : ' + str(v) + '\n')

    log_file_name = os.path.join(directory, "output.log")
    # Save log information
    sys.stdout = Logger(log_file_name)

    figdirectory = "./fig_{}".format(env_name)
    if not os.path.exists(figdirectory):
        os.mkdir(figdirectory)
    ###################################
    
    env = Environ()
    state_dim = 2
    action_dim = 1
    max_action = 1
    
    policy = TD3(lr, state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer()
    
    if random_seed:
        print("Random Seed: {}".format(random_seed))
        # env.seed(random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    
    # logging variables:
    avg_reward = 0
    ep_reward = 0
    log_f = open(os.path.join(directory, "log.txt"),"w+")
    
    # training procedure:
    for episode in range(1, max_episodes+1):
        state = env.reset()
        for t in range(max_timesteps):
            # select action and add exploration noise:
            action = policy.select_action(state)
            action = action + np.random.normal(0, exploration_noise, size=action.shape)
            action = action.clip(env.low, env.high)
            
            # take action in env:
            next_state, reward, done, _ = env.step(action)
            
            replay_buffer.add((state, action, reward, next_state, float(done)))
            state = next_state
            
            avg_reward += reward
            ep_reward += reward
            
            # if episode is done then update policy:
            if done or t==(max_timesteps-1):
                policy.update(replay_buffer, t, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay)
                break

        # logging updates:
        log_f.write('{},{}\n'.format(episode, ep_reward))
        log_f.flush()
        ep_reward = 0
        
        
        filename = "TD3_{}_{}_ep{}".format(env_name, random_seed, episode)
        if episode % 20==0:
            policy.save(directory, filename)
        
        # print avg reward every log interval:
        if episode % log_interval == 0:
            avg_reward = int(avg_reward / log_interval)
            print("Episode: {}\tAverage Reward: {}".format(episode, avg_reward))
            avg_reward = 0


if __name__ == '__main__':
    train()
    
