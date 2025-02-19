# -*- coding: utf-8 -*-
from __future__ import division
import bz2
from datetime import datetime
import os
import pickle
import json 

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
import random

from agent import Agent
from env import Env
from memory import ReplayMemory
from test import test
from rtpt import RTPT
from parsers import parser

# Setup
args = parser.parse_args()



# Initialize TensorBoard writer
log_dir = os.path.join("logs", args.game)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

chkpt_dir = os.path.join(log_dir, 'checkpoints')
os.makedirs(chkpt_dir, exist_ok=True)



model_desc = args.algo + "_" + args.act_f + "_" + str(args.seed)
logging_dir = os.path.join(log_dir, model_desc)
os.makedirs(logging_dir, exist_ok=True)
# Save args as a JSON file
args_dict = vars(args)
args_json_path = os.path.join(log_dir, f'{model_desc}_args.json')
memory_path = os.path.join(chkpt_dir, f'{model_desc}_memory.pkl')
if args.memory:
  memory_path = args.memory
with open(args_json_path, 'w') as f:
    json.dump(args_dict, f, indent=4)
print(" " * 26 + "Options")
for k, v in vars(args).items():
    print(" " * 26 + k + ": " + str(v))
  
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available() and not args.disable_cuda:
  args.device = torch.device('cuda')
  torch.cuda.manual_seed(args.seed)
  torch.backends.cudnn.enabled = args.enable_cudnn
else:
  args.device = torch.device('cpu')

writer = SummaryWriter(log_dir=logging_dir)


metrics = {'steps': [], 'rewards': [], 'Qs': [], 'best_avg_reward': -float('inf')}
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available() and not args.disable_cuda:
  args.device = torch.device('cuda')
  torch.cuda.manual_seed(args.seed)
  torch.backends.cudnn.enabled = args.enable_cudnn
else:
  args.device = torch.device('cpu')


# Simple ISO 8601 timestamped logger
def log(s):
  print('[' + str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')) + '] ' + s)


def load_memory(memory_path, disable_bzip):
  # memory_path = os.path.join(memory_dir, 'mem.pkl')
  if disable_bzip:
    with open(memory_path, 'rb') as pickle_file:
      return pickle.load(pickle_file)
  else:
    with bz2.open(memory_path, 'rb') as zipped_pickle_file:
      return pickle.load(zipped_pickle_file)


def save_memory(memory, memory_path, disable_bzip):
  # memory_path = os.path.join(memory_dir, 'mem.pkl')
  if disable_bzip:
    with open(memory_path, 'wb') as pickle_file:
      pickle.dump(memory, pickle_file)
  else:
    with bz2.open(memory_path, 'wb') as zipped_pickle_file:
      pickle.dump(memory, zipped_pickle_file)


# Environment
env = Env(args)
env.train()
action_space = env.action_space()

# Agent
dqn = Agent(args, env)

# If a model is provided, and evaluate is false, presumably we want to resume, so try to load memory
if args.model is not None and not args.evaluate:
  if not args.memory:
    raise ValueError('Cannot resume training without memory save path. Aborting...')
  elif not os.path.exists(args.memory):
    raise ValueError('Could not find memory file at {path}. Aborting...'.format(path=args.memory))

  mem = load_memory(memory_path, args.disable_bzip_memory)

else:
  mem = ReplayMemory(args, args.memory_capacity)

priority_weight_increase = (1 - args.priority_weight) / (args.T_max - args.learn_start)


# Construct validation memory
val_mem = ReplayMemory(args, args.evaluation_size)
T, done = 0, True
while T < args.evaluation_size:
  if done:
    state = env.reset()

  next_state, _, done = env.step(np.random.randint(0, action_space))
  val_mem.append(state, -1, 0.0, done)
  state = next_state
  T += 1

if args.evaluate:
  dqn.eval()  # Set DQN (online network) to evaluation mode
  avg_reward, avg_Q = test(args, 0, dqn, val_mem, metrics, log_dir, evaluate=True)  # Test
  print('Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))
else:
  # Training loop
  dqn.train()
  done = True
  rtpt = RTPT('QD', f"{args.game[:3]}S{args.seed}_{args.act_f}", args.T_max//args.evaluation_interval)
  rtpt.start()
  for T in trange(1, args.T_max + 1):
    if done:
      state = env.reset()

    if T % args.replay_frequency == 0:
      dqn.reset_noise()  # Draw a new set of noisy weights

    action = dqn.act(state)  # Choose an action greedily (with noisy weights)
    next_state, reward, done = env.step(action)  # Step
    if args.reward_clip > 0:
      reward = max(min(reward, args.reward_clip), -args.reward_clip)  # Clip rewards
    mem.append(state, action, reward, done)  # Append transition to memory

    # Train and test
    if T >= args.learn_start:
      mem.priority_weight = min(mem.priority_weight + priority_weight_increase, 1)  # Anneal importance sampling weight β to 1

      if T % args.replay_frequency == 0:
        dqn.learn(mem)  # Train with n-step distributional double-Q learning

      if T % args.evaluation_interval == 0:
        dqn.eval()  # Set DQN (online network) to evaluation mode
        avg_reward, avg_Q = test(args, T, dqn, val_mem, metrics, log_dir)  # Test
        log('T = ' + str(T) + ' / ' + str(args.T_max) + ' | Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))
        dqn.train()  # Set DQN (online network) back to training mode
        rtpt.step(avg_reward)
        writer.add_scalar('avg_reward', np.mean(metrics['rewards'][-1]), T)
        writer.add_scalar('avg_Q', np.mean(metrics['Qs'][-1]), T)
        writer.add_scalar('best_avg_reward', metrics['best_avg_reward'], T)


        # If memory path provided, save it
        if args.memory is not None:
          save_memory(mem, memory_path, args.disable_bzip_memory)

      # Update target network
      if T % args.target_update == 0:
        dqn.update_target_net()

      # Checkpoint the network
      if (args.checkpoint_interval != 0) and (T % args.checkpoint_interval == 0):
        dqn.save(chkpt_dir, f'{model_desc}_checkpoint.pth')
        save_memory(mem, memory_path, args.disable_bzip_memory)

    state = next_state


pickle.dump(metrics, open(os.path.join(logging_dir, f'metrics.pkl'), 'wb'))
dqn.save(chkpt_dir, f'{model_desc}_final.pth')

writer.close()
env.close()
