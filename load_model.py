import os
import random
import numpy as np
import torch
from env import Env
from agent import Agent
from parsers import parser
import matplotlib.pyplot as plt


args = parser.parse_args()


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available() and not args.disable_cuda:
  args.device = torch.device('cuda')
  torch.cuda.manual_seed(args.seed)
  torch.backends.cudnn.enabled = args.enable_cudnn
else:
  args.device = torch.device('cpu')

env = Env(args)
env.eval()

dqn = Agent(args, env)

log_dir = os.path.join("logs", args.game)
chkpt_dir = os.path.join(log_dir, 'checkpoints')
model_desc = args.algo + "_" + args.act_f + "_" + str(args.seed)

dqn.load(chkpt_dir, f'{model_desc}_final.pth')

x = torch.arange(-2, 2, 0.01).to(device=args.device)
import ipdb; ipdb.set_trace()
fig, axes = plt.subplots(2, 2, figsize=(12, 6))
axes = axes.flatten()
for act, ax in enumerate(axes):
  exec(f"y = dqn.online_net.act_f{act+1}(x).cpu().detach().numpy()")
  ax.plot(x.cpu().numpy(), y)
  ax.set_title(f"Rat {act}")
plt.show()