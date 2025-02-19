import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
from argparse import ArgumentParser
import os

sns.set_style('whitegrid')
    
parser = ArgumentParser()
parser.add_argument('--game', type=str, default='space_invaders')

args = parser.parse_args()


rewards = []
for actf in ['relu', 'rat']:
    for seed in range(5):
        try:
            results_path = os.path.join('logs', args.game, f'Rainbow_{actf}_{seed}',  
                                        f'metrics.pkl')
            with open(results_path, 'rb') as f:
                data = pickle.load(f)
        except FileNotFoundError:
            continue
        rewards.append(np.mean(data['rewards'], 1))

    steps = data['steps']
    # Plot rewards
    m_rew = np.mean(rewards, 0)
    std_rew = np.std(rewards, 0)
    plt.plot(steps, m_rew, label=f'Rainbow_{actf}')
    plt.fill_between(steps, m_rew-std_rew, m_rew+std_rew, alpha=0.3)
plt.title(f'{args.game}')
plt.ylabel('Reward')
plt.xlabel('Step')
plt.legend()
plt.show()