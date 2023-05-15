# %%
import pandas as pd
import numpy as np
import plotly.express as px
from tqdm import tqdm
import pickle

from modules.funcs import (simulator,
                           clean_up_relevant_data)
from modules.policies import (baseline_policy,
                              moving_avg_policy,
                              likelihood_naive,
                              random_policy)


import gymnasium as gym
from gymnasium import spaces

# PREPARE DATA
df = pd.read_csv('data/data.csv')
DF_DIST, DF_TRN = clean_up_relevant_data(df)
OPTIONS = [60, 54, 48, 36]
PROB_DATA = {'actions': {60: [60, 54, 48, 36],
                         54: [54, 48, 36],
                         48: [48, 36],
                         36: [36]},
             'start_inventory': 2000,
             'start_price': 60,
             'total_duration': 15}



class MarkdownEnv(gym.Env):

    def __init__(self, problem_dat):

        # Metadata
        self.prices = [0,1,2,3]
        self.start_inv = problem_dat['start_inventory']
        self.start_pc = problem_dat['start_price']
        self.tot_dur = problem_dat['total_duration']

        # State Space
        self.state_space = spaces.Dict({
            "week": spaces.Discrete(self.tot_dur), 
            "curr_price": spaces.Discrete(4),
            "curr_sales": spaces.Discrete(300), 
            "tot_sales": spaces.Discrete(self.start_inv+1)
        })

        # Action Space
        self.action_space = spaces.Discrete(4)

        # Map actions to real world implications
        self._action_to_price = {0: 60, 1: 54,
                                 2: 48,3: 36,}
        
    def _get_state(self):
        return {
            'week': self.week,
            'curr_price': self.curr_price,
            'curr_sales': self.sales,
            'tot_sales': self.tot_sales
        }

    def reset(self, seed=None):
        super().reset(seed=seed)

        # Generate Random Distribution
        self.dist = {self._action_to_price[opt]: {
            'mean': max(self.np_random.normal(DF_DIST[self._action_to_price[opt]]['mean_mean'], 
                                              DF_DIST[self._action_to_price[opt]]['mean_sd']), 0),
            'sd': max(self.np_random.normal(DF_DIST[self._action_to_price[opt]]['sd_mean'], 
                                            DF_DIST[self._action_to_price[opt]]['sd_sd']), 0),
            'sd': 1
        } for opt in self.prices}

        # Initialize 
        self.week = 1
        self.tot_sales = 0
        self.curr_price = 0
        self.tot_revenue = 0
        self.curr_inventory = self.start_inv - self.tot_sales

        # Sales this week
        sales = np.round(self.np_random.normal(
            self.dist[self._action_to_price[self.curr_price]]['mean'],
            self.dist[self._action_to_price[self.curr_price]]['sd']
        ), 0)
        sales = np.max([0, sales])
        sales = np.min([self.curr_inventory, sales])
        sales = int(sales)

        # Save Inventory Data
        self.sales = sales
        self.tot_sales += self.sales
        self.curr_inventory = self.start_inv - self.tot_sales
        self.tot_revenue += self.sales * self._action_to_price[self.curr_price]

        # Return Data
        state = self._get_state()
        return state

    def step(self, action):
        # Change based on action
        if action >= self.curr_price:
            self.curr_price = action
 
        # Sales this week
        sales = np.round(self.np_random.normal(
            self.dist[self._action_to_price[self.curr_price]]['mean'],
            self.dist[self._action_to_price[self.curr_price]]['sd']
        ), 0)
        sales = np.max([0, sales])
        sales = np.min([self.curr_inventory, sales])
        sales = int(sales)

        # Update State
        self.week += 1
        self.sales = sales
        self.tot_sales += self.sales
        self.curr_inventory = self.start_inv - self.tot_sales
        self.tot_revenue += self.sales * self._action_to_price[self.curr_price]

        # Return Data
        state = self._get_state()
        if self.week == (self.tot_dur-1):
            terminated = True
        else: 
            terminated = False
        reward = self.tot_revenue if terminated else 0


        return state, reward, terminated, False


# %% Reinforcement Learning
# Prepare Environment
env = MarkdownEnv(PROB_DATA)
q_table = np.zeros([
    env.state_space['week'].n,
    env.state_space['curr_price'].n,
    env.state_space['curr_sales'].n,
    env.state_space['tot_sales'].n,
    env.action_space.n
])
expl = 0.2
upd = 0.1

# Q Learning
rewards = []
for i in tqdm(range(5000000)):
    state = env.reset()
    terminated = False

    while not terminated:
        if np.random.uniform(0,1) < expl:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[(state['week'],
                                        state['curr_price'],
                                        state['curr_sales'],
                                        state['tot_sales'],)])
            
        # PREVIOUS VALUE
        old_estimate = q_table[state['week'], state['curr_price'], state['curr_sales'], state['tot_sales'],action]

        # GET NEW VALUE ESTIMATE
        new_state, reward, terminated, truncated  = env.step(action) 
        next_estimate = np.max(q_table[(new_state['week'],new_state['curr_price'], 
                                        new_state['curr_sales'], new_state['tot_sales'],)])
        new_estimate = ((1-upd) * old_estimate) + (upd * (reward + next_estimate - old_estimate))
        q_table[(state['week'], state['curr_price'], 
                 state['curr_sales'], state['tot_sales'], action)] = new_estimate

        # Update State
        state = new_state

    rewards.append(reward)

print(np.count_nonzero(q_table))

with open('data/q_policy.npy', 'wb') as file:
    np.save(file, q_table) 

#%%
fig = px.line(rewards)
fig.update_traces(line={'width': 0.01})
fig.show(renderer='browser')

# %% 
