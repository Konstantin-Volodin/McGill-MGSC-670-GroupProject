# %%
import pandas as pd
import numpy as np
import plotly.express as px
from tqdm import tqdm
from modules.funcs import (clean_up_relevant_data,
                           get_price_dependency)

import sklearn.pipeline
import sklearn.preprocessing
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler

import gymnasium as gym
from gymnasium import spaces

# PREPARE DATA
df = pd.read_csv('data/data.csv')
DF_DIST, DF_TRN = clean_up_relevant_data(df)
MEAN_DEPENDENCY, SD_DEPENDENCY = get_price_dependency(pd.DataFrame(DF_DIST))
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
        self.prices = [0, 1, 2, 3]
        self.start_inv = problem_dat['start_inventory']
        self.start_pc = problem_dat['start_price']
        self.tot_dur = problem_dat['total_duration']

        # State Space
        self.state_space = spaces.Dict({
            "curr_price": spaces.Discrete(4),
            "curr_sales": spaces.Discrete(300),
            "tot_sales": spaces.Discrete(self.start_inv+1),
            "week": spaces.Discrete(self.tot_dur),
        })

        # Action Space
        self.action_space = spaces.Discrete(4)

        # Map actions to real world implications
        self._action_to_price = {0: 60, 1: 54,
                                 2: 48, 3: 36, }

    def _get_state(self):
        return {
            'curr_price': self.curr_price,
            'curr_sales': self.sales,
            'tot_sales': self.tot_sales,
            'week': self.week,
        }

    def reset(self, seed=None):
        # super().reset(seed=seed)

        # Generate Random Distribution
        self.dist = {60: {'mean': max(self.np_random.normal(DF_DIST[60]['mean_mean'],
                                                            DF_DIST[60]['mean_sd']), 0),
                          'sd': max(self.np_random.normal(DF_DIST[60]['sd_mean'],
                                                          DF_DIST[60]['sd_sd']), 0)}}

        for opt in OPTIONS[1:]:
            self.dist[opt] = {'mean': self.dist[60]['mean'] * MEAN_DEPENDENCY[(60, opt)],
                              'sd': self.dist[60]['sd'] * SD_DEPENDENCY[(60, opt)], }

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


class QL_est():
    """Quantization Estimation"""

    def __init__(self, env, quantizer):
        self.quantizer = quantizer
        self.q_val = np.zeros([env.state_space['curr_price'].n,
                               self.quantizer.n_bins,
                               self.quantizer.n_bins,
                               env.state_space['week'].n,
                               env.action_space.n])

    def quantize_state(self, state):
        """ Converts a state to quantized data """
        state = np.array(list(state.values()))
        state[1:3] = quantizer.transform([state[1:3]])[0]
        return (state)

    def update(self, s, a, r, sp, upd, disc):
        """ Updates the policy table """
        s_q = self.quantize_state(s)
        sp_q = self.quantize_state(sp)

        # Gets Values
        old_est = self.q_val[s_q[0], s_q[1], s_q[2], s_q[3], a]
        next_est = np.max(self.q_val[sp_q[0], sp_q[1], sp_q[2], sp_q[3]])

        # Updates
        self.q_val[s_q[0], s_q[1], s_q[2], s_q[3]] = old_est + upd * (r + disc * next_est - old_est)

    def get_action(self, state):
        """ Gets the action of the estimator """
        state_q = self.quantize_state(state)
        action = np.argmax(self.q_val[state_q[0], state_q[1],
                                      state_q[2], state_q[3]])
        return action


class QL_func_estimator():
    """Value Function Linear Approximator"""

    def __init__(self):
        self.models = []
        for _ in range(env.action_space.n):
            model = SGDRegressor(learning_rate="constant")
            model.partial_fit([self.featurize_state(env.reset())], [0])
            self.models.append(model)

    def featurize_state(self, state):
        state = list(state.values())
        scaled = scaler.transform([state])
        featurized = featurizer.transform(scaled)
        return featurized[0]

    def predict(self, s, a=None):
        """
        Makes value function predictions.

        Args:
            s: state to make a prediction for
            a: (Optional) action to make a prediction for

        Returns
            If an action a is given this returns a single number as the prediction.
            If no action is given this returns a vector or predictions for all actions
            in the environment where pred[i] is the prediction for action i.

        """
        if a is not None:
            prediction = self.models[a].predict([self.featurize_state(s)])
            return prediction[0]

        else:
            predictions = np.array([self.models[i].predict([self.featurize_state(s)]) for i in range(env.action_space.n)])
            return predictions.reshape(-1)

    def update(self, s, a, y):
        """
        Updates the estimator parameters for a given state and action towards
        the target y.
        """
        self.models[a].partial_fit([self.featurize_state(s)], [y])

    def get_action(self, observation):
        """
        Returns the best action given the current policy value
        """
        q_values = self.predict(observation)
        best_action = np.argmax(q_values)
        return best_action


# %% Reinforcement Learning
##### NAIVE Q LEARNING #####
# Setup
env = MarkdownEnv(PROB_DATA)

# Prepare Transformation Pipelines
sample_vals = np.array([list(env.state_space.sample().values()) for x in range(100000)])
quantizer = sklearn.preprocessing.KBinsDiscretizer(n_bins=7, encode='ordinal')
quantizer.fit(sample_vals[:, 1:3])

# Policy Estimator
nv_est = QL_est(env, quantizer)

# Optimization
rewards = []
exp = 0.1
disc = 0.95
upd = 0.1
epochs = 5000
for i in tqdm(range(epochs)):

    state = env.reset()
    terminated = False

    while not terminated:

        # Exploration
        if np.random.uniform(0, 1) < exp:
            action = env.action_space.sample()
        # Exploitation
        else:
            action = nv_est.get_action(state)

        # Updates Policy
        new_state, reward, terminated, _ = env.step(action)
        nv_est.update(state, action, reward, new_state, upd, disc)

        # Update State
        state = new_state

    rewards.append(reward)

# %%
print(np.count_nonzero(nv_est.q_val))
fig = px.line(rewards)
fig.update_traces(line={'width': 0.1})
fig.show(renderer='browser')

# %%

# ##### APPROXIMATION Q LEARNING #####
# # Setup
# env = MarkdownEnv(PROB_DATA)
# observation_examples = np.array([list(env.state_space.sample().values()) for x in range(10000)])
# scaler = sklearn.preprocessing.StandardScaler()
# scaler.fit(pd.DataFrame.from_records(observation_examples))
# featurizer = sklearn.pipeline.FeatureUnion([
#         ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
#         ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
#         ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
#         ("rbf4", RBFSampler(gamma=0.5, n_components=100))])
# featurizer.fit(scaler.transform(pd.DataFrame.from_records(observation_examples)))
# estimator = QL_func_estimator()
# policy = qlearn_policy(estimator)

# # Optimization
# rewards = []
# exploration = 0.2
# discount_factor = 1
# epochs = 100000
# for i in tqdm(range(10000)):

#     policy = qlearn_policy(estimator)
#     state = env.reset()
#     terminated = False

#     while not terminated:

#         # Exploration
#         if np.random.uniform(0,1) < exploration:
#             action = env.action_space.sample()
#         # Exploitation
#         else:
#             # action = np.argmax(q_table[(state['week'],
#             #                             state['curr_price'],
#             #                             state['curr_sales'],
#             #                             state['tot_sales'],)])
#             action = policy(state)

#         # Perform the action -> Get the reward and observe the next state
#         new_state, reward, terminated, _ = env.step(action)
#         q_values_new_state = estimator.predict(new_state)
#         td_target = reward + discount_factor * np.max(q_values_new_state)
#         estimator.update(state, action, td_target)

#         # PREVIOUS VALUE
#         # old_estimate = q_table[state['week'], state['curr_price'], state['curr_sales'], state['tot_sales'],action]

#         # # GET NEW VALUE ESTIMATE
#         # new_state, reward, terminated, truncated  = env.step(action)
#         # next_estimate = np.max(q_table[(new_state['week'],new_state['curr_price'],
#         #                                 new_state['curr_sales'], new_state['tot_sales'],)])
#         # new_estimate = ((1-upd) * old_estimate) + (upd * (reward + next_estimate - old_estimate))
#         # q_table[(state['week'], state['curr_price'],
#         #         state['curr_sales'], state['tot_sales'], action)] = new_estimate

#         # Update State
#         state = new_state

#             # Y['value'].append(reward)

#     rewards.append(reward)

# print(np.count_nonzero(q_table))

# with open('data/q_policy.npy', 'wb') as file:
#     np.save(file, q_table)

# #%%
# fig = px.line(rewards)
# fig.update_traces(line={'width': 0.1})
# fig.show(renderer='browser')

# # %%

# %%
