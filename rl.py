# %%
import pandas as pd
import numpy as np
import plotly.express as px
from tqdm import tqdm
from modules.funcs import (clean_up_relevant_data,
                           get_price_dependency)

import sklearn.pipeline
import sklearn.preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import AdaBoostRegressor

import gymnasium as gym
from gymnasium import spaces
import pickle

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
        state[1:4] = quantizer.transform([state[1:4]])[0]
        return (state)

    def update(self, s, a, r, sp, upd, disc):
        """ Updates the policy table """
        s_q = self.quantize_state(s)
        sp_q = self.quantize_state(sp)

        # Gets Values
        old_est = self.q_val[s_q[0], s_q[1], s_q[2], s_q[3], a]
        next_est = np.max(self.q_val[sp_q[0], sp_q[1], sp_q[2], sp_q[3]])

        # Updates
        self.q_val[s_q[0], s_q[1], s_q[2], s_q[3]] = (1-upd) * old_est + upd * (r + disc * next_est - old_est)

    def get_action(self, state):
        """ Gets the action of the estimator """
        state_q = self.quantize_state(state)
        action = np.argmax(self.q_val[state_q[0], state_q[1],
                                      state_q[2], state_q[3]])
        return action


class QL_func_estimator():
    """Value Function Linear Approximator"""

    def __init__(self, env, tranformation, inp_model):
        self.models = []
        self.X_tr = []
        self.Y_tr = []
        self.transformer = tranformation
        self.actions = env.action_space.n
        for _ in range(env.action_space.n):
            model = inp_model()
            model.fit([self.featurize_state(env.reset())], [0])
            self.models.append(model)
            self.X_tr.append([])
            self.Y_tr.append([])

    def featurize_state(self, state):
        state = list(state.values())
        transformed = self.transformer.transform([state])
        return transformed[0]

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
            predictions = np.array([self.models[i].predict([self.featurize_state(s)]) for i in range(self.actions)])
            return predictions.reshape(-1)

    def save_data(self, s, a, r, sn, disc):
        """Saves data for later refitting of the data"""
        nest_state_val = np.max(self.predict(sn))
        self.X_tr[a].append(self.featurize_state(s))
        self.Y_tr[a].append(r + disc * nest_state_val)

    def update(self, upd):
        """Updates the estimator given currently saved data"""
        for a in range(self.actions):
            self.models[a].fit(self.X_tr[a], self.Y_tr[a])

            # self.X_tr[a] = []
            # self.Y_tr[a] = []

            idx = np.random.randint(len(self.X_tr[a]), size=int(len(self.X_tr[a]) * (1-upd)))
            self.X_tr[a] = np.array(self.X_tr[a] )
            self.Y_tr[a] = np.array(self.Y_tr[a] )

            self.X_tr[a] = self.X_tr[a][idx]
            self.Y_tr[a] = self.Y_tr[a][idx]

            self.X_tr[a] = list(self.X_tr[a] )
            self.Y_tr[a] = list(self.Y_tr[a] )

    def get_action(self, observation):
        """Returns the best action given the current policy value"""
        q_values = self.predict(observation)
        best_action = np.argmax(q_values)
        return best_action


#%%
if __name__ == '__main__':

    # %% Reinforcement Learning
    # #### NAIVE Q LEARNING #####
    # # Setup
    # env = MarkdownEnv(PROB_DATA)

    # # Prepare Transformation Pipelines
    # sample_vals = np.array([list(env.state_space.sample().values()) for x in range(100000)])
    # quantizer = sklearn.preprocessing.KBinsDiscretizer(n_bins=10, encode='ordinal')
    # quantizer.fit(sample_vals[:, 1:4])

    # # Policy Estimator
    # ql_est = QL_est(env, quantizer)

    # # Optimization
    # rewards = []
    # exp = 0.5
    # disc = 0.95
    # upd = 0.05
    # epochs = 100000
    # for i in tqdm(range(epochs)):

    #     state = env.reset()
    #     terminated = False

    #     while not terminated:

    #         # Exploration
    #         if np.random.uniform(0, 1) < exp:
    #             action = env.action_space.sample()
    #         # Exploitation
    #         else:
    #             action = ql_est.get_action(state)

    #         # Updates Policy
    #         new_state, reward, terminated, _ = env.step(action)
    #         ql_est.update(state, action, reward, new_state, upd, disc)

    #         # Update State
    #         state = new_state

    #     rewards.append(reward)

    # rew_ma = pd.DataFrame(rewards).rolling(int(epochs*0.02)).mean()
    # fig = px.line(rew_ma)
    # fig.update_traces(line={'width': 1})
    # fig.show(renderer='browser')

    # %%
    ##### APPROXIMATION Q LEARNING #####
    # Setup
    env = MarkdownEnv(PROB_DATA)
    _action_to_price = {0: 60, 1: 54, 2: 48, 3: 36, }
    _price_to_action = {60: 0, 54: 1, 48: 2, 36: 3}

    # Prepare Transformation Pipelines
    sample_vals = np.array([list(env.state_space.sample().values()) for x in range(100000)])
    cat_pipe = Pipeline([ 
        ('encode', sklearn.preprocessing.OneHotEncoder(drop='first')) 
    ])
    num_pipe = Pipeline([ 
        ('transform', sklearn.preprocessing.RobustScaler()), 
    ])
    pre_pipe = ColumnTransformer([ 
        ("categorical", cat_pipe, [0]),
        ("numerical", num_pipe, [1,2,3]),
    ])
    feature_pipe = Pipeline([ 
        ('polynomial_features', sklearn.preprocessing.PolynomialFeatures()) 
    ])
    full_pipe = Pipeline([ 
        ('preprocessing', pre_pipe),
        ('transformation', feature_pipe) 
    ])
    full_pipe.fit(sample_vals)

    # Policy Estimator
    qlf_est = QL_func_estimator(env, full_pipe, AdaBoostRegressor)
    export_file = 'rl_approx_ada_tr'

    # Optimization
    rewards_epoch = []
    rewards_all = []
    exp = 0.975
    disc = 0.95
    epochs = 250
    upd = 0.05
    minibatch = 32
    for i in range(epochs):

        # Generate Minibatch Data
        rewards_mb = []
        for ii in range(minibatch):

            state = env.reset()
            terminated = False

            while not terminated:

                # Exploration
                if np.random.uniform(0, 1) < (exp**i):
                    action = env.action_space.sample()
                # Exploitation
                else:
                    action = qlf_est.get_action(state)

                # Updates Policy
                new_state, reward, terminated, _ = env.step(action)
                qlf_est.save_data(state, action, reward, new_state, disc)

                # Update State
                state = new_state
                

            rewards_mb.append(reward)
            rewards_all.append(reward)

        rewards_epoch.append(np.mean(rewards_mb))

        # Update Model with New Minibatch
        print(f"Epoch {i+1}: {rewards_epoch[-1]}")
        qlf_est.rev = rewards_epoch[-1]
        with open(f'data/{export_file}_{i+1}.pkl', 'wb') as outp:
            pickle.dump(qlf_est, outp, pickle.HIGHEST_PROTOCOL)
        qlf_est.update(upd)


    # %%
    # fig = px.line(rewards_epoch)
    # fig.update_traces(line={'width': 1})
    # fig.show(renderer='browser')

    # # %%
    # rew_ma = pd.DataFrame(rewards_all).rolling(int(len(rewards_all)*0.05)).mean()
    # fig = px.line(rewards_all)
    # fig.update_traces(line={'width': 1})
    # fig.show(renderer='browser')
    # %%
