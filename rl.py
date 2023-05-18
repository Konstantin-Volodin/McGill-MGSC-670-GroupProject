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
from sklearn.
from sklearn.ensemble import (AdaBoostRegressor,
                              GradientBoostingRegressor)

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
            "total_sales": spaces.Discrete(2000),
            "week": spaces.Discrete(self.tot_dur+1),
            "week0_price": spaces.Discrete(4),
            "week0_sales": spaces.Discrete(300),
            "week10_price": spaces.Discrete(4),
            "week10_sales": spaces.Discrete(300),
            "week11_price": spaces.Discrete(4),
            "week11_sales": spaces.Discrete(300),
            "week12_price": spaces.Discrete(4),
            "week12_sales": spaces.Discrete(300),
            "week13_price": spaces.Discrete(4),
            "week13_sales": spaces.Discrete(300),
            "week14_price": spaces.Discrete(4),
            "week14_sales": spaces.Discrete(300),
            "week1_price": spaces.Discrete(4),
            "week1_sales": spaces.Discrete(300),
            "week2_price": spaces.Discrete(4),
            "week2_sales": spaces.Discrete(300),
            "week3_price": spaces.Discrete(4),
            "week3_sales": spaces.Discrete(300),
            "week4_price": spaces.Discrete(4),
            "week4_sales": spaces.Discrete(300),
            "week5_price": spaces.Discrete(4),
            "week5_sales": spaces.Discrete(300),
            "week6_price": spaces.Discrete(4),
            "week6_sales": spaces.Discrete(300),
            "week7_price": spaces.Discrete(4),
            "week7_sales": spaces.Discrete(300),
            "week8_price": spaces.Discrete(4),
            "week8_sales": spaces.Discrete(300),
            "week9_price": spaces.Discrete(4),
            "week9_sales": spaces.Discrete(300),
        })

        # Action Space
        self.action_space = spaces.Discrete(4)

        # Map actions to real world implications
        self._action_to_price = {0: 60, 1: 54,
                                 2: 48, 3: 36, }

    def _get_state(self):
        return {
            "total_sales": self.tot_sales,
            "week": self.week,
            "week0_price": self.price_history[0],
            "week0_sales": self.sales_history[0],
            "week10_price": self.price_history[10],
            "week10_sales": self.sales_history[10],
            "week11_price": self.price_history[11],
            "week11_sales": self.sales_history[11],
            "week12_price": self.price_history[12],
            "week12_sales": self.sales_history[12],
            "week13_price": self.price_history[13],
            "week13_sales": self.sales_history[13],
            "week14_price": self.price_history[14],
            "week14_sales": self.sales_history[14],
            "week1_price": self.price_history[1],
            "week1_sales": self.sales_history[1],
            "week2_price": self.price_history[2],
            "week2_sales": self.sales_history[2],
            "week3_price": self.price_history[3],
            "week3_sales": self.sales_history[3],
            "week4_price": self.price_history[4],
            "week4_sales": self.sales_history[4],
            "week5_price": self.price_history[5],
            "week5_sales": self.sales_history[5],
            "week6_price": self.price_history[6],
            "week6_sales": self.sales_history[6],
            "week7_price": self.price_history[7],
            "week7_sales": self.sales_history[7],
            "week8_price": self.price_history[8],
            "week8_sales": self.sales_history[8],
            "week9_price": self.price_history[9],
            "week9_sales": self.sales_history[9],
        }

    def reset(self, seed=None):

        # Generate Random Distribution
        self.dist = {60: {'mean': max(self.np_random.normal(DF_DIST[60]['mean_mean'],
                                                            DF_DIST[60]['mean_sd']), 0),
                          'sd': max(self.np_random.normal(DF_DIST[60]['sd_mean'],
                                                          DF_DIST[60]['sd_sd']), 0)}}

        for opt in OPTIONS[1:]:
            self.dist[opt] = {'mean': self.dist[60]['mean'] * MEAN_DEPENDENCY[(60, opt)],
                              'sd': self.dist[60]['sd'] * SD_DEPENDENCY[(60, opt)], }

        # Initialize
        self.week = 0
        self.sales_history = {i: -1 for i in range(self.tot_dur)}
        self.price_history = {i: -1 for i in range(self.tot_dur)}
        self.tot_sales = 0
        self.curr_price = 0
        self.curr_inventory = self.start_inv - self.tot_sales
        self.tot_revenue = 0

        # Sales this week
        sales = np.round(self.np_random.normal(
            self.dist[self._action_to_price[self.curr_price]]['mean'],
            self.dist[self._action_to_price[self.curr_price]]['sd']
        ), 0)
        sales = np.max([0, sales])
        sales = np.min([self.curr_inventory, sales])
        sales = int(sales)

        # Save Inventory Data
        self.sales_history[self.week] = sales
        self.price_history[self.week] = 0
        self.tot_sales += sales
        self.curr_inventory = self.start_inv - self.tot_sales
        self.tot_revenue += sales * self._action_to_price[self.curr_price]

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
        self.sales_history[self.week] = sales
        self.price_history[self.week] = self.curr_price
        self.tot_sales += sales
        self.curr_inventory = self.start_inv - self.tot_sales
        self.tot_revenue += sales * self._action_to_price[self.curr_price]

        # Return Data
        state = self._get_state()
        if self.week == (self.tot_dur-1):
            terminated = True
        else:
            terminated = False
        reward = self.tot_revenue if terminated else 0

        return state, reward, terminated, False


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
            self.X_tr[a] = np.array(self.X_tr[a])
            self.Y_tr[a] = np.array(self.Y_tr[a])

            self.X_tr[a] = self.X_tr[a][idx]
            self.Y_tr[a] = self.Y_tr[a][idx]

            self.X_tr[a] = list(self.X_tr[a])
            self.Y_tr[a] = list(self.Y_tr[a])

    def get_action(self, observation):
        """Returns the best action given the current policy value"""
        q_values = self.predict(observation)
        best_action = np.argmax(q_values)
        return best_action


# %%
if __name__ == '__main__':

    # %% Reinforcement Learning
    ##### APPROXIMATION Q LEARNING #####
    # Setup
    env = MarkdownEnv(PROB_DATA)
    _action_to_price = {0: 60, 1: 54, 2: 48, 3: 36, }
    _price_to_action = {60: 0, 54: 1, 48: 2, 36: 3}

    # Generate Sample Data
    sample_vals = []
    for i in tqdm(range(10000)):
        state = env.reset()
        sample_vals.append(list(state.values()))
        terminated = False
        while not terminated:
            action = env.action_space.sample()
            state, reward, terminated, _ = env.step(action)
            sample_vals.append(list(state.values()))
    sample_vals = np.array(sample_vals)
    sample_vals.shape

    # Prepare Transformation Pipelines
    cat_pipe = Pipeline([('encode', sklearn.preprocessing.OneHotEncoder(drop='first'))])
    num_pipe = Pipeline([('transform', sklearn.preprocessing.RobustScaler()), ])
    pre_pipe = ColumnTransformer([("categorical", cat_pipe, [1]),
                                  ("numerical", num_pipe, [i for i in range(32) if i != 1]), ])
    feature_pipe = Pipeline([('polynomial_features', sklearn.preprocessing.PolynomialFeatures(interaction_only=True))])
    full_pipe = Pipeline([('preprocessing', pre_pipe),
                          #   ('transformation', feature_pipe),
                          ])
    full_pipe.fit(sample_vals)

    # Policy Estimator
    qlf_est = QL_func_estimator(env, full_pipe, GradientBoostingRegressor)
    export_file = 'rl_approx_gbr_notr'

    # Optimization
    rewards_epoch = []
    rewards_all = []
    exp = 0.1
    disc = 0.95
    epochs = 500
    upd = 0.01
    minibatch = 32
    for i in range(epochs):

        # Generate Minibatch Data
        rewards_mb = []
        for ii in range(minibatch):

            state = env.reset()
            terminated = False

            while not terminated:

                # Exploration
                if np.random.uniform(0, 1) < exp:
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
