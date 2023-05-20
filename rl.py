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
from sklearn.ensemble import (AdaBoostRegressor,
                              GradientBoostingRegressor)
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import SGDRegressor

import gymnasium as gym
from gymnasium import spaces
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
            "week": spaces.Discrete(self.tot_dur+1),
            "curr_price": spaces.Discrete(4),
            "sales_total": spaces.Discrete(2000),
            "sales_mean": spaces.Discrete(300),
            "sales_sd": spaces.Discrete(200),
            # "week0_price": spaces.Discrete(4),
            # "week1_price": spaces.Discrete(4),
            # "week2_price": spaces.Discrete(4),
            # "week3_price": spaces.Discrete(4),
            # "week4_price": spaces.Discrete(4),
            # "week5_price": spaces.Discrete(4),
            # "week6_price": spaces.Discrete(4),
            # "week7_price": spaces.Discrete(4),
            # "week8_price": spaces.Discrete(4),
            # "week9_price": spaces.Discrete(4),
            # "week10_price": spaces.Discrete(4),
            # "week11_price": spaces.Discrete(4),
            # "week12_price": spaces.Discrete(4),
            # "week13_price": spaces.Discrete(4),
            # "week14_price": spaces.Discrete(4),
            "week0_sales": spaces.Discrete(300),
            "week1_sales": spaces.Discrete(300),
            "week2_sales": spaces.Discrete(300),
            "week3_sales": spaces.Discrete(300),
            "week4_sales": spaces.Discrete(300),
            "week5_sales": spaces.Discrete(300),
            "week6_sales": spaces.Discrete(300),
            "week7_sales": spaces.Discrete(300),
            "week8_sales": spaces.Discrete(300),
            "week9_sales": spaces.Discrete(300),
            "week10_sales": spaces.Discrete(300),
            "week11_sales": spaces.Discrete(300),
            "week12_sales": spaces.Discrete(300),
            "week13_sales": spaces.Discrete(300),
            "week14_sales": spaces.Discrete(300),
        })

        # Action Space
        self.action_space = spaces.Discrete(4)

        # Map actions to real world implications
        self._action_to_price = {0: 60, 1: 54,
                                 2: 48, 3: 36, }

    def _get_state(self):
        return {
            "week": self.week,
            "curr_price": self.curr_price,
            "sales_total": self.sales_total,
            "sales_mean": self.sales_mean,
            "sales_sd": self.sales_sd,
            # "week0_price": spaces.Discrete(4),
            # "week1_price": spaces.Discrete(4),
            # "week2_price": spaces.Discrete(4),
            # "week3_price": spaces.Discrete(4),
            # "week4_price": spaces.Discrete(4),
            # "week5_price": spaces.Discrete(4),
            # "week6_price": spaces.Discrete(4),
            # "week7_price": spaces.Discrete(4),
            # "week8_price": spaces.Discrete(4),
            # "week9_price": spaces.Discrete(4),
            # "week10_price": spaces.Discrete(4),
            # "week11_price": spaces.Discrete(4),
            # "week12_price": spaces.Discrete(4),
            # "week13_price": spaces.Discrete(4),
            # "week14_price": spaces.Discrete(4),
            "week0_sales": self.sales_history[0] if len(self.sales_history) > 0 else -1,
            "week1_sales": self.sales_history[1] if len(self.sales_history) > 1 else -1,
            "week2_sales": self.sales_history[2] if len(self.sales_history) > 2 else -1,
            "week3_sales": self.sales_history[3] if len(self.sales_history) > 3 else -1,
            "week4_sales": self.sales_history[4] if len(self.sales_history) > 4 else -1,
            "week5_sales": self.sales_history[5] if len(self.sales_history) > 5 else -1,
            "week6_sales": self.sales_history[6] if len(self.sales_history) > 6 else -1,
            "week7_sales": self.sales_history[7] if len(self.sales_history) > 7 else -1,
            "week8_sales": self.sales_history[8] if len(self.sales_history) > 8 else -1,
            "week9_sales": self.sales_history[9] if len(self.sales_history) > 9 else -1,
            "week10_sales": self.sales_history[10] if len(self.sales_history) > 10 else -1,
            "week11_sales": self.sales_history[11] if len(self.sales_history) > 11 else -1,
            "week12_sales": self.sales_history[12] if len(self.sales_history) > 12 else -1,
            "week13_sales": self.sales_history[13] if len(self.sales_history) > 13 else -1,
            "week14_sales": self.sales_history[14] if len(self.sales_history) > 14 else -1,
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
        self.curr_price = 0
        self.sales_history = []
        self.sales_total = 0
        self.sales_mean = 0
        self.sales_sd = 0
        self.curr_inventory = self.start_inv - self.sales_total
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
        self.sales_history.append(sales)
        self.sales_total += sales
        self.sales_mean = np.mean(self.sales_history)
        self.sales_sd = np.std(self.sales_history)
        self.curr_inventory = self.start_inv - self.sales_total
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
        self.sales_history.append(sales)
        self.sales_total += sales
        self.sales_mean = np.mean(self.sales_history)
        self.sales_sd = np.std(self.sales_history)
        self.curr_inventory = self.start_inv - self.sales_total
        self.tot_revenue += sales * self._action_to_price[self.curr_price]

        # Return Data
        state = self._get_state()
        if self.week == (self.tot_dur-1):
            terminated = True
        else:
            terminated = False
        reward = self.tot_revenue if terminated else 0

        return state, reward, terminated, False


class QlFuncEstimator():
    """Value Function Linear Approximator"""

    def __init__(self, env, tranformation, model_params):
        self.models = []
        self.loss = []
        self.optim = []

        self.X_tr = []
        self.Y_tr = []
        self.transformer = tranformation
        self.actions = env.action_space.n

        for _ in range(env.action_space.n):
            model = FNN(model_params[0], model_params[1], model_params[2])
            nn.init.kaiming_normal_(model.hidden1.weight, nonlinearity='relu')
            nn.init.kaiming_normal_(model.hidden2.weight, nonlinearity='relu')
            nn.init.kaiming_normal_(model.output.weight, nonlinearity='relu')
            loss_function = nn.MSELoss()
            optimizer = optim.Adam(model.parameters())

            self.models.append(model)
            self.loss.append(loss_function)
            self.optim.append(optimizer)
            self.X_tr.append([])
            self.Y_tr.append([])

    def featurize_state(self, state):
        state = pd.DataFrame(state, index=[0])
        transformed = self.transformer.transform(state)
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
            prediction = self.models[0](torch.tensor(self.featurize_state(s), dtype=torch.float32))
            return prediction.tolist()[0]

        else:
            predictions = np.array([self.models[i](torch.tensor(self.featurize_state(s), dtype=torch.float32)).tolist()[0] for i in range(4)])
            return predictions

    def save_data(self, s, a, r, sn, disc):
        """Saves data for later refitting of the data"""
        nest_state_val = np.max(self.predict(sn))
        self.X_tr[a].append(self.featurize_state(s))
        self.Y_tr[a].append(r + disc * nest_state_val)

    def update(self):
        """Updates the estimator given currently saved data"""
        for a in range(self.actions):
            if len(self.X_tr[a]) > 0:

                # Prepare Data
                x_tr = torch.Tensor(np.array(self.X_tr[a]))
                y_tr = torch.Tensor(np.array([np.array(self.Y_tr[a])])).T

                old_err = 0
                count = 0
                while True:
                # for i in tqdm(range(500),leave=False):

                    # Forward pass
                    output = self.models[a](x_tr)
                    loss_val = self.loss[a](output, y_tr)

                    # Backward pass
                    self.optim[a].zero_grad()
                    loss_val.backward()

                    # Update weights
                    self.optim[a].step()

                    # Stop Condition
                    new_err = loss_val.item()
                    if abs(old_err - new_err) / (np.mean([old_err, new_err]) ) <= 0.001:
                        break
                    old_err = new_err
                    count += 1

                # Delete data
                self.X_tr[a] = []
                self.Y_tr[a] = []

    def get_action(self, state):
        """Returns the best action given the current policy value"""
        q_values = self.predict(state)
        best_action = np.argmax(q_values)
        return best_action


class FNN(nn.Module):
    def __init__(self, input_size, hidden, output_size):
        super(FNN, self).__init__()
        self.hidden1 = nn.Linear(input_size, hidden)
        self.hidden2 = nn.Linear(hidden, hidden)
        self.hidden3 = nn.Linear(hidden, hidden)
        self.output = nn.Linear(hidden, output_size)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x = self.dropout(x)
        x = self.hidden1(x)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.hidden3(x)
        x = self.output(x)

        return x


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
    for i in tqdm(range(1000)):
        state = env.reset()
        sample_vals.append(state)
        terminated = False
        while not terminated:
            action = env.action_space.sample()
            state, reward, terminated, _ = env.step(action)
            sample_vals.append(state)
    sample_vals = pd.DataFrame.from_records(sample_vals)

    # Prepare Transformation Pipelines
    cat_cols = ['week', 'curr_price']
    num_cols = sample_vals.columns.drop(cat_cols).to_list()
    inter_cols = ['week'] + [f"week{i}_sales" for i in range(15)]

    cat_pipe = Pipeline([('encode', sklearn.preprocessing.OneHotEncoder(drop='first'))])
    num_pipe = Pipeline([('transform', sklearn.preprocessing.RobustScaler()), ])
    inter_pipe = Pipeline([('polynomial_features', sklearn.preprocessing.PolynomialFeatures(interaction_only=True))])
    notr_pipe = ColumnTransformer([ ("categorical", cat_pipe, cat_cols),
                                ("numerical", num_pipe, num_cols),] )
    tr_pipe = ColumnTransformer([ ("categorical", cat_pipe, cat_cols),
                                ("numerical", num_pipe, num_cols), 
                                ("interaction", inter_pipe, inter_cols)] )
    notr_pipe.fit(sample_vals)
    tr_pipe.fit(sample_vals)

    # Policy Estimator (Options)
    online = True
    # qlf_est = QlFuncEstimator(env, notr_pipe, MLPRegressor, online)
    # qlf_est = QlFuncEstimator(env, notr_pipe, (35, 256, 256, 1))
    # export_file = 'rl_approx_nn_notr'
    qlf_est = QlFuncEstimator(env, tr_pipe, (172, 64, 1))
    export_file = 'rl_approx_nn_tr'
    # qlf_est = QL_func_estimator(env, tr_pipe, MLPRegressor, online)
    # export_file = 'rl_approx_nn_tr'

    # online = True
    # qlf_est = QL_func_estimator(env, notr_pipe, SGDRegressor, online)
    # export_file = 'rl_approx_sgd_notr'
    # qlf_est = QL_func_estimator(env, tr_pipe, SGDRegressor, online)
    # export_file = 'rl_approx_sgd_tr'

    # online = False
    # qlf_est = QL_func_estimator(env, notr_pipe, GradientBoostingRegressor, online)
    # export_file = 'rl_approx_gbr_notr'
    # qlf_est = QL_func_estimator(env, tr_pipe, GradientBoostingRegressor, online)
    # export_file = 'rl_approx_gbr_tr'

    # online = False
    # qlf_est = QL_func_estimator(env, notr_pipe, AdaBoostRegressor, online)
    # export_file = 'rl_approx_ada_notr'
    # qlf_est = QL_func_estimator(env, tr_pipe, AdaBoostRegressor, online)
    # export_file = 'rl_approx_ada_tr'

    print(export_file)

    # Optimization
    rewards_epoch = []
    rewards_all = []
    exp = 0.01
    disc = 0.9
    epochs = 5000
    upd = 0.05
    minibatch = 16
    for i in range(epochs):

        # Generate Minibatch Data
        rewards_mb = []
        for ii in tqdm(range(minibatch), leave=False):

            state = env.reset()
            action = 0
            terminated = False

            while not terminated:
                
                # If last action
                if action == 3:
                    action = 3
                else:
                    # Exploration
                    if np.random.uniform(0, 1) < exp:
                        # action = min(action + 1, 3)
                        action = env.action_space.sample()
                    # Exploitation
                    else:
                        action = qlf_est.get_action(state)

                # Updates Policy
                new_state, reward, terminated, _ = env.step(action)
                qlf_est.save_data(state, action, reward, new_state, disc)
                # qlf_est.update()

                # Update State
                state = new_state

            
            # qlf_est.update()
            rewards_mb.append(reward)
            rewards_all.append(reward)

        rewards_epoch.append(np.mean(rewards_mb))

        # Update Model with New Minibatch
        print(f"Epoch {i+1}: {rewards_epoch[-1]}")
        qlf_est.rev = rewards_epoch[-1]

        with open(f'data/{export_file}_{i+1}.pkl', 'wb') as outp:
            pickle.dump(qlf_est, outp, pickle.HIGHEST_PROTOCOL)

        # if not qlf_est.online:
        qlf_est.update()

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
