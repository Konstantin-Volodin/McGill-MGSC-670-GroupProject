# %%
import pandas as pd
import numpy as np
import numpy.random as npr
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.io as pio
from tqdm import tqdm
import pickle
from rl import QlFuncEstimator, FNN 
np.seterr(all="ignore")

np.seterr(all="ignore")
from modules.funcs import (validator,
                           clean_up_relevant_data,
                           get_price_dependency)
from modules.policies import (baseline_policy,
                              moving_avg_longterm,
                              moving_avg_shorterm,
                              likelihood_naive,
                              likelihood_shared_distribution,
                              likelihood_price_dependency,
                              random_policy,
                              rl_policy)

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

with open('data/rl_approx_nn_tr_2475.pkl', 'rb') as inp:
    RL_APPROX = pickle.load(inp)

# %%
# VALIDATION
res = {'seed': [], 'revenue': [], 'perf_revenue': [], 'difference': [], 'policy': [], 'param': []}
for i in tqdm(range(10)):

    # BASELINE POLICY
    ind_res = validator(PROB_DATA, i, baseline_policy, kwargs={})
    res['seed'].append(i)
    res['revenue'].append(ind_res[0])
    res['perf_revenue'].append(ind_res[1])
    res['difference'].append(ind_res[2])
    res['policy'].append('baseline')
    res['param'].append('none')

    # RANDOM POLICY
    ind_res = validator(PROB_DATA, i, random_policy, kwargs={})
    res['seed'].append(i)
    res['revenue'].append(ind_res[0])
    res['perf_revenue'].append(ind_res[1])
    res['difference'].append(ind_res[2])
    res['policy'].append('random')
    res['param'].append('none')

    # MOVING AVERAGE LONGTERM POLICY
    ind_res = validator(PROB_DATA, i, moving_avg_longterm, kwargs={'num_observation': 6})
    res['seed'].append(i)
    res['revenue'].append(ind_res[0])
    res['perf_revenue'].append(ind_res[1])
    res['difference'].append(ind_res[2])
    res['policy'].append('moving_avg_longterm')
    res['param'].append(6)

    # MOVING AVERAGE SHORTERM POLICY
    ind_res = validator(PROB_DATA, i, moving_avg_shorterm, kwargs={'num_observation': 3})
    res['seed'].append(i)
    res['revenue'].append(ind_res[0])
    res['perf_revenue'].append(ind_res[1])
    res['difference'].append(ind_res[2])
    res['policy'].append('moving_avg_shorterm')
    res['param'].append(3)

    # NAIVE LIKELIHOOD POLICY
    ind_res = validator(PROB_DATA, i, likelihood_naive, kwargs={'req_likelihood': 0.04})
    res['seed'].append(i)
    res['revenue'].append(ind_res[0])
    res['perf_revenue'].append(ind_res[1])
    res['difference'].append(ind_res[2])
    res['policy'].append('likelihood_naive')
    res['param'].append(0.04)

    # SHARED LIKELIHOOD POLICY
    ind_res = validator(PROB_DATA, i, likelihood_shared_distribution,
                        kwargs={'req_likelihood': 0.03, 'mean_dependency': MEAN_DEPENDENCY, 'sd_dependency': SD_DEPENDENCY})
    res['seed'].append(i)
    res['revenue'].append(ind_res[0])
    res['perf_revenue'].append(ind_res[1])
    res['difference'].append(ind_res[2])
    res['policy'].append('likelihood_shared')
    res['param'].append(0.03)

    # PRICE DEPENDENCY LIKELIHOOD POLICY
    ind_res = validator(PROB_DATA, i, likelihood_price_dependency, kwargs={'mean_dependency': MEAN_DEPENDENCY, 'sd_dependency': SD_DEPENDENCY})
    res['seed'].append(i)
    res['revenue'].append(ind_res[0])
    res['perf_revenue'].append(ind_res[1])
    res['difference'].append(ind_res[2])
    res['policy'].append('likelihood_price')
    res['param'].append('none')

    # REINFORCEMENT LEARNING POLICY
    ind_res = validator(PROB_DATA, i, rl_policy, kwargs={'rl_object': RL_APPROX})
    res['seed'].append(i)
    res['revenue'].append(ind_res[0])
    res['perf_revenue'].append(ind_res[1])
    res['difference'].append(ind_res[2])
    res['policy'].append(f'rl_policy')
    res['param'].append('none')


res = pd.DataFrame(res)
res.to_csv('data/validation_result.csv', index=False)


# %%
# REVENUE DISTRIBUTION
fig = px.box(res, y='revenue', facet_col='policy', color='policy',
             boxmode='overlay', points='all')
fig.show(renderer='browser')
# pio.write_image(fig, 'images/validation_revenue_distribution_box.png', scale=1, width=1800, height=900)

# DIFFERENCE DISTRIBUTION
fig = px.box(res, y='difference', facet_col='policy', color='policy',
             boxmode='overlay', points='all')
fig.show(renderer='browser')
# pio.write_image(fig, 'images/validation_difference_distribution_box.png', scale=1, width=1800, height=900)

# %%
