# %%
import pandas as pd
import numpy as np
import numpy.random as npr
import plotly.express as px
from tqdm import tqdm

from modules.funcs import (simulator,
                           clean_up_relevant_data)
from modules.policies import (baseline_policy,
                              decrease_policy,
                              likelihood_naive,)

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


# %% FOR TESTING
distributions = {opt: {'mean': max(npr.normal(DF_DIST[opt]['mean_mean'],
                                              DF_DIST[opt]['mean_sd']), 0),
                       'sd': max(npr.normal(DF_DIST[opt]['sd_mean'],
                                            DF_DIST[opt]['sd_sd']), 0)} for opt in OPTIONS}
ind_res = simulator(PROB_DATA, distributions, likelihood_naive, kwargs={'req_likelihood': 0.5})
ind_res

# %% 100 simulations
res = {'repl': [], 'week': [], 'sales': [], 'price': [], 'policy': [], 'param': []}
for i in tqdm(range(500)):
    distributions = {opt: {'mean': max(npr.normal(DF_DIST[opt]['mean_mean'],
                                                  DF_DIST[opt]['mean_sd']), 0),
                           'sd': max(npr.normal(DF_DIST[opt]['sd_mean'],
                                                DF_DIST[opt]['sd_sd']), 0)} for opt in OPTIONS}

    # BASELINE POLICY
    ind_res = simulator(PROB_DATA, distributions, baseline_policy, kwargs={})
    res['repl'].extend([i for j in range(PROB_DATA['total_duration'])])
    res['policy'].extend(['baseline' for j in range(PROB_DATA['total_duration'])])
    res['param'].extend(['none' for j in range(PROB_DATA['total_duration'])])
    res['week'].extend(ind_res[0])
    res['sales'].extend(ind_res[1])
    res['price'].extend(ind_res[2])

    # ALWAYS DECREASE POLICY
    ind_res = simulator(PROB_DATA, distributions, decrease_policy, kwargs={})
    res['repl'].extend([i for j in range(PROB_DATA['total_duration'])])
    res['policy'].extend(['always_decrease' for j in range(PROB_DATA['total_duration'])])
    res['param'].extend(['none' for j in range(PROB_DATA['total_duration'])])
    res['week'].extend(ind_res[0])
    res['sales'].extend(ind_res[1])
    res['price'].extend(ind_res[2])

    LIKELIHOOD_ESTIMATOR_PROB_OPTIONS = np.linspace(0,1,101)
    for LIKELIHOOD_ESTIMATOR_PROB in LIKELIHOOD_ESTIMATOR_PROB_OPTIONS:
        # NAIVE LIKELIHOOD
        ind_res = simulator(PROB_DATA, distributions, likelihood_naive, kwargs={'req_likelihood': LIKELIHOOD_ESTIMATOR_PROB})
        res['repl'].extend([i for j in range(PROB_DATA['total_duration'])])
        res['policy'].extend(['likelihood' for j in range(PROB_DATA['total_duration'])])
        res['param'].extend([LIKELIHOOD_ESTIMATOR_PROB for j in range(PROB_DATA['total_duration'])])
        res['week'].extend(ind_res[0])
        res['sales'].extend(ind_res[1])
        res['price'].extend(ind_res[2])

res = pd.DataFrame(res)
res['revenue'] = res['sales'] * res['price']

# %% Aggregated results
res_agg = res.groupby(['policy', 'param', 'repl']).agg({'revenue': 'sum'}).reset_index()
res_agg = res_agg.groupby(['policy', 'param']).agg({'revenue': ['mean', 'std']}).reset_index()
res_agg.columns = ['policy', 'param', 'revenue_mean', 'revenue_sd']
res_agg.sort_values('revenue_mean')

# %% Plot evolutions
res_baseline = res.query(f'policy == "baseline"')
res_baseline['cum_revenue'] = res_baseline.groupby('repl')['revenue'].transform(pd.Series.cumsum)

res_decrease = res.query(f'policy == "always_decrease"')
res_decrease['cum_revenue'] = res_decrease.groupby('repl')['revenue'].transform(pd.Series.cumsum)

res_likelihood = res.query(f'policy == "likelihood" and param == 0.03')
res_likelihood['cum_revenue'] = res_likelihood.groupby('repl')['revenue'].transform(pd.Series.cumsum)

res_all = pd.concat([res_baseline, res_decrease, res_likelihood])

# Simulations
fig = px.line(res_all, x='week', y='cum_revenue', color='policy',
              facet_col='policy', facet_col_wrap=2,
              )
fig.update_traces(opacity=0.05)
fig.show(renderer='browser')

# Aggregate
res_all_agg = res_all.\
    groupby(['week', 'policy']).\
    agg({'cum_revenue': 'mean'}).\
    reset_index()

fig = px.line(res_all_agg, x='week', y='cum_revenue', color='policy',)
fig.show(renderer='browser')

# %%
