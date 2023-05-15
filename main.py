# %%
import pandas as pd
import numpy as np
import numpy.random as npr
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import kaleido
from tqdm import tqdm

from modules.funcs import (simulator,
                           clean_up_relevant_data,
                           get_price_dependency)
from modules.policies import (baseline_policy,
                              moving_avg_longterm,
                              moving_avg_shorterm,
                              likelihood_naive,
                              likelihood_price_dependency)

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


# %% FOR TESTING
distributions = {opt: {'mean': max(npr.normal(DF_DIST[opt]['mean_mean'],
                                              DF_DIST[opt]['mean_sd']), 0),
                       'sd': max(npr.normal(DF_DIST[opt]['sd_mean'],
                                            DF_DIST[opt]['sd_sd']), 0)} for opt in OPTIONS}
#ind_res = simulator(PROB_DATA, distributions, likelihood_naive, kwargs={'req_likelihood': 0.5})
#ind_res = simulator(PROB_DATA, distributions, moving_avg_policy, kwargs={'num_observation': 5})
ind_res = simulator(PROB_DATA, distributions, likelihood_price_dependency, kwargs={'mean_dependency': MEAN_DEPENDENCY, 'sd_dependency': SD_DEPENDENCY})
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

    # MOVING AVERAGE LONGTERM POLICY
    NUM_OBSERVATION_OPTIONS = np.linspace(3,15,13)
    for NUM_OBSERVATION in NUM_OBSERVATION_OPTIONS:
        ind_res = simulator(PROB_DATA, distributions, moving_avg_longterm, kwargs={'num_observation': NUM_OBSERVATION})
        res['repl'].extend([i for j in range(PROB_DATA['total_duration'])])
        res['policy'].extend(['moving_avg_longterm' for j in range(PROB_DATA['total_duration'])])
        res['param'].extend([NUM_OBSERVATION for j in range(PROB_DATA['total_duration'])])
        res['week'].extend(ind_res[0])
        res['sales'].extend(ind_res[1])
        res['price'].extend(ind_res[2])

    # MOVING AVERAGE SHORTERM POLICY
    NUM_OBSERVATION_OPTIONS = np.linspace(3,15,13)
    for NUM_OBSERVATION in NUM_OBSERVATION_OPTIONS:
        ind_res = simulator(PROB_DATA, distributions, moving_avg_shorterm, kwargs={'num_observation': NUM_OBSERVATION})
        res['repl'].extend([i for j in range(PROB_DATA['total_duration'])])
        res['policy'].extend(['moving_avg_shorterm' for j in range(PROB_DATA['total_duration'])])
        res['param'].extend([NUM_OBSERVATION for j in range(PROB_DATA['total_duration'])])
        res['week'].extend(ind_res[0])
        res['sales'].extend(ind_res[1])
        res['price'].extend(ind_res[2])

    # NAIVE LIKELIHOOD POLICY
    LIKELIHOOD_ESTIMATOR_PROB_OPTIONS = np.linspace(0,1,101)
    for LIKELIHOOD_ESTIMATOR_PROB in LIKELIHOOD_ESTIMATOR_PROB_OPTIONS:
        ind_res = simulator(PROB_DATA, distributions, likelihood_naive, kwargs={'req_likelihood': LIKELIHOOD_ESTIMATOR_PROB})
        res['repl'].extend([i for j in range(PROB_DATA['total_duration'])])
        res['policy'].extend(['likelihood_naive' for j in range(PROB_DATA['total_duration'])])
        res['param'].extend([LIKELIHOOD_ESTIMATOR_PROB for j in range(PROB_DATA['total_duration'])])
        res['week'].extend(ind_res[0])
        res['sales'].extend(ind_res[1])
        res['price'].extend(ind_res[2])

    # PRICE DEPENDENCY LIKELIHOOD POLICY
    ind_res = simulator(PROB_DATA, distributions, likelihood_price_dependency, kwargs={'mean_dependency': MEAN_DEPENDENCY, 'sd_dependency': SD_DEPENDENCY})
    res['repl'].extend([i for j in range(PROB_DATA['total_duration'])])
    res['policy'].extend(['likelihood_price' for j in range(PROB_DATA['total_duration'])])
    res['param'].extend(['none' for j in range(PROB_DATA['total_duration'])])
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
#res_agg[res_agg.policy == "moving_avg_longterm"].sort_values('revenue_mean')
#res_agg[res_agg.policy == "moving_avg_shorterm"].sort_values('revenue_mean')

# %% Plot evolutions
res_baseline = res.query(f'policy == "baseline"')
res_baseline['cum_revenue'] = res_baseline.groupby('repl')['revenue'].transform(pd.Series.cumsum)

res_moving_avg_longterm = res.query(f'policy == "moving_avg_longterm" and param == 6')
res_moving_avg_longterm['cum_revenue'] = res_moving_avg_longterm.groupby('repl')['revenue'].transform(pd.Series.cumsum)

res_moving_avg_shorterm = res.query(f'policy == "moving_avg_shorterm" and param == 3')
res_moving_avg_shorterm['cum_revenue'] = res_moving_avg_shorterm.groupby('repl')['revenue'].transform(pd.Series.cumsum)

res_likelihood_naive = res.query(f'policy == "likelihood_naive" and param == 0.03')
res_likelihood_naive['cum_revenue'] = res_likelihood_naive.groupby('repl')['revenue'].transform(pd.Series.cumsum)

res_likelihood_price = res.query(f'policy == "likelihood_price"')
res_likelihood_price['cum_revenue'] = res_likelihood_price.groupby('repl')['revenue'].transform(pd.Series.cumsum)

res_all = pd.concat([res_baseline, 
                     res_moving_avg_longterm, 
                     res_moving_avg_shorterm, 
                     res_likelihood_naive, 
                     res_likelihood_price])

# Simulations
fig = px.line(res_all, x='week', y='cum_revenue', color='policy',
              facet_col='policy', facet_col_wrap=2,
              )
fig.update_traces(opacity=0.05)
fig.show(renderer='browser')
fig.write_image('images/simulation_results.png')

# Aggregate
res_all_agg = res_all.\
    groupby(['week', 'policy']).\
    agg({'cum_revenue': 'mean'}).\
    reset_index()

fig = px.line(res_all_agg, x='week', y='cum_revenue', color='policy',)
fig.show(renderer='browser')
fig.write_image('images/revenue_aggregation.png')

# Prices versus Sales
res_all_agg_versus = res_all.\
    groupby(['week', 'policy']).\
    agg({'price': 'mean', 'sales': 'mean'}).\
    reset_index()
fig = px.bar(res_all_agg_versus, x='week', y='sales', color='policy',
              facet_col='policy', facet_col_wrap=2,
              )
fig.add_trace(go.Scatter(x=res_all_agg_versus.query(f'policy == "baseline"').week, 
                         y=res_all_agg_versus.query(f'policy == "baseline"').price, 
                         name = 'price_baseline',
                         mode='lines'), row = 3, col = 1)
fig.add_trace(go.Scatter(x=res_all_agg_versus.query(f'policy == "likelihood_naive"').week, 
                         y=res_all_agg_versus.query(f'policy == "likelihood_naive"').price, 
                         name = 'price_likelihood_naive',
                         mode='lines'), row = 3, col = 2)
fig.add_trace(go.Scatter(x=res_all_agg_versus.query(f'policy == "likelihood_price"').week, 
                         y=res_all_agg_versus.query(f'policy == "likelihood_price"').price, 
                         name = 'price_likelihood_price',
                         mode='lines'), row = 2, col = 1)
fig.add_trace(go.Scatter(x=res_all_agg_versus.query(f'policy == "moving_avg_longterm"').week, 
                         y=res_all_agg_versus.query(f'policy == "moving_avg_longterm"').price,
                         name = 'price_moving_avg_longterm',
                         mode='lines'), row = 2, col = 2)
fig.add_trace(go.Scatter(x=res_all_agg_versus.query(f'policy == "moving_avg_shorterm"').week, 
                         y=res_all_agg_versus.query(f'policy == "moving_avg_shorterm"').price,
                         name = 'price_moving_avg_shorterm',
                         mode='lines'), row = 1, col = 1)

fig.show(renderer='browser')
fig.write_image('images/prices_versus_sales.png')


# Revenue Distribution - Hist and Rug
res_rev_all = res_all.groupby(['repl','policy']).agg({'revenue': 'sum'}).reset_index()
res_rev = []
for policy in res_rev_all.policy.unique():
    res_rev.append(res_rev_all[res_rev_all.policy == policy]['revenue'])

fig = ff.create_distplot(res_rev, res_rev_all.policy.unique(), 
                         bin_size=1000, curve_type='normal')
fig.show(renderer='browser')
fig.write_image('images/revenue_distribution_hist.png')


# Revenue Distribution - Boxplot
fig = px.box(res_rev_all, y='revenue', facet_col='policy', color='policy',
             boxmode='overlay', points='all')
fig.show(renderer='browser')
fig.write_image('images/revenue_distribution_box.png')

# %%
