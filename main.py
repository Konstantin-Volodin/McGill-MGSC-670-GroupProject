# %%
import pandas as pd
import numpy as np
import scipy.stats as scs
import plotly.express as px
from tqdm import tqdm


# %%
# READ DATA
df = pd.read_csv('data/data.csv')

# %% DISTRIBUTION of INIT DISTRIBUTIONS
df_dist = df.groupby(['item', 'price']).\
    agg({'sales': ['mean', 'std']}).\
    sort_values(['item', 'price'], ascending=False).\
    reset_index()
df_dist.columns = ['item', 'price', 'sales_mean', 'sales_sd']

df_dist = df_dist.\
    groupby('price').\
    agg({'sales_mean': ['mean', 'std'],
         'sales_sd': ['mean', 'std']}).\
    reset_index()
df_dist.columns = ['price', 'mean_mean', 'mean_sd', 'sd_mean', 'sd_sd']
DF_DIST = df_dist.set_index('price').to_dict('index')

# %% DISTRIBUTIONS of TRANSITIONS
df_trn = df.groupby(['item', 'price']).\
    agg({'sales': ['mean', 'std']}).\
    sort_values(['item', 'price'], ascending=False).\
    groupby('item').pct_change().\
    dropna().reset_index()
df_trn.columns = ['item', 'price', 'mean_change', 'sd_change']

df_trn = df_trn.groupby('price').\
    agg({'mean_change': ['mean', 'std'],
         'sd_change': ['mean', 'std']}).\
    reset_index()
df_trn.columns = ['price', 'mean_change_mean', 'mean_change_sd',
                  'sd_change_mean', 'sd_change_sd']
DF_TRN = df_trn.set_index('price').to_dict('index')
DF_TRN

# %%


def simulator(distr, policy):
    curr_price = 60
    curr_inventory = TOTAL_INV

    hist_sales = []
    hist_prices = []
    hist_weeks = []

    for i in range(TOTAL_DUR):
        # Update Distribution
        curr_dist = distr[curr_price]

        # Sales this week
        sales = np.round(np.random.normal(curr_dist['mean'],
                                          curr_dist['sd']), 0)
        sales = np.max([0, sales])
        sales = np.min([curr_inventory, sales])
        curr_inventory = curr_inventory - sales

        # Save Data
        hist_sales.append(sales)
        hist_prices.append(curr_price)
        hist_weeks.append(i+1)

        # Action
        curr_price = policy(curr_price, hist_sales, hist_prices)

    return (hist_weeks, hist_sales, hist_prices)


def baseline_policy(curr_price, sales, prices):
    return (curr_price)


def decreate_policy(curr_price, sales, prices):
    if curr_price == 36:
        return (ACTIONS[curr_price][0])
    else:
        return (ACTIONS[curr_price][1])


def naive_likelihood_policy(curr_price, sales, prices):
    # Get Price Relevant Data
    indexes = np.where(np.array(prices) == curr_price)[0]

    # Need to gather more data
    if len(indexes) <= 1:
        return (curr_price)

    # Get Distribution Estimate
    curr_mean = np.mean(np.array(sales)[indexes])
    curr_sd = np.std(np.array(sales)[indexes])

    # Get Sales Requirements
    left_over_inv = TOTAL_INV - sum(sales)
    daily_req = left_over_inv/(TOTAL_DUR-len(sales))

    # Check if probability meets it
    prob_it_fits = 1-scs.norm.cdf(daily_req, loc=curr_mean, scale=curr_sd)

    if prob_it_fits < LIKELIHOOD_ESTIMATOR_PROB and curr_price != 36:
        return (ACTIONS[curr_price][1])

    else:
        return (ACTIONS[curr_price][0])


def mean_likelihood_policy(curr_price, sales, prices):
    # Get Price Relevant Data
    indexes = np.where(np.array(prices) == curr_price)[0]

    # Need to gather more data
    if len(indexes) <= 1:
        return (curr_price)

    # Get Distribution Estimate
    curr_mean = np.mean(np.array(sales)[indexes])
    curr_sd = np.std(np.array(sales)[indexes])

    # Get Sales Requirements
    left_over_inv = TOTAL_INV - sum(sales)
    daily_req = left_over_inv/(TOTAL_DUR-len(sales))

    # Check if probability meets it
    prob_it_fits = 1-scs.norm.cdf(daily_req, loc=curr_mean, scale=curr_sd)

    if prob_it_fits < LIKELIHOOD_ESTIMATOR_PROB and curr_price != 36:
        return (ACTIONS[curr_price][1])

    else:
        return (ACTIONS[curr_price][0])


# %% FOR TESTING
ACTIONS = {60: [60, 54, 48, 36],
           54: [54, 48, 36],
           48: [48, 36],
           36: [36]}
TOTAL_INV = 2000
TOTAL_DUR = 15
LIKELIHOOD_ESTIMATOR_PROB = 0.1

distributions = {60: {'mean': max(np.random.normal(DF_TRN[36]['mean_change_mean'], 19.646415), 0),
                      'sd': max(np.random.normal(21.363060, 10.900014), 0)},
                 54: {'mean': max(np.random.normal(98.777778, 29.547931), 0),
                      'sd': max(np.random.normal(34.059664, 15.195045), 0)},
                 48: {'mean': max(np.random.normal(119.100000, 22.958931), 0),
                      'sd': max(np.random.normal(43.171959, 31.978993), 0)},
                 36: {'mean': max(np.random.normal(181.366667, 12.537942), 0),
                      'sd': max(np.random.normal(79.846521, 46.166815), 0)}, }

ind_res = simulator(distributions, naive_likelihood_policy)

# %% 100 simulations
ACTIONS = {60: [60, 54, 48, 36],
           54: [54, 48, 36],
           48: [48, 36],
           36: [36]}
TOTAL_INV = 2000
TOTAL_DUR = 15
LIKELIHOOD_ESTIMATOR_PROB_OPTIONS = np.linspace(0, 1, 101)

res = {'repl': [], 'week': [], 'sales': [], 'price': [], 'policy': [], 'param': []}
for i in tqdm(range(500)):
    distributions = {60: {'mean': max(np.random.normal(73.897354, 19.646415), 0),
                          'sd': max(np.random.normal(21.363060, 10.900014), 0)},
                     54: {'mean': max(np.random.normal(98.777778, 29.547931), 0),
                          'sd': max(np.random.normal(34.059664, 15.195045), 0)},
                     48: {'mean': max(np.random.normal(119.100000, 22.958931), 0),
                          'sd': max(np.random.normal(43.171959, 31.978993), 0)},
                     36: {'mean': max(np.random.normal(181.366667, 12.537942), 0),
                          'sd': max(np.random.normal(79.846521, 46.166815), 0)}, }

    # BASELINE POLICY
    ind_res = simulator(distributions, baseline_policy)
    res['repl'].extend([i for j in range(TOTAL_DUR)])
    res['policy'].extend(['baseline' for j in range(TOTAL_DUR)])
    res['param'].extend(['none' for j in range(TOTAL_DUR)])
    res['week'].extend(ind_res[0])
    res['sales'].extend(ind_res[1])
    res['price'].extend(ind_res[2])

    # ALWAYS DECREASE POLICY
    ind_res = simulator(distributions, decreate_policy)
    res['repl'].extend([i for j in range(TOTAL_DUR)])
    res['policy'].extend(['always_decrease' for j in range(TOTAL_DUR)])
    res['param'].extend(['none' for j in range(TOTAL_DUR)])
    res['week'].extend(ind_res[0])
    res['sales'].extend(ind_res[1])
    res['price'].extend(ind_res[2])

    # LIKELIHOOD
    for LIKELIHOOD_ESTIMATOR_PROB in LIKELIHOOD_ESTIMATOR_PROB_OPTIONS:
        ind_res = simulator(distributions, naive_likelihood_policy)
        res['repl'].extend([i for j in range(TOTAL_DUR)])
        res['policy'].extend(['likelihood' for j in range(TOTAL_DUR)])
        res['param'].extend([LIKELIHOOD_ESTIMATOR_PROB for j in range(TOTAL_DUR)])
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
