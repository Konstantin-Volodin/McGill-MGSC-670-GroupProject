import pandas as pd
import numpy as np
import itertools


def clean_up_relevant_data(df):
    # DISTRIBUTION OF INIT DISTRIBUTIONS
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

    # DISTRIBUTIONS OF TRANSITIONS
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

    return (DF_DIST, DF_TRN)

def get_price_dependency(df_dist):
    prices = df_dist.columns.to_list()
    prices.sort(reverse = True)
    combos = list(itertools.combinations(prices, 2))

    mean_dependency = {}
    sd_dependency = {}
    for i, j in combos:
        mean_dependency[(i, j)] = df_dist.loc['mean_mean', j]/df_dist.loc['mean_mean', i]
        sd_dependency[(i, j)] = df_dist.loc['sd_mean', j]/df_dist.loc['sd_mean', i]
    
    return (mean_dependency, sd_dependency)

def simulator(problem_dat, distr, policy, **kwargs):
    curr_price = problem_dat['start_price']
    curr_inventory = problem_dat['start_inventory']
    tot_dur = problem_dat['total_duration']
    actions = problem_dat['actions']

    hist_sales = []
    hist_prices = []
    hist_weeks = []

    for i in range(tot_dur):
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
        curr_price = policy(problem_dat, actions, hist_sales, hist_prices, kwargs=kwargs['kwargs'])

    return (hist_weeks, hist_sales, hist_prices)
