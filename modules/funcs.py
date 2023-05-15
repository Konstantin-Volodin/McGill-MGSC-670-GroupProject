import pandas as pd
import numpy as np
import itertools
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium import webdriver


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
    prices.sort(reverse=True)
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


def validator(problem_dat, seed, policy, **kwargs):

    # Open website
    url = "http://www.randhawa.us/games/retailer/nyu.html"
    driver = webdriver.Firefox()
    driver.get(url)
    driver.find_element(By.ID, "seedUser").send_keys(seed)

    # Simulation Params
    tot_dur = problem_dat['total_duration']
    actions = problem_dat['actions']

    # Simulate
    for i in range(tot_dur):

        # Retrieve Data
        result_text = driver.find_element(By.ID, "result-table").text
        hist_sales = []
        hist_prices = []
        hist_weeks = []

        for i in result_text.split('\n')[1:]:
            vals = i.split(' ')
            hist_sales.append(int(vals[2]))
            hist_prices.append(int(vals[1]))
            hist_weeks.append(int(vals[0]))

        # End
        if hist_weeks[-1] == 15:
            break

        # Generate Action
        action = policy(problem_dat, actions, hist_sales, hist_prices, kwargs=kwargs['kwargs'])

        # Execute Action
        previous_price = hist_prices[-1]

        if action == previous_price:
            button = driver.find_element(By.ID, "maintainButton")
        elif action == 54:
            button = driver.find_element(By.ID, "tenButton")
        elif action == 48:
            button = driver.find_element(By.ID, "twentyButton")
        elif action == 36:
            button = driver.find_element(By.ID, "fortyButton")
        button.click()

    # Save Data
    tot_rev = int(driver.find_element(By.ID, 'rev').text[1:].replace(',', ''))
    perf_rev = int(driver.find_element(By.ID, 'perfect').text[1:].replace(',', ''))
    diff_rev = (perf_rev - tot_rev)/perf_rev

    driver.close()
    return (tot_rev, perf_rev, diff_rev)
