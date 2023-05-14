import numpy as np
import scipy.stats as scs
import itertools


def baseline_policy(problem_dat, actions, sales, prices, **kwargs):
    curr_price = prices[-1]
    return (curr_price)


def moving_avg_policy(problem_dat, actions, sales, prices, **kwargs):

    # Get Price Relevant Data
    curr_price = prices[-1]
    curr_week = len(sales) + 1

    # No change for the first N rounds
    N = kwargs['kwargs']['num_observation']
    if curr_week <= N:
        return curr_price
    
    sales_pred = sales.copy()
    for i in range(curr_week, problem_dat['total_duration']):
        new_demand = np.round(sum(sales_pred[i-3:i])/3)
        left_over = problem_dat['start_inventory'] - sum(sales_pred)
        sales_pred.append(min(new_demand, left_over))

    if sum(sales_pred) < problem_dat['start_inventory'] and curr_price != 36:
        return (actions[curr_price][1])
    else:
        return (actions[curr_price][0])


def likelihood_naive(problem_dat, actions, sales, prices, **kwargs):

    # Get Price Relevant Data
    curr_price = prices[-1]
    indexes = np.where(np.array(prices) == curr_price)[0]

    # If need to gather more data
    if len(indexes) <= 1:
        return (curr_price)

    # Get Distribution Estimate
    curr_mean = np.mean(np.array(sales)[indexes])
    curr_sd = np.std(np.array(sales)[indexes])

    # Get Sales Requirements
    left_over_inv = problem_dat['start_inventory'] - sum(sales)
    daily_req = left_over_inv/(problem_dat['total_duration']-len(sales))

    # Check if probability meets it
    prob_it_fits = 1-scs.norm.cdf(daily_req, loc=curr_mean, scale=curr_sd)

    if prob_it_fits < kwargs['kwargs']['req_likelihood'] and curr_price != 36:
        return (actions[curr_price][1])
    else:
        return (actions[curr_price][0])

def likelihood_price_dependency(problem_dat, actions, sales, prices, **kwargs):

    # Get Price Relevant Data
    curr_price = prices[-1]
    indexes = np.where(np.array(prices) == curr_price)[0]

    # If need to gather more data
    if len(indexes) <= 1 or curr_price == 36:
        return (curr_price)
    
    # Get Distribution Estimate (Current and Next)
    curr_mean = np.mean(np.array(sales)[indexes])
    curr_sd = np.std(np.array(sales)[indexes])

    mean_dependency = kwargs['kwargs']['mean_dependency']
    sd_dependency = kwargs['kwargs']['sd_dependency']

    next_price = actions[curr_price][1]
    next_mean = curr_mean * mean_dependency[(curr_price, next_price)]
    next_sd = curr_sd * sd_dependency[(curr_price, next_price)]

    # Expected Revenue from Different Distribution
    exp_rev_curr = 0
    exp_rev_next = 0
    for i in range(len(sales), problem_dat['total_duration']):
        exp_rev_curr += np.random.normal(curr_mean, curr_sd) * curr_price
        exp_rev_next += np.random.normal(next_mean, next_sd) * next_price

    if exp_rev_curr >= exp_rev_next:
        return (actions[curr_price][0])
    else:
        return (actions[curr_price][1])


# def likelihood_pe_estimate(curr_price, sales, prices, **kwargs):
#     # Get Price Relevant Data
#     indexes = np.where(np.array(prices) == curr_price)[0]

#     # Need to gather more data
#     if len(indexes) <= 1:
#         return (curr_price)

#     # Get Distribution Estimate
#     curr_mean = np.mean(np.array(sales)[indexes])
#     curr_sd = np.std(np.array(sales)[indexes])

#     # Get Sales Requirements
#     left_over_inv = TOTAL_INV - sum(sales)
#     daily_req = left_over_inv/(TOTAL_DUR-len(sales))

#     # Check if probability meets it
#     prob_it_fits = 1-scs.norm.cdf(daily_req, loc=curr_mean, scale=curr_sd)

#     if prob_it_fits < LIKELIHOOD_ESTIMATOR_PROB and curr_price != 36:
#         return (ACTIONS[curr_price][1])

#     else:
#         return (ACTIONS[curr_price][0])