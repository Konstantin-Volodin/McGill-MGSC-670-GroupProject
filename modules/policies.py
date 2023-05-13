import numpy as np
import scipy.stats as scs


def baseline_policy(problem_dat, actions, sales, prices, **kwargs):
    curr_price = prices[-1]
    return (curr_price)


def decrease_policy(problem_dat, actions, sales, prices, **kwargs):
    curr_price = prices[-1]
    if curr_price == 36:
        return (actions[curr_price][0])
    else:
        return (actions[curr_price][1])


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
