import numpy as np
import scipy.stats as scs
import itertools


def baseline_policy(problem_dat, actions, sales, prices, **kwargs):
    curr_price = prices[-1]
    return (curr_price)


def moving_avg_longterm(problem_dat, actions, sales, prices, **kwargs):

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


def moving_avg_shorterm(problem_dat, actions, sales, prices, **kwargs):

    # Get Price Relevant Data
    curr_price = prices[-1]
    curr_week = len(sales) + 1

    # No change for the first N rounds
    N = kwargs['kwargs']['num_observation']
    if curr_week <= N:
        return curr_price

    indexes = np.where(np.array(prices) == curr_price)[0]
    # If need to gather more data
    if len(indexes) < 3:
        return (curr_price)

    new_demand = np.round(sum(sales[curr_week-3:curr_week])/3)
    left_over_inv = problem_dat['start_inventory'] - sum(sales)
    daily_req = left_over_inv/(problem_dat['total_duration']-len(sales))

    if new_demand < daily_req and curr_price != 36:
        return (actions[curr_price][1])
    else:
        return (actions[curr_price][0])


def random_policy(problem_dat, actions, sales, prices, **kwargs):
    curr_price = prices[-1]
    return (np.random.choice(problem_dat['actions'][curr_price]))


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


def likelihood_shared_distribution(problem_dat, actions, sales, prices, **kwargs):

    # Get Price Relevant Data
    curr_price = prices[-1]

    # Normalize sales to 60 prices
    normalized_sales = []
    if curr_price == 60:
        normalized_sales = sales

    # Normalize sales to 54 prices
    elif curr_price == 54:
        indexes = np.array(np.where(np.array(prices) == 60)[0])
        vals = np.array(sales)[indexes] * kwargs['kwargs']['mean_dependency'][(60, 54)]
        normalized_sales.extend(vals)

        indexes = np.array(np.where(np.array(prices) == 54)[0])
        vals = np.array(sales)[indexes]
        normalized_sales.extend(vals)

    # Normalize sales to 48 prices
    elif curr_price == 48:
        indexes = np.array(np.where(np.array(prices) == 60)[0])
        vals = np.array(sales)[indexes] * kwargs['kwargs']['mean_dependency'][(60, 48)]
        normalized_sales.extend(vals)

        indexes = np.array(np.where(np.array(prices) == 54)[0])
        vals = np.array(sales)[indexes] * kwargs['kwargs']['mean_dependency'][(54, 48)]
        normalized_sales.extend(vals)

        indexes = np.array(np.where(np.array(prices) == 48)[0])
        vals = np.array(sales)[indexes]
        normalized_sales.extend(vals)

    # Normalize sales to 36 prices
    elif curr_price == 36:
        indexes = np.array(np.where(np.array(prices) == 60)[0])
        vals = np.array(sales)[indexes] * kwargs['kwargs']['mean_dependency'][(60, 36)]
        normalized_sales.extend(vals)

        indexes = np.array(np.where(np.array(prices) == 54)[0])
        vals = np.array(sales)[indexes] * kwargs['kwargs']['mean_dependency'][(54, 36)]
        normalized_sales.extend(vals)

        indexes = np.array(np.where(np.array(prices) == 48)[0])
        vals = np.array(sales)[indexes] * kwargs['kwargs']['mean_dependency'][(48, 36)]
        normalized_sales.extend(vals)

        indexes = np.array(np.where(np.array(prices) == 36)[0])
        vals = np.array(sales)[indexes]
        normalized_sales.extend(vals)

    # Get Distribution Estimate
    curr_mean = np.mean(np.array(normalized_sales))
    curr_sd = np.std(np.array(normalized_sales))

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


def rl_policy(problem_dat, actions, sales, prices, **kwargs):

    # Mapping
    _action_to_price = {0: 60, 1: 54, 2: 48, 3: 36, }
    _price_to_action = {60: 0, 54: 1, 48: 2, 36: 3}

    # Model
    rl_model = kwargs['kwargs']['rl_object']

    # Change the data
    state = {"week": len(sales)-1,
             "curr_price": _price_to_action[prices[-1]],
             "sales_total": np.sum(sales),
             "sales_mean": np.mean(sales),
             "sales_sd": np.std(sales),
             "week0_sales": sales[0] if len(sales) > 0 else -1,
             "week1_sales": sales[1] if len(sales) > 1 else -1,
             "week2_sales": sales[2] if len(sales) > 2 else -1,
             "week3_sales": sales[3] if len(sales) > 3 else -1,
             "week4_sales": sales[4] if len(sales) > 4 else -1,
             "week5_sales": sales[5] if len(sales) > 5 else -1,
             "week6_sales": sales[6] if len(sales) > 6 else -1,
             "week7_sales": sales[7] if len(sales) > 7 else -1,
             "week8_sales": sales[8] if len(sales) > 8 else -1,
             "week9_sales": sales[9] if len(sales) > 9 else -1,
             "week10_sales": sales[10] if len(sales) > 10 else -1,
             "week11_sales": sales[11] if len(sales) > 11 else -1,
             "week12_sales": sales[12] if len(sales) > 12 else -1,
             "week13_sales": sales[13] if len(sales) > 13 else -1,
             "week14_sales": sales[14] if len(sales) > 14 else -1,
        }

    # Get Action
    action = rl_model.get_action(state)
    action = _action_to_price[action]

    if action <= prices[-1]:
        return action
    else:
        return prices[-1]
