# %%
import pandas as pd
import numpy as np
import plotly.express as px
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler


# %% IMDB DATA
df_2022 = pd.read_excel('Dataset\.xlsx', sheet_name='2022')
df_2021 = pd.read_excel('Dataset\imdb_data.xlsx', sheet_name='2021')
df_2020 = pd.read_excel('Dataset\imdb_data.xlsx', sheet_name='2020')
df = pd.concat([df_2022, df_2021, df_2020])
df = df.reset_index(drop=True)
df = df[['Date', 'Top 10 Gross', '%± YD', '#1 Release', 'Gross']]
df.columns = ['date', 'top_10_gross', 'day_change', 'n1_release', 'gross']

# PREPROCESSING
df['date'] = pd.to_datetime(df['date'])
df['year'] = df.date.dt.year.astype('str')
df['weekday'] = df.date.dt.day_name()
# df = df[df.year != '2020']

# RELEASE DATE & LAG
df = df.merge(df.groupby('n1_release').date.min(), how='left', on='n1_release', suffixes=('','_start'))
df['date_lag'] = (df.date - df.date_start).dt.days
df['date_lag_adj'] = df.date_lag + 0.00001

# GROSSED LAG
df = df.merge(df.query('date_lag == 0')[['n1_release', 'gross']], how='left', on='n1_release', suffixes=('','_start'))
df['gross_lag'] = df.gross / df.gross_start

# REMOVE OUTLIERS
outliers = ['The Wretched', 'The Croods: A New Age']
df = df.query(f'n1_release not in {outliers}')

# %% Visualization (Sales)
fig  = px.line(df, x='date', y='top_10_gross', facet_row='year', color='year')
fig.update_xaxes(matches=None, showticklabels=True)
fig.show(renderer='browser')


# %% Sales Decrease over Time (Individual)
fig = px.scatter(df.query('date_lag <= 50'), 
                 x='date_lag_adj', y='gross_lag', color='year', 
                 hover_data='n1_release',
                 trendline="lowess",
                 trendline_scope="overall", trendline_color_override="black",)
fig.show(renderer='browser')


# %% Sales Decrease over Time (Aggregate)
dfa = df.query('date_lag <= 50').groupby('date_lag').gross.mean().reset_index()
fig = px.line(dfa, x='date_lag', y='gross')
fig.show(renderer='browser')


# %% Impact of Discounts given various elasticities
default_ticket = 15
dfa = df.query('date_lag <= 50').groupby('date_lag').gross.mean().reset_index()
dfa['elasticity'] = 'none'
dfa['price'] = default_ticket
dfa['demand'] = dfa.gross/default_ticket

def get_demand(demand, new_price, elasticity):
    price_change = (new_price/default_ticket)-1
    demand_change = price_change * demand * elasticity
    new_demand = demand + demand_change
    return(new_demand)

# Estimate Sales for Various Elasticities
dfaf = pd.DataFrame()
prices = np.round(np.linspace(20, 10, 21),2)
elasticities = np.round(np.linspace(-0.5, -1.5, 101),2)
for p in tqdm(prices):
    for e in elasticities:
        dfan = dfa.copy()
        dfan['demand'] = dfa.demand.apply(lambda x: get_demand(x, p, e))        
        dfan['elasticity'] = e
        dfan['price'] = p
        dfan['gross'] = dfan.demand * p
        dfan['Sales Change'] = (dfan.gross - dfa.gross)/(dfa.gross)
 
        dfaf = pd.concat([dfaf, dfan])
dfaf = dfaf.reset_index(drop=True)

# Finds Best Price for each elasticity
dfab = dfaf.groupby(['elasticity','price']).gross.sum().reset_index()
dfab = dfab.iloc[dfab.groupby('elasticity').gross.idxmax().values].reset_index(drop=True)
dfab['Sales Improvement'] = (dfab['gross'] / dfa.gross.sum())-1

# Plot
fig = px.scatter(dfab, color='price',
                 x='elasticity', y='Sales Improvement', 
                 title='Impact of Price Changes on Various Elasticities',
                 color_continuous_scale=px.colors.sequential.Viridis,
                 template='none')
fig.show(renderer='browser')
# %%
