import pandas as pd
from pathlib import Path
from plotnine import *
import os

file_loc = os.path.dirname(os.path.realpath(__file__))/Path('output')
file_name = 'output.csv'
path_to_file = Path(file_loc)/Path(file_name)

dat = pd.read_csv(path_to_file,
                  dtype={
                      'date_time': 'string',
                      'SoC': 'float64',
                      'state': 'string',
                      'price': 'float64'
                  },
                  parse_dates=['date_time'])

#Graph 1: plotting the state of charge over time, coloring by what state the battery was in
p1 = (
    ggplot(dat,aes(x = 'date_time', y = 'SoC',color = 'state'))
    +geom_point()
    + theme(axis_text_x = element_text(angle = 45, vjust = 1, hjust = 1),
            figure_size=(16,8))
)
p1.save(file_loc/Path('graphs\\charge_over_time.png'))

#Graph 2: proportion of time spent in each state
state_list = dat['state'].value_counts().index.to_list()
state_cat = pd.Categorical(dat['state'],categories=state_list)
dat = dat.assign(state_cat = state_cat)

p2 = (
    ggplot(dat, aes(x = state_cat))+
    geom_bar()
)
p2.save(file_loc/Path('graphs\\state_proportions.png'))

#Graph 3: Cumulative sum of prices over time
price_cumsum = dat['price'].cumsum(axis = 0).to_list()
dat = dat.assign(price_cumsum = price_cumsum)

p3 = (
    ggplot(dat,aes(x = 'date_time', y = 'price_cumsum'))
    + geom_line()
    + theme(axis_text_x = element_text(angle = 45, vjust = 1, hjust = 1),
            figure_size=(16,8))
)

p3.save(file_loc/Path('graphs\\cumulative_profit.png'))

#Graph 4: revenue by product
state_revs = dat[['state','price']].groupby(['state']).sum(['price'])#.sort_values(by = 'price')

p4 = (
    ggplot(dat, aes(x = 'reorder(state,price)', y = 'price'))
    +geom_col()
)
p4.save(file_loc/Path('graphs\\rev_by_product.png'))
