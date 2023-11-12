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
                      'product': 'string',
                      'price': 'float64'
                  },
                  parse_dates=['date_time'])

def checkDA(x):
    if x[:2] == 'DA':
        return 'yes'
    else:
        return 'no'

def repDA(x):
    if x[:2] == 'DA':
        return x[2:]

print(dat.dtypes)
dat['isDA'] = dat['product'].apply(checkDA)
dat['product'] = dat['product'].str.replace('DA','')
print(dat.describe())
#Graph 1: plotting the state of charge over time, coloring by what product the battery was in
p1 = (
    ggplot(dat,aes(x = 'date_time', y = 'SoC',color = 'product'))
    +geom_point()
    + theme(axis_text_x = element_text(angle = 45, vjust = 1, hjust = 1),
            figure_size=(16,8))
)
p1.save(file_loc/Path('graphs\\charge_over_time.png'))

#Graph 1.2: plotting whether the model chose to participate in the DA market
p1_2 = (
    ggplot(dat,aes(x = 'date_time', y = 'SoC',color = 'isDA'))
    +geom_point()
    + theme(axis_text_x = element_text(angle = 45, vjust = 1, hjust = 1),
            figure_size=(16,8))
)
p1_2.save(file_loc/Path('graphs\\DA_participation.png'))

#Graph 2: proportion of time spent in each product
product_list = dat['product'].value_counts().index.to_list()
product_cat = pd.Categorical(dat['product'],categories=product_list)
dat = dat.assign(product_cat = product_cat)

p2 = (
    ggplot(dat, aes(x = product_cat))+
    geom_bar()
)
p2.save(file_loc/Path('graphs\\product_proportions.png'))

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
product_revs = dat[['product','price']].groupby(['product']).sum(['price'])#.sort_values(by = 'price')

p4 = (
    ggplot(dat, aes(x = 'reorder(product,price)', y = 'price'))
    +geom_col()
)
p4.save(file_loc/Path('graphs\\rev_by_product.png'))

p5 = (
    ggplot(dat,aes(x = 'date_time', y = 'price',color = 'product'))
    +geom_point()
    + theme(axis_text_x = element_text(angle = 45, vjust = 1, hjust = 1),
            figure_size=(16,8))
)
p5.save(file_loc/Path('graphs\\price_over_time.png'))
