import os
from os.path import join
import pandas as pd

path = './results'

files = [f for f in os.listdir(path)]

df = pd.DataFrame()

for file in files:
    path_to_file = join(path, file)
    tmp = pd.read_csv(path_to_file, index_col=0)
    df = df.append(tmp, ignore_index=True)

# to_plot = df.groupby(by=['num_goals', 'branch'], observed=True).mean().reset_index(level=[0, 1], inplace=False)

import seaborn as sns
import matplotlib.pyplot as plt

print(df['branch'].unique())

to_plot = df[df['branch'].isin(
    [
        '[4]',
        # '[4, 4]',
        '[4, 4, 4]',
        # '[4, 4, 4, 4]',
        # '[4, 4, 4, 4, 4]',
        '[4, 4, 4, 1, 1, 1]',
        # '[1, 1, 1, 1, 1, 1, 1, 4, 4, 4]',
        # '[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]',
        # '[1]',
        # '[1, 1, 1]',
        '[1, 1, 1, 4, 4, 4]',
        '[4, 4, 4, 1, 1, 1, 1, 1, 1, 1]'

    ])]

sns.lineplot(x="num_goals", y="score", hue='branch',
             err_style="bars", data=to_plot)
import numpy as np


sns.scatterplot(x=np.arange(1, 11), y=np.arange(1, 11), marker='_',
                color='red', label="best possible")
plt.show()
