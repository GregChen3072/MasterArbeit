# Importing required libraries
import pandas as pd
import numpy as np

df = pd.DataFrame(dict(
    A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    B=range(10)
))

#print(df_2.sample(n=2, weights='A', random_state=1).reset_index(drop=True))
# define total sample size desired
N = 5

# perform stratified random sampling
res = df.groupby('A', group_keys=False).apply(lambda x: x.sample(
    int(
        np.rint(N*len(x)/len(df)))
)
).sample(frac=1)

print(res)
print(res.index)

rest_df = df.drop(res.index)

res_2 = res.groupby('A', group_keys=False).apply(lambda x: x.sample(
    int(
        np.rint(N*len(x)/len(df)))
)
).sample(frac=1)

print(res_2)
print(res_2.index)
