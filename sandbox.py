import time

# import numpy as np
# import pandas as pd
#
# data = np.zeros((3, 6), dtype=np.int16)
# col_index = pd.MultiIndex.from_product([['A', 'B', 'C'], ['t1', 't2']])
# df = pd.DataFrame(data, columns=col_index)
#
# print('----------------------------')
# print(df)
# print('----------------------------')
# print('----------------------------')
#
# print(df['A'])
# print('----------------------------')
# print('----------------------------')
# print('----------------------------')
# # df[:, (slice(None), 't1')]
# # print(df[:, 't2'])
# tf = df.transpose()
# print(tf)
#
# print(tf.loc[('A', 't2')])
# print(tf[('A', 't2'), :])
# print(tf.loc[(slice(None), 't2'), :])
# # print(tf.loc[(slice(None), 't2')])
# #
# #
# #
import numpy as np
import pandas as pd
# from pandas.core.layout import array_layout
# array_layout.order = 'F'
print("pandas version: ", pd.__version__)
print([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], order='C')

print("Numpy array is C-contiguous: ", array.flags.c_contiguous)
print(array)


dataframe = pd.DataFrame(array, index = pd.MultiIndex.from_tuples([('A', 'U'), ('A', 'V'), ('B', 'W')], names=['dim_one', 'dim_two']))
print("DataFrame is C-contiguous: ", dataframe.values.flags.c_contiguous)
print(dataframe.values)

dataframe_copy = dataframe.copy()
print("Copy of DataFrame is C-contiguous: ", dataframe_copy.values.flags.c_contiguous)
print(dataframe_copy.values)

dataframe_copy = dataframe_copy.copy()
print("Copy of Copy of DataFrame is C-contiguous: ", dataframe_copy.values.flags.c_contiguous)
print(dataframe_copy.values)

aggregated_dataframe = dataframe.groupby('dim_one').sum()
print("Aggregated copy of copy DataFrame is C-contiguous: ", aggregated_dataframe.values.flags.c_contiguous)
print(aggregated_dataframe.values)

aggregated_dataframe = dataframe.groupby('dim_one').sum()
print("Aggregated DataFrame is C-contiguous: ", aggregated_dataframe.values.flags.c_contiguous)
print(aggregated_dataframe.values)

print("===========================")


# from pandas.core.layout import ArrayLayout
# ArrayLayout().order = 'C'

print("pandas version: ", pd.__version__)
print([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], order='C')

print("Numpy array is C-contiguous: ", array.flags.c_contiguous)
print(array)


dataframe = pd.DataFrame(array, index = pd.MultiIndex.from_tuples([('A', 'U'), ('A', 'V'), ('B', 'W')], names=['dim_one', 'dim_two']))
print("DataFrame is C-contiguous: ", dataframe.values.flags.c_contiguous)
print(dataframe.values)

dataframe_copy = dataframe.copy()
print("Copy of DataFrame is C-contiguous: ", dataframe_copy.values.flags.c_contiguous)
print(dataframe_copy.values)

dataframe_copy = dataframe_copy.copy()
print("Copy of Copy of DataFrame is C-contiguous: ", dataframe_copy.values.flags.c_contiguous)
print(dataframe_copy.values)

aggregated_dataframe = dataframe.groupby('dim_one').sum()
print("Aggregated copy of copy DataFrame is C-contiguous: ", aggregated_dataframe.values.flags.c_contiguous)
print(aggregated_dataframe.values)

aggregated_dataframe = dataframe.groupby('dim_one').sum()
print("Aggregated DataFrame is C-contiguous: ", aggregated_dataframe.values.flags.c_contiguous)
print(aggregated_dataframe.values)

## Output in Jupyter Notebook
# pandas version:  0.23.4
# Numpy array is C-contiguous:  True
# DataFrame is C-contiguous:  True
# Copy of DataFrame is C-contiguous:  False
# Aggregated DataFrame is C-contiguous:  False


import numpy as np
import pandas as pd

names = ['super', 'natural', 'multi', 'index', 'construction']
index = pd.MultiIndex.from_product([list(name) for name in names], names=names)
columns = pd.RangeIndex(10000)
values = np.random.randn(len(index), len(columns))

def major_order(values):
    '''
    Return the major ordering as string 'C' or 'F'.

    :param values: `numpy.ndarray` or `pandas.DataFrame`
    :return: `str`
    '''
    if isinstance(values, pd.DataFrame):
        values = values.values
    return {True: 'C', False: 'F'}[values.flags.c_contiguous]



print('Initial arrays` major-order: ', major_order(values))
data = pd.DataFrame(values, index=index, columns=columns)
print('Initial dataframes` major-order: ', major_order(values))
print('Shape of data: ', data.shape)

#%%timeit
aggregated = data.groupby(['natural', 'multi', 'index']).sum()
# 228 ms ± 9.87 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

print('Shape of aggregated dataframe: ', aggregated.shape)
print('Aggregated datas` major-order: ', major_order(aggregated))
data_copy = data.copy()
print('Data copy major-order: ', major_order(data_copy))

# %%timeit
aggregated_of_copy = data_copy.groupby(['natural', 'multi', 'index']).sum()
# 3.86 s ± 92.9 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

aggregated_of_copy = data_copy.groupby(['natural', 'multi', 'index']).sum()
print('Data copy major-order: ', major_order(aggregated_of_copy))

