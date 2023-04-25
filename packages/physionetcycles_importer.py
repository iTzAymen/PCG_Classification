import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def getTrainingSet(path='Dataset/PhysioNetCycles'):
    training_df = pd.read_csv(path + '/REFERENCE.csv')
    training_df['relative_path'] = './Dataset/' + training_df['relative_path']
    training_df['original_file'] = training_df['file_name'].str.split(
        '_', 1).str[0]
    return training_df.reset_index(drop=True)


def getSplit(df, pct, seed=0):
    grouped = df.groupby('original_file')
    # Shuffle the groups
    shuffled_groups = []
    for name, group in grouped:
        shuffled_groups.append(group)
    np.random.seed(seed)
    np.random.shuffle(shuffled_groups)

    num_rows = len(shuffled_groups)
    partition_sizes = [int(num_rows * p) for p in pct]

    # Split the DataFrame into partitions using the calculated sizes
    group_partitions = []
    start_idx = 0
    for size in partition_sizes:
        group_partitions.append(shuffled_groups[start_idx:start_idx+size])
        start_idx += size

    partitions = []
    for group in group_partitions:
        partitions.append(pd.concat(group).reset_index(drop=True))

    # Return a tuple of DataFrames for each partition
    return tuple(partitions)


def getStratifiedSplit(df, test_pct, seed=0):

    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=test_pct, random_state=seed)

    original_df = df[['original_file', 'classID']
                     ].drop_duplicates().reset_index(drop=True)
    X = original_df['original_file']
    y = original_df['classID']

    train_idx, test_idx = next(sss.split(X, y))
    train_df = original_df.loc[train_idx].reset_index(drop=True)
    test_df = original_df.loc[test_idx].reset_index(drop=True)

    final_train_df = df[df['original_file'].isin(
        train_df['original_file'].unique())].reset_index(drop=True)
    final_test_df = df[df['original_file'].isin(
        test_df['original_file'].unique())].reset_index(drop=True)

    # Return a tuple of DataFrames for each partition
    return final_train_df, final_test_df
