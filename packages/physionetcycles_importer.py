import pandas as pd
import numpy as np


def getTrainingSet(path='Dataset/PhysioNetCycles'):
    training_df = pd.read_csv(path + '/REFERENCE.csv')
    training_df['relative_path'] = './Dataset/' + training_df['relative_path']
    training_df['original_file'] = training_df['file_name'].str.split(
        '_', 1).str[0]
    return training_df.reset_index(drop=True)


def getSplit(df, pct):
    grouped = df.groupby('original_file')
    # Shuffle the groups
    shuffled_groups = []
    for name, group in grouped:
        shuffled_groups.append(group)
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
