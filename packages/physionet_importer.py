import pandas as pd
from os import walk


def getOneSet(path):
    reference_df = pd.read_csv(
        path + '/REFERENCE.csv', header=None, names=['file_name', 'classID'])
    # add .wav file extension to the file name
    reference_df['file_name'] = reference_df['file_name'] + '.wav'
    # insert new column 'relative_path' that indicates the path to the audio file
    reference_df['relative_path'] = path + \
        '/' + reference_df['file_name'].astype(str)

    # convert all -1 classIDs to 0
    reference_df.loc[reference_df['classID'].astype(int) == -1, 'classID'] = 0
    return reference_df


def getTrainingSet(path='Dataset/PhysioNet'):
    _, folders, _ = next(walk(path + '/training'))
    training_df = pd.DataFrame()

    for f in folders:
        f_df = getOneSet(path + '/training/' + f)
        training_df = pd.concat([training_df, f_df])

    return training_df.reset_index(drop=True)


def getValidationSet(path='Dataset/PhysioNet'):
    return getOneSet(path + '/validation')

print(getTrainingSet()['classID'].value_counts())