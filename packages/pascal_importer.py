import pandas as pd
from os import walk

class_weights = [0] * 4


def fillPaths(d, path, classID):
    for (dirpath, dirnames, filenames) in walk(path):
        relative_path = map(lambda self: dirpath + '/' + self, filenames)
        d['relative_path'].extend(relative_path)
        temp = [classID] * len(filenames)
        d['classID'].extend(temp)
        d['file_name'].extend(filenames)
        print(f"{path} [{len(filenames)}]")
        class_weights[classID] = 1/len(filenames)
        break
    return d


def getData(dirname='Dataset', binary=False, artifacts=True):
    d = {'relative_path': [], 'classID': [], 'file_name': []}
    if binary:
        fillPaths(dirname + '/Atraining_extrahls', 1)
        fillPaths(dirname + '/Atraining_murmur', 1)
        fillPaths(dirname + '/Atraining_normal', 0)
    else:
        if artifacts:
            fillPaths(dirname + '/Atraining_artifact', 3)
        fillPaths(dirname + '/Atraining_extrahls', 2)
        fillPaths(dirname + '/Atraining_murmur', 1)
        fillPaths(dirname + '/Atraining_normal', 0)
    

    df = pd.DataFrame(data=d)
    df = df[df.file_name != '.DS_Store']

    return df

def getPreprocessedData(dirname='Dataset/PASCAL'):
    d = {'relative_path': [], 'classID': [], 'file_name': []}
    d = fillPaths(d, dirname + '/normal', 0)
    d = fillPaths(d, dirname + '/abnormal', 1)
    df = pd.DataFrame(data=d)
    df = df[df['file_name'].str.endswith('.wav')].reset_index(drop=True)
    return df