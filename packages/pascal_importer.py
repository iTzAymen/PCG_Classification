import pandas as pd
from os import walk

d = {'relative_path': [], 'classID': [], 'file_name': []}
class_weights = [0] * 4


def fillPaths(path, classID):
    for (dirpath, dirnames, filenames) in walk(path):
        relative_path = map(lambda self: dirpath + '/' + self, filenames)
        d['relative_path'].extend(relative_path)
        temp = [classID] * len(filenames)
        d['classID'].extend(temp)
        d['file_name'].extend(filenames)
        print(f"{path} [{len(filenames)}]")
        class_weights[classID] = 1/len(filenames)
        break


def getData(dirname='Dataset'):
    fillPaths(dirname + '/Atraining_extrahls', 3)
    fillPaths(dirname + '/Atraining_murmur', 2)
    fillPaths(dirname + '/Atraining_normal', 1)
    fillPaths(dirname + '/Atraining_artifact', 0)

    df = pd.DataFrame(data=d)
    df = df[df.file_name != '.DS_Store']

    return df
