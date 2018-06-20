# -*- coding:utf-8 -*-
import pandas as pd

def fileIO(path):
    data=pd.read_csv(path)
    print(data.columns)
    print(data.describe())


if __name__=='__main__':
    path=input('Please input the file path: ')
    fileIO(path)