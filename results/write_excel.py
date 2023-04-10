import os
import numpy as np
import pandas as pd 

source = ['a','b','c','d']
target = ['a','b','c','d']
dic = {'a': 1, 'b': 2, 'c': 3, 'd' : 4}

def test(source_name,target_name,dic):
    acc = ord(source_name)-ord(target_name)
    excel_file = '/data/huangtao/projects/subsapce-attack/results/excel/result.txt'
    if os.path.exists(excel_file):
        df = np.loadtxt(excel_file,delimiter=",")
        if source_name not in dic:
            dic[source_name] = len(dic)+1
        if target_name not in dic:
            dic[target_name] = len(dic)+1
        df[dic[target_name]][dic[source_name]]=acc
        np.savetxt(excel_file, df, fmt='%.01f', delimiter=",")
    else:
        df = np.zeros(10,10)
        np.savetxt(excel_file, df, fmt='%.01f', delimiter=",")
    return 
# import pdb;pdb.set_trace()
for source_name in source:
    for tagrte_name in target:
        test(source_name,tagrte_name,dic)

