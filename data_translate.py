
from collections import defaultdict
import bisect
import json

import numpy as np
import matplotlib.pyplot as plt
import math
 
for model in {"Squeezenet","Googlenet","MobilenetV1","MobilenetV2"}:

    for batch in range(2,17):
        file_name=f'heuristic/{model}_{batch}.heuristic.json'
        writer_name=f'new_data/{model}_{batch}.in'
        writer_file = open(writer_name, 'w')
        
        
        file=dict()
        with open(file_name) as f:
            file=json.load(f)
        for (t,info) in file.items():
            print(t,info['alloc'],info['free'],info['size'])
            writer_file.write(f"{t}  {info['alloc']} {info['free']} {info['size']}\n")
            # print(info['alloc'])
            # print(info['free'])
            # print(info['size'])
            
        writer_file.close()