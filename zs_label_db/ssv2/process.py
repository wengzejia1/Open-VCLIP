import os
import json

label_dic = json.load(open('./something-something-v2-labels.json', 'r'))
label2index = {}
index2label = {}

for i, j in label_dic.items():
    # i = i.replace('something', '') 
    label2index[i] = int(j)
    index2label[int(j)] = i
    
json.dump(label2index, open('./ssv2-cls2index.json', 'w'))
json.dump(index2label, open('./ssv2-index2cls.json', 'w'))
