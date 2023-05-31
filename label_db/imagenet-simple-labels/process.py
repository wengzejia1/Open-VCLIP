import json

a = json.load(open('./imagenet-simple-labels.json','r'))
cls2idx = {}
idx2cls = {}

for i, cls in enumerate(a):
    cls2idx[cls] = i
    idx2cls[i] = cls

json.dump(cls2idx, open('../inet-cls2index.json', 'w'))
json.dump(idx2cls, open('../inet-index2cls.json', 'w'))
