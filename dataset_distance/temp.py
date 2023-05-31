import json

a = json.load(open('../label_db/ucf101/ucf101-cls2index.json', 'r'))
with open('./ucf_101_labels.csv', 'w') as f:
    f.write('id,name')
    f.write('\n')
    for i, key in enumerate(a):
        f.write('%d,%s'%(i, key))
        f.write('\n')
