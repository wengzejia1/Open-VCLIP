import json
import re

if __name__ == '__main__':
    lines = open('./testlist01.txt', 'r').readlines()
    cls_dir = {}
    cls_info = []
    flag = 0
    
    for line in lines:
        cls = line.split('/')[0]
        if cls not in cls_dir:
            cls_dir[cls] = None
            cls_info.append([cls, flag])
            flag += 1
        
    del(cls_dir)
    cls2index = {}
    index2cls = {}
    final_cls2index = {}
    final_index2cls = {}

    for info in cls_info:
        cls = info[0]
        final_cls = ' '.join(re.findall('[A-Z][^A-Z]*', cls))
        index = info[1]
        if final_cls == 'Yo Yo':
            final_cls = 'Yoyo'

        cls2index[info[0]] = info[1]
        index2cls[info[1]] = info[0]
        
        final_cls2index[final_cls] = info[1]
        final_index2cls[info[1]] = final_cls

        
    json.dump(final_cls2index, open('./ucf101-cls2index.json', 'w'))
    json.dump(final_index2cls, open('./ucf101-index2cls.json', 'w'))
    
        
    with open('./test.csv', 'w') as f:
        for line in lines:
            line = line.strip()
            record = line + ',' + str(cls2index[line.split('/')[0]])
            f.write(record)
            f.write('\n')
    
    with open('./train.csv', 'w') as f:
        train_lines = open('./trainlist01.txt', 'r').readlines() 
        for line in train_lines:
            line = line.split(' ')[0]
            line = line.strip()
            record = line + ',' + str(cls2index[line.split('/')[0]])
            f.write(record)
            f.write('\n')
     

