import json
import re
import random
import os

if __name__ == '__main__':
    random.seed(2023)

    # rand 10 to split 50 classes
    for i in range(10):
        os.makedirs('./%d'%i)
        # rand 50 from 101
        rand_idx = [i for i in range(101)]
        random.shuffle(rand_idx)
        rand_idx = sorted(rand_idx[:50])

        lines = open('./testlist01.txt', 'r').readlines()
        cls_dir = {}
        cls_info = []
        flag = 0
        real_flag = 0

        for line in lines:
            cls = line.split('/')[0]
            if cls not in cls_dir:
                cls_dir[cls] = None
                if flag in rand_idx:
                    cls_info.append([cls, real_flag])
                    real_flag += 1
                flag += 1
            
        del(cls_dir)
        
        cls2index = {}
        index2cls = {}
        final_cls2index = {}
        final_index2cls = {}

        for info in cls_info:
            cls = info[0]
            final_cls = ' '.join(re.findall('[A-Z][^A-Z]*', cls))
            if final_cls == 'Yo Yo':
                final_cls = 'Yoyo'

            index = info[1]

            cls2index[info[0]] = info[1]
            index2cls[info[1]] = info[0]
            
            final_cls2index[final_cls] = info[1]
            final_index2cls[info[1]] = final_cls

            
        json.dump(final_cls2index, open('./%d/ucf101-cls2index.json'%i, 'w'))
        json.dump(final_index2cls, open('./%d/ucf101-index2cls.json'%i, 'w'))
        
        test_f = open('./%d/test.csv'%i, 'w') 
        for line in lines:
            line = line.strip()
            if line.split('/')[0] not in cls2index:
                continue
            record = line + ',' + str(cls2index[line.split('/')[0]])
            test_f.write(record)
            test_f.write('\n')
        
        train_lines = open('./trainlist01.txt', 'r').readlines() 
        for line in train_lines:
            line = line.split(' ')[0]
            line = line.strip()
            if line.split('/')[0] not in cls2index:
                continue
            record = line + ',' + str(cls2index[line.split('/')[0]])
            test_f.write(record)
            test_f.write('\n')

