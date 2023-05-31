import json
import os
import random
import os

if __name__ == '__main__':
    random.seed(2023)

    # rand 10 to split 25 classes
    for split in range(10):
        os.makedirs('./%d'%split)
        # rand 25 from 51
        rand_idx = [i for i in range(51)]
        random.shuffle(rand_idx)
        rand_idx = sorted(rand_idx[:25])

        namelist = sorted([' '.join(i.split('_')[:-2]) for i in os.listdir('./raw_txt')])
        real_flag = 0
        real_namelist = []

        cls2index = {}
        index2cls = {}
        for idx, clsname in enumerate(namelist):
            if idx in rand_idx:
                cls2index[clsname] = real_flag
                index2cls[real_flag] = clsname
                real_flag += 1
                real_namelist.append(clsname)

        json.dump(cls2index, open('./%d/hmdb-cls2index.json'%split, 'w'))
        json.dump(index2cls, open('./%d/hmdb-index2cls.json'%split, 'w'))
        
                    
        with open('./%d/test.csv'%split, 'w') as f:
            
            for idx, clsname in enumerate(real_namelist):
                prefix = clsname.replace(' ', '_')
                lines = [i.strip().split(' ') for i in open(os.path.join(
                        'raw_txt', 
                        '%s_test_split1.txt'%prefix 
                    )).readlines()]
                lines = [i for i in lines if int(i[1]) == 2 or int(i[1]) == 1]
                
                for i in lines:
                    f.write('%s,%d'%(os.path.join(
                            prefix, i[0]
                        ), idx))
                    f.write('\n')
                    
        """        
        with open('./train.csv', 'w') as f:
            
            for idx, clsname in enumerate(real_namelist):
                prefix = clsname.replace(' ', '_')
                lines = [i.strip().split(' ') for i in open(os.path.join(
                        'raw_txt', 
                        '%s_test_split1.txt'%prefix 
                    )).readlines()]
                lines = [i for i in lines if int(i[1]) == 0]
             
                for i in lines:
                    f.write('%s,%d'%(os.path.join(
                            prefix, i[0]
                        ), idx))
                    f.write('\n')
                
        """
