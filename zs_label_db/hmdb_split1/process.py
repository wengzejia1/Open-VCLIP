import json
import os

if __name__ == '__main__':
    namelist = sorted([' '.join(i.split('_')[:-2]) for i in os.listdir('./raw_txt')])
    cls2index = {}
    index2cls = {}
    for idx, clsname in enumerate(namelist):
        cls2index[clsname] = idx
        index2cls[idx] = clsname
     
    json.dump(cls2index, open('../hmdb-cls2index.json', 'w'))
    json.dump(index2cls, open('../hmdb-index2cls.json', 'w'))

    clsdefine = json.load(open('./classes_label_defn.json','r'))
    detail_cls2index = {}
    detail_index2cls = {}
    for idx, clsname in enumerate(namelist):
        detail_clsname = None
        for line in clsdefine:
            if clsname == line['word']:
                detail_clsname = clsname + '. ' + line['cleaned_defn']
                detail_clsname = detail_clsname[:40]
            
        if detail_clsname is None:
            print("stupid")
            exit()
        else:
            detail_cls2index[detail_clsname] = idx
            detail_index2cls[idx] = detail_clsname
    
    json.dump(detail_cls2index, open('../hmdb-detail-cls2index.json', 'w'))
    json.dump(detail_index2cls, open('../hmdb-detail-index2cls.json', 'w'))
        
    with open('./test.csv', 'w') as f:
        
        for idx, clsname in enumerate(namelist):
            prefix = clsname.replace(' ', '_')
            lines = [i.strip().split(' ') for i in open(os.path.join(
                    'raw_txt', 
                    '%s_test_split1.txt'%prefix 
                )).readlines()]
            lines = [i for i in lines if int(i[1]) == 2]
            
            for i in lines:
                f.write('%s,%d'%(os.path.join(
                        prefix, i[0]
                    ), idx))
                f.write('\n')
                

    with open('./train.csv', 'w') as f:
        
        for idx, clsname in enumerate(namelist):
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
            

