import os
from tqdm import tqdm

ROOT = '/share_io02_ssd/jia/kinetics-compress'

if __name__ == '__main__':
    # process test csv
    
    lines = open('old_train.csv', 'r').readlines()
    with open('train.csv', 'w') as f:
        for line in lines:
            line = line.split(' ')
            line = [i.strip() for i in line]
            head = line[0].split('/')[-2:]
            head = os.path.join('train', head[0], head[1])
            tail = line[1]
            new_line = head + ',' + tail
            f.write(new_line)
            f.write('\n')
    
    lines = open('old_val.csv', 'r').readlines()
    with open('val.csv', 'w') as f:
        for line in lines:
            line = line.split(' ')
            line = [i.strip() for i in line]
            head = line[0].split('/')[-2:]
            head = os.path.join('val', head[0], head[1])
            tail = line[1]
            new_line = head + ',' + tail
            f.write(new_line)
            f.write('\n')
        

        
