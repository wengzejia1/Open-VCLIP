import os
from tqdm import tqdm

ROOT = '/share_io02_ssd/jia/kinetics-compress'

if __name__ == '__main__':
    # process test csv
    with open('test.csv', 'w') as output_f:
        filename = '../full_splits/test.csv'
        f = open(filename, 'r')
        lines = f.readlines()
        f_num = len(lines)
        for idx, line in tqdm(enumerate(lines)):
            line = line.strip()
            pth, cls = line.split(",")
            cls = int(cls)
            # remove test pth tail
            assert pth.split('.')[1] == 'mp4'
            pth = pth.split('.')[0][:-14] + '.' + pth.split('.')[1]
            pth = pth.replace(' ', '_')
            
            if (os.path.exists(os.path.join(ROOT, pth))):
                pass
            elif (os.path.exists(os.path.join(ROOT, pth[:-4]+'.mkv'))):
                pth = pth[:-4]+'.mkv'
            elif (os.path.exists(os.path.join(ROOT, pth+'.mkv'))):
                pth = pth+'.mkv'
            elif (os.path.exists(os.path.join(ROOT, pth[:-4]+'.webm'))):
                pth = pth[:-4]+'webm'
            elif os.path.exists(os.path.join(ROOT, pth+'.webm')):
                pth = pth + '.webm'
            else:
                print('missing video')
                print(pth)
                exit()
            
            new_line = '%s,%s'%(pth, cls)
            output_f.write(new_line)
            output_f.write('\n')
    
    # process train csv
    with open('train.csv', 'w') as output_f:
        filename = '../full_splits/train.csv'
        f = open(filename, 'r')
        lines = f.readlines()
        f_num = len(lines)
        for idx, line in tqdm(enumerate(lines)):
            line = line.strip()
            pth, cls = line.split(",")
            cls = int(cls)
            pth = pth.replace(' ', '_')
        
            if (os.path.exists(os.path.join(ROOT, pth))):
                pass
            else:
                print('missing video')
                print(pth)

            new_line = '%s,%s'%(pth, cls)
            output_f.write(new_line)
            output_f.write('\n')

 
        



