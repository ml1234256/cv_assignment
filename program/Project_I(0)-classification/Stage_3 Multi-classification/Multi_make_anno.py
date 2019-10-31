# It's empty. Surprise!
# Please complete this by yourself.
import pandas as pd
import os
from PIL import Image

ROOTS = '../Dataset/'
PHASE = ['train', 'val']
CLASSES = ['Mammals', 'Birds'] # [0, 1]
SPECIES = ['rabbits', 'rats', 'chickens'] # [0, 1, 2]

DATA_info = {
    'train': {'path': [], 'classes':[], 'species':[]},
    'val': {'path': [], 'classes': [], 'species': []}
}

for p in PHASE:
    for s in SPECIES:
        DATA_DIR = ROOTS + p + '/' + s
        DATA_NAME = os.listdir(DATA_DIR)

        for item in DATA_NAME:
            Image_path = os.path.join(DATA_DIR, item)
            try:
                img = Image.open(Image_path)
            except OSError:
                pass
            DATA_info[p]['path'].append(Image_path)
            if s == 'rabbits':
                DATA_info[p]['classes'].append(0)
                DATA_info[p]['species'].append(0)
            elif s == 'rats':
                DATA_info[p]['classes'].append(0)
                DATA_info[p]['species'].append(1)
            else:
                DATA_info[p]['classes'].append(1)
                DATA_info[p]['species'].append(2)
    annotation = pd.DataFrame(DATA_info[p])
    #print(annotation)
    annotation.to_csv('Multi_%s_annotation.csv' % p)
    print('Multi_%s_annotation file is saved.' % p)


# if __name__ == '__main__':
#     print(os.path.exists(Image_path))

