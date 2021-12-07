
# preprocess user study data
import os
import csv
import random
import shutil

in_dir = 'generate_samples_debug/'
out_dir = 'generate_samples_labelbox/'
file_name = 'image_iou_list.txt'
img_count = 75 

with open(os.path.join(in_dir, file_name)) as f:
    reader = csv.reader(f, delimiter="\t")
    d = list(reader)

random.shuffle(d)
shutil.rmtree(out_dir)
os.makedirs(out_dir) #, exist_ok=True)

with open(out_dir + file_name, 'wt') as f:
    writer = csv.writer(f, delimiter="\t")
    for row in d[0:img_count]:
        shutil.copy(row[0], out_dir)
        row[0] = os.path.basename(row[0])
        writer.writerow(row)
