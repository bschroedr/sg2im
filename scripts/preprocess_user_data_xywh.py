
# preprocess user study data
import os
import csv
import pdb

#in_dir = 'generate_samples_xywh/'
in_dir = 'generate_samples_xywh/'
out_dir = 'generate_samples_labelbox/'
in_file_name_iou = 'image_iou_list.txt'
#in_file_name_xywh = 'image_iou_xywh_list.txt'
in_file_name_xywh = 'image_iou_xywh_list2.txt' 
img_count = 75 

fnames = []
 
with open(os.path.join(out_dir, in_file_name_iou)) as f:
    reader = csv.reader(f, delimiter="\t")
    d = list(reader)
    fnames = [row[0] for row in d]

f.close()
# load file contain iou/xywh data
# make dictionary out of this
iou_xywh = {}
with open(os.path.join(in_dir, in_file_name_xywh)) as fs:
    readers = csv.reader(fs, delimiter="\t")
    d = list(readers)
    for row in d:
        row[0] = os.path.basename(row[0])
        iou_xywh[row[0]] = row

fs.close()

# reconstruct data matrix to be read in by torch
all_data = [iou_xywh[n] for n in fnames]

with open(os.path.join(out_dir, in_file_name_xywh), 'wt') as ff:
    writer = csv.writer(ff, delimiter="\t")
    for row in all_data:
        writer.writerow(row)

ff.close()
