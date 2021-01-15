
import sys
sys.path.append("./") # in order to include sg2im as module
import argparse
import pprint
import os
import json
import math
from collections import defaultdict
from collections import namedtuple
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.models as models
from sg2im.data.vg_eval import VgSceneGraphDataset, vg_collate_fn
import pdb

VG_DIR = '/Users/brigit/datasets/vg'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='vg', choices=['vg', 'coco'])

# Optimization hyperparameters
parser.add_argument('--batch_size', default=32, type=int)

# Dataset options common to both VG and COCO
parser.add_argument('--loader_num_workers', default=4, type=int)
parser.add_argument('--include_relationships', default=True, type=bool)

# VG-specific options
parser.add_argument('--vg_dir', default=VG_DIR)
parser.add_argument('--use_split', default='val', choices=['val', 'test'])
parser.add_argument('--num_val_samples', default=1024, type=int)
parser.add_argument('--max_objects_per_image', default=10, type=int)
parser.add_argument('--vg_use_orphaned_objects', default=True, type=bool)
parser.add_argument('--random_seed', default=42, type=int)


def generate_db(args, loader, vocab):
  num_samples = 0 
  #triple_data_to_entry = {}
  image_id_to_entries = {}
  e = 0
  img_count = 0
  db = [] 
  # define named tuple 
  #TripleData = namedtuple("TripleData", ["relationship_id", "image_id"]) 
  # iterate over all batches of images
  with torch.no_grad():
    for batch in loader:
      if torch.cuda.is_available():
        batch = [tensor.cuda() for tensor in batch]
      else:
        batch = [tensor for tensor in batch]
     
      print('Batch loading data')

      # get batch
      imgs, objs, boxes, triples, obj_to_img, triple_to_img, num_attrs, attrs, urls = batch

      num_samples += args.batch_size 
      if num_samples > args.num_val_samples:
        break

      for i in range(0, args.batch_size): 
        objs_index = np.where(obj_to_img.numpy() == i) # objects indices for image in batch
        objs_img = objs[objs_index] # object class ids for image
        #obj_names = np.array(vocab['object_idx_to_name'])[objs_img] # object class labels 
        obj_boxes = boxes[objs_index] 
        img_url = urls[i]
        # temporary
        img_id = img_count
        img_count += 1

        # process all triples (e.g. relationships) in image
        tr_index = np.where(triple_to_img == i)
        tr_img = triples[tr_index]
        num_triples = len(tr_img)

        # s,o: indices for "objs" array (yields 'object_idx' for 'object_idx_to_name')
        # p: use this value as is (yields 'pred_idx' for 'pred_idx_to_name')
        s, p, o, rel_ids = np.split(tr_img, 4, axis=1)

        #pdb.set_trace()
        for n in range(0, num_triples):
          subj_index = s[n]
          subj = np.array(vocab['object_idx_to_name'])[objs[subj_index]]
          pred = np.array(vocab['pred_idx_to_name'])[p[n]]
          obj_index = o[n]
          obj = np.array(vocab['object_idx_to_name'])[objs[obj_index]]
          if obj == '__image__': # singleton objects
            continue
          rel_id = rel_ids[n].item()
          subj_box = boxes[subj_index].tolist()
          obj_box = boxes[obj_index].tolist()
          relationship = {}
          relationship['metadata'] = {'image_url': img_url, 'vg_scene_id': img_id, 'vg_relationship_id': rel_id}
          relationship['data'] = {'s_box': subj_box, 'o_box': obj_box, 'subject': subj, 'predicate': pred, 'object': obj}
          db.append(relationship)
          pprint.pprint(relationship)
          # keep track of which relationship and image belong to each db entry
          #triple_data_to_entry[TripleData(relationship_id=rel_id, image_id=img_id)] = e
          #image_id_to_entries[img_id].update({e})
          if img_id not in image_id_to_entries:
            image_id_to_entries[img_id] = [e]
          elif img_id in image_id_to_entries:
            image_id_to_entries[img_id] += [e]
          e += 1

  return db, image_id_to_entries

def generate_queries(args, db, image_id_to_entries, num_queries=1000):
  print('Database has ', len(db), ' documents.')
  queries = {}
  q = 0
  num_entries = len(db)
  # need to associate with number of queries
  img_ids = range(0,100) # permute randomly based upon num_queries
  entries_to_delete = []
  for i in img_ids: 
    query_id = image_id_to_entries[i][0] # query is first triple in set
    queries[q] = db[query_id]
    q += 1
    entries_to_delete += image_id_to_entries[i]

  # delete queries from db
  for e in entries_to_delete:
    db[e] = ['void']
  db = [x for x in db if x != ['void']]

  return db, queries

def build_vg_dsets(args):
  with open(args.vocab_json, 'r') as f:
    vocab = json.load(f)
  dset_kwargs = {
    'vocab': vocab,
    'h5_path': args.val_h5,
    'image_dir': None, # None indicates omit images 
    'max_samples': args.num_val_samples,
    'max_objects': args.max_objects_per_image,
    'use_orphaned_objects': args.vg_use_orphaned_objects,
    'include_relationships': args.include_relationships,
    'seed': args.random_seed
  }

  dset_kwargs['h5_path'] = args.val_h5
  del dset_kwargs['max_samples']
  vg_dset = VgSceneGraphDataset(**dset_kwargs)

  return vocab, vg_dset

def build_loaders(args):
  if args.dataset == 'vg':
    vocab, vg_dset = build_vg_dsets(args)
    collate_fn = vg_collate_fn

  loader_kwargs = {
    'batch_size': args.batch_size,
    'num_workers': args.loader_num_workers,
    'shuffle': False,
    'collate_fn': collate_fn,
  }

  val_loader = DataLoader(vg_dset, **loader_kwargs)
  return vocab, val_loader

def main(args):

  # data loader for VG
  vocab, val_loader = build_loaders(args)
  # create test db 
  db, image_id_to_entries = generate_db(args, val_loader, vocab)  
  # generate query db
  db, queries = generate_queries(args, db, image_id_to_entries)
  # write documents and queries
  with open('docs.json', 'w') as json_file:
    json.dump(db, json_file)
  with open('queries.json', 'w') as json_file:
    json.dump(queries, json_file)
  
if __name__ == '__main__':
  args = parser.parse_args()
  # where VG data is located
  args.val_h5 = os.path.join(VG_DIR, args.use_split + '.h5')
  args.vocab_json = os.path.join(VG_DIR, 'vocab.json')
  main(args)
