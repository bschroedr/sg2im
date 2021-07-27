import os
import pdb
import json
import numpy as np
from tqdm import tqdm

# ConceptNet word embeddings -  better than GLOVE! 
WORD_EMBED='/home/brigit/sandbox/sg2im_brigit/word_embeddings/numberbatch-en-19.08.txt'
JSON_VOCAB='/home/brigit/sandbox/sg2im_brigit/datasets/vg/vocab.json'

# load word embeddings
embeddings_dict = {}
head = True
with open(WORD_EMBED, 'r') as f:
  for line in tqdm(f):
    if head:
      head = False
      continue
    values = line.split()
    word = values[0]
    vector = np.asarray(values[1:], "float32")
    embeddings_dict[word] = vector

# load VG vocab for objects and relationships
with open(JSON_VOCAB) as f:
  vocab = json.load(f)

objects = list(vocab['object_name_to_idx'].keys())
predicates = list(vocab['pred_name_to_idx'].keys())
predicates_clean = []

# clean predicate list - to deal with cases like 'walking on'
for p in predicates:
   vals = p.split()
   if len(vals) <= 2: # 'walking on', 'eating'
     predicates_clean.append(vals[0])
   elif len(vals) == 3: # 'on top of'
     predicates_clean.append(vals[1])

# 

pdb.set_trace()
dist = []
for obj in objects:
  if obj in embeddings_dict:
    key = obj
  else: 
    key = "none"
  for other_obj in objects:
    if other_obj in embeddings_dict:
      dist.append(np.dot(embeddings_dict[key], embeddings_dict[other_obj]))
    else:
      dist.append(10000000) # do we want two queries that have 'none' to be similar?
  print('sort synonyms')
  sort_idx = np.argsort(dist)
  sort_objs = np.array(objects)[sort_idx]
  sort_dist = np.array(dist)[sort_idx]
  pdb.set_trace()
  
    

for pred in predicates_clean:
  if obj in embeddings_dict:
    key = obj
  else: 
    key = "none"
