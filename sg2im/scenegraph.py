
import os
import pdb
import numpy as np

# pass in copy of objs? 
def corrupt_graph(objs, triples, num_attrs, attrs, vocab, random_seed=None):

  # either s,p,o, s_attrib, o_attrib
  max_corruptable_elements = 5 
  # max num of objs, preds, attrs in vocab
  max_objs = len(vocab['object_idx_to_name'])
  max_preds = len(vocab['pred_idx_to_name'])
  max_attrs = len(vocab['attribute_idx_to_name'])

  # objs is all objects in batch: s/o index into this list  
  num_triples = len(triples)
  s, p, o = np.split(triples, 3, axis=1)
  num_triples = len(triples)
  
  # object ids that index into model vocab
  subj_objs = objs[s]
  obj_objs = objs[o] 
 
  for n in range(0, num_triples): 
    # debug
    subj = np.array(vocab['object_idx_to_name'])[subj_objs[n]]
    pred = np.array(vocab['pred_idx_to_name'])[p[n]]
    obj = np.array(vocab['object_idx_to_name'])[obj_objs[n]]
    print(tuple([subj, pred, obj]))
    pdb.set_trace()

    # let's corrupt some part of each triple to avoid exact matches - 
    # randomly selected
    element = np.random.randint(0, max_corruptable_elements-1)
    element = 0
    if element == 0: # s
         # add new obj to objs
         new_obj = select_object(max_objs)
         objs += new_obj
         s[n] = len(objs)-1
    elif element == 1: # p
         p[n] = select_predicate(max_preds)
    elif element == 2: # o
         new_obj = select_object(max_objs)
         objs += new_obj
         s[n] = len(objs)-1
    elif element == 3: # s_attrib
         a = select_attribute(max_attrs)
    elif element == 4: # o_attrib
         a = select_attribute(max_attrs)
 
  pdb.set_trace()
  return 0


def select_object(num_objs):
  return np.random.randint(0, num_objs-1)
  pdb.set_trace() 

def select_predicate(num_preds):
  return np.random.randint(0, num_preds-1)

def select_attrib(num_attrs):
  return np.random.randint(0, num_attrs-1) 
