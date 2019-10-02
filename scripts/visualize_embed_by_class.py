
import numpy as np
import sg2im.db_utils as db_utils
import pdb
from tsne import bh_sne
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

db = db_utils.read_fr_JSON('tree.json')
#db = db_utils.read_fr_JSON('person.json')

# assign each triplet an id 
all_embeds = []
all_ids = []
all_objs = []
id = 0

vocab = dict()
vocab['pred_idx_to_name'] = [
      '__in_image__',
      'left of',
      'right of',
      'above',
      'below',
      'inside',
      'surrounding',
      'behind',
      'infront of',
      'on',
      'under',
      'adjacent right',
      'adjacent left',
      'adjacent right stuff',
      'adjacent left stuff'
    ]
vocab['pred_name_to_idx'] = {}
for idx, name in enumerate(vocab['pred_idx_to_name']):
  vocab['pred_name_to_idx'][name] = idx

for k in db.keys():
  print(k, ": ", len(db[k]))
  s = db_utils.string_to_tuple(k)
  p = s[1]
  o = s[2]
  for i in range(len(db[k])):
    p = vocab['pred_name_to_idx'][p] 
    # visualize only (original) mutually exclusive relationships
    if p > 6:
      continue
    #all_embeds += [db[k][i]['object_embed']]
    all_embeds += [db[k][i]['predicate_embed']]
    #all_embeds += [db[k][i]['subject_embed']]
    all_ids.append(p)
    sc = db[k][i]['object_supercat']
    all_objs.append(sc + ' ' + o)
    #all_objs.append(o)

# tsne
X = np.array(all_embeds, dtype=np.float64).squeeze()
y = np.array(all_ids)
from sklearn.manifold import TSNE
X_2d = TSNE(n_components=2, perplexity=30).fit_transform(X)
#X_2d = bh_sne(X)

pdb.set_trace()

# plot by relation type
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='Set1' )
#plt.title('tree by clustered by object type (object embedding)')
plt.title('tree by clustered by predicate type (predicate embedding)')
plt.colorbar()
for i, txt in enumerate(all_objs):
    plt.annotate('  '+txt, (X_2d[i, 0], X_2d[i, 1]))
plt.show()

