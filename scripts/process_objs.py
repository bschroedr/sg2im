
import pdb
import random
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import sg2im.db_utils as db_utils

coco_object_db_json = 'coco_object_db_multimap.json' 
db = db_utils.read_fr_JSON(coco_object_db_json)

# delete singlton objects
if '__image__' in db.keys():
  del db['__image__']
    
sort_objs_by_count = sorted(db, key=lambda x: db[x].get('count',0), reverse=True)
sort_count = [db[k]['count'] for k in sort_objs_by_count]

# plot distribution of sorted objects 
fig = plt.figure()
stc = sort_objs_by_count
sc = sort_count
x_pos = np.arange(len(sc))
plt.bar(x_pos, sc, align='center', alpha=0.5)
plt.xticks(x_pos, stc)
plt.yticks(np.arange(0, max(sc)+1, 100))
plt.xlim((0,len(sc)))
plt.ylabel("object count")
plt.xlabel("object label")
plt.title("MS-COCO: Object Distribution")
plt.setp(plt.gca().get_xticklabels(), rotation=90, horizontalalignment='right', fontsize=6)
plt.tight_layout()
plt.show()
