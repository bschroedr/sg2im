
import pdb
import numpy as np
from sklearn.svm import SVC, LinearSVC # "Support Vector Classifier" 

# https://www.mlgworkshop.org/2018/papers/MLG2018_paper_50.pdf
#  For classification, we follow the data splits of
# [33], with only 140 and 120 nodes for training, respectively, for the
# Cora and Citeseer dataset. Test partitions include 1000 nodes.

src_dir = '/nfs/site/home/brigitsc/layout_models/multimap_aug/'
#src_dir = '/nfs/site/home/brigitsc/layout_models/baseline/'

# load embeddings and category ids
#tr_embeds = np.loadtxt(src_dir + 'all_train_word_embed.txt')
#te_embeds = np.loadtxt(src_dir + 'all_val_word_embed.txt')
tr_embeds = np.loadtxt(src_dir + 'all_train_embed.txt')
tr_ids = np.loadtxt(src_dir + 'all_train_ids.txt') 
te_embeds = np.loadtxt(src_dir + 'all_val_embed.txt')
te_ids = np.loadtxt(src_dir + 'all_val_ids.txt') 
num_elems = len(tr_ids)

# randomize
p = np.random.permutation(num_elems)
tr_embeds = tr_embeds[p]
tr_ids = tr_ids[p]

# train/test splits val
num_train = num_elems
#num_train = 20000 # 
#num_train = 100
#tr_embeds = tr_embeds[0:num_train,:] 
#tr_ids = tr_ids[0:num_train]
#te_embed = all_embeds[num_train:,:]
#te_ids = all_ids[num_train:]
 
# classes
y = tr_ids
X = tr_embeds

# "Support Vector Classifier" 
# https://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane_unbalanced.html
print('training classifier with', num_train, 'samples')
clf = LinearSVC(random_state=0, tol=1e-4, max_iter=3000) 
#clf = SVC(kernel='linear', random_state=0, tol=1e-5) 
# fitting X samples and y classes 
clf.fit(X, y)  
# predict
print('evaluating on val set')
pr = clf.predict(te_embeds)
#clf.score(self, X, y[, sample_weight])	Returns the mean accuracy on the given test data and labels.
mean_accur = clf.score(te_embeds, te_ids)
print('mean accuracy = ', mean_accur)

# class ids are sorted by object frequency!
class_accur = dict()
class_accur['id'] = [] 
class_accur['accur'] = [] 
# record mean accuracy
class_accur['accur'] += [mean_accur]
class_accur['id'] += [999]

unique_ids, u_idx = np.unique(te_ids, return_index=True)
unique_ids = unique_ids[np.argsort(u_idx)]
top_k = 10
count = 0
for u in unique_ids:
  idx = np.where(np.array(te_ids) == u)
  # get class accuracy 
  a = np.sum(pr[idx] == te_ids[idx])/len(pr[idx])
  class_accur['accur'] += [a]
  class_accur['id'] += [u]
  if count < top_k:
    print(u, ':', a)
    count += 1

np.save(src_dir + 'class_accur.npy', class_accur)
# to load results
#r = np.load(src_dir + 'class_accur.npy')
#r = r.item() # r is structured array
#mean_accur = r[-1]
