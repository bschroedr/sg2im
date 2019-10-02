
import pdb
import numpy as np
from sklearn.svm import SVC # "Support Vector Classifier" 

# https://www.mlgworkshop.org/2018/papers/MLG2018_paper_50.pdf
#  For classification, we follow the data splits of
# [33], with only 140 and 120 nodes for training, respectively, for the
# Cora and Citeseer dataset. Test partitions include 1000 nodes.

# load embeddings and category ids
tr_embeds = np.loadtxt('all_train_word_embed.txt')
tr_ids = np.loadtxt('all_train_ids.txt') 
te_embeds = np.loadtxt('all_val_word_embed.txt')
te_ids = np.loadtxt('all_val_ids.txt') 
num_elem = len(tr_ids)

pdb.set_trace()

# randomize
p = np.random.permutation(num_elem)
tr_embeds = tr_embeds[p]
tr_ids = tr_ids[p]

# train/test splits val
num_train = 2000 # ~10% of ~20K objects in train set 
#num_train = 500
#num_train = 3000
tr_embeds = tr_embeds[0:num_train,:] 
tr_ids = tr_ids[0:num_train]
#te_embed = all_embeds[num_train:,:]
#te_ids = all_ids[num_train:]
 
# classes
y = tr_ids
X = tr_embeds

# "Support Vector Classifier" 
# https://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane_unbalanced.html
print('training classifier with', num_train, 'samples')
clf = SVC(kernel='linear', random_state=0, tol=1e-5) 
# fitting X samples and y classes 
clf.fit(X, y)  
# predict
print('evaluating on val set')
#pr = clf.predict(te_embed)
#clf.score(self, X, y[, sample_weight])	Returns the mean accuracy on the given test data and labels.
mean_accur = clf.score(te_embed, te_ids)

print('mean accuracy = ', mean_accur)
