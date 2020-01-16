
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pdb

#mean_recall_topK100_mmap_aug_so.txt  mean_recall_topK100_multimap_p.txt   mean_recall_topK100_tr_symm.txt
#mean_recall_topK100_multimap.txt     mean_recall_topK100_multimap_s.txt   mean_recall_topK100_tr_symm_so.txt
#mean_recall_topK100_multimap_aug.txt mean_recall_topK100_multimap_so.txt

topK_recall = 100
labels = []
labels += [ ['mean_recall_topK100_tr_symm.txt', 'tr_symm_da_s+p+o'] ]
labels += [ ['mean_recall_topK100_multimap_aug.txt', 'tr_da_s+p+o'] ]
labels += [ ['mean_recall_topK100_multimap.txt', 'tr_s+p+o'] ]
labels += [ ['mean_recall_topK100_tr_symm_so.txt', 'tr_symm_da_s+o'] ]
labels += [ ['mean_recall_topK100_mmap_aug_so.txt', 'tr_da_s+o'] ]
labels += [ ['mean_recall_topK100_multimap_so.txt', 'tr_s+o'] ]
labels += [ ['mean_recall_topK100_multimap_aug_s.txt', 'baseline_tr_da_s'] ]
labels += [ ['mean_recall_topK100_multimap_aug_p.txt', 'baseline_tr_da_p'] ]
labels += [ ['mean_recall_topK100_multimap_aug_o.txt', 'baseline_tr_da_o'] ]


# plot recall
fig = plt.figure()
plt.style.use('seaborn-whitegrid')
plt.ylim(0,0.6)
plt.xlim(0, topK_recall)
plt.xlabel('k')
plt.ylabel('Recall at k')
plt.title('Recall@k: Triplet Retrieval')
legend = []
x = np.arange(1,topK_recall+1)
for i in range(0, len(labels)):
  mean_recall = np.loadtxt(labels[i][0])
  plt.plot(x, mean_recall)
  legend += [labels[i][1]]
plt.legend(legend)
plt.show()
