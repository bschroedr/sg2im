
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pdb

#mean_recall_topK100_mmap_aug_so.txt  mean_recall_topK100_multimap_p.txt   mean_recall_topK100_tr_symm.txt
#mean_recall_topK100_multimap.txt     mean_recall_topK100_multimap_s.txt   mean_recall_topK100_tr_symm_so.txt
#mean_recall_topK100_multimap_aug.txt mean_recall_topK100_multimap_so.txt

topK_recall = 100
labels = []
###labels += [ ['mean_recall_topK100_multimap_so.txt', 'Triplet-s+o'] ]
###labels += [ ['mean_recall_topK100_tr_symm_so.txt', 'Triplet-DA-s+o'] ]
##labels += [ ['mean_recall_topK100_multimap.txt', 'Triplet-s+p+o'] ]
##labels += [ ['mean_recall_topK100_multimap_spo_random.txt', 'Triplet-s+random+o'] ]
##labels += [ ['mean_recall_topK100_baseline_spo.txt', 'NoTriplet-s+p+o'] ]
###labels += [ ['mean_recall_topK100_baseline_so.txt', 'NoTriplet-s+o'] ]
##labels += [ ['mean_recall_topK100_tr_symm.txt', 'Triplet-DA-s+p+o'] ]
#labels += [ ['mean_recall_topK100_multimap_so.txt', 'tr_s+o'] ]
#labels += [ ['mean_recall_topK100_tr_symm_so.txt', 'tr_symm_da_s+o'] ]
#labels += [ ['mean_recall_topK100_multimap.txt', 'tr_s+p+o'] ]
#labels += [ ['mean_recall_topK100_tr_symm.txt', 'tr_symm_da_s+p+o'] ]
#labels += [ ['mean_recall_topK100_multimap_aug.txt', 'tr_da_s+p+o'] ]
#labels += [ ['mean_recall_topK100_mmap_aug_so.txt', 'tr_da_s+o'] ]
###labels += [ ['mean_recall_topK100_multimap_aug_s.txt', 'Baseline-Triplet-s'] ]
###labels += [ ['mean_recall_topK100_multimap_aug_o.txt', 'Baseline-Triplet-o'] ]
###labels += [ ['mean_recall_topK100_multimap_aug_p.txt', 'Baseline-Triplet-p'] ]
###labels += [ ['mean_recall_topK100_multimap_aug_s_random.txt', 'Random'] ]
#labels += [ ['mean_recall_topK100_multimap_aug_s.txt', 'baseline_tr_da_s'] ]
#labels += [ ['mean_recall_topK100_multimap_aug_o.txt', 'baseline_tr_da_o'] ]
#labels += [ ['mean_recall_topK100_multimap_aug_p.txt', 'baseline_tr_da_p'] ]
#labels += [ ['mean_recall_topK100_tr_symm_person.txt', 'tr_symm_da_person'] ]
#labels += [ ['mean_recall_topK100_multimap_aug_person.txt', 'tr_da_person'] ]
### s+p+o models
labels += [ ['mean_recall_topK100_multimap_skateboard.txt', 'Triplet-skateboard (3)'] ]
labels += [ ['mean_recall_topK100_multimap_zebra.txt', 'Triplet-zebra (2)'] ]
labels += [ ['mean_recall_topK100_multimap_person.txt', 'Triplet-person (1)'] ]
#labels += [ ['mean_recall_topK100_multimap_person.txt', 'tr_person'] ]
#labels += [ ['mean_recall_topK100_tr_symm_dog.txt', 'tr_symm_da_dog'] ]
#labels += [ ['mean_recall_topK100_multimap_aug_dog.txt', 'tr_da_dog'] ]
#labels += [ ['mean_recall_topK100_multimap_dog.txt', 'tr_dog'] ]
#labels += [ ['mean_recall_topK100_tr_symm_zebra.txt', 'tr_symm_da_zebra'] ]
#labels += [ ['mean_recall_topK100_multimap_aug_zebra.txt', 'tr_da_zebra'] ]
#labels += [ ['mean_recall_topK100_multimap_zebra.txt', 'tr_zebra'] ]
#labels += [ ['mean_recall_topK100_tr_symm_skateboard.txt', 'tr_symm_da_skateboard'] ]
#labels += [ ['mean_recall_topK100_multimap_aug_skateboard.txt', 'tr_da_skateboard'] ]
#labels += [ ['mean_recall_topK100_multimap_skateboard.txt', 'tr_skateboard'] ]


# plot recall
fig = plt.figure()
plt.style.use('seaborn-whitegrid')
plt.ylim(0,0.8)
plt.xlim(0, topK_recall)
plt.xlabel('k')
plt.ylabel('Recall at k')
#plt.title('Recall@k: Triplet Retrieval')
legend = []
x = np.arange(1,topK_recall+1)
for i in range(0, len(labels)):
  mean_recall = np.loadtxt(labels[i][0])
  plt.plot(x, mean_recall, label=labels[i][1])
  #plt.plot(x, mean_recall)
  legend += [labels[i][1]]
l = plt.legend(frameon=True)
l.get_frame().set_facecolor('white')
#plt.legend(legend)
fig.savefig('recall_at_k_plot.png')
plt.show()
