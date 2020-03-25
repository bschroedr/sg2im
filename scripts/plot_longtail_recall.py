
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pdb

#mean_recall_topK100_mmap_aug_so.txt  mean_recall_topK100_multimap_p.txt   mean_recall_topK100_tr_symm.txt
#mean_recall_topK100_multimap.txt     mean_recall_topK100_multimap_s.txt   mean_recall_topK100_tr_symm_so.txt
#mean_recall_topK100_multimap_aug.txt mean_recall_topK100_multimap_so.txt

topK_recall = 100
labels = []
hlabels = []
# 20% classes
hlabels += [ ['mean_recall_topK100_multimap_person_both.txt', 'Triplet-person (H)'] ]
hlabels += [ ['mean_recall_topK100_multimap_sky_both.txt', 'Triplet-sky-other (H)'] ]
#labels += [ ['mean_recall_topK100_multimap_tree_both.txt', 'Triplet-tree (H)'] ]
labels += hlabels
#long tail classes
labels += [ ['mean_recall_topK100_multimap_skateboard_both.txt', 'Triplet-skateboard'] ]
labels += [ ['mean_recall_topK100_multimap_zebra_both.txt', 'Triplet-zebra'] ]
labels += [ ['mean_recall_topK100_multimap_skis_both.txt', 'Triplet-skis'] ]
labels += [ ['mean_recall_topK100_multimap_truck_both.txt', 'Triplet-truck'] ]

labels += [ ['mean_recall_topK100_multimap_cabinet_both.txt', 'Triplet-cabinet'] ]
labels += [ ['mean_recall_topK100_multimap_toilet_both.txt', 'Triplet-toilet'] ]
labels += [ ['mean_recall_topK100_multimap_bowl_both.txt', 'Triplet-bowl'] ]
labels += [ ['mean_recall_topK100_multimap_pavement_both.txt', 'Triplet-pavement'] ]
labels += [ ['mean_recall_topK100_multimap_banana_both.txt', 'Triplet-banana'] ]
labels += [ ['mean_recall_topK100_multimap_motorcycle_both.txt', 'Triplet-motorcycle'] ]

# self repeats - don't use
#labels += [ ['mean_recall_topK100_multimap_laptop_both.txt', 'Triplet-laptop'] ]
#labels += [ ['mean_recall_topK100_multimap_kite_both.txt', 'Triplet-kite'] ]
# too close to motorcycle
#labels += [ ['mean_recall_topK100_multimap_sink_both.txt', 'Triplet-sink'] ]
#labels += [ ['mean_recall_topK100_multimap_door_both.txt', 'Triplet-door-stuff'] ]

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

# plot head vs. long-tail
lt_mean_recall = []
ltlabels = labels[2:]
for i in range(0, len(ltlabels)):
  print(ltlabels[i][1])
  r = np.loadtxt(ltlabels[i][0])
  print(r)
  lt_mean_recall += [r]
lt_mean_recall = np.stack(lt_mean_recall)
lt_mean_recall = np.mean(lt_mean_recall, axis=0)

pdb.set_trace()
h_mean_recall = []
for i in range(0, len(hlabels)):
  print(ltlabels[i][1])
  h_mean_recall += [np.loadtxt(hlabels[i][0])]
pdb.set_trace()
h_mean_recall = np.stack(h_mean_recall)
h_mean_recall = np.mean(h_mean_recall, axis=0)

fig = plt.figure()
plt.style.use('seaborn-whitegrid')
plt.ylim(0,0.8)
plt.xlim(0, topK_recall)
plt.xlabel('k')
plt.ylabel('Recall at k')
legend = []
x = np.arange(1,topK_recall+1)
plt.plot(x, lt_mean_recall, label='Long-tail classes')
plt.plot(x, h_mean_recall, label='Head classes')
l = plt.legend(frameon=True)
l.get_frame().set_facecolor('white')
#plt.legend(legend)
fig.savefig('recall_at_k_plot_freq.png')
plt.show() 
pdb.set_trace()
