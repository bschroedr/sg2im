#!/bin/sh

interval=5000
i=$1 
i_stop=$2
model=$3
sep="_"

while [ $i -le $i_stop ]
do
        file=/home/brigit/layout_models/$model/checkpoint_with_model_$i.pt
	if [ ! -f "$file" ]; then
		echo $file does not exist!
	else
        	OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 python scripts/extract_embeddings_val_tsne_vg_ss.py --checkpoint=/home/brigit/layout_models/$model/checkpoint_with_model_$i.pt --batch_size 32 --num_val_samples 1025 --image_size 128,128  --model_label "$model$sep$i"  --dataset vg  --visualize_retrieval 0  
	fi
        i=$(($i+$interval))
done

txt_files=$(ls recall_$model*.txt)
rm $model.txt
cat $txt_files >> $model.txt
