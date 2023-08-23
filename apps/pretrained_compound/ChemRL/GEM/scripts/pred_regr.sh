#!/bin/bash
cd $(dirname $0)
cd ..

source ~/.bashrc
source ./scripts/utils.sh

root_path="$(pwd)/../../../.."
export PYTHONPATH="$root_path/":$PYTHONPATH
datasets="qm7"
compound_encoder_config="model_configs/geognn_l8.json"
init_model="./pretrain_models-chemrl_gem/regr.pdparams"
log_prefix="log/pretrain"
thread_num=4
count=0
conda activate paddlehelix
for dataset in $datasets; do
	echo "==> $dataset"
	data_path="./chemrl_downstream_datasets/$dataset"
	cached_data_path="./cached_data/$dataset"
	if [ ! -f "$cached_data_path.done" ]; then
		rm -r $cached_data_path
		python predict.py \
				--task=data \
				--num_workers=20 \
				--dataset_name=$dataset \
				--data_path=$data_path \
				--cached_data_path=$cached_data_path \
				--compound_encoder_config=$compound_encoder_config \
				--model_config="model_configs/down_mlp2.json"
		if [ $? -ne 0 ]; then
			echo "Generate data failed for $dataset"
			exit 1
		fi
		touch $cached_data_path.done
	fi

	model_config="model_configs/down_mlp3.json"
	if [ "$dataset" == "qm8" ] || [ "$dataset" == "qm9" ]; then
		batch_size=256
	elif [ "$dataset" == "freesolv" ]; then
		batch_size=30
	else
		batch_size=256
	fi

	{
		CUDA_VISIBLE_DEVICES=0 python predict.py \
			--batch_size=$batch_size \
			--dataset_name=$dataset \
			--data_path=$data_path \
			--cached_data_path=$cached_data_path \
			--split_type=scaffold \
			--compound_encoder_config=$compound_encoder_config \
			--model_config=$model_config > pred.log 
	} &
done
wait

