#!/bin/bash

# set DEPTH
if [ $1 == 'Eurlex-4K' ]; then
	DATASET=Eurlex-4K
elif [ $1 == 'Wiki10-31K' ]; then
	DATASET=Wiki10-31K
elif [ $1 == 'AmazonCat-13K' ]; then
	DATASET=AmazonCat-13K
elif [ $1 == 'Wiki-500K' ]; then
	DATASET=Wiki-500K
else
	echo "unknown dataset for the experiment!"
	exit
fi
GPUS=0
LABEL_EMB_LIST=( elmo pifa )
ALGO_LIST=( 0 5 )
SEED_LIST=( 0 1 2 )
MATCHER=xttention

for idx in "${!LABEL_EMB_LIST[@]}"; do
	ALGO=${ALGO_LIST[$idx]}
	LABEL_EMB=${LABEL_EMB_LIST[$idx]}
	for SEED in "${SEED_LIST[@]}"; do
		# neural matcher
		OUTPUT_DIR=pretrained_models/${DATASET}/${LABEL_EMB}-a${ALGO}-s${SEED}
		CUDA_VISIBLE_DEVICES=${GPUS} python -u -m xbert.matcher.attention \
			-i ${OUTPUT_DIR}/data-bin-${MATCHER}/data_dict.pt \
			-o ${OUTPUT_DIR}/matcher/${MATCHER} \
			--do_eval --cuda \
			--init_checkpoint_dir ${OUTPUT_DIR}/matcher/${MATCHER}
		
		# ranker (default: matcher=hierarchical linear)
		OUTPUT_DIR=pretrained_models/${DATASET}/${LABEL_EMB}-a${ALGO}-s${SEED}
		python -m xbert.ranker predict \
			-m ${OUTPUT_DIR}/ranker \
			-x datasets/${DATASET}/X.tst.npz \
			-y datasets/${DATASET}/Y.tst.npz \
			-c ${OUTPUT_DIR}/matcher/${MATCHER}/C_eval_pred.npz \
			-o ${OUTPUT_DIR}/ranker/tst.pred.xttention.npz
	done
done

