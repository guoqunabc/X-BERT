#!/bin/bash

# set DEPTH
if [ $1 == 'Eurlex-4K' ]; then
	DATASET=Eurlex-4K
	DEPTH=6
	TRAIN_BATCH_SIZE=128
	LOG_INTERVAL=10
	EVAL_INTERVAL=50
	NUM_TRAIN_EPOCHS=12
elif [ $1 == 'Wiki10-31K' ]; then
	DATASET=Wiki10-31K
	DEPTH=9
	TRAIN_BATCH_SIZE=128
	LOG_INTERVAL=10
	EVAL_INTERVAL=40
	NUM_TRAIN_EPOCHS=20
elif [ $1 == 'AmazonCat-13K' ]; then
	DATASET=AmazonCat-13K
	DEPTH=8
	TRAIN_BATCH_SIZE=256
	LOG_INTERVAL=100
	EVAL_INTERVAL=1300
	NUM_TRAIN_EPOCHS=6
elif [ $1 == 'Wiki-500K' ]; then
	DATASET=Wiki-500K
	DEPTH=13
	TRAIN_BATCH_SIZE=256
	LOG_INTERVAL=100
	EVAL_INTERVAL=1500
	NUM_TRAIN_EPOCHS=9
else
	echo "unknown dataset for the experiment!"
	exit
fi
GPUS=0,1,4,5,6,7
GPUS=0
LABEL_EMB_LIST=( elmo pifa )
ALGO_LIST=( 0 5 )
SEED_LIST=( 0 1 2 )
MATCHER=xttention

for idx in "${!LABEL_EMB_LIST[@]}"; do
	ALGO=${ALGO_LIST[$idx]}
	LABEL_EMB=${LABEL_EMB_LIST[$idx]}
	for SEED in "${SEED_LIST[@]}"; do

		# indexer
		OUTPUT_DIR=save_models/${DATASET}/${LABEL_EMB}-a${ALGO}-s${SEED}
		mkdir -p $OUTPUT_DIR/indexer
		python -m xbert.indexer \
			-i datasets/${DATASET}/L.${LABEL_EMB}.npz \
			-o ${OUTPUT_DIR}/indexer \
			-d ${DEPTH} --algo ${ALGO} --seed ${SEED} --max-iter 20
		
		# preprocess data_bin for neural matcher
		OUTPUT_DIR=save_models/${DATASET}/${LABEL_EMB}-a${ALGO}-s${SEED}
		mkdir -p $OUTPUT_DIR/data-bin-${MATCHER}
		python -m xbert.preprocess \
			-m ${MATCHER} \
			-i datasets/${DATASET} \
			-c ${OUTPUT_DIR}/indexer/code.npz \
			-o ${OUTPUT_DIR}/data-bin-${MATCHER}
		
		# neural matcher
		OUTPUT_DIR=save_models/${DATASET}/${LABEL_EMB}-a${ALGO}-s${SEED}
		mkdir -p ${OUTPUT_DIR}/matcher/${MATCHER}
		CUDA_VISIBLE_DEVICES=${GPUS} python -u -m xbert.matcher.attention \
			-i ${OUTPUT_DIR}/data-bin-${MATCHER}/data_dict.pt \
			-o ${OUTPUT_DIR}/matcher/${MATCHER} \
			--do_train --do_eval --cuda --stop_by_dev \
			--train_batch_size ${TRAIN_BATCH_SIZE} \
			--num_train_epochs ${NUM_TRAIN_EPOCHS} \
			--log_interval ${LOG_INTERVAL} \
			--eval_interval ${EVAL_INTERVAL} \
			|& tee ${OUTPUT_DIR}/matcher/${MATCHER}.log

		# ranker (default: matcher=hierarchical linear)
		OUTPUT_DIR=save_models/${DATASET}/${LABEL_EMB}-a${ALGO}-s${SEED}
		mkdir -p $OUTPUT_DIR/ranker
		python -m xbert.ranker train \
			-x datasets/${DATASET}/X.trn.npz \
			-y datasets/${DATASET}/Y.trn.npz \
			-c ${OUTPUT_DIR}/indexer/code.npz \
			-o ${OUTPUT_DIR}/ranker

		python -m xbert.ranker predict \
			-m ${OUTPUT_DIR}/ranker \
			-x datasets/${DATASET}/X.tst.npz \
			-y datasets/${DATASET}/Y.tst.npz \
			-c ${OUTPUT_DIR}/matcher/${MATCHER}/C_eval_pred.npz \
			-o ${OUTPUT_DIR}/ranker/tst.prediction.npz
		exit	
	done
done
