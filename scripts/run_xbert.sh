#!/bin/bash

# set DEPTH
if [ $1 == 'Eurlex-4K' ]; then
	DATASET=Eurlex-4K
	DEPTH=6
	TRAIN_BATCH_SIZE=32
	EVAL_BATCH_SIZE=64
	LOG_INTERVAL=100
	EVAL_INTERVAL=400
	NUM_TRAIN_EPOCHS=20
	LEARNING_RATE=5e-5
	WARMUP_RATE=0.1
elif [ $1 == 'Wiki10-31K' ]; then
	DATASET=Wiki10-31K
	DEPTH=9
	TRAIN_BATCH_SIZE=32
	EVAL_BATCH_SIZE=64
	LOG_INTERVAL=50
	EVAL_INTERVAL=150
	NUM_TRAIN_EPOCHS=15
	LEARNING_RATE=5e-5
	WARMUP_RATE=0.1
elif [ $1 == 'AmazonCat-13K' ]; then
	DATASET=AmazonCat-13K
	DEPTH=8
	TRAIN_BATCH_SIZE=36
	EVAL_BATCH_SIZE=64
	LOG_INTERVAL=1000
	EVAL_INTERVAL=10000
	NUM_TRAIN_EPOCHS=5
	LEARNING_RATE=5e-5
	WARMUP_RATE=0.1
elif [ $1 == 'Wiki-500K' ]; then
	DATASET=Wiki-500K
	DEPTH=13
	TRAIN_BATCH_SIZE=36
	EVAL_BATCH_SIZE=64
	LOG_INTERVAL=1000
	EVAL_INTERVAL=15000
	NUM_TRAIN_EPOCHS=12
	LEARNING_RATE=8e-5
	WARMUP_RATE=0.2
else
	echo "unknown dataset for the experiment!"
	exit
fi
GPUS=0,1,2,3
GPUS=0,2,3,5,6,7
LABEL_EMB_LIST=( elmo pifa )
ALGO_LIST=( 0 5 )
SEED_LIST=( 0 1 2 )
MATCHER=xbert

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
		CUDA_VISIBLE_DEVICES=${GPUS} python -u -m xbert.matcher.bert \
			-i ${OUTPUT_DIR}/data-bin-${MATCHER}/data_dict.pt \
			-o ${OUTPUT_DIR}/matcher/${MATCHER} \
			--bert_model bert-base-uncased \
			--do_train --do_eval --stop_by_dev \
			--learning_rate ${LEARNING_RATE} \
			--warmup_proportion ${WARMUP_RATE} \
			--train_batch_size ${TRAIN_BATCH_SIZE} \
			--eval_batch_size ${EVAL_BATCH_SIZE} \
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
	done
done
