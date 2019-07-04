#!/bin/bash

# set DEPTH
if [ $1 == 'Eurlex-4K' ]; then
	DATASET=Eurlex-4K
	DEPTH=6
elif [ $1 == 'Wiki10-31K' ]; then
	DATASET=Wiki10-31K
	DEPTH=9
elif [ $1 == 'AmazonCat-13K' ]; then
	DATASET=AmazonCat-13K
	DEPTH=8
elif [ $1 == 'Wiki-500K' ]; then
	DATASET=Wiki-500K
	DEPTH=13
else
	echo "unknown dataset for the experiment!"
	exit
fi

LABEL_EMB_LIST=( elmo pifa )
ALGO_LIST=( 0 5 )
SEED_LIST=( 0 1 2 )

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
			-o ${OUTPUT_DIR}/ranker/tst.prediction.npz
	done
done

