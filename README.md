# eXtreme Multi-label Text Classification with BERT

This is a README for the experimental code in our paper
>[X-BERT: eXtreme Multi-label Text Classification with BERT](#)

>Wei-Cheng Chang, Hsiang-Fu Yu, Kai Zhong, Yiming Yang, Inderjit Dhillon

>Preprint 2019



## Installation

### Requirements
  * conda 
  * python=3.6
  * cuda=9.0
  * Pytorch=0.4.1
  * pytorch-pretrained-BERT=0.6.2
  * allenlp=0.8.4
  
### Depedencies via Conda Environment
	
	> conda create -n xbert-env python=3.6
	> source activate xbert-env
	> (xbert-env) conda install scikit-learn
	> (xbert-env) conda install pytorch=0.4.1 cuda90 -c pytorch
	> (xbert-env) pip install pytorch-pretrained-bert==0.6.2
	> (xbert-env) pip install allennlp==0.8.4
	> (xbert-env) pip install -e .
	
**Warning: you need to install pytorch=0.4.1 based on the cuda version on your machine.

**Notice: the following examples are executed under the ```> (xbert-env)``` conda virtual environment

## Reproduce Evaulation Results in the Paper
We demonstrate how to reproduce the evaluation results in our paper
by downloading the raw dataset and pretrained models.

### Download Dataset (Eurlex-4K, Wiki10-31K, AmazonCat-13K, Wiki-500K)
Change directory into ./datasets folder, download and unzip each dataset

```bash
cd ./datasets
bash download-data.sh Eurlex-4K
bash download-data.sh Wiki10-31K
bash download-data.sh AmazonCat-13K
bash download-data.sh Wiki-500K
cd ../
```

Each dataset contains the following files
- ```X.trn.npz, X.val.npz, X.tst.npz```: data tf-idf sparse matrix 
- ```Y.trn.npz, Y.val.npz, Y.tst.npz```: label sparse matrix
- ```L.elmo.npz, L.pifa.npz```: label embedding matrix
- ```mlc2seq/{train,valid.test}.txt```: each line is label_ids \tab raw_text 
- ```mlc2seq/label_vocab.txt```: each line is label_count \tab label_text
  
### Download Pretrained Models (Indexing codes, matcher models and ranker models)
Change directory into ./pretrained_models folder, download and unzip models for each dataset
	
```bash
cd ./pretrained_models
bash download-model.sh Eurlex-4K
bash download-model.sh Wiki10-31K
bash download-model.sh AmazonCat-13K
bash download-model.sh Wiki-500K
cd ../
```

### Prediction and Evaluation Pipeline
load indexing codes, generate predicted codes from pretrained matchers, and predict labels from pretrained rankers.

```bash
export DATASETS=Eurlex-4K
bash scripts/run_linear_eval.sh ${DATASETS}
bash scripts/run_xbert_eval.sh ${DATASETS}
bash scripts/run_xttention_eval.sh ${DATASETS}
```

- ```DATASETS```: the dataset name such as Eurlex-4K, Wiki10-31K, AmazonCat-13K, or Wiki-500K.
	
### Ensemble prediction and Evaluation

``` bash
python -m xbert.evaluator \
  -y [path to Y.tst.npz] \
  -e prediction-path [prediction-path ... ] 
```

For example, given the ranker prediction files (tst.pred.xbert.npz), 

``` bash
python -m xbert.evaluator \
  -y datasets/Eurlex-4K/Y.tst.npz \
  -e pretrained_models/Eurlex-4K/*/ranker/tst.pred.xbert.npz
```

which computes the metric for the X-BERT ensemble of the label_emb={elmo,pifa} and seed={0,1,2} combinations.



## Pipeline for running X-BERT on a new dataset

### Generate label embedding
We support ELMo and PIFA label embedding given the file label_vocab.txt.

```bash
cd ./datasets/
python label_embedding.py --dataset ${DATASET} --embed-type ${LABEL_EMB}
cd ../	
```

- `DATASETS`: the customized dataset name which contains the necessary files as described in [download dataset section]
- `LABEL_EMB`: currently support either elmo or pifa

### Semantic Indexing and Linear Ranker
Before training deep neural matcher, we first obtain indexed label codes and linear ranker.
The following example assume to have a similar structure as the `pretrained_models` folder.

#### Semantic Label Indexing
An example usage would be: 

```bash
OUTPUT_DIR=save_models/${DATASET}/${LABEL_EMB}-a${ALGO}-s${SEED}
mkdir -p ${OUTPUT_DIR}/indexer
python -m xbert.indexer \
  -i datasets/${DATASET}/L.${LABEL_EMB}.npz \
  -o ${OUTPUT_DIR}/indexer \
  -d ${DEPTH} --algo ${ALGO} --seed ${SEED} \
  --max-iter 20
```

- `ALGO`: clustering algorithm. 0 for KMEANS, 5 for SKMEANS
- `DEPTH`: The depth of hierarchical 2-means
- `SEED`: random seed

#### Linear Ranker training
An example usage would be:

```bash
OUTPUT_DIR=save_models/${DATASET}/${LABEL_EMB}-a${ALGO}-s${SEED}
mkdir -p $OUTPUT_DIR/ranker
python -m xbert.ranker train \
  -x datasets/${DATASET}/X.trn.npz \
  -y datasets/${DATASET}/Y.trn.npz \
  -c ${OUTPUT_DIR}/indexer/code.npz \
  -o ${OUTPUT_DIR}/ranker
```

#### Linear Ranker Prediction
An example usage would be:

```bash
OUTPUT_DIR=save_models/${DATASET}/${LABEL_EMB}-a${ALGO}-s${SEED}
mkdir -p $OUTPUT_DIR/ranker
python -m xbert.ranker predict \
  -m ${OUTPUT_DIR}/ranker \
  -x datasets/${DATASET}/X.tst.npz \
  -y datasets/${DATASET}/Y.tst.npz \
  -c ${OUTPUT_DIR}/matcher/${MATCHER}/C_eval_pred.npz \
  -o ${OUTPUT_DIR}/ranker/tst.prediction.npz
```

### Neural Matching via XBERT or Xttention
#### Create Data Binary as Preprocessing
Before training, we need to generate preprocessed data as binary pickle files.
	
```bash
OUTPUT_DIR=save_models/${DATASET}/${LABEL_EMB}-a${ALGO}-s${SEED}
mkdir -p $OUTPUT_DIR/data-bin-${MATCHER}
CUDA_VISIBLE_DEVICES=GPUS python -m xbert.preprocess \
  -m ${MATCHER} \
  -i datasets/${DATASET} \
  -c ${OUTPUT_DIR}/indexer/code.npz \
  -o ${OUTPUT_DIR}/data-bin-${MATCHER}
```

-`GPUS`: the available gpu_id
-`MATCHER`: currently support `xttention` or `xbert`

#### Training XBERT
Set hyper-parameters properly, an example would be
``` bash
GPUS=0,1,2,3,4,5
MATCHER=xbert
TRAIN_BATCH_SIZE=36
EVAL_BATCH_SIZE=64
LOG_INTERVAL=1000
EVAL_INTERVAL=10000
NUM_TRAIN_EPOCHS=12
LEARNING_RATE=5e-5
WARMUP_RATE=0.1
```
Users can also check `scripts/run_xbert.sh` to see the detailed setting for each datasets used in the paper.

We are now ready to run the xbert models:

```bash
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
  > |& tee ${OUTPUT_DIR}/matcher/${MATCHER}.log
```

#### Training Xttention
Set hyper-parameters properly, an example would be
``` bash
GPUS=0
MATCHER=xttention
TRAIN_BATCH_SIZE=128
LOG_INTERVAL=100
EVAL_INTERVAL=1000
NUM_TRAIN_EPOCHS=10
```
Users can also check `scripts/run_xttention.sh` to see the detailed setting for each datasets used in the paper.

We are now ready to run the xttention models:

```bash
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
  > |& tee ${OUTPUT_DIR}/matcher/${MATCHER}.log
```	


## Acknowledge

Some portions of this repo is borrowed from the following repos:
- [pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT)
- [liblinear](https://github.com/cjlin1/liblinear)
- [TRMF](https://github.com/rofuyu/exp-trmf-nips16)
