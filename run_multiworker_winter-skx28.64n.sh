TENSORFLOW_MODELS_DIR=$HOME/TensorFlow/models
BERT_BASE_DIR=/data/01/kushal/bert/uncased_L-24_H-1024_A-16
export PYTHONPATH=$PYTHONPATH:$TENSORFLOW_MODELS_DIR

DATA_DIR=/data/01/kushal/bert/BERT_TF_RECORDS/tfrec/*

source /opt/rh/devtoolset-8/enable
source $HOME/.gcc7.3.0.rc
source $HOME/.openmpi.rc

NUM_NODES=64
NUM_WORKERS_PER_NODE=2
NUM_WORKERS=$((NUM_NODES * NUM_WORKERS_PER_NODE))
echo $NUM_WORKERS
OMP_NUM_THREADS=20

HOSTFILE=~/winter.hosts.$NUM_NODES

MPI=/nfs/pdx/home/kdatta1/openmpi/bin/mpirun
#PYTHON=/nfs/pdx/home/kdatta1/anaconda3/envs/py3.6-tf2.0/bin/python
PYTHON=/nfs/pdx/home/kdatta1/anaconda3/bin/python3

KMP_AFFINITY="compact,1,0,granularity=fine"

HOROVOD_FUSION_THRESHOLD=524288000 OMP_NUM_THREADS=$OMP_NUM_THREADS KMP_BLOCKTIME=0 KMP_AFFINITY=$KMP_AFFINITY \
$MPI -np $NUM_WORKERS --hostfile $HOSTFILE --map-by socket \
  -x KMP_AFFINITY=$KMP_AFFINITY -x PYTHONPATH -x PATH -x LD_LIBRARY_PATH \
  $PYTHON run_pretraining_horovod.py \
  --input_file=$DATA_DIR \
  --output_dir=/data/01/kushal/bert/pretraining_output \
  --do_train=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --train_batch_size=32 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=20 \
  --num_warmup_steps=5 \
  --learning_rate=2e-5 \
  --inter_op=2 \
  --intra_op=$OMP_NUM_THREADS \
  --use_horovod=True

# --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
#pssh -h $HOSTFILE 'kill -9 $(ps -eaf | grep vmstat | awk '{print $2}')'
