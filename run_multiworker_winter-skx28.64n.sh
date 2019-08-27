sudo sh -c "/usr/bin/echo 3 > /proc/sys/vm/drop_caches"
~/scripts/clean_caches.sh
vmstat 1 > /tmp/vmstatlog 2>&1 &

TENSORFLOW_MODELS_DIR=$HOME/TensorFlow/models
BERT_BASE_DIR=/data/01/kushal/bert/uncased_L-24_H-1024_A-16
export PYTHONPATH=$PYTHONPATH:$TENSORFLOW_MODELS_DIR

export MAX_SEQ_LENGTH=512
export NUM_NODES=64
export NUM_WORKERS_PER_NODE=2
export OMP_NUM_THREADS=40
export HOROVOD_FUSION_THRESHOLD=$((128*1024*1024))

export NUM_WORKERS=$((NUM_NODES * NUM_WORKERS_PER_NODE))
export DATA_DIR=/data/01/kushal/bert/BERT_TF_RECORDS_$MAX_SEQ_LENGTH/tfrec/*
export OUTPUT_DIR=/data/01/kushal/bert/pretraining_output

export PATH=$HOME/GCC/gcc-7.3.0/bin:/nfs/pdx/home/kdatta1/openmpi/bin:/nfs/pdx/home/kdatta1/anaconda3/envs/py3.6-tf2.0/bin:/nfs/pdx/home/kdatta1/anaconda3/condabin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/usr/local/bin:/usr/local/bin:/usr/local/bin:/usr/local/bin
export LD_LIBRARY_PATH=$HOME/GCC/gcc-7.3.0/lib:$HOME/GCC/gcc-7.3.0/lib64:/usr/lib64

HOSTFILE=~/winter.hosts.$NUM_NODES

MPI=/nfs/pdx/home/kdatta1/openmpi/bin/mpiexec
PYTHON=/nfs/pdx/home/kdatta1/anaconda3/envs/py3.6-tf2.0/bin/python

which $MPI
which $PYTHON

pssh -h $HOSTFILE -i -P "rm -rf $OUTPUT_DIR/*"

$MPI -np $NUM_WORKERS --hostfile $HOSTFILE --map-by ppr:$NUM_WORKERS_PER_NODE:node \
  --bind-to numa --report-bindings \
  -x KMP_BLOCKTIME=0 -x PYTHONPATH=$PYTHONPATH -x PATH=$PATH \
  -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH  \
  -x HOROVOD_FUSION_THRESHOLD=$HOROVOD_FUSION_THRESHOLD \
  -x OMP_NUM_THREADS=$OMP_NUM_THREADS \
  numactl -l $PYTHON run_pretraining_horovod.py \
  --input_file=$DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --do_train=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --train_batch_size=16 \
  --max_seq_length=$MAX_SEQ_LENGTH \
  --max_predictions_per_seq=20 \
  --num_train_steps=20 \
  --num_warmup_steps=2 \
  --learning_rate=2e-5 \
  --timeline=True \
  --inter_op=1 \
  --intra_op=$OMP_NUM_THREADS \
  --use_horovod=True

kill -9 $(ps -eaf | grep vmstat | awk '{print $2}')
min=$(cat /tmp/vmstatlog | sed '/memory/d' | sed '/free/d' | awk -v min=9999999999 '{if($4<min){min=$4}}END{print min} ')
top=$(cat /tmp/vmstatlog | sed '/memory/d' | sed '/free/d' | head -n 1 | awk '{print $4}')
echo "Peak memory (KB):" $((top-min))
