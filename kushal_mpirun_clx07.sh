#sudo /tmp/clean_caches.sh
#vmstat 1 > /tmp/vmstatlog 2>&1 & 

TENSORFLOW_MODELS_DIR=$HOME/TensorFlow/models
BERT_BASE_DIR=/data/01/kushal/bert/uncased_L-24_H-1024_A-16
export PYTHONPATH=$PYTHONPATH:$TENSORFLOW_MODELS_DIR

#DATA_DIR=/data/01/kushal/Wikipedia/BERT_TF_RECORDS/tfrec/*
DATA_DIR=/data/01/sun/BERT_wiki/ExtractedText_24Apr2019/tfrec/*

NUM_WORKERS=4

HOROVOD_FUSION_THRESHOLD=524288000 KMP_BLOCKTIME=1 OMP_NUM_THREADS=12 mpiexec -np $NUM_WORKERS --map-by socket \
  -x KMP_BLOCKTIME -x OMP_NUM_THREADS=12 \
  numactl -l python run_pretraining_horovod.py \
  --input_file=$DATA_DIR \
  --output_dir=/data/01/kushal/bert/pretraining_output \
  --do_train=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --train_batch_size=32 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=20 \
  --num_warmup_steps=5 \
  --learning_rate=2e-5 \
  --inter_op=2 \
  --intra_op=12 \
  --use_horovod=True

kill -9 $(ps -eaf | grep vmstat | awk '{print $2}')
