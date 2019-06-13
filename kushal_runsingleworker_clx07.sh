#sudo /tmp/clean_caches.sh
#vmstat 1 > /tmp/vmstatlog 2>&1 & 

TENSORFLOW_MODELS_DIR=$HOME/TensorFlow/models
BERT_BASE_DIR=/data/01/kushal/bert/uncased_L-24_H-1024_A-16
export PYTHONPATH=$PYTHONPATH:$TENSORFLOW_MODELS_DIR

#DATA_DIR=/data01/kushal/bert/tf_examples.tfrecord
DATA_DIR=/data/01/sun/BERT_wiki/ExtractedText_24Apr2019/tfrec/*

#KMP_BLOCKTIME=0 numactl -m 0 -C 0-19,40-59 python run_pretraining.py \
KMP_BLOCKTIME=1 KMP_AFFINITY="fine,granularity=proclist[0-23],explicit" taskset -c 0-23 numactl -m 0 python run_pretraining.py \
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
  --learning_rate=2e-5

kill -9 $(ps -eaf | grep vmstat | awk '{print $2}')
