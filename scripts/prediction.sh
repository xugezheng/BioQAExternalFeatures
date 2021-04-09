export MODEL_DIR=/home/BioQAExternalFeatures/model/model_biobert_v1.1_pubmed
export DATA_DIR=/home/BioQAExternalFeatures/data
export OUTPUT_DIR=/home/BioQAExternalFeatures/output

nohup python run_factoid_pos_ner.py \
     --do_train=False\
     --do_predict=True \
     --vocab_file=$MODEL_DIR/vocab.txt \
     --bert_config_file=$MODEL_DIR/bert_config.json \
     --init_checkpoint=$MODEL_DIR/model.ckpt-1000000 \
     --max_seq_length=384 \
     --train_batch_size=8 \
     --learning_rate=5e-6 \
     --doc_stride=128 \
     --num_train_epochs=4.0 \
     --do_lower_case=False \
     --train_file=$DATA_DIR/ner_pos_train-v1.1.json \
     --predict_file=$DATA_DIR/ner_pos_BioASQ-test-factoid-6b-1.json \
     --output_dir=$OUTPUT_DIR >> $OUTPUT_DIR/1.log 2>&1 &
