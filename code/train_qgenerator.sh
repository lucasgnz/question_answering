python -m nmt.nmt \
    --copynet --share_vocab \
    --src=doc --tgt=q \
    --vocab_prefix=../data/squad/voc  \
    --train_prefix=../data/squad/train \
    --dev_prefix=../data/squad/dev  \
    --test_prefix=../data/squad/test \
    --embed_prefix=../data/glove \
    --out_dir=models/qgen_model \
    --num_train_steps=12000 \
    --steps_per_stats=100 \
    --num_layers=2 \
    --num_units=100 \
    --dropout=0.2 \
    --metrics=bleu \
    --encoder_type=bi \
    --attention=bahdanau \
    --optimizer=adam \
    --check_special_token=True \
    --learning_rate=0.01 \
    --src_max_len=200 \
    --batch_size=8 \