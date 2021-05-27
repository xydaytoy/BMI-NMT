code_dir=THUMT-BMI
work_dir=THUMT
data_dir=en_de_data
export PYTHONPATH=$work_dir/$code_dir:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3

signature=ende_ft_0.15_0.8

output_dir=$work_dir/train/$signature

if [ ! -d $output_dir ]; then
    mkdir $output_dir
    chmod 777 $output_dir -R
fi

python $work_dir/$code_dir/thumt/bin/trainer.py \
  --model transformer \
  --output $output_dir \
  --input $data_dir/train.en.shuf $data_dir/train.de.shuf $data_dir/ende_mi.txt \
  --vocabulary $data_dir/dict.en.txt $data_dir/dict.de.txt \
  --validation $data_dir/valid.en \
  --references $data_dir/valid.de.tok \
  --checkpoint $work_dir/ende_base_10w \
  --parameters=device_list=[0,1,2,3],eval_steps=90000000,train_steps=200000,batch_size=4096,max_length=128,constant_batch_size=False,optimizer=Adam,adam_beta1=0.9,adam_beta2=0.98,residual_dropout=0.1,attention_dropout=0.1,relu_dropout=0.1,num_encoder_layers=6,layer_preprocess=none,layer_postprocess=layer_norm,update_cycle=2,hidden_size=512,filter_size=2048,num_heads=8,label_smoothing=0.1,warmup_steps=4000,learning_rate=1.0,save_checkpoint_steps=10000,keep_checkpoint_max=200,position_info_type=absolute,shared_embedding_and_softmax_weights=True,shared_source_target_embedding=True,mi_base=0.8,mi_delta=0.15,zero_step=True

