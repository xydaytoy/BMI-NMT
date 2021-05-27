code_dir=THUMT-BMI
work_dir=THUMT
data_dir=zh_en_data
export PYTHONPATH=$work_dir/$code_dir:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3

signature=zhen_ft_0.15_0.8

output_dir=$work_dir/train/$signature

if [ ! -d $output_dir ]; then
    mkdir $output_dir
    chmod 777 $output_dir -R
fi

python $work_dir/$code_dir/thumt/bin/trainer.py \
  --model transformer \
  --output $output_dir \
  --input $data_dir/train.cn.shuf $data_dir/train.en.shuf $data_dir/zhen_mi.txt \
  --vocabulary $data_dir/vocab.cn $data_dir/vocab.en \
  --validation $data_dir/newstest2018.cn.bpe \
  --references $data_dir/newstest2018-zhen-ref.en \
  --parameters=device_list=[0,1,2,3],eval_steps=90000000,train_steps=200000,batch_size=4096,max_length=128,optimizer=Adam,adam_beta1=0.9,adam_beta2=0.98,warmup_steps=4000,learning_rate=1.0,constant_batch_size=False,residual_dropout=0.1,attention_dropout=0.1,relu_dropout=0.1,num_encoder_layers=6,layer_preprocess=none,layer_postprocess=layer_norm,update_cycle=2,hidden_size=512,filter_size=2048,num_heads=8,label_smoothing=0.1,save_checkpoint_steps=5000,keep_checkpoint_max=200,position_info_type=absolute,shared_embedding_and_softmax_weights=True,shared_source_target_embedding=False,mi_base=0.9,mi_delta=0.15,zero_step=False