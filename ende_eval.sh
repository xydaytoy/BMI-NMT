code_dir=THUMT-BMI
work_dir=THUMT
data_dir=en_de_data
export PYTHONPATH=$work_dir/$code_dir:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3

signature=ende_ft_0.15_0.8

test_names='newstest2014'

output_dir=$work_dir/results/$signature

if [ ! -d $output_dir ]; then
    mkdir $output_dir
    chmod 777 $output_dir -R
fi

for idx in `seq 160000 10000 200000`; do
  echo model_checkpoint_path: \"model.ckpt-$idx\" > $work_dir/train/$signature/checkpoint
  python $work_dir/$code_dir/thumt/bin/translator.py \
    --models transformer \
    --input $data_dir/test.en \
    --output $output_dir/"$test_names".out.$idx \
    --vocabulary $data_dir/dict.en.txt $data_dir/dict.de.txt \
    --checkpoints $work_dir/train/$signature \
    --parameters=device_list=[0],decode_alpha=0.4,beam_size=7,decode_batch_size=128
  echo evaluating with checkpoint-$idx

done
