model_name=MICN
dataset=MIX_large # MIX_large
train_epochs=100
early_cycle_threshold=100
learning_rate=0.00005
master_port=25129
num_process=2
batch_size=4
n_heads=4
seq_len=1
accumulation_steps=4
lstm_layers=6
e_layers=4
d_layers=2 
d_model=256
d_ff=64
dropout=0
charge_discharge_length=300
patience=5 # Eearly stopping patience
lradj=constant
loss=MSE
patch_len=50
stride=50
seed=2021



checkpoints=/path/to/your/saving/folder # the save path of checkpoints
data=Dataset_original
root_path=./dataset
comment='MICN' 
task_name=classification

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu  --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name $task_name \
  --data $data \
  --is_training 1 \
  --root_path $root_path \
  --model_id MICN \
  --model $model_name \
  --features MS \
  --seq_len $seq_len \
  --label_len 50 \
  --factor 3 \
  --enc_in 3 \
  --dec_in 1 \
  --c_out 3 \
  --des 'Exp' \
  --seed $seed \
  --itr 1 \
  --class_num 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --accumulation_steps $accumulation_steps \
  --charge_discharge_length $charge_discharge_length \
  --dataset $dataset \
  --num_workers 32 \
  --e_layers $e_layers \
  --lstm_layers $lstm_layers \
  --d_layers $d_layers \
  --patience $patience \
  --n_heads $n_heads \
  --early_cycle_threshold $early_cycle_threshold \
  --dropout $dropout \
  --lradj $lradj \
  --loss $loss \
  --checkpoints $checkpoints 

