model_name=iTransformer
dataset=MIX_large # MIX_large
train_epochs=100
early_cycle_threshold=100
learning_rate=0.001
master_port=25149
num_process=2
batch_size=32
n_heads=8
seq_len=1
accumulation_steps=1
lstm_layers=1
e_layers=6
d_layers=2
d_model=128
d_ff=256
dropout=0
charge_discharge_length=300
patience=5 # Eearly stopping patience
lradj=constant
loss=MSE
seed=2021



checkpoints=/path/to/your/saving/folder # the save path of checkpoints
data=Dataset_original
root_path=./dataset
comment='iTransformer' 
task_name=classification

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu  --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name $task_name \
  --data $data \
  --is_training 1 \
  --root_path $root_path \
  --model_id iTransformer \
  --model $model_name \
  --features MS \
  --seq_len $seq_len \
  --label_len 50 \
  --factor 3 \
  --enc_in 3 \
  --dec_in 1 \
  --seed $seed \
  --c_out 1 \
  --des 'Exp' \
  --itr 1 \
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

