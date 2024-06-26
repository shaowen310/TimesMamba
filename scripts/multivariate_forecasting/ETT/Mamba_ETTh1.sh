# export CUDA_VISIBLE_DEVICES=0
# model_name=Mamba

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 32 \
  --r_ff 4 \
  --revin_affine \
  --channel_independence \
  --dropout 0.1 \
  --batch_size 32 \
  --learning_rate 5e-3 \
  --train_epochs 10 \
  --itr 1 >&1 | tee logs/ETTh1_96_${model_name}.log

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_192 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 32 \
  --r_ff 4 \
  --revin_affine \
  --channel_independence \
  --dropout 0.1 \
  --batch_size 32 \
  --learning_rate 1e-3 \
  --train_epochs 10 \
  --itr 1 >&1 | tee logs/ETTh1_192_${model_name}.log

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_336 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 32 \
  --r_ff 4 \
  --revin_affine \
  --channel_independence \
  --dropout 0.1 \
  --batch_size 32 \
  --learning_rate 5e-4 \
  --train_epochs 10 \
  --itr 1 >&1 | tee logs/ETTh1_336_${model_name}.log

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_720 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 64 \
  --r_ff 4 \
  --revin_affine \
  --channel_independence \
  --dropout 0.1 \
  --batch_size 32 \
  --learning_rate 5e-4 \
  --train_epochs 10 \
  --itr 1 >&1 | tee logs/ETTh1_720_${model_name}.log
