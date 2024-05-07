# export CUDA_VISIBLE_DEVICES=6
# model_name=Mamba

pred_lens=(96 192 336 720)

for pred_len in "${pred_lens[@]}"
do
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --use_mark \
  --seq_len 96 \
  --pred_len $pred_len \
  --e_layers 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 512 \
  --r_ff 2 \
  --dropout 0.1 \
  --batch_size 32 \
  --learning_rate 1e-3 \
  --train_epochs 10 \
  --itr 1 >&1 | tee logs/Traffic_${pred_len}_${model_name}.log
done
