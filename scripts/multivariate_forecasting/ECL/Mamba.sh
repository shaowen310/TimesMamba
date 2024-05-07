# export CUDA_VISIBLE_DEVICES=6
# model_name=Mamba

pred_lens=(96 192 336 720)

for pred_len in "${pred_lens[@]}"
do
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len $pred_len \
  --e_layers 1 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 256 \
  --r_ff 4 \
  --revin_affine \
  --dropout 0.1 \
  --batch_size 32 \
  --learning_rate 1e-3 \
  --train_epochs 10 \
  --itr 1 >&1 | tee logs/ECL_${pred_len}_${model_name}.log
done
