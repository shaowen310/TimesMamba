# export CUDA_VISIBLE_DEVICES=6
# model_name=iBMamba

pred_lens=(96 192 336 720)
lrs=(1e-4 5e-5)
d_models=(256)
e_layerss=(3)

for pred_len in "${pred_lens[@]}"; do
  for e_layers in "${e_layerss[@]}"; do
    for d_model in "${d_models[@]}"; do
      for lr in "${lrs[@]}"; do
        python -u run.py \
          --is_training 1 \
          --root_path ./dataset/weather/ \
          --data_path weather.csv \
          --model_id weather_96_$pred_len \
          --model $model_name \
          --data custom \
          --features M \
          --seq_len 96 \
          --pred_len $pred_len \
          --e_layers $e_layers \
          --enc_in 21 \
          --dec_in 21 \
          --c_out 21 \
          --des 'Exp' \
          --d_model $d_model \
          --r_ff 4 \
          --dropout 0.1 \
          --batch_size 32 \
          --learning_rate $lr \
          --train_epochs 10 \
          --itr 1 >&1 | tee Weather_${pred_len}_${model_name}_el${e_layers}_dm${d_model}_lr$lr.log
      done
    done
  done
done
