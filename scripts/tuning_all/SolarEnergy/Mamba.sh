# model_name=STMamba

pred_lens=(96 192 336 720)
lrs=(1e-4)
d_models=(256)
e_layerss=(2)

for pred_len in "${pred_lens[@]}"; do
  for e_layers in "${e_layerss[@]}"; do
    for d_model in "${d_models[@]}"; do
      for lr in "${lrs[@]}"; do
        python -u run.py \
          --is_training 1 \
          --root_path ./dataset/Solar/ \
          --data_path solar_AL.txt \
          --model_id solar_96_$pred_len \
          --model $model_name \
          --data Solar \
          --features M \
          --seq_len 96 \
          --pred_len $pred_len \
          --e_layers $e_layers \
          --enc_in 137 \
          --dec_in 137 \
          --c_out 137 \
          --des 'Exp' \
          --d_model $d_model \
          --r_ff 4 \
          --dropout 0.1 \
          --batch_size 32 \
          --learning_rate $lr \
          --train_epochs 10 \
          --itr 1 >&1 | tee SolarE_${pred_len}_${model_name}_el${e_layers}_dm${d_model}_tol1e-3_lr$lr.log
      done
    done
  done
done
