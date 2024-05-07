# export CUDA_VISIBLE_DEVICES=0
# model_name=STMamba

pred_lens=(12 24 48 96)
lrs=(1e-3 5e-4 1e-4)
d_models=(512)
e_layerss=(3)

for pred_len in "${pred_lens[@]}"; do
  for e_layers in "${e_layerss[@]}"; do
    for d_model in "${d_models[@]}"; do
      for lr in "${lrs[@]}"; do
        python -u run.py \
          --is_training 1 \
          --root_path ./dataset/PEMS/ \
          --data_path PEMS07.npz \
          --model_id PEMS07_96_$pred_len \
          --model $model_name \
          --data PEMS \
          --features M \
          --seq_len 96 \
          --pred_len $pred_len \
          --e_layers $e_layers \
          --enc_in 883 \
          --dec_in 883 \
          --c_out 883 \
          --des 'Exp' \
          --d_model $d_model \
          --r_ff 4 \
          --learning_rate $lr \
          --train_epochs 10 \
          --no_norm \
          --batch_size 32 \
          --itr 1 >&1 | tee PEMS07_${pred_len}_${model_name}_el${e_layers}_dm${d_model}_lr$lr.log
      done
    done
  done
done
