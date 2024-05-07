# export CUDA_VISIBLE_DEVICES=0
# model_name=STMamba

pred_lens=(12 24 48 96)
use_norms=(1 1 0 0)
lrs=(1e-3 5e-4 1e-4)
d_models=(512)
e_layerss=(3)

for ((i = 0; i <= ${#pred_lens[@]}; i++)); do
  for e_layers in "${e_layerss[@]}"; do
    for d_model in "${d_models[@]}"; do
      for lr in "${lrs[@]}"; do
        python -u run.py \
          --is_training 1 \
          --root_path ./dataset/PEMS/ \
          --data_path PEMS08.npz \
          --model_id PEMS08_96_${pred_lens[i]} \
          --model $model_name \
          --data PEMS \
          --features M \
          --seq_len 96 \
          --pred_len ${pred_lens[i]} \
          --e_layers $e_layers \
          --enc_in 170 \
          --dec_in 170 \
          --c_out 170 \
          --des 'Exp' \
          --d_model $d_model \
          --r_ff 4 \
          --learning_rate $lr \
          --train_epochs 10 \
          --batch_size 32 \
          --itr 1 >&1 | tee PEMS08_${pred_lens[i]}_${model_name}_el${e_layers}_dm${d_model}_lr$lr.log
      done
    done
  done
done
