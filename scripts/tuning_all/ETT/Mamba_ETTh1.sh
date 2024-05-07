# export CUDA_VISIBLE_DEVICES=0
# model_name=Mamba

pred_lens=(96 192 336 720)
lrs=(5e-3)
d_models=(32)
e_layerss=(1)

for pred_len in "${pred_lens[@]}"; do
  for e_layers in "${e_layerss[@]}"; do
    for d_model in "${d_models[@]}"; do
      for lr in "${lrs[@]}"; do
        python -u run.py \
          --is_training 1 \
          --root_path ./dataset/ETT-small/ \
          --data_path ETTh1.csv \
          --model_id ETTh1_96_$pred_len \
          --model $model_name \
          --data ETTh1 \
          --features M \
          --seq_len 96 \
          --pred_len $pred_len \
          --e_layers $e_layers \
          --enc_in 7 \
          --dec_in 7 \
          --c_out 7 \
          --des 'Exp' \
          --d_model $d_model \
          --r_ff 4 \
          --revin_affine \
          --channel_independence \
          --ssm_expand 0 \
          --dropout 0.1 \
          --batch_size 32 \
          --learning_rate $lr \
          --train_epochs 10 \
          --itr 1 >&1 | tee ETTh1_${pred_len}_${model_name}_el${e_layers}_dm${d_model}_rff4_lr$lr.log
      done
    done
  done
done
