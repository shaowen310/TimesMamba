export CUDA_VISIBLE_DEVICES=0
export model_name=TimesMamba

# multivariate_forecasting

bash ./scripts/multivariate_forecasting/ETT/Mamba_ETTh1.sh
bash ./scripts/multivariate_forecasting/ECL/Mamba.sh
bash ./scripts/multivariate_forecasting/Traffic/Mamba.sh
