export CUDA_VISIBLE_DEVICES=0
export model_name=TimesMamba

# tuning all

bash ./scripts/tuning_all/ETT/Mamba_ETTh1.sh
# bash ./scripts/tuning_all/ETT/Mamba_ETTh2.sh
# bash ./scripts/tuning_all/ETT/Mamba_ETTm1.sh
# bash ./scripts/tuning_all/ETT/Mamba_ETTm2.sh
bash ./scripts/tuning_all/ECL/Mamba.sh
bash ./scripts/tuning_all/Traffic/Mamba.sh
# bash ./scripts/tuning_all/Weather/Mamba.sh
# bash ./scripts/tuning_all/Exchange/Mamba.sh
# bash ./scripts/tuning_all/SolarEnergy/Mamba.sh
# bash ./scripts/tuning_all/PEMS/Mamba_03.sh
# bash ./scripts/tuning_all/PEMS/Mamba_04.sh
# bash ./scripts/tuning_all/PEMS/Mamba_07.sh
# bash ./scripts/tuning_all/PEMS/Mamba_08.sh
