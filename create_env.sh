conda create -n timesmamba python=3.11 -y
conda activate timesmamba
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install packaging
pip install "causal-conv1d>=1.1.0"
pip install "mamba-ssm>=1.1.0"
pip install pandas
pip install scikit-learn
pip install timm
pip install reformer-pytorch
pip install matplotlib
