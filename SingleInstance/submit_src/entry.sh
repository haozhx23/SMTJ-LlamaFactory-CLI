

curl -L https://github.com/peak/s5cmd/releases/download/v2.2.2/s5cmd_2.2.2_Linux-64bit.tar.gz | tar -xz s5cmd
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory && pip install -e ".[torch,metrics]"
cd ..
pip install flash-attn --no-build-isolation
pip install -r requirements.txt


chmod +x ./s5cmd
./s5cmd cp $MODEL_ID_OR_S3_PATH /tmp/initial-model-path/
FORCE_TORCHRUN=1 llamafactory-cli train $CONF_YAML_NAME
./s5cmd cp /tmp/tuned-model-path/ $MODEL_SAVE_PATH_S3



