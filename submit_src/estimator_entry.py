import os
import json
import socket
import torch

if __name__ == "__main__":

    os.system("git clone https://github.com/hiyouga/LLaMA-Factory.git")
    os.system('''cd LLaMA-Factory && pip install -e ".[torch,metrics]"''')
    os.system("pip install flash-attn --no-build-isolation")
    
    hosts = json.loads(os.environ['SM_HOSTS'])
    current_host = os.environ['SM_CURRENT_HOST']
    host_rank = hosts.index(current_host)

    os.environ['TORCHRUN_NODE_NUMBER'] = str(len(hosts))
    os.environ['TORCHRUN_MASTER'] = hosts[0]
    os.environ['TORCHRUN_NODE_INDEX'] = str(host_rank)
    
    # os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'
    
    # backend env config
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # os.environ['NCCL_DEBUG'] = 'INFO'
    # os.environ['FI_PROVIDER'] = 'efa'
    # os.environ['NCCL_PROTO'] = 'simple'
    # os.environ['HCCL_OVER_OFI'] = '1'

    os.environ['FI_PROVIDER'] = 'efa'
    os.environ['FI_EFA_USE_DEVICE_RDMA'] = '1' # for P4
    os.environ['OFI_NCCL_PROTOCOL'] = 'RDMA' # for P5
    
    os.system("wandb disabled")
    
    # Copy model from s3 to GPU instance
    os.system('chmod +x ./s5cmd')
    model_s3_path = os.environ['MODEL_ID_OR_S3_PATH']
    os.system(f'./s5cmd cp {model_s3_path} /tmp/initial-model-path/')

    # # Copy data from s3 to GPU instance
    # data_s3_path = os.environ['DATA_S3_PATH']
    # os.system(f'./s5cmd cp {data_s3_path} /tmp/data-path/')

    # Run customized script
    # os.system('chmod +x compatible_script.sh')
    # os.system(f'''FORCE_TORCHRUN=1 NNODES=2 NODE_RANK={host_rank} MASTER_ADDR=algo-1 MASTER_PORT=7777 llamafactory-cli train {os.environ['CONF_YAML_NAME']}''')

    os.system(f'''FORCE_TORCHRUN=1 NNODES={len(hosts)} NODE_RANK={host_rank} MASTER_ADDR={hosts[0]} MASTER_PORT=7777 llamafactory-cli train {os.environ['CONF_YAML_NAME']}''')


    # # Copy model from 1 GPU instance (if "stage3_gather_16bit_weights_on_model_save": true)
    trained_s3_uri = os.environ['MODEL_SAVE_PATH_S3']
    if 0 == host_rank:
        os.system(f'./s5cmd cp /tmp/tuned-model-path/ {trained_s3_uri}')
