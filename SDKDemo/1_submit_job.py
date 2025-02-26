import sagemaker
import boto3, os
# from sagemaker import get_execution_role


from sagemaker.pytorch import PyTorch
from sagemaker.estimator import Estimator


sess = sagemaker.Session()
role = "arn:aws:iam::633205212955:role/service-role/AmazonSageMaker-ExecutionRole-20220923T160810"
sagemaker_default_bucket = sess.default_bucket()
region = 'us-east-1'

# https://github.com/aws/deep-learning-containers/blob/master/available_images.md
# image_uri = f'763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-training:2.2.0-gpu-py310-cu121-ubuntu20.04-sagemaker'
image_uri = f'763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-training:2.3.0-gpu-py311-cu121-ubuntu20.04-sagemaker'
# image_uri = f'763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-training:2.4.0-gpu-py311-cu124-ubuntu22.04-sagemaker'
# image_uri = f'763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-training:2.5.1-gpu-py311-cu124-ubuntu22.04-sagemaker'

# instance_type = "ml.g5.8xlarge"    # 1 * A10g (24G/GPU)
instance_type = "ml.g5.12xlarge"     # 4 * A10g (24G/GPU)
# instance_type = "ml.g5.48xlarge"    # 8 * A10g (24G/GPU)
# instance_type = "ml.p4d.24xlarge"   # 8 * A100 (40G/GPU)
# instance_type = "ml.p5.48xlarge"    # 8 * H100 (80G/GPU)
# instance_type = "ml.g6e.12xlarge"    # 4 * L40s (80G/GPU)
# instance_type = "ml.g6e.48xlarge"    # 8 * L40s (80G/GPU)

instance_count = 2                  # 1 or Multi-node


envs = {
    'MODEL_ID_OR_S3_PATH': f's3://{sagemaker_default_bucket}/Qwen2-0.5B-Instruct/*',
    'MODEL_SAVE_PATH_S3': f's3://{sagemaker_default_bucket}/output-model/250220/',
    'CONF_YAML_NAME': 'qwen2_full_sft.yaml'
}


smp_estimator = Estimator(role=role,
    sagemaker_session=sess,
    base_job_name='qwen-training',
    entry_point="estimator_entry.py",
    source_dir='submit_src/',
    instance_type=instance_type,
    instance_count=instance_count,
    environment=envs,
    hyperparameters={},
    image_uri=image_uri,
    max_run=7200,
    keep_alive_period_in_seconds=3600,
    disable_output_compression=True,
)

smp_estimator.fit()