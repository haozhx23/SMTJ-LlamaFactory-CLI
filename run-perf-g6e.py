import sagemaker
import boto3, os
from sagemaker import get_execution_role

sess = sagemaker.Session()
# role = get_execution_role()
role = 'arn:aws:iam::633205212955:role/service-role/AmazonSageMaker-ExecutionRole-20220923T160810'
sagemaker_default_bucket = sess.default_bucket()
region = sess.boto_session.region_name

import boto3
logs_client = boto3.client('logs')
log_group_name = "/aws/sagemaker/TrainingJobs"

from sagemaker.pytorch import PyTorch
from sagemaker.estimator import Estimator

# https://github.com/aws/deep-learning-containers/blob/master/available_images.md
image_uri = f'763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-training:2.3.0-gpu-py311-cu121-ubuntu20.04-sagemaker'



import itertools
import yaml
from string import Template
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

class ConfigGenerator:
    def __init__(self, template_path: str):
        self.template_path = Path(template_path)
        self.template = self._read_template()
        
    def _read_template(self) -> str:
        """读取yaml模板文件"""
        return self.template_path.read_text()
    
    @staticmethod
    def generate_combinations(variables: Dict[str, List[Any]]) -> List[Dict]:
        """生成所有可能的参数组合"""
        keys = variables.keys()
        values = variables.values()
        return [dict(zip(keys, combo)) 
                for combo in itertools.product(*values)]
    
    def generate_configs(self, 
                        variables: Dict[str, List[Any]], 
                        output_dir: str,
                        filename_template: str = "config_{index}.yaml"):
        """生成所有配置文件"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        combinations = self.generate_combinations(variables)
        generated_files = []
        
        for i, params in enumerate(combinations):
            # 使用模板生成文件名
            filename = filename_template.format(index=i, **params)
            output_path = output_dir / filename
            
            # 替换参数并保存
            config_content = Template(self.template).safe_substitute(params)
            output_path.write_text(config_content)
            
            generated_files.append({
                'path': str(output_path),
                'conf_file_name': str(filename),
                'params': params
            })
            
        return generated_files

# 使用示例
if __name__ == '__main__':
    # 变量定义
    variables = {
        'ds_conf_name': ["ds_z2_config","ds_z3_config","ds_z2_offload_config","ds_z3_offload_config"],
        # 'ds_conf_name': ["ds_z2_offload_config","ds_z3_offload_config"],
        # 'micro_bs': [1,2,4,8]
        'micro_bs': [10,12,16]
    }
    
    # 初始化生成器
    generator = ConfigGenerator('submit_src/llama3_full_sft_template.yaml')
    
    # 生成配置文件
    configs = generator.generate_configs(
        variables=variables,
        output_dir='submit_src/configs/',
        filename_template="genConf_{ds_conf_name}_mbs{micro_bs}.yaml"
    )

    # inst_list = ['ml.p3dn.24xlarge', "ml.g5.48xlarge", "ml.g6e.48xlarge", "ml.p4d.24xlarge"]
    inst_list = ['ml.g6e.48xlarge']

    for inst in inst_list:
        
        for config in configs:

            print('---config---:', config)
            tmpstr = config['params']['ds_conf_name'].replace('ds_','').replace('_config','').replace('_','-')

            logfilename = '-'.join([tmpstr, str(config['params']['micro_bs']), inst.split('.')[1]])
            print('---logfilename---:', logfilename)

            envs = {
                # "DATA_S3_PATH": f's3://{sagemaker_default_bucket}/qwen2-train-dataset/*',
                'MODEL_ID_OR_S3_PATH': f's3://llm-artifacts-us-east-1/MTLM-llama-3-8b-instruct/*',
                'MODEL_SAVE_PATH_S3': f's3://{sagemaker_default_bucket}/output-model/241201/',
                'CONF_YAML_NAME': f'''configs/{config['conf_file_name']}'''
            }

            print('ENV', envs)

            hypers = {
            }

            smp_estimator = Estimator(role=role,
                sagemaker_session=sess,
                base_job_name=f'perf-{logfilename}',
                entry_point="estimator_entry.py",
                source_dir='submit_src/',
                instance_type=inst,
                instance_count=1,
                environment=envs,
                hyperparameters=hypers,
                image_uri=image_uri,
                max_run=1800,
                keep_alive_period_in_seconds=1800,
                enable_remote_debug=True,
                disable_output_compression=True,
            )

            # formatted_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            # training_job_name = f'perf-{logfilename}-{formatted_time}'

            try:
                smp_estimator.fit()
            except Exception as e:
                print('---training job breaks---')
                print(e)
                pass

            # break

            training_job_name = smp_estimator.latest_training_job.job_name
            print('---_current_job_name---:', training_job_name)
            
            response = logs_client.describe_log_streams(
                logGroupName=log_group_name,
                logStreamNamePrefix=training_job_name
            )

            with open(f'smest_logs/log-{logfilename}.logs', 'w') as f:
                for stream in response['logStreams']:
                    log_stream_name = stream['logStreamName']
                    logs = logs_client.get_log_events(
                        logGroupName=log_group_name,
                        logStreamName=log_stream_name
                    )
                    
                    for event in logs['events']:
                        print(event['message'])
                        f.write(event['message'] + '\n')


            # break