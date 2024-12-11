import sagemaker
from sagemaker import get_execution_role

sess = sagemaker.Session()
# role = get_execution_role()
role = 'arn:aws:iam::633205212955:role/service-role/AmazonSageMaker-ExecutionRole-20220923T160810'
sagemaker_default_bucket = sess.default_bucket()
region = sess.boto_session.region_name

from sagemaker.estimator import Estimator
image_uri = f'763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-training:2.3.0-gpu-py311-cu121-ubuntu20.04-sagemaker'

import boto3
logs_client = boto3.client('logs')
log_group_name = "/aws/sagemaker/TrainingJobs"

from config_gens import *

# 变量定义
variables = {
    'zero_conf': ["ds_z1","ds_z2","ds_z3","ds_z2_offload","ds_z3_offload"],
    'micro_bs': [1,2,4,8],
    'accum_steps': [2,4,8]
}

# 初始化生成器
generator = ConfigGenerator('submit_src/llama3_full_dpo_template.yaml')

# 生成配置文件
configs = generator.generate_configs(
    variables=variables,
    output_dir='submit_src/configs/',
    filename_template="genConf_{zero_conf}_mbs{micro_bs}_acm{accum_steps}.yaml"
)

configs_dict = {i: value for i, value in enumerate(configs)}


GLB_BS = 64

for inst in [1,2]:
    for config_i in configs_dict.keys():
        
        config = configs_dict[config_i]

        gen_bs = config['params']['micro_bs']
        gen_accum = config['params']['accum_steps']

        if inst*8*gen_bs*gen_accum != GLB_BS:
            continue
        
        # skip list
        skip_list = [## z1 and big bs
                     'z1_mbs8', 
                     'z2_mbs1', 
                     ## test done
                     'z3_mbs1',
                     ## will be slower
                     'z2_offload_mbs1',
                     'z3_offload_mbs1',
                     ## oom

                     ## done
                     'z1_mbs1',
                     'z1_mbs2'
                     ]
        # for sl in skip_list:
        #     if sl not in config['conf_file_name']:
        #         continue

        if any(sl in config['conf_file_name'] for sl in skip_list):
            continue

        print('---------PROGRESS---------: ', config_i)
        print('---config---:', config)
        namestr = f'inst{inst}-' + config['conf_file_name'].replace('genConf_ds_','').replace('.yaml','').replace('_','-')
        print('---JOB NAME---:', namestr)

        envs = {
            # "DATA_S3_PATH": f's3://{sagemaker_default_bucket}/qwen2-train-dataset/*',
            # 'MODEL_ID_OR_S3_PATH': f's3://llm-artifacts-us-east-1/MTLM-llama-3-8b-instruct/*', 
            'MODEL_ID_OR_S3_PATH': f's3://llm-artifacts-us-east-1/Llama-3.2-3B-Instruct/*',
            'MODEL_SAVE_PATH_S3': f's3://{sagemaker_default_bucket}/output-model/241201/',
            'CONF_YAML_NAME': f'''configs/{config['conf_file_name']}'''
        }

        instance_type = "ml.p5.48xlarge"
        
        smp_estimator = Estimator(role=role,
            sagemaker_session=sess,
            base_job_name=namestr,
            entry_point="estimator_entry.py",
            source_dir='submit_src/',
            instance_type=instance_type,
            instance_count=inst,
            environment=envs,
            hyperparameters={},
            image_uri=image_uri,
            max_run=7200,
            keep_alive_period_in_seconds=1800,
            enable_remote_debug=True,
            disable_output_compression=True,
        )

        # smp_estimator.fit()

        try:
            smp_estimator.fit()
        except Exception as e:
            print('---training job breaks---')
            print(e)
            continue

        # break

        training_job_name = smp_estimator.latest_training_job.job_name
        print('---_current_job_name---:', training_job_name)
        
        response = logs_client.describe_log_streams(
            logGroupName=log_group_name,
            logStreamNamePrefix=training_job_name
        )

        with open(f'smest_logs/log-{namestr}.logs', 'w') as f:
            for stream in response['logStreams']:
                log_stream_name = stream['logStreamName']
                logs = logs_client.get_log_events(
                    logGroupName=log_group_name,
                    logStreamName=log_stream_name
                )
                
                for event in logs['events']:
                    print(event['message'])
                    f.write(event['message'] + '\n')

