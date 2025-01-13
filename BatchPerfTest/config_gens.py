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


if __name__ == '__main__':
    # 变量定义
    variables = {
        'ds_conf_name': ["ds_z2_config","ds_z3_config","ds_z2_offload_config","ds_z3_offload_config"],
        # 'ds_conf_name': ["ds_z2_offload_config","ds_z3_offload_config"],
        'micro_bs': [1,2,4,8]
    }
    
    # 初始化生成器
    generator = ConfigGenerator('submit_src/llama3_full_sft_template.yaml')
    
    # 生成配置文件
    configs = generator.generate_configs(
        variables=variables,
        output_dir='submit_src/configs/',
        filename_template="genConf_{ds_conf_name}_mbs{micro_bs}.yaml"
    )