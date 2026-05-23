import os
import json
import gzip
import argparse
import pickle
import pprint

def read_datasets(dataset_path):
    """
    遍历并读取指定路径下的数据集文件 (支持 .json, .json.gz, .pickle, .pkl)。
    """
    print(f"开始扫描数据集路径: {os.path.abspath(dataset_path)}\n")
    
    if not os.path.exists(dataset_path):
        print(f"路径不存在: {dataset_path}")
        return

    dataset_files = []
    if os.path.isfile(dataset_path):
        if dataset_path.endswith(('.json', '.json.gz', '.pickle', '.pkl')):
            dataset_files.append(dataset_path)
    else:
        for root, _, files in os.walk(dataset_path):
            for file in files:
                if file.endswith(('.json', '.json.gz', '.pickle', '.pkl')):
                    dataset_files.append(os.path.join(root, file))
                
    if not dataset_files:
        print("未找到任何 .json, .json.gz, .pickle, 或 .pkl 数据集文件。")
        return
        
    for file_path in dataset_files:
        try:
            if file_path.endswith('.json.gz'):
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    data = json.load(f)
                file_type = "GZIP JSON"
            elif file_path.endswith('.pickle') or file_path.endswith('.pkl'):
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                file_type = "PICKLE"
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                file_type = "JSON"
                
            # 简单展示数据基本信息
            if isinstance(data, dict):
                data_info = f"字典, 包含键数量: {len(data.keys())}"
            elif isinstance(data, list):
                data_info = f"列表, 长度: {len(data)}"
            else:
                data_info = type(data).__name__
                
            print(f"[成功] 读取了 {file_type} 数据集: {file_path}")
            print(f"       -> 数据信息: {data_info}")
            print(f"       -> 详细内容 (部分/全部预览):")
            
            # 使用 pprint 将内容格式化打印出来
            # 为了防止数据过大刷屏，如果是字典或列表，默认设置了 depth 和 width，您可以根据需要取消注释修改
            pprint.pprint(data, indent=2, depth=1, width=80)
            
        except Exception as e:
            print(f"[错误] 无法读取文件 {file_path}。原因: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="读取并分析 datasets 路径下的数据集文件")
    parser.add_argument(
        "--path", 
        type=str, 
        default="/home/dministrator/EmbodiedBench-problemsolving/data/datasets",
        help="要扫描的数据集路径 (默认: data/datasets)"
    )
    args = parser.parse_args()
    
    read_datasets(args.path)
