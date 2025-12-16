#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
依次执行三个TA注意力机制的对照实验：
1. T=10，不禁止发放脉冲（对照实验）
2. T=20，禁止发放脉冲
3. T=20，不禁止发放脉冲（对照实验）
"""

import os
import sys
import subprocess
import time
import re
from datetime import datetime

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(script_dir))  # 添加MA_SNN到路径
os.chdir(script_dir)

def run_experiment(name, cmd_args, description, log_path):
    """运行单个实验，并记录日志"""
    print("\n" + "="*80)
    print(f"开始实验: {name}")
    print(f"描述: {description}")
    print(f"命令: python Att_SNN_CNN.py {' '.join(cmd_args)}")
    print(f"日志文件: {log_path}")
    print("="*80 + "\n")
    
    start_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        # 创建日志文件并写入实验信息
        with open(log_path, 'w', encoding='utf-8') as log_file:
            log_file.write("=" * 80 + "\n")
            log_file.write(f"实验: {name}\n")
            log_file.write(f"描述: {description}\n")
            log_file.write(f"开始时间: {start_datetime}\n")
            log_file.write(f"执行命令: python Att_SNN_CNN.py {' '.join(cmd_args)}\n")
            log_file.write("=" * 80 + "\n\n")
            log_file.flush()
            
            # 执行命令，实时输出到终端和文件
            process = subprocess.Popen(
                ["python", "Att_SNN_CNN.py"] + cmd_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # 实时读取输出并同时写入文件和打印到终端
            for line in process.stdout:
                # 检查是否为进度条行（包含 "数字%|" 且通常包含 "it/s" 或 "s/it"）
                is_progress_bar = re.search(r'\d+%\|', line) and ('it/s' in line or 's/it' in line)
                
                if is_progress_bar:
                    # 如果是进度条：只打印到终端，不写入文件
                    print(line, end='', flush=True)
                else:
                    # 如果是普通日志：打印到终端 + 写入文件
                    print(line, end='', flush=True)
                    log_file.write(line)
                    log_file.flush()
            
            # 等待进程完成
            return_code = process.wait()
            
            end_time = time.time()
            duration = end_time - start_time
            end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 写入结束信息
            log_file.write("\n" + "=" * 80 + "\n")
            log_file.write(f"结束时间: {end_datetime}\n")
            log_file.write(f"耗时: {duration:.2f} 秒 ({duration/60:.2f} 分钟)\n")
            log_file.write(f"返回代码: {return_code}\n")
            log_file.write("=" * 80 + "\n")
        
        if return_code == 0:
            print("\n" + "="*80)
            print(f"实验 '{name}' 完成!")
            print(f"开始时间: {start_datetime}")
            print(f"结束时间: {end_datetime}")
            print(f"耗时: {duration:.2f} 秒 ({duration/60:.2f} 分钟)")
            print(f"日志已保存到: {log_path}")
            print("="*80 + "\n")
            return True, duration
        else:
            print("\n" + "="*80)
            print(f"实验 '{name}' 失败!")
            print(f"错误代码: {return_code}")
            print(f"日志已保存到: {log_path}")
            print("="*80 + "\n")
            return False, duration
        
    except KeyboardInterrupt:
        print("\n" + "="*80)
        print(f"实验 '{name}' 被用户中断!")
        print(f"日志已保存到: {log_path}")
        print("="*80 + "\n")
        # 确保日志文件记录了中断信息
        try:
            with open(log_path, 'a', encoding='utf-8') as log_file:
                log_file.write("\n" + "=" * 80 + "\n")
                log_file.write("用户中断实验\n")
                log_file.write(f"中断时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                log_file.write("=" * 80 + "\n")
        except:
            pass
        return False, 0
    except Exception as e:
        print("\n" + "="*80)
        print(f"实验 '{name}' 发生错误: {str(e)}")
        print(f"日志已保存到: {log_path}")
        print("="*80 + "\n")
        return False, 0

def main():
    """主函数"""
    print("\n" + "="*80)
    print("TA注意力机制对照实验脚本")
    print("="*80)
    print(f"当前工作目录: {os.getcwd()}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    # 创建结果目录（类似run_all_experiments.py）
    from DVSGestures.CNN import Config
    config_temp = Config.configs()
    base_result_dir = config_temp.recordPath  # 基础Result目录
    
    # 生成文件夹名称：包含时间和主要参数
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"TA_experiments_{timestamp}_epoch100_dt25"
    result_dir = os.path.join(base_result_dir, folder_name)
    
    # 创建文件夹
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        print(f"创建结果目录: {result_dir}")
    
    # 定义三个实验（只执行这三个，不是8个）
    experiments = [
        {
            "name": "实验1: T=10 (不禁止发放脉冲)",
            "args": ["--attention", "TA", "--T", "10", "--dt", "25", "--num_epochs", "100"],
            "description": "T=10，不禁止发放脉冲，作为对照实验，训练100个epoch",
            "log_name": "exp1_T10_no_disable"
        },
        {
            "name": "实验2: T=20 (禁止发放脉冲)",
            "args": ["--attention", "TA", "--T", "20", "--dt", "25", "--disable_spike", "--num_epochs", "100"],
            "description": "T=20，禁止发放脉冲，从20步中选择10步禁止发放脉冲，训练100个epoch",
            "log_name": "exp2_T20_disable_spike"
        },
        {
            "name": "实验3: T=20 (不禁止发放脉冲)",
            "args": ["--attention", "TA", "--T", "20", "--dt", "25", "--num_epochs", "100"],
            "description": "T=20，不禁止发放脉冲，作为对照实验，训练100个epoch",
            "log_name": "exp3_T20_no_disable"
        }
    ]
    
    results = []
    total_start_time = time.time()
    
    # 依次执行实验
    for i, exp in enumerate(experiments, 1):
        # 创建日志文件路径
        log_filename = f"{exp['log_name']}_{timestamp}.txt"
        log_path = os.path.join(result_dir, log_filename)
        
        # 在命令行参数中加入结果目录，确保生成的图表/模型/日志落在指定文件夹
        cmd_args_with_dir = exp["args"] + ["--result_dir", result_dir]
        
        success, duration = run_experiment(exp["name"], cmd_args_with_dir, exp["description"], log_path)
        results.append({
            "name": exp["name"],
            "success": success,
            "duration": duration,
            "log_path": log_path
        })
        
        # 如果实验失败，询问是否继续
        if not success:
            response = input(f"实验 '{exp['name']}' 失败，是否继续执行下一个实验? (y/n): ")
            if response.lower() != 'y':
                print("\n用户选择停止执行。")
                break
        
        # 实验之间的间隔（可选）
        if i < len(experiments):
            print(f"\n等待5秒后开始下一个实验...\n")
            time.sleep(5)
    
    # 打印总结
    total_duration = time.time() - total_start_time
    total_end_time = datetime.now()
    
    print("\n" + "="*80)
    print("所有实验执行完成!")
    print("="*80)
    print(f"总耗时: {total_duration:.2f} 秒 ({total_duration/60:.2f} 分钟)")
    print(f"开始时间: {datetime.fromtimestamp(total_start_time).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"结束时间: {total_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"结果保存目录: {result_dir}")
    print("\n实验结果总结:")
    print("-" * 80)
    
    success_count = 0
    fail_count = 0
    for i, result in enumerate(results, 1):
        status = "✓ 成功" if result["success"] else "✗ 失败"
        duration_str = f"{result['duration']:.2f}秒" if result["duration"] > 0 else "N/A"
        print(f"{i}. {result['name']}: {status} (耗时: {duration_str})")
        print(f"   日志文件: {result['log_path']}")
        if result["success"]:
            success_count += 1
        else:
            fail_count += 1
    
    print("-" * 80)
    print(f"成功: {success_count}, 失败: {fail_count}")
    print(f"\n所有实验结果保存在: {result_dir}")
    print("  包括：模型文件(.t7)、CSV文件、TXT摘要文件、实验日志文件")
    print("="*80 + "\n")
    
    # 保存总结到文件
    summary_path = os.path.join(result_dir, f"summary_{timestamp}.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("TA注意力机制对照实验总结\n")
        f.write("=" * 80 + "\n")
        f.write(f"开始时间: {datetime.fromtimestamp(total_start_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"结束时间: {total_end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总耗时: {total_duration:.2f} 秒 ({total_duration/60:.2f} 分钟)\n")
        f.write(f"成功: {success_count}, 失败: {fail_count}\n")
        f.write("\n详细结果:\n")
        f.write("-" * 80 + "\n")
        for i, result in enumerate(results, 1):
            status = "成功" if result["success"] else "失败"
            duration_str = f"{result['duration']:.2f}秒" if result["duration"] > 0 else "N/A"
            f.write(f"{i}. {result['name']}: {status} (耗时: {duration_str})\n")
            f.write(f"   日志文件: {result['log_path']}\n")
        f.write("=" * 80 + "\n")
    
    print(f"总结已保存到: {summary_path}\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断程序执行。")
        sys.exit(1)

