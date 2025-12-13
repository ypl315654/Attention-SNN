"""
批量运行所有注意力机制实验的脚本

用法：
    python run_all_experiments.py
    或
    python run_all_experiments.py --num_epochs 100 --dt 25 --T 20

配置说明：
    - 默认会依次运行注意力机制 no, TA, CA, SA, CSA, TSA, TCA, TCSA 的所有实验
    - 可以通过命令行参数配置默认的epoch、dt、T等参数
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(script_dir))  # 添加MA_SNN到路径

# 注意力机制选项列表（按顺序运行）
ATTENTION_OPTIONS = ['no', 'TA', 'CA', 'SA', 'CSA', 'TSA', 'TCA', 'TCSA']


def run_experiment(attention, num_epochs=None, dt=None, T=None, batch_size=None, 
                   lr=None, c_ratio=None, t_ratio=None, result_dir=None, batch_result_dir=None):
    """
    运行单个实验
    
    Args:
        attention: 注意力机制字符串 ('no', 'TA', 'CA', 等)
        num_epochs: 训练轮数
        dt: 时间步长
        T: 时间窗口
        batch_size: 批次大小
        lr: 学习率
        c_ratio: 通道比例
        t_ratio: 时间比例
        result_dir: 单个实验的结果目录路径（用于保存实验的模型、CSV、TXT文件）
        batch_result_dir: 批量实验的日志目录（用于保存batch_log文件）
    """
    print("\n" + "=" * 80)
    print(f"开始运行实验: {attention}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # 构建命令行参数
    cmd = [sys.executable, os.path.join(script_dir, "Att_SNN_CNN.py")]
    cmd.extend(["--attention", attention])
    
    if num_epochs is not None:
        cmd.extend(["--num_epochs", str(num_epochs)])
    if dt is not None:
        cmd.extend(["--dt", str(dt)])
    if T is not None:
        cmd.extend(["--T", str(T)])
    if batch_size is not None:
        cmd.extend(["--batch_size", str(batch_size)])
    if lr is not None:
        cmd.extend(["--lr", str(lr)])
    if c_ratio is not None:
        cmd.extend(["--c_ratio", str(c_ratio)])
    if t_ratio is not None:
        cmd.extend(["--t_ratio", str(t_ratio)])
    
    # 指定结果保存目录（每个实验的结果会保存到这里）
    if result_dir is not None:
        cmd.extend(["--result_dir", result_dir])
    
    # 创建日志文件路径（保存到批量实验目录）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"batch_log_{attention}_{timestamp}.txt"
    log_path = os.path.join(batch_result_dir, log_filename)
    
    print(f"执行命令: {' '.join(cmd)}")
    print(f"日志文件: {log_path}")
    print("-" * 80)
    
    # 运行实验，将输出保存到日志文件
    try:
        with open(log_path, 'w', encoding='utf-8') as log_file:
            # 写入实验信息
            log_file.write("=" * 80 + "\n")
            log_file.write(f"实验: {attention}\n")
            log_file.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"执行命令: {' '.join(cmd)}\n")
            log_file.write("=" * 80 + "\n\n")
            log_file.flush()
            
            # 运行命令，同时输出到终端和文件
            process = subprocess.Popen(
                cmd, 
                cwd=script_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # 实时读取输出并同时写入文件和打印到终端
            # 过滤进度条相关输出（tqdm格式的进度条）
            import re
            
            # 用于跟踪进度条，只写入最终的完整进度条（100%完成的）
            last_progress_line = None
            
            for line in process.stdout:
                # 检查是否为进度条行
                # 特征：包含 "数字%|" (如 1%|, 100%|) 且通常包含 "it/s" 或 "s/it"
                # 用户日志示例: Train:Epoch[1/1]:   1%|▏         | 1/74 ...
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
            
            # 写入结束信息
            log_file.write("\n" + "=" * 80 + "\n")
            log_file.write(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"返回代码: {return_code}\n")
            log_file.write("=" * 80 + "\n")
        
        if return_code == 0:
            print("\n" + "=" * 80)
            print(f"实验 {attention} 完成！")
            print(f"日志已保存到: {log_path}")
            print("=" * 80 + "\n")
            return True
        else:
            print("\n" + "=" * 80)
            print(f"实验 {attention} 失败！返回代码: {return_code}")
            print(f"日志已保存到: {log_path}")
            print("=" * 80 + "\n")
            return False
            
    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print(f"用户中断：实验 {attention} 被中断")
        print(f"日志已保存到: {log_path}")
        print("=" * 80 + "\n")
        # 确保日志文件记录了中断信息
        try:
            with open(log_path, 'a', encoding='utf-8') as log_file:
                log_file.write("\n" + "=" * 80 + "\n")
                log_file.write("用户中断实验\n")
                log_file.write(f"中断时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                log_file.write("=" * 80 + "\n")
        except:
            pass
        return False
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"实验 {attention} 发生错误: {str(e)}")
        print(f"日志已保存到: {log_path}")
        print("=" * 80 + "\n")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='批量运行所有注意力机制实验',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认配置运行所有实验
  python run_all_experiments.py
  
  # 指定训练轮数为100
  python run_all_experiments.py --num_epochs 100
  
  # 指定多个参数
  python run_all_experiments.py --num_epochs 200 --dt 25 --T 20 --batch_size 32
  
  # 只运行部分实验 (例如从TA开始到TSA结束)
  python run_all_experiments.py --start_from TA --end_at TSA
        """
    )
    
    parser.add_argument('--num_epochs', type=int, default=300,
                        help='训练轮数 (默认: 300)')
    parser.add_argument('--dt', type=int, default=25,
                        help='时间步长 dt (默认: 25)')
    parser.add_argument('--T', type=int, default=20,
                        help='时间窗口 T (默认: 20)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小 (默认: 16)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率 (默认: 1e-4)')
    parser.add_argument('--c_ratio', type=int, default=8,
                        help='通道比例 (默认: 8)')
    parser.add_argument('--t_ratio', type=int, default=5,
                        help='时间比例 (默认: 5)')
    parser.add_argument('--start_from', type=str, 
                        choices=ATTENTION_OPTIONS, default='no',
                        help='从哪个注意力机制开始运行 (默认: no)')
    parser.add_argument('--end_at', type=str, 
                        choices=ATTENTION_OPTIONS, default='TCSA',
                        help='运行到哪个注意力机制 (默认: TCSA)')
    parser.add_argument('--auto_continue', action='store_true',
                        help='自动继续下一个实验，不需要确认')
    
    args = parser.parse_args()
    
    # 创建以时间和参数命名的文件夹
    from DVSGestures.CNN import Config
    config_temp = Config.configs()
    base_result_dir = config_temp.recordPath  # 基础Result目录
    
    # 生成文件夹名称：包含时间和主要参数
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"batch_{timestamp}_epoch{args.num_epochs}_dt{args.dt}_T{args.T}"
    result_dir = os.path.join(base_result_dir, folder_name)
    
    # 创建文件夹
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        print(f"创建结果目录: {result_dir}")
    
    # 确定运行范围
    start_idx = ATTENTION_OPTIONS.index(args.start_from)
    end_idx = ATTENTION_OPTIONS.index(args.end_at)
    experiments_to_run = ATTENTION_OPTIONS[start_idx:end_idx + 1]
    
    # 显示配置信息
    print("\n" + "=" * 80)
    print("批量实验配置")
    print("=" * 80)
    print(f"实验列表: {', '.join(experiments_to_run)}")
    print(f"实验数量: {len(experiments_to_run)}")
    print(f"训练轮数: {args.num_epochs}")
    print(f"时间步长 dt: {args.dt}")
    print(f"时间窗口 T: {args.T}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"通道比例: {args.c_ratio}")
    print(f"时间比例: {args.t_ratio}")
    print(f"自动继续: {args.auto_continue}")
    print(f"结果保存目录: {result_dir}")
    print("=" * 80)
    
    # 确认开始
    if not args.auto_continue:
        confirm = input("\n确认开始批量运行？(y/n): ").strip().lower()
        if confirm not in ['y', 'yes', '']:
            print("已取消")
            return
    
    # 运行实验
    start_time = datetime.now()
    success_count = 0
    fail_count = 0
    results = {}
    
    for i, attention in enumerate(experiments_to_run, 1):
        print(f"\n进度: {i}/{len(experiments_to_run)}")
        success = run_experiment(
            attention=attention,
            num_epochs=args.num_epochs,
            dt=args.dt,
            T=args.T,
            batch_size=args.batch_size,
            lr=args.lr,
            c_ratio=args.c_ratio,
            t_ratio=args.t_ratio,
            result_dir=result_dir,
            batch_result_dir=result_dir  # batch日志也保存在同一个文件夹
        )
        
        results[attention] = "成功" if success else "失败"
        if success:
            success_count += 1
        else:
            fail_count += 1
        
        # 询问是否继续（如果不是自动模式且不是最后一个）
        if not args.auto_continue and i < len(experiments_to_run):
            continue_input = input(f"\n实验 {attention} 完成。继续下一个实验？(y/n，默认y): ").strip().lower()
            if continue_input in ['n', 'no']:
                print(f"\n用户选择停止。已完成 {i}/{len(experiments_to_run)} 个实验")
                break
    
    # 总结
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "=" * 80)
    print("批量实验总结")
    print("=" * 80)
    print(f"总实验数: {len(experiments_to_run)}")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    print(f"总耗时: {duration}")
    print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n详细结果:")
    for attention, status in results.items():
        print(f"  {attention}: {status}")
    print(f"\n所有实验结果保存在: {result_dir}")
    print("  包括：模型文件(.t7)、CSV文件、TXT摘要文件、批量日志文件")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

