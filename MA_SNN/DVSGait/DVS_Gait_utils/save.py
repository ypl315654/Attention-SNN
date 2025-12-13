import pandas as pd
import os
from datetime import datetime


def save_csv(config):
    config.epoch_list.append(config.best_epoch)
    config.acc_test_list.append(config.best_acc)

    lists = [config.loss_train_list,
             config.loss_test_list,
             config.acc_train_list,
             config.acc_test_list]
    csv = pd.DataFrame(
        data=lists,
        index=['Train_Loss',
               'Test_Loss',
               'Train_Accuracy',
               'Test_Accuracy'],
        columns=config.epoch_list)
    csv.index.name = 'Epochs'

    if not os.path.exists(config.recordPath):
        os.makedirs(config.recordPath)
    csv.to_csv(config.recordPath + os.sep + config.recordNames)
    
    # 新增：保存txt文件
    save_txt(config)


def save_txt(config):
    """保存实验摘要信息到txt文件"""
    txt_name = config.name + ".txt"
    txt_path = config.recordPath + os.sep + txt_name
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("实验摘要信息\n")
        f.write("=" * 60 + "\n\n")
        
        # 基本信息
        f.write(f"实验名称: {config.name}\n")
        f.write(f"保存时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 模型配置
        f.write("-" * 60 + "\n")
        f.write("模型配置\n")
        f.write("-" * 60 + "\n")
        f.write(f"注意力机制: {config.attention}\n")
        f.write(f"时间步长 dt: {config.dt} ms\n")
        f.write(f"时间窗口 T: {config.T}\n")
        f.write(f"批次大小: {config.batch_size}\n")
        f.write(f"学习率: {config.lr}\n")
        f.write(f"总训练轮数: {config.num_epochs}\n")
        f.write(f"通道比例 c_ratio: {config.c_ratio}\n")
        f.write(f"时间比例 t_ratio: {config.t_ratio}\n\n")
        
        # LIF神经元参数
        f.write("-" * 60 + "\n")
        f.write("LIF神经元参数\n")
        f.write("-" * 60 + "\n")
        f.write(f"Alpha (衰减因子): {config.alpha}\n")
        f.write(f"Beta: {config.beta}\n")
        f.write(f"Vreset (重置电压): {config.Vreset}\n")
        f.write(f"Vthres (阈值电压): {config.Vthres}\n")
        f.write(f"模式选择: {config.mode_select}\n")
        f.write(f"TR模型: {config.TR_model}\n\n")
        
        # 训练结果
        f.write("-" * 60 + "\n")
        f.write("训练结果\n")
        f.write("-" * 60 + "\n")
        f.write(f"最佳准确率: {config.best_acc:.4f}%\n")
        f.write(f"最佳轮次: {config.best_epoch}\n")
        if len(config.acc_train_list) > 0:
            f.write(f"最终训练准确率: {config.acc_train_list[-1]:.4f}% (epoch {len(config.acc_train_list)})\n")
        if len(config.acc_test_list) > 0:
            f.write(f"最终测试准确率: {config.acc_test_list[-1]:.4f}% (epoch {len(config.acc_test_list)})\n")
        if len(config.loss_train_list) > 0:
            f.write(f"最终训练损失: {config.loss_train_list[-1]:.6f}\n")
        if len(config.loss_test_list) > 0:
            f.write(f"最终测试损失: {config.loss_test_list[-1]:.6f}\n")
        f.write("\n")
        
        # 训练历史（可选：显示前5个和后5个epoch）
        f.write("-" * 60 + "\n")
        f.write("训练历史摘要\n")
        f.write("-" * 60 + "\n")
        if len(config.epoch_list) > 10:
            f.write("前5个epoch:\n")
            for i in range(min(5, len(config.epoch_list))):
                epoch = config.epoch_list[i]
                if i < len(config.acc_train_list) and i < len(config.acc_test_list):
                    f.write(f"  Epoch {epoch}: Train Acc={config.acc_train_list[i]:.2f}%, "
                           f"Test Acc={config.acc_test_list[i]:.2f}%, "
                           f"Train Loss={config.loss_train_list[i]:.6f}, "
                           f"Test Loss={config.loss_test_list[i]:.6f}\n")
            f.write("...\n")
            f.write("后5个epoch:\n")
            start_idx = max(5, len(config.epoch_list) - 5)
            for i in range(start_idx, len(config.epoch_list)):
                epoch = config.epoch_list[i]
                if i < len(config.acc_train_list) and i < len(config.acc_test_list):
                    f.write(f"  Epoch {epoch}: Train Acc={config.acc_train_list[i]:.2f}%, "
                           f"Test Acc={config.acc_test_list[i]:.2f}%, "
                           f"Train Loss={config.loss_train_list[i]:.6f}, "
                           f"Test Loss={config.loss_test_list[i]:.6f}\n")
        else:
            f.write("所有epoch:\n")
            for i, epoch in enumerate(config.epoch_list):
                if i < len(config.acc_train_list) and i < len(config.acc_test_list):
                    f.write(f"  Epoch {epoch}: Train Acc={config.acc_train_list[i]:.2f}%, "
                           f"Test Acc={config.acc_test_list[i]:.2f}%, "
                           f"Train Loss={config.loss_train_list[i]:.6f}, "
                           f"Test Loss={config.loss_test_list[i]:.6f}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"模型文件: {config.modelNames}\n")
        f.write(f"CSV文件: {config.recordNames}\n")
        f.write(f"TXT文件: {txt_name}\n")
        f.write("=" * 60 + "\n")
