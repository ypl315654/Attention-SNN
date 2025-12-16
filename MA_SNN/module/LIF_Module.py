import torch
from torch import nn
from module.LIF import *
from module.Attention import *
from module.utils import *
import torch.nn.functional as F


class AttLIF(nn.Module):
    def __init__(
        self,
        inputSize,
        hiddenSize,
        spikeActFun,
        attention="TA",
        onlyLast=False,
        useBatchNorm=False,
        useLayerNorm=False,
        init_method=None,
        scale=0.3,
        pa_dict=None,
        pa_train_self=False,
        bias=True,
        reduction=16,
        T=60,
        p=0,
        track_running_stats=False,
        mode_select="spike",
        mem_act=torch.relu,
        TR_model="NTR",
        t_ratio=16,
        disable_spike=False
    ):
        super().__init__()
        self.onlyLast = onlyLast

        self.useBatchNorm = useBatchNorm

        self.network = nn.Sequential()
        self.attention_flag = attention
        self.disable_spike = disable_spike  # 是否禁止发放脉冲
        self.linear = nn.Linear(
            in_features=inputSize,
            out_features=hiddenSize,
            bias=bias,
        )

        if self.useBatchNorm:
            self.BNLayer = nn.BatchNorm1d(
                num_features=hiddenSize, track_running_stats=track_running_stats
            )

        if init_method is not None:
            paramInit(model=self.linear, method=init_method)
        if self.attention_flag == "TA":
            self.attention = TA(T, hiddenSize, t_ratio=t_ratio, fc=True)
        elif self.attention_flag == "no":
            pass

        self.network.add_module(
            "IF",
            IFCell(
                inputSize,
                hiddenSize,
                spikeActFun,
                bias=bias,
                scale=scale,
                pa_dict=pa_dict,
                pa_train_self=pa_train_self,
                p=p,
                mode_select=mode_select,
                mem_act=mem_act,
                TR_model=TR_model,
            ),
        )

    def forward(self, data):

        for layer in self.network:
            layer.reset()

        b, t, _ = data.size()
        output = self.linear(data.reshape(b * t, -1))

        if self.useBatchNorm:
            output = self.BNLayer(output)

        outputsum = output.reshape(b, t, -1)

        if self.attention_flag == "no":
            data = outputsum
            attention_weights = None
        elif self.attention_flag == "TA":
            # TA 返回 (原始输入, 注意力权重)
            data, attention_weights = self.attention(outputsum)
        else:
            # 其他注意力机制保持原样
            data = self.attention(outputsum)
            attention_weights = None

        # 如果是TA，预先计算哪些时间步允许发放脉冲
        spike_mask_per_step = None
        if self.attention_flag == "TA" and attention_weights is not None and self.disable_spike:
            # attention_weights 形状: (b, timeWindows, channels)
            # 计算每个时间步的平均权重（跨通道维度）
            all_step_weights = attention_weights.mean(dim=2)  # (b, timeWindows)
            
            # 选择top T//2 个时间步允许发放脉冲
            k = max(1, t // 2)  # 至少选择1步
            _, top_indices = torch.topk(all_step_weights, k, dim=1)  # (b, k)
            
            # 为每个时间步创建掩码
            spike_mask_per_step = []
            for step in range(t):
                is_top_k = (top_indices == step).any(dim=1)  # (b,)
                spike_mask_per_step.append(is_top_k.float())  # (b,)

        for step in range(list(data.size())[1]):
            out = data[:, step, :]
            
            # 如果是TA注意力且启用禁止发放脉冲，应用掩码
            if self.attention_flag == "TA" and spike_mask_per_step is not None and self.disable_spike:
                # 获取当前时间步的掩码
                spike_mask = spike_mask_per_step[step]  # (b,)
                spike_mask = spike_mask.view(b, 1)  # (b, 1) 用于广播
                
                # 正常通过LIF神经元（膜电位会更新）
                for layer in self.network:
                    out = layer(out)
                
                # 应用掩码：禁止发放脉冲的位置输出为0
                out = out * spike_mask
            else:
                # 正常处理
                for layer in self.network:
                    out = layer(out)
            
            output = out

            if step == 0:
                temp = list(output.size())
                temp.insert(1, list(data.size())[1])
                outputsum = torch.zeros(temp)
                if outputsum.device != data.device:
                    outputsum = outputsum.to(data.device)
            outputsum[:, step, :] = output

        if self.onlyLast:
            return output
        else:
            return outputsum


class ConvAttLIF(nn.Module):
    def __init__(
        self,
        inputSize,
        hiddenSize,
        kernel_size,
        spikeActFun,
        h=128,
        w=128,
        attention="TA",
        bias=True,
        onlyLast=False,
        padding=1,
        useBatchNorm=False,
        init_method=None,
        scale=0.02,
        pa_dict=None,
        pa_train_self=False,
        reduction=16,
        T=60,
        stride=1,
        pooling_kernel_size=1,
        p=0,
        track_running_stats=False,
        mode_select="spike",
        mem_act=torch.relu,
        TR_model="NTR",
        c_ratio=16,
        t_ratio=16,
        disable_spike=False
    ):
        super().__init__()

        self.onlyLast = onlyLast
        self.attention_flag = attention
        self.disable_spike = disable_spike  # 是否禁止发放脉冲

        self.conv2d = nn.Conv2d(
            in_channels=inputSize,
            out_channels=hiddenSize,
            kernel_size=kernel_size,
            bias=True,
            padding=padding,
            stride=stride,
        )

        if init_method is not None:
            paramInit(model=self.conv2d, method=init_method)

        self.useBatchNorm = useBatchNorm

        if self.useBatchNorm:
            self.BNLayer = nn.BatchNorm2d(
                hiddenSize, track_running_stats=track_running_stats
            )

        self.pooling_kernel_size = pooling_kernel_size
        if self.pooling_kernel_size > 1:
            self.pooling = nn.AvgPool2d(kernel_size=pooling_kernel_size)

        if self.attention_flag == "TCSA":
            self.attention = TCSA(T, hiddenSize, c_ratio=c_ratio, t_ratio=t_ratio)
        elif self.attention_flag == "TSA":
            self.attention = TSA(T, hiddenSize, t_ratio=t_ratio)
        elif self.attention_flag == "TCA":
            self.attention = TCA(T, hiddenSize, c_ratio=c_ratio, t_ratio=t_ratio)
        elif self.attention_flag == "CSA":
            self.attention = CSA(T, hiddenSize, c_ratio=c_ratio)
        elif self.attention_flag == "TA":
            self.attention = TA(T, hiddenSize, t_ratio=t_ratio, disable_spike=self.disable_spike)
        elif self.attention_flag == "CA":
            self.attention = CA(T, hiddenSize, c_ratio=c_ratio)
        elif self.attention_flag == "SA":
            self.attention = SA(T, hiddenSize)
        elif self.attention_flag == "no":
            pass
        self.network = nn.Sequential()
        self.network.add_module(
            "ConvIF",
            ConvIFCell(
                inputSize=inputSize,
                hiddenSize=hiddenSize,
                kernel_size=kernel_size,
                bias=bias,
                spikeActFun=spikeActFun,
                padding=padding,
                scale=scale,
                pa_dict=pa_dict,
                pa_train_self=pa_train_self,
                p=p,
                mode_select=mode_select,
                mem_act=mem_act,
                TR_model=TR_model,
            ),
        )

    def forward(self, data):

        for layer in self.network:
            layer.reset()

        b, t, c, h, w = data.size()
        out = data.reshape(b * t, c, h, w)
        output = self.conv2d(out)

        if self.useBatchNorm:
            output = self.BNLayer(output)

        if self.pooling_kernel_size > 1:
            output = self.pooling(output)

        _, c, h, w = output.size()
        outputsum = output.reshape(b, t, c, h, w)

        if self.attention_flag == "no":
            data = outputsum
            attention_weights = None
        elif self.attention_flag == "TA":
            # TA 返回 (原始输入, 注意力权重)
            data, attention_weights = self.attention(outputsum)
        else:
            # 其他注意力机制保持原样
            data = self.attention(outputsum)
            attention_weights = None

        # 如果是TA，预先计算哪些时间步允许发放脉冲
        spike_mask_per_step = None
        if self.attention_flag == "TA" and attention_weights is not None and self.disable_spike:
            # attention_weights 形状: (b, timeWindows, channels, 1, 1)
            # 计算每个时间步的平均权重（跨通道和空间维度）
            all_step_weights = attention_weights.mean(dim=(2, 3, 4))  # (b, timeWindows)
            
            # 选择top T//2 个时间步允许发放脉冲
            k = max(1, t // 2)  # 至少选择1步
            _, top_indices = torch.topk(all_step_weights, k, dim=1)  # (b, k)
            
            # 为每个时间步创建掩码
            spike_mask_per_step = []
            for step in range(t):
                is_top_k = (top_indices == step).any(dim=1)  # (b,)
                spike_mask_per_step.append(is_top_k.float())  # (b,)

        for step in range(list(data.size())[1]):
            out = data[:, step, :, :, :]
            
            # 如果是TA注意力且启用禁止发放脉冲，应用掩码
            if self.attention_flag == "TA" and spike_mask_per_step is not None and self.disable_spike:
                # 获取当前时间步的掩码
                spike_mask = spike_mask_per_step[step]  # (b,)
                spike_mask = spike_mask.view(b, 1, 1, 1)  # (b, 1, 1, 1) 用于广播
                
                # 正常通过LIF神经元（膜电位会更新）
                for layer in self.network:
                    out = layer(out)
                
                # 应用掩码：禁止发放脉冲的位置输出为0
                out = out * spike_mask
            else:
                # 正常处理
                for layer in self.network:
                    out = layer(out)
            
            output = out

            if step == 0:
                temp = list(output.size())
                temp.insert(1, list(data.size())[1])
                outputsum = torch.zeros(temp)
                if outputsum.device != data.device:
                    outputsum = outputsum.to(data.device)

            outputsum[:, step, :, :, :] = output

        if self.onlyLast:
            return output
        else:
            return outputsum
