"""
计算时间统计分析工具
用于统计LLM冻结部分前向传播的计算时间和LoRA矩阵反向更新的计算时间
数据集和模型的加载方式使用fed_trainer.py文件中的方式
"""

import os
import sys
import time
import copy
import torch
import numpy as np
import argparse
from torch.utils.data import Subset, DataLoader
from torch import nn

# 设置使用GPU 3
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from model_utils import get_model_tokenizer, get_model_lora
from dataset_utils import get_fed_data_info, Metrics_trainer
from task import Task
from transformers import TrainingArguments, Trainer, Seq2SeqTrainer, Seq2SeqTrainingArguments, \
    DataCollatorWithPadding, DataCollatorForTokenClassification, DataCollatorForLanguageModeling


class TimingTrainer(Trainer):
    def __init__(self, *args, timing_stats=None, lora_params_size=0, upload_speed=5.0, download_speed=50.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.timing_stats = timing_stats
        self.lora_params_size = lora_params_size
        self.upload_speed = upload_speed
        self.download_speed = download_speed

    def training_step(self, model, inputs, *args, **kwargs):
        model.train()

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        data_transfer_start = time.perf_counter()
        inputs = self._prepare_inputs(inputs)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        data_transfer_end = time.perf_counter()
        data_transfer_time = data_transfer_end - data_transfer_start

        if self.timing_stats is not None:
            self.timing_stats['data_transfer_times'].append(data_transfer_time)
            self.timing_stats['total_data_transfer_time'] += data_transfer_time

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        forward_start = time.perf_counter()
        loss = self.compute_loss(model, inputs)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        forward_end = time.perf_counter()
        forward_time = forward_end - forward_start

        if self.args.gradient_accumulation_steps > 1 and self.args.deepspeed is None:
            loss = loss / self.args.gradient_accumulation_steps

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        backward_start = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        backward_end = time.perf_counter()
        backward_time = backward_end - backward_start

        if self.timing_stats is not None:
            self.timing_stats['forward_times'].append(forward_time)
            self.timing_stats['backward_times'].append(backward_time)
            self.timing_stats['total_forward_time'] += forward_time
            self.timing_stats['total_backward_time'] += backward_time
            self.timing_stats['num_batches'] += 1

            upload_time = (self.lora_params_size / (1024**2)) * 8 / self.upload_speed
            download_time = (self.lora_params_size / (1024**2)) * 8 / self.download_speed
            self.timing_stats['upload_times'].append(upload_time)
            self.timing_stats['download_times'].append(download_time)
            self.timing_stats['total_upload_time'] += upload_time
            self.timing_stats['total_download_time'] += download_time

        return loss.detach()

    def optimizer_step(self, optimizer, model, optimizer_idx, closure=None, **kwargs):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        param_update_start = time.perf_counter()
        super().optimizer_step(optimizer, model, optimizer_idx, closure=closure, **kwargs)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        param_update_end = time.perf_counter()
        param_update_time = param_update_end - param_update_start
        if self.timing_stats is not None:
            self.timing_stats['param_update_times'].append(param_update_time)
            self.timing_stats['total_param_update_time'] += param_update_time


class TimingSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, timing_stats=None, lora_params_size=0, upload_speed=5.0, download_speed=50.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.timing_stats = timing_stats
        self.lora_params_size = lora_params_size
        self.upload_speed = upload_speed
        self.download_speed = download_speed

    def training_step(self, model, inputs, *args, **kwargs):
        model.train()

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        data_transfer_start = time.perf_counter()
        inputs = self._prepare_inputs(inputs)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        data_transfer_end = time.perf_counter()
        data_transfer_time = data_transfer_end - data_transfer_start

        if self.timing_stats is not None:
            self.timing_stats['data_transfer_times'].append(data_transfer_time)
            self.timing_stats['total_data_transfer_time'] += data_transfer_time

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        forward_start = time.perf_counter()
        loss = self.compute_loss(model, inputs)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        forward_end = time.perf_counter()
        forward_time = forward_end - forward_start

        if self.args.gradient_accumulation_steps > 1 and self.args.deepspeed is None:
            loss = loss / self.args.gradient_accumulation_steps

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        backward_start = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        backward_end = time.perf_counter()
        backward_time = backward_end - backward_start

        if self.timing_stats is not None:
            self.timing_stats['forward_times'].append(forward_time)
            self.timing_stats['backward_times'].append(backward_time)
            self.timing_stats['total_forward_time'] += forward_time
            self.timing_stats['total_backward_time'] += backward_time
            self.timing_stats['num_batches'] += 1

            upload_time = (self.lora_params_size / (1024**2)) * 8 / self.upload_speed
            download_time = (self.lora_params_size / (1024**2)) * 8 / self.download_speed
            self.timing_stats['upload_times'].append(upload_time)
            self.timing_stats['download_times'].append(download_time)
            self.timing_stats['total_upload_time'] += upload_time
            self.timing_stats['total_download_time'] += download_time

        return loss.detach()

    def optimizer_step(self, optimizer, model, optimizer_idx, closure=None, **kwargs):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        param_update_start = time.perf_counter()
        super().optimizer_step(optimizer, model, optimizer_idx, closure=closure, **kwargs)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        param_update_end = time.perf_counter()
        param_update_time = param_update_end - param_update_start
        if self.timing_stats is not None:
            self.timing_stats['param_update_times'].append(param_update_time)
            self.timing_stats['total_param_update_time'] += param_update_time


class TimingAnalyzer:
    """用于分析前向传播和反向更新计算时间的类"""
    
    def __init__(self, args):
        self.args = args
        self.data_collator = None
        self.model = None
        self.tokenizer = None
        self.task = None
        self.lora_params_size = 0  # LoRA参数大小（字节）
        self.upload_speed = args.upload_speed  # 上传网速 (Mbps)
        self.download_speed = args.download_speed  # 下载网速 (Mbps)
        self.timing_stats = {
            'forward_times': [],
            'backward_times': [],
            'upload_times': [],  # 上传时间列表
            'download_times': [],  # 下载时间列表
            'data_transfer_times': [],  # 数据传输时间列表
            'param_update_times': [],  # 参数更新时间列表
            'total_forward_time': 0.0,
            'total_backward_time': 0.0,
            'total_upload_time': 0.0,
            'total_download_time': 0.0,
            'total_data_transfer_time': 0.0,
            'total_param_update_time': 0.0,
            'num_batches': 0,
            'avg_forward_time': 0.0,
            'avg_backward_time': 0.0,
            'avg_upload_time': 0.0,
            'avg_download_time': 0.0,
            'avg_data_transfer_time': 0.0,
            'avg_param_update_time': 0.0,
        }
    
    def set_data_collator(self, tokenizer, task):
        """根据任务类型设置数据collator - 使用与fed_trainer.py相同的方式"""
        if task == Task.SequenceClassification:
            self.data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        elif task == Task.TokenClassification:
            self.data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
        elif task == Task.QuestionAnswering:
            self.data_collator = None
        else:
            self.data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    
    def load_model_and_data(self):
        """加载模型和数据集 - 使用与fed_trainer.py相同的方式"""
        print("=" * 80)
        print("加载Tokenizer...")
        print("=" * 80)
        self.tokenizer = get_model_tokenizer(
            model=self.args.model, 
            max_length=self.args.max_length
        )
        
        print("=" * 80)
        print("加载数据集...")
        print("=" * 80)
        data, partition_map, num_labels, metric, validation_key, \
            task, label_names, validation_dataset, grids = get_fed_data_info(
                args=self.args, 
                tokenizer=self.tokenizer
            )
        self.task = task
        
        print("=" * 80)
        print("加载模型并添加LoRA...")
        print("=" * 80)
        self.model, trainable_parameters, untrain_part = get_model_lora(
            model=self.args.model,
            lora_alpha=self.args.lora_alpha,
            lora_rank=self.args.lora_rank,
            num_labels=num_labels,
            task=task,
            method=self.args.method
        )
        
        print("\n模型架构:")
        print(self.model)
        print(f"\nLM Head (冻结) 部分: {untrain_part}")
        
        # 统计LoRA参数
        lora_params = 0
        print("\n可训练的LoRA层:")
        for layer_name in self.model.state_dict():
            if 'lora' in layer_name:
                print(f"  - {layer_name}")
                lora_params += self.model.state_dict()[layer_name].numel()
        print(f"LoRA总参数数: {lora_params}")
        
        # 计算LoRA参数大小（字节），假设float32
        self.lora_params_size = lora_params * 4  # 4 bytes per float32 parameter
        print(f"LoRA参数大小: {self.lora_params_size / (1024**2):.2f} MB")
        
        self.set_data_collator(self.tokenizer, self.task)
        
        return data, partition_map
    
    def analyze_timing(self, data, data_indices, device='cuda:0'):
        """分析前向传播和反向更新的计算时间"""
        print("\n" + "=" * 80)
        print("开始计算时间分析...")
        print("=" * 80)

        train_data = Subset(data["train"], data_indices)
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)
        save_steps = sys.maxsize

        if self.task in [Task.SequenceClassification, Task.TokenClassification, Task.QuestionAnswering, Task.CausalLM]:
            training_args = TrainingArguments(
                output_dir='./save/model',
                save_steps=save_steps,
                num_train_epochs=self.args.epochs,
                per_device_train_batch_size=self.args.batch_size,
                do_train=True,
                learning_rate=self.args.lr,
                ddp_find_unused_parameters=False,
                lr_scheduler_type="constant",
                logging_steps=1,
                dataloader_num_workers=0,
                dataloader_pin_memory=False,
                disable_tqdm=True,
                report_to=[],
                logging_strategy="no",
                save_strategy="no",
            )
            trainer = TimingTrainer(
                model=self.model,
                processing_class=self.tokenizer,
                args=training_args,
                train_dataset=train_data,
                data_collator=self.data_collator,
                optimizers=(optimizer, None),
                timing_stats=self.timing_stats,
                lora_params_size=self.lora_params_size,
                upload_speed=self.upload_speed,
                download_speed=self.download_speed,
            )
        else:
            training_args = Seq2SeqTrainingArguments(
                output_dir='save',
                save_steps=save_steps,
                num_train_epochs=self.args.epochs,
                per_device_train_batch_size=self.args.batch_size,
                do_train=True,
                learning_rate=self.args.lr,
                lr_scheduler_type="constant",
                logging_steps=1,
                dataloader_num_workers=0,
                dataloader_pin_memory=False,
                disable_tqdm=True,
                report_to=[],
                logging_strategy="no",
                save_strategy="no",
            )
            trainer = TimingSeq2SeqTrainer(
                model=self.model,
                processing_class=self.tokenizer,
                args=training_args,
                train_dataset=train_data,
                data_collator=self.data_collator,
                optimizers=(optimizer, None),
                timing_stats=self.timing_stats,
                lora_params_size=self.lora_params_size,
                upload_speed=self.upload_speed,
                download_speed=self.download_speed,
            )

        self.model.to(device)
        trainer.train()
        self.model.cpu()
        
        if self.timing_stats['num_batches'] > 0:
            self.timing_stats['avg_forward_time'] = (
                self.timing_stats['total_forward_time'] / self.timing_stats['num_batches']
            )
            self.timing_stats['avg_backward_time'] = (
                self.timing_stats['total_backward_time'] / self.timing_stats['num_batches']
            )
            self.timing_stats['avg_upload_time'] = (
                self.timing_stats['total_upload_time'] / self.timing_stats['num_batches']
            )
            self.timing_stats['avg_download_time'] = (
                self.timing_stats['total_download_time'] / self.timing_stats['num_batches']
            )
            self.timing_stats['avg_data_transfer_time'] = (
                self.timing_stats['total_data_transfer_time'] / self.timing_stats['num_batches']
            )
            self.timing_stats['avg_param_update_time'] = (
                self.timing_stats['total_param_update_time'] / self.timing_stats['num_batches']
            )
    
    def print_results(self):
        """打印分析结果"""
        print("\n" + "=" * 80)
        print("计算时间统计结果")
        print("=" * 80)
        
        stats = self.timing_stats
        
        def safe_stats(values):
            if not values:
                return 'N/A', 'N/A', 'N/A', 'N/A'
            return (
                f"{min(values)*1000:.2f} ms",
                f"{max(values)*1000:.2f} ms",
                f"{np.std(values)*1000:.2f} ms",
                f"{np.mean(values)*1000:.2f} ms"
            )

        print(f"\n总批次数: {stats['num_batches']}")
        print(f"\nLLM冻结部分前向传播时间:")
        print(f"  - 总时间: {stats['total_forward_time']:.4f} 秒 ({stats['total_forward_time']*1000:.2f} ms)")
        forward_min, forward_max, forward_std, forward_mean = safe_stats(stats['forward_times'])
        print(f"  - 最小时间: {forward_min}")
        print(f"  - 最大时间: {forward_max}")
        print(f"  - 标准差: {forward_std}")
        
        print(f"\nLoRA矩阵反向更新时间:")
        print(f"  - 总时间: {stats['total_backward_time']:.4f} 秒 ({stats['total_backward_time']*1000:.2f} ms)")
        backward_min, backward_max, backward_std, backward_mean = safe_stats(stats['backward_times'])
        print(f"  - 最小时间: {backward_min}")
        print(f"  - 最大时间: {backward_max}")
        print(f"  - 标准差: {backward_std}")
        
        print(f"\n参数更新时间:")
        print(f"  - 总时间: {stats['total_param_update_time']:.4f} 秒 ({stats['total_param_update_time']*1000:.2f} ms)")
        param_min, param_max, param_std, param_mean = safe_stats(stats['param_update_times'])
        print(f"  - 最小时间: {param_min}")
        print(f"  - 最大时间: {param_max}")
        print(f"  - 标准差: {param_std}")
        
        print(f"\n数据传输时间 (CPU到GPU):")
        print(f"  - 总时间: {stats['total_data_transfer_time']:.4f} 秒 ({stats['total_data_transfer_time']*1000:.2f} ms)")
        data_transfer_min, data_transfer_max, data_transfer_std, data_transfer_mean = safe_stats(stats['data_transfer_times'])
        print(f"  - 最小时间: {data_transfer_min}")
        print(f"  - 最大时间: {data_transfer_max}")
        print(f"  - 标准差: {data_transfer_std}")
        
        print(f"\nLoRA参数上传时间 (网速: {self.upload_speed} Mbps):")
        print(f"  - 总时间: {stats['total_upload_time']:.4f} 秒 ({stats['total_upload_time']*1000:.2f} ms)")
        upload_min, upload_max, upload_std, upload_mean = safe_stats(stats['upload_times'])
        print(f"  - 最小时间: {upload_min}")
        print(f"  - 最大时间: {upload_max}")
        print(f"  - 标准差: {upload_std}")
        
        print(f"\nLoRA参数下载时间 (网速: {self.download_speed} Mbps):")
        print(f"  - 总时间: {stats['total_download_time']:.4f} 秒 ({stats['total_download_time']*1000:.2f} ms)")
        download_min, download_max, download_std, download_mean = safe_stats(stats['download_times'])
        print(f"  - 最小时间: {download_min}")
        print(f"  - 最大时间: {download_max}")
        print(f"  - 标准差: {download_std}")
        
        print(f"\n时间比例:")
        total_time = (stats['total_data_transfer_time'] + stats['total_forward_time'] + 
                     stats['total_backward_time'] + stats['total_param_update_time'] + 
                     stats['total_upload_time'] + stats['total_download_time'])
        if total_time == 0:
            data_transfer_ratio = forward_ratio = backward_ratio = param_update_ratio = upload_ratio = download_ratio = 0.0
        else:
            data_transfer_ratio = stats['total_data_transfer_time'] / total_time * 100
            forward_ratio = stats['total_forward_time'] / total_time * 100
            backward_ratio = stats['total_backward_time'] / total_time * 100
            param_update_ratio = stats['total_param_update_time'] / total_time * 100
            upload_ratio = stats['total_upload_time'] / total_time * 100
            download_ratio = stats['total_download_time'] / total_time * 100
        print(f"  - 数据传输: {data_transfer_ratio:.2f}%")
        print(f"  - 前向传播: {forward_ratio:.2f}%")
        print(f"  - 反向更新: {backward_ratio:.2f}%")
        print(f"  - 参数更新: {param_update_ratio:.2f}%")
        print(f"  - 参数上传: {upload_ratio:.2f}%")
        print(f"  - 参数下载: {download_ratio:.2f}%")
        
        print(f"\n平均每个样本的计算时间:")
        avg_data_transfer_per_batch = stats['avg_data_transfer_time'] * 1000
        avg_forward_per_batch = stats['avg_forward_time'] * 1000
        avg_backward_per_batch = stats['avg_backward_time'] * 1000
        avg_param_update_per_batch = stats['avg_param_update_time'] * 1000
        avg_upload_per_batch = stats['avg_upload_time'] * 1000
        avg_download_per_batch = stats['avg_download_time'] * 1000
        print(f"  - 数据传输: {avg_data_transfer_per_batch:.2f} ms/batch")
        print(f"  - 前向传播: {avg_forward_per_batch:.2f} ms/batch")
        print(f"  - 反向更新: {avg_backward_per_batch:.2f} ms/batch")
        print(f"  - 参数更新: {avg_param_update_per_batch:.2f} ms/batch")
        print(f"  - 参数上传: {avg_upload_per_batch:.2f} ms/batch")
        print(f"  - 参数下载: {avg_download_per_batch:.2f} ms/batch")
        
        print("\n" + "=" * 80)
    
    def save_results(self, output_file='timing_results.txt'):
        """保存结果到文件"""
        stats = self.timing_stats

        def safe_stats(values):
            if not values:
                return 'N/A', 'N/A', 'N/A', 'N/A'
            return (
                f"{min(values)*1000:.2f} ms",
                f"{max(values)*1000:.2f} ms",
                f"{np.std(values)*1000:.2f} ms",
                f"{np.mean(values)*1000:.2f} ms",
            )
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("计算时间统计结果\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"模型: {self.args.model}\n")
            f.write(f"数据集: {self.args.dataset}\n")
            f.write(f"批大小: {self.args.batch_size}\n")
            f.write(f"Epoch数: {self.args.epochs}\n")
            f.write(f"总批次数: {stats['num_batches']}\n\n")
            
            f.write("LLM冻结部分前向传播时间:\n")
            f.write(f"  - 总时间: {stats['total_forward_time']:.4f} 秒 ({stats['total_forward_time']*1000:.2f} ms)\n")
            forward_min, forward_max, forward_std, forward_mean = safe_stats(stats['forward_times'])
            f.write(f"  - 平均时间/批次: {stats['avg_forward_time']*1000:.2f} ms\n")
            f.write(f"  - 最小时间: {forward_min}\n")
            f.write(f"  - 最大时间: {forward_max}\n")
            f.write(f"  - 标准差: {forward_std}\n\n")
            
            f.write("LoRA矩阵反向更新时间:\n")
            f.write(f"  - 总时间: {stats['total_backward_time']:.4f} 秒 ({stats['total_backward_time']*1000:.2f} ms)\n")
            backward_min, backward_max, backward_std, backward_mean = safe_stats(stats['backward_times'])
            f.write(f"  - 平均时间/批次: {stats['avg_backward_time']*1000:.2f} ms\n")
            f.write(f"  - 最小时间: {backward_min}\n")
            f.write(f"  - 最大时间: {backward_max}\n")
            f.write(f"  - 标准差: {backward_std}\n\n")
            
            f.write("参数更新时间:\n")
            f.write(f"  - 总时间: {stats['total_param_update_time']:.4f} 秒 ({stats['total_param_update_time']*1000:.2f} ms)\n")
            param_min, param_max, param_std, param_mean = safe_stats(stats['param_update_times'])
            f.write(f"  - 平均时间/批次: {stats['avg_param_update_time']*1000:.2f} ms\n")
            f.write(f"  - 最小时间: {param_min}\n")
            f.write(f"  - 最大时间: {param_max}\n")
            f.write(f"  - 标准差: {param_std}\n\n")
            
            f.write("数据传输时间 (CPU到GPU):\n")
            f.write(f"  - 总时间: {stats['total_data_transfer_time']:.4f} 秒 ({stats['total_data_transfer_time']*1000:.2f} ms)\n")
            data_transfer_min, data_transfer_max, data_transfer_std, data_transfer_mean = safe_stats(stats['data_transfer_times'])
            f.write(f"  - 平均时间/批次: {stats['avg_data_transfer_time']*1000:.2f} ms\n")
            f.write(f"  - 最小时间: {data_transfer_min}\n")
            f.write(f"  - 最大时间: {data_transfer_max}\n")
            f.write(f"  - 标准差: {data_transfer_std}\n\n")
            
            f.write(f"LoRA参数上传时间 (网速: {self.upload_speed} Mbps):\n")
            f.write(f"  - 总时间: {stats['total_upload_time']:.4f} 秒 ({stats['total_upload_time']*1000:.2f} ms)\n")
            upload_min, upload_max, upload_std, upload_mean = safe_stats(stats['upload_times'])
            f.write(f"  - 平均时间/批次: {stats['avg_upload_time']*1000:.2f} ms\n")
            f.write(f"  - 最小时间: {upload_min}\n")
            f.write(f"  - 最大时间: {upload_max}\n")
            f.write(f"  - 标准差: {upload_std}\n\n")
            
            f.write(f"LoRA参数下载时间 (网速: {self.download_speed} Mbps):\n")
            f.write(f"  - 总时间: {stats['total_download_time']:.4f} 秒 ({stats['total_download_time']*1000:.2f} ms)\n")
            download_min, download_max, download_std, download_mean = safe_stats(stats['download_times'])
            f.write(f"  - 平均时间/批次: {stats['avg_download_time']*1000:.2f} ms\n")
            f.write(f"  - 最小时间: {download_min}\n")
            f.write(f"  - 最大时间: {download_max}\n")
            f.write(f"  - 标准差: {download_std}\n\n")
            
            total_time = (stats['total_forward_time'] + stats['total_backward_time'] + 
                         stats['total_upload_time'] + stats['total_download_time'])
            if total_time == 0:
                forward_ratio = backward_ratio = upload_ratio = download_ratio = 0.0
            else:
                forward_ratio = stats['total_forward_time'] / total_time * 100
                backward_ratio = stats['total_backward_time'] / total_time * 100
                upload_ratio = stats['total_upload_time'] / total_time * 100
                download_ratio = stats['total_download_time'] / total_time * 100
            f.write("时间比例:\n")
            f.write(f"  - 前向传播: {forward_ratio:.2f}%\n")
            f.write(f"  - 反向更新: {backward_ratio:.2f}%\n")
            f.write(f"  - 参数上传: {upload_ratio:.2f}%\n")
            f.write(f"  - 参数下载: {download_ratio:.2f}%\n\n")
            
            avg_data_transfer_per_batch = stats['avg_data_transfer_time'] * 1000
            avg_forward_per_batch = stats['avg_forward_time'] * 1000
            avg_backward_per_batch = stats['avg_backward_time'] * 1000
            avg_param_update_per_batch = stats['avg_param_update_time'] * 1000
            avg_upload_per_batch = stats['avg_upload_time'] * 1000
            avg_download_per_batch = stats['avg_download_time'] * 1000
            f.write("平均batch计算时间:\n")
            f.write(f"  - 数据传输: {avg_data_transfer_per_batch:.2f} ms/batch\n")
            f.write(f"  - 前向传播: {avg_forward_per_batch:.2f} ms/batch\n")
            f.write(f"  - 反向更新: {avg_backward_per_batch:.2f} ms/batch\n")
            f.write(f"  - 参数更新: {avg_param_update_per_batch:.2f} ms/batch\n")
            f.write(f"  - 参数上传: {avg_upload_per_batch:.2f} ms/batch\n")
            f.write(f"  - 参数下载: {avg_download_per_batch:.2f} ms/batch\n")
        
        print(f"\n结果已保存到: {output_file}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='LLM计算时间分析工具')
    
    # 模型参数
    parser.add_argument('--model', type=str, default='distilbert-base-multilingual-cased',
                       choices=['distilbert-base-multilingual-cased', 'roberta-base', 'roberta-large', 'llama-2-7B'],
                       help='模型名称')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA alpha参数')
    parser.add_argument('--lora_rank', type=int, default=8, help='LoRA rank')
    parser.add_argument('--method', type=str, default='raw', help='方法名称')
    
    # 数据集参数
    parser.add_argument('--dataset', type=str, default='sst2', help='数据集名称')
    parser.add_argument('--subdataset', type=str, default='sst2', help='子数据集名称')
    parser.add_argument('--max_length', type=int, default=128, help='最大序列长度')
    parser.add_argument('--batch_size', type=int, default=32, help='批大小')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=2, help='训练epoch数')
    parser.add_argument('--lr', type=float, default=5e-5, help='学习率')
    parser.add_argument('--random_seed', type=int, default=42, help='随机种子')
    parser.add_argument('--upload_speed', type=float, default=5.0, help='上传网速 (Mbps)')
    parser.add_argument('--download_speed', type=float, default=50.0, help='下载网速 (Mbps)')
    
    # 其他参数
    parser.add_argument('--num_client', type=int, default=1, help='客户端数量（仅用于兼容）')
    parser.add_argument('--pmin', type=float, default=0.1, help='最小比例（仅用于兼容）')
    parser.add_argument('--pmax', type=float, default=0.9, help='最大比例（仅用于兼容）')
    parser.add_argument('--sample_fraction', type=float, default=1.0, help='采样比例（仅用于兼容）')
    parser.add_argument('--comm_round', type=int, default=1, help='通信轮数（仅用于兼容）')
    parser.add_argument('--weight', type=float, default=0.0, help='权重衰减（仅用于兼容）')
    parser.add_argument('--proportion', type=float, default=0.1, help='压缩比例（仅用于兼容）')
    parser.add_argument('--point', type=int, default=10, help='切换点（仅用于兼容）')
    parser.add_argument('--residual', action='store_true', help='是否使用残差（仅用于兼容）')
    parser.add_argument('--early_stop', action='store_true', help='是否提前停止（仅用于兼容）')
    parser.add_argument('--partition', action='store_true', help='iid:True, non-iid:False')
    parser.add_argument('--label', action='store_true', help='using label to partition')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Alpha for the dirichlet distribution for data partitioning')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    print("\n" + "=" * 80)
    print("LLM计算时间分析工具")
    print("=" * 80)
    print(f"\n配置信息:")
    print(f"  - 模型: {args.model}")
    print(f"  - 数据集: {args.dataset}")
    print(f"  - 批大小: {args.batch_size}")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - 学习率: {args.lr}")
    print(f"  - LoRA Alpha: {args.lora_alpha}")
    print(f"  - LoRA Rank: {args.lora_rank}")
    print(f"  - 上传网速: {args.upload_speed} Mbps")
    print(f"  - 下载网速: {args.download_speed} Mbps")
    print(f"  - 设备: GPU 3 (cuda:0)")
    
    # 设置随机种子
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    # 创建分析器
    analyzer = TimingAnalyzer(args)
    
    # 加载模型和数据
    data, partition_map = analyzer.load_model_and_data()
    
    # 获取第一个客户端的数据索引（用于演示）
    data_indices = partition_map[0] if 0 in partition_map else list(range(len(data["train"])))
    print(f"\n使用 {len(data_indices)} 个样本进行计时分析")
    
    # 确定设备
    device = 'cuda:0'  # 使用GPU 3（CUDA_VISIBLE_DEVICES=3后变为cuda:0）
    print(f"使用设备: {device}")
    
    # 执行计时分析
    analyzer.analyze_timing(data, data_indices, device=device)
    
    # 打印结果
    analyzer.print_results()
    
    # 保存结果
    if args.dataset == 'glue':
        output_file = f"timing_results_{args.model}_{args.dataset}_{args.subdataset}.txt"
    else:
        output_file = f"timing_results_{args.model}_{args.dataset}.txt"
    analyzer.save_results(output_file)


if __name__ == '__main__':
    main()
