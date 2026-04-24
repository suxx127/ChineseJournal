import os
import sys
import copy
import torch
import numpy as np
import math
import time
from model_utils import get_model_path, get_automodel, get_untrain_part, set_requires_grad, safe_replace_lora_layers
from torch import nn
from torch.nn.utils import prune
import random

from task import Task
from fedcomp import Fedcomp
from accelerate import Accelerator
from torch.utils.data import Subset
from safetensors.torch import save_file
from cvlc import Fed_cvlc, get_coefficient_of_variation
from model_utils import get_model_tokenizer, get_model_lora
from dataset_utils import get_fed_data_info, Metrics_trainer
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding, Seq2SeqTrainer, \
    Seq2SeqTrainingArguments, DataCollatorForTokenClassification, DataCollatorForLanguageModeling
from peft import LoraConfig, TaskType, get_peft_model
from utils import TrainAutoEncoder, prune_lora_layers


class Fed_trainer(object):
    def __init__(self, args, random_list):
        super().__init__()
        self.accelerator = Accelerator()
        self.data_collator = None
        self.metric_trainer = None
        self.gobal_model = None
        self.gobal_grad = None
        self.args = args
        self.training_metric = {}
        self.random_list = random_list
        self.accu_gra = None
        self.communication = 0
        
        # 时间统计相关
        self.timing_stats = {}  # 从文件中读取的timing统计
        self.client_timings = {}  # 记录每个客户端的耗时 {client_id: {'compute': xx, 'communication': xx}}
        self.round_timings = []  # 记录每轮的总耗时
        self.client_allocated_timings = {}  # 预分配的客户端信息 {client_id: {'forward_time': xx, 'backward_time': xx, 'upload_speed': xx, 'download_speed': xx}}
        self.total_training_time = 0.0  # 到当前回合为止的总训练时间
        
        # HeLoRA 相关
        self.client_ranks = {}  # 记录每个客户端的秩大小 {client_id: rank}

    def get_grad(self, model):
        grad_parts = []
        param_parts = []
        name_grad = {}
        name_param = {}
        name_paramlast = {}
        global_state = self.gobal_model.state_dict()
        model_state = model.state_dict()

        for layer, global_param in global_state.items():
            if layer.find("num_batches_tracked") != -1:
                continue
            # if 'lora' in layer or any(part in layer for part in self.untrain_part):
            if 'lora' in layer:
                param_now = model_state[layer].detach().cpu()
                param_last = global_param.detach().cpu()
                param_g = param_last - param_now
                param_parts.append(param_now.reshape(-1))
                grad_parts.append(param_g.reshape(-1))
                name_grad[layer] = param_g
                name_param[layer] = param_now
                name_paramlast[layer] = param_last

        if grad_parts:
            grad = torch.cat(grad_parts)
            param = torch.cat(param_parts)
        else:
            grad = torch.empty(0)
            param = torch.empty(0)

        print("the proportion of zeros in gradients is ", torch.sum(grad < 1e-5) / grad.numel())
        return grad, param, name_grad, name_param, name_paramlast

    def paramAggre(self, grad, gobal_model=None):
        if gobal_model is None:
            gobal_model = self.gobal_model
        grad = grad.cuda()
        current_index = 0
        model = copy.deepcopy(gobal_model)
        current_state_dict = model.state_dict()
        for name, param in current_state_dict.items():
            if 'lora' in name:
                param = param.cuda()
                numel = param.data.numel()
                size = param.data.size()
                current_state_dict[name] = grad[current_index:current_index + numel].view(size)
                current_index += numel
        model.load_state_dict(current_state_dict)
        return model
    
    def substitute(self, grad, gobal_model=None):
        if gobal_model is None:
            gobal_model = self.gobal_model
        grad = grad.cuda()
        current_index = 0
        index = 0
        model = copy.deepcopy(gobal_model)
        current_state_dict = model.state_dict()
        for name, param in current_state_dict.items():
            if 'lora' in name:
                param = param.cuda()
                if index % 2 == 0:
                    _, mA = param.shape
                    nameA = name
                else:
                    mB, r = param.shape
                    current_grad = grad[current_index: current_index + mA * mB].reshape(mB, mA)
                    current_index += mA * mB
                    print("the shape of matrix is ", current_grad.shape)
                    print("the size and non-zero element of matrix are ", mA * mB, torch.sum(current_grad != 0))
                    
                    U, S, Vh = torch.svd_lowrank(current_grad.to(torch.float32), q=r)
                    print("the shape of U S V are ", U.shape, S.shape, Vh.shape)
                    S = torch.diag(S)
                    B = torch.matmul(U, S)
                    A = Vh.T
                    # B = torch.matmul(U, torch.sqrt(S))
                    # A = torch.matmul(torch.sqrt(S), Vh.T)
                    # U, S, Vh = torch.linalg.svd(current_grad.reshape(mB, mA))
                    # U_r = U[:, :r]
                    # S_r = torch.diag(S[:r])
                    # Vh_r = Vh[:r, :]
                    # B = torch.matmul(U_r, torch.sqrt(S_r))
                    # A = torch.matmul(torch.sqrt(S_r), Vh_r)

                    current_state_dict[nameA] = A
                    current_state_dict[name] = B
                index += 1
                # numel = param.data.numel()
                # size = param.data.size()
                # current_state_dict[name] = \
                #     torch.subtract(param.data.detach(), grad[current_index:current_index + numel].view(size))
                # current_index += numel
        model.load_state_dict(current_state_dict)
        return model
    
    def updateW(self, grad, gobal_model=None):
        if gobal_model is None:
            gobal_model = self.gobal_model
        current_index = 0
        model = copy.deepcopy(gobal_model)
        current_state_dict = model.state_dict()
        for name, param in current_state_dict.items():
            if any(target in name for target in self.target_modules) and 'weight' in name and 'lora' not in name:
                m, n = current_state_dict[name].shape
                print("the shape of {} is {}".format(name, (m, n)))
                lora_res = grad[current_index: current_index + m * n].detach()
                current_state_dict[name] += self.args.factor * self.args.lora_alpha / math.sqrt(self.args.lora_rank) * lora_res.reshape(m, n)
                current_index += m * n
            # elif any(part in name for part in self.untrain_part):
            #     numel = current_state_dict[name].numel()
            #     current_state_dict[name] -= grad[current_index: current_index + numel].reshape(current_state_dict[name].shape)
            #     current_index += numel
        # for name, param in current_state_dict.items():
        #     if 'lora_A' in name:
        #         # torch.nn.init.kaiming_normal_(current_state_dict[name])
        #         torch.nn.init.kaiming_uniform_(current_state_dict[name], a=math.sqrt(5 / 5)) # distilbert
        #         # torch.nn.init.normal_(current_state_dict[name], mean=0, std=1. / self.args.lora_rank)
        #     if 'lora_B' in name:
        #         torch.nn.init.zeros_(current_state_dict[name])
        model.load_state_dict(current_state_dict)
        return model
    
    def combine(self, grad, gobal_model=None):
        if gobal_model is None:
            gobal_model = self.gobal_model
        current_index = 0
        with torch.no_grad():
            for name, param in gobal_model.named_parameters():
                if 'lora' not in name:
                    continue
                numel = param.numel()
                grad_slice = grad[current_index:current_index + numel].view_as(param)
                if grad_slice.device != param.device:
                    grad_slice = grad_slice.to(param.device, non_blocking=True)
                param.sub_(grad_slice)
                current_index += numel
        return gobal_model

    def aggregate(self, grad_dist: dict, cohorts: list, partition_map: dict):
        model_gra = torch.zeros_like(grad_dist[cohorts[0]])
        weights = {}
        data_sum = 0

        for client in cohorts:
            data_size = len(partition_map[client])
            if self.args.method == 'HeLoRA' and client in self.client_ranks:
                weights[client] = data_size * self.client_ranks[client]
            else:
                weights[client] = data_size
            data_sum += weights[client]

        for client in cohorts:
            w = weights[client] / data_sum
            model_gra += (w * grad_dist[client])
        
        return model_gra

    def set_data_collator(self, tokenizer, task):
        if task == Task.SequenceClassification:
            self.data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        elif task == Task.TokenClassification:
            self.data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
        elif task == Task.QuestionAnswering:
            self.data_collator = None
        else:
            self.data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    def run(self):
        print("configaration is ", self.args)
        
        # 固定随机种子，保证可重复性和不同算法的公平比较
        random_seed = 42
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
        
        # 可训练的低秩矩阵比例
        proportions = [random.uniform(self.args.pmin, self.args.pmax) for _ in range(self.args.num_client)]
        
        tokenizer = get_model_tokenizer(model=self.args.model, max_length=self.args.max_length)
        data, partition_map, num_labels, metric, validation_key, \
            task, label_names, validation_dataset, grids = get_fed_data_info(args=self.args, tokenizer=tokenizer)
        self.metric_trainer = Metrics_trainer(metric_name=metric, label_names=label_names, grids=grids)
        self.gobal_model, trainable_parameters, self.untrain_part = get_model_lora(model=self.args.model, lora_alpha=self.args.lora_alpha,
                                                                lora_rank=self.args.lora_rank, num_labels=num_labels,
                                                                task=task, method=self.args.method)
        print("the model architecture is ")
        print(self.gobal_model)

        num_param = 0
        print(self.untrain_part)
        print("the name of update layers are")
        for layer in self.gobal_model.state_dict():
            if layer.find("num_batches_tracked") != -1:
                continue
            # if 'lora' in layer or any(part in layer for part in self.untrain_part):
            if 'lora' in layer:
                print(layer)
                num_param += self.gobal_model.state_dict()[layer].numel() 
        # 计算LoRA参数大小（字节）
        lora_params_size = num_param * 4  # 假设float32，4字节/参数
        print(f"LoRA参数大小: {lora_params_size / (1024**2):.2f} MB")

        # 加载时间统计信息
        self.load_timing_stats()
        
        # 为所有客户端预分配计算和通信时间（网速仅在此处生成一次）
        self.allocate_client_timings(partition_map, lora_params_size)
        
        # 如果是 HeLoRA 方法，分配客户端秩大小（使用已分配好的网速）
        if self.args.method == 'HeLoRA':
            self.allocate_client_ranks_helora(partition_map, lora_params_size)
                
        self.accu_gra = torch.zeros((self.args.num_client, num_param))
        self.set_data_collator(tokenizer=tokenizer, task=task)
        matrix = self.get_model_matrix_num()
        
        # coefficient_of_variation = []
        self.init_training_metric(metrics=metric)
        compress = Fed_cvlc(args=self.args, trainable_parameters=trainable_parameters, matrix_num=matrix)
        # Train
        for rnd in range(self.args.comm_round):
            if self.accelerator.is_local_main_process:
                print(f'ROUND:{rnd}')
            # 初始化该轮的客户端耗时记录
            round_client_timings = {}
            np.random.seed(self.random_list[rnd])
            cohorts = np.random.choice(self.args.num_client, int(self.args.num_client * self.args.sample_fraction),
                                    replace=False).tolist()
            
            # 如果是 FFTHM 方法，计算每个客户端的训练比例p以平衡时间
            # 计算每个客户端的训练比例p
            client_proportions = {}
            if self.args.method == 'FFTHM':
                # 计算每个选中客户端的原始总时间（p=1时）
                client_times = {}
                for client in cohorts:
                    allocated = self.client_allocated_timings[client]
                    forward = allocated['forward_time']
                    backward = allocated['backward_time']
                    upload_speed = allocated['upload_speed']
                    download_speed = allocated['download_speed']
                    # 估算传输字节数（简化使用lora_params_size）
                    transmitted_bytes = lora_params_size
                    upload_speed_bytes = upload_speed * 1024 * 1024 / 8
                    download_speed_bytes = download_speed * 1024 * 1024 / 8
                    upload_time = transmitted_bytes / upload_speed_bytes
                    download_time = lora_params_size / download_speed_bytes
                    total_time = forward + backward + upload_time + download_time
                    client_times[client] = total_time
                
                # 找出资源最丰富的客户端（总时间最短）
                richest_client = min(client_times, key=client_times.get)
                T_max = client_times[richest_client]
                
                for client in cohorts:
                    if client == richest_client:
                        client_proportions[client] = 1.0
                    else:
                        forward = self.client_allocated_timings[client]['forward_time']
                        backward = self.client_allocated_timings[client]['backward_time']
                        upload_time = transmitted_bytes / (self.client_allocated_timings[client]['upload_speed'] * 1024 * 1024 / 8)
                        download_time = lora_params_size / (self.client_allocated_timings[client]['download_speed'] * 1024 * 1024 / 8)
                        denominator = backward + upload_time
                        if denominator > 0:
                            p = (T_max - forward - download_time) / denominator
                            p = max(0.1, min(1.0, p))  # 限制在0.1到1.0之间
                        else:
                            p = 1.0
                        client_proportions[client] = p
            else:
                for client in cohorts:
                    client_proportions[client] = 1.0
            
            grad_dist = {}
            for i, client in enumerate(cohorts):
                self.accelerator.print(f'CLIENT:{client}')
                local_model = copy.deepcopy(self.gobal_model)
                local_model = self.train(data=data, data_indices=partition_map[client],
                                        model=local_model,
                                        tokenizer=tokenizer, task=task, client_idx=i,
                                        update_proportion=client_proportions[client], client_id=client)
                # if rnd == 0 and client == cohorts[0]:
                #     self.copy_model_expect_lora(local_model)
                
                allocated_timing = self.client_allocated_timings[client]
                forward_time = allocated_timing['forward_time']
                backward_time = allocated_timing['backward_time']
                upload_speed = allocated_timing['upload_speed']
                download_speed = allocated_timing['download_speed']

                grad, param, name_grad, name_param, name_paramlast = self.get_grad(local_model)
                if self.args.method == 'pq':
                    grad_dist[client] = compress.do_compress(grad, param, name_grad, name_paramlast, proportion=self.args.proportion, rnd=rnd)
                else:
                    grad += self.accu_gra[client]
                    grad_dist[client] = compress.do_compress(grad, param, name_grad, name_paramlast, proportion=self.args.proportion, rnd=rnd)
                    self.accu_gra[client] = grad - grad_dist[client]
                
                # 估算本轮传输的数据量并计算通信时间
                transmitted_bytes = self.estimate_transmitted_bytes(grad_dist[client], lora_params_size)
                download_bytes = lora_params_size
                upload_speed_bytes_per_sec = upload_speed * 1024 * 1024 / 8
                download_speed_bytes_per_sec = download_speed * 1024 * 1024 / 8
                upload_time = transmitted_bytes / upload_speed_bytes_per_sec
                download_time = download_bytes / download_speed_bytes_per_sec
                
                # 如果是 FFTHM 方法，按训练比例p缩放反向传播时间和上传时间
                if self.args.method == 'FFTHM':
                    p = client_proportions[client]
                    backward_time_scaled = backward_time * p * math.sqrt(p)
                    upload_time_scaled = upload_time * p
                    download_time_scaled = download_time  # 下载时间不变
                elif self.args.method == 'HeLoRA':
                    # HeLoRA 方法根据客户端秩大小缩放时间
                    rank_ratio = self.client_ranks.get(client, self.args.lora_rank) / self.args.lora_rank
                    backward_time_scaled = backward_time * rank_ratio
                    upload_time_scaled = upload_time * rank_ratio
                    download_time_scaled = download_time * rank_ratio
                else:
                    backward_time_scaled = backward_time
                    upload_time_scaled = upload_time
                    download_time_scaled = download_time
                
                communication_time = upload_time_scaled + download_time_scaled
                total_client_time = forward_time + backward_time_scaled + communication_time
                
                round_client_timings[client] = {
                    'forward_time': forward_time,
                    'backward_time': backward_time_scaled,
                    'communication_time': communication_time,
                    'total_time': total_client_time,
                    'upload_speed': upload_speed,
                    'download_speed': download_speed,
                    'upload_bytes': transmitted_bytes,
                    'download_bytes': download_bytes
                }
                
                print(f'  Client {client} 前向耗时: {forward_time:.4f}s, '
                      f'反向耗时: {backward_time:.4f}s, '
                      f'通信耗时: {communication_time:.4f}s, '
                      f'(上传速率: {upload_speed:.2f}Mbps, 下载速率: {download_speed:.2f}Mbps)')
            grad = self.aggregate(grad_dist=grad_dist, cohorts=cohorts, partition_map=partition_map)
            
            # 计算该轮的最大耗时（联邦学习中的瓶颈）
            if round_client_timings:
                max_forward_time = max([t['forward_time'] for t in round_client_timings.values()])
                max_backward_time = max([t['backward_time'] for t in round_client_timings.values()])
                max_communication_time = max([t['communication_time'] for t in round_client_timings.values()])
                max_total_time = max([t['total_time'] for t in round_client_timings.values()])
                max_client = max(round_client_timings.items(), key=lambda x: x[1]['total_time'])[0]
                
                self.round_timings.append({
                    'round': rnd,
                    'max_forward_time': max_forward_time,
                    'max_backward_time': max_backward_time,
                    'max_communication_time': max_communication_time,
                    'max_total_time': max_total_time,
                    'bottleneck_client': max_client
                })
                
                # 累加到总训练时间
                self.total_training_time += max_total_time
                
                print(f"\n[轮次 {rnd} 时间统计]")
                print(f"  最大前向耗时: {max_forward_time:.4f}s")
                print(f"  最大反向耗时: {max_backward_time:.4f}s")
                print(f"  最大通信耗时: {max_communication_time:.4f}s")
                print(f"  本轮总耗时: {max_total_time:.4f}s (瓶颈客户端: {max_client})")
                print(f"  累计训练时间: {self.total_training_time:.4f}s")
                print()
            
            self.gobal_grad = grad
            self.gobal_model = self.combine(grad)
            result, predictions = self.test_metric(data=data, tokenizer=tokenizer,
                                                validation_key=validation_key, task=task)
            
            
            if self.args.method == 'raw':
                name_model = {}
                global_state = self.gobal_model.state_dict()
                for layer in global_state:
                    if 'lora' in layer:
                        name_model[layer] = global_state[layer].detach().cpu()
                file_path = 'para/' + self.args.model + '_' + str(rnd) + '.pt'
                torch.save(name_model, file_path)
        
            if predictions is not None:
                result = self.metric_trainer.compute_metrics_predictions(predictions=predictions, data=data,
                                                                        result=result,
                                                                        validation_dataset=validation_dataset)
            self.record_metric(metrics=metric, result=result)

            if self.args.early_stop:
                if "accuracy" in metric and result['eval_acc'] >= 0.6:
                    print(f"Stopping training as accuracy has reached {result['eval_acc']} after {rnd + 1} rounds.")
                    break
        
        # # 保存和打印时间统计结果
        # self.save_round_timings()
        
        return

    def train(self, data, data_indices, model, tokenizer, task, client_idx, update_proportion, client_id=None):
        # 用于对低秩矩阵进行冻结
        if self.args.method == 'FFTHM':
            model = safe_replace_lora_layers(model, update_proportion)
        
        # 如果是 HeLoRA 方法，执行秩截断
        if self.args.method == 'HeLoRA' and client_id is not None:
            target_rank = self.client_ranks.get(client_id, self.args.lora_rank)
            if target_rank < self.args.lora_rank:
                model = self.truncate_lora_rank(model, target_rank, client_id)
        
        model.train()
        train_data = Subset(data["train"], data_indices)
        save_steps = sys.maxsize
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        if task in [Task.SequenceClassification, Task.TokenClassification, Task.QuestionAnswering, Task.CausalLM]:
            training_args = TrainingArguments(output_dir='./save/model', save_steps=save_steps,
                                            #   save_strategy='epoch',
                                              num_train_epochs=self.args.epochs,
                                              per_device_train_batch_size=self.args.batch_size, do_train=True,
                                              learning_rate=self.args.lr,
                                              ddp_find_unused_parameters=False,
                                              lr_scheduler_type="constant",
                                              logging_steps=10)
            trainer = Trainer(
                model=model,
                tokenizer=tokenizer,
                args=training_args,
                train_dataset=train_data,
                data_collator=self.data_collator,
                optimizers=(optimizer, None)
            )
        else:
            training_args = Seq2SeqTrainingArguments(output_dir='save', save_steps=sys.maxsize,
                                                     num_train_epochs=self.args.epochs,
                                                     per_device_train_batch_size=self.args.batch_size, do_train=True,
                                                     learning_rate=self.args.lr,
                                                     lr_scheduler_type="constant",
                                                     ddp_find_unused_parameters=False,
                                                     logging_steps=10)
            trainer = Seq2SeqTrainer(
                model=model,
                tokenizer=tokenizer,
                args=training_args,
                train_dataset=train_data,
                data_collator=self.data_collator,
                optimizers=(optimizer, None)
            )
        trainer.train()
        
        # 如果是 HeLoRA 方法，恢复被截断的 LoRA 矩阵
        if self.args.method == 'HeLoRA' and client_id is not None:
            model = self.fill_truncated_lora(model, client_id)
        
        return model

    def test_metric(self, data, tokenizer, validation_key, task):
        self.gobal_model.eval()
        predictions = None
        eval_data = data[validation_key]
        if task in [Task.SequenceClassification, Task.TokenClassification]:
            training_args = TrainingArguments(output_dir='save', save_steps=sys.maxsize,
                                              per_device_eval_batch_size=self.args.batch_size,
                                              do_eval=True, seed=self.args.random_seed)
            trainer = Trainer(
                model=self.gobal_model,
                args=training_args,
                tokenizer=tokenizer,
                compute_metrics=self.metric_trainer.compute_metrics,
                eval_dataset=eval_data
            )
            results = trainer.evaluate(metric_key_prefix="eval")
        elif task in [Task.QuestionAnswering]:
            training_args = TrainingArguments(output_dir='save', save_steps=sys.maxsize,
                                              per_device_eval_batch_size=self.args.batch_size,
                                              do_predict=True, seed=self.args.random_seed)
            trainer = Trainer(
                model=self.gobal_model,
                args=training_args,
                tokenizer=tokenizer,
                compute_metrics=self.metric_trainer.compute_metrics,
                eval_dataset=eval_data
            )
            predictions, _, results = trainer.predict(eval_data)
        else:
            training_args = Seq2SeqTrainingArguments(output_dir='save', save_steps=sys.maxsize,
                                                     per_device_eval_batch_size=self.args.batch_size,
                                                     save_total_limit=1,
                                                     do_predict=True, seed=self.args.random_seed)
            trainer = Seq2SeqTrainer(
                model=self.gobal_model,
                args=training_args,
                tokenizer=tokenizer,
                compute_metrics=self.metric_trainer.compute_metrics,
                eval_dataset=eval_data
            )
            results = trainer.evaluate(max_length=self.args.max_length + 3)
        if self.accelerator.is_local_main_process:
            print(results)
        return results, predictions

    def save_model(self):
        file_name = '{}_{}_{}_{}_{}.safetensors'.format(self.args.model, self.args.alpha, self.args.num_client,
                                                        self.args.epochs, self.args.comm_round)
        save_file(self.gobal_model.state_dict(), os.path.join('./save', file_name))

    def save_metric(self, metrics: list):
        for metric in metrics:
            if metric == "matthews_correlation":
                metric_list = self.training_metric['matthews_correlation']
                name = "matthews_correlation_{}_{}_{}_{}_{}_{}_{}.txt".format(self.args.method,
                                                                              self.args.blocks,
                                                                              self.args.alpha, self.args.comm_round,
                                                                              self.args.bit,
                                                                              self.args.dataset, self.args.subdataset)
                save_file_metric(name, metric_list)
            elif metric == "pearson":
                metric_list = self.training_metric['pearson']
                name = "pearson_{}_{}_{}_{}_{}_{}_{}_{}.txt".format(self.args.method, self.args.blocks,
                                                                    self.args.alpha, self.args.comm_round,
                                                                    self.args.bit,
                                                                    self.args.dataset, self.args.subdataset,
                                                                    self.args.model)
                save_file_metric(name, metric_list)
            elif metric == "spearman":
                metric_list = self.training_metric['spearman']
                name = "spearman_{}_{}_{}_{}_{}_{}_{}_{}.txt".format(self.args.method, self.args.blocks,
                                                                     self.args.alpha, self.args.comm_round,
                                                                     self.args.bit,
                                                                     self.args.dataset, self.args.subdataset,
                                                                     self.args.model)
                save_file_metric(name, metric_list)
            elif metric == "accuracy":
                metric_list = self.training_metric['accuracy']
                name = "accuracy_{}_{}_{}_{}_{}_{}_{}_{}.txt".format(self.args.method, self.args.blocks,
                                                                     self.args.alpha, self.args.comm_round,
                                                                     self.args.bit,
                                                                     self.args.dataset, self.args.subdataset,
                                                                     self.args.model)
                save_file_metric(name, metric_list)
            elif metric == "f1" or metric == "f1_m":
                metric_list = self.training_metric['f1']
                name = "f1_{}_{}_{}_{}_{}_{}_{}_{}.txt".format(self.args.method, self.args.blocks,
                                                               self.args.alpha, self.args.comm_round, self.args.bit,
                                                               self.args.dataset, self.args.subdataset,
                                                               self.args.model)
                save_file_metric(name, metric_list)
            elif metric == "precision":
                metric_list = self.training_metric['precision']
                name = "precision.txt"
                save_file_metric(name, metric_list)
            elif metric == "exact_match":
                metric_list = self.training_metric['exact_match']
                name = "exact_match_{}_{}_{}_{}_{}_{}_{}.txt".format(self.args.method, self.args.blocks,
                                                                     self.args.alpha, self.args.comm_round,
                                                                     self.args.bit,
                                                                     self.args.dataset, self.args.subdataset)
                save_file_metric(name, metric_list)
            elif metric == "nDCG":
                metric_list = self.training_metric['ndcg']
                name = "ndcg.txt"
                save_file_metric(name, metric_list)
            else:
                exit('No metric {} erro in save_metric'.format(metric))

        return

    def init_training_metric(self, metrics: list):
        for metric in metrics:
            if metric == "f1_m":
                metric = 'f1'
            self.training_metric[metric] = []

    def record_metric(self, metrics: list, result: dict):
        for metric in metrics:
            if metric == "matthews_correlation":
                self.training_metric[metric].append(result['eval_matthews_correlation'])
            elif metric == "pearson":
                self.training_metric[metric].append(result['eval_pearson'])
            elif metric == "spearman":
                self.training_metric[metric].append(result['eval_spearmanr'])
            elif metric == "accuracy":
                self.training_metric[metric].append(result['eval_acc'])
            elif metric in ["f1", "f1_m"]:
                metric = 'f1'
                self.training_metric[metric].append(result['eval_f1'])
            elif metric == "precision":
                self.training_metric[metric].append(result['eval_precision'])
            elif metric == "exact_match":
                self.training_metric[metric].append(result['eval_exact_match'])
            elif metric == "nDCG":
                self.training_metric[metric].append(result['eval_ndcg'])
            else:
                exit('No metric {} erro in record_metric'.format(metric))

    def get_model_matrix_num(self):
        num = 0
        for layer in self.gobal_model.state_dict():
            if layer.find("num_batches_tracked") != -1:
                continue
            if 'lora' in layer:
                num += 1
        return num
    
    def get_packet_num(self, layers):
        traffic = 0
        for layer in self.gobal_model.state_dict():
            if layer.find("num_batches_tracked") != -1:
                continue
            if 'lora' in layer or any(part in layer for part in self.untrain_part):
                num = self.gobal_model.state_dict()[layer].numel()
                traffic += int(self.args.proportion * num) * (32 + math.log2(num) + math.log2(layers))
        return int(traffic / 1500 / 8)

    def copy_model_expect_lora(self, model):
        state_dict_a = self.gobal_model.state_dict()
        state_dict_b = model.state_dict()

        for name, param in state_dict_b.items():
            if 'lora' not in name and name in state_dict_b:
                state_dict_a[name].copy_(param)
        self.gobal_model.load_state_dict(state_dict_a)
        for name, param in self.gobal_model.named_parameters():
            if 'lora' not in name:
                param.requires_grad = False
    
    def load_timing_stats(self):
        """从timing_analyzer.py生成的文件中读取时间统计"""
        if self.args.dataset == 'glue':
            timing_file = f'timing_results_{self.args.model}_{self.args.dataset}_{self.args.subdataset}.txt'
        else:
            timing_file = f'timing_results_{self.args.model}_{self.args.dataset}.txt'
        try:
            with open(timing_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    if '平均时间/批次' in line and '前向传播' in lines[i-2]:
                        # 提取前向传播平均时间 (ms)
                        forward_time_str = line.split(':')[1].strip().split(' ')[0]
                        self.timing_stats['avg_forward_time'] = float(forward_time_str) / 1000  # 转为秒
                    elif '平均时间/批次' in line and '反向更新' in lines[i-2]:
                        # 提取反向更新平均时间 (ms)
                        backward_time_str = line.split(':')[1].strip().split(' ')[0]
                        self.timing_stats['avg_backward_time'] = float(backward_time_str) / 1000  # 转为秒
            print(f"已从 {timing_file} 加载时间统计")
        except FileNotFoundError:
            print(f"警告: 找不到 {timing_file}，使用默认值")
            self.timing_stats['avg_forward_time'] = 0.01  # 默认10ms
            self.timing_stats['avg_backward_time'] = 0.01  # 默认10ms
    
    def allocate_client_ranks_helora(self, partition_map, lora_params_size):
        """为 HeLoRA 方法分配客户端秩大小
        根据客户端的计算和通信资源分为三个档次：
        - 第一档次：秩 = lora_rank（计算+通信资源最优）
        - 第二档次：秩 = lora_rank / 2
        - 第三档次：秩 = lora_rank / 4（计算+通信资源最受限）
        
        注意：此函数应该在 allocate_client_timings 之后调用，
              以便使用已分配好的网速和计算时间
        """
        if self.args.method != 'HeLoRA':
            return
        
        print("\n" + "="*80)
        print("HeLoRA 秩分配（按计算+通信资源）")
        print("="*80)
        
        # 计算每个客户端的总资源得分（计算+通信）
        client_scores = []
        for client_id in range(self.args.num_client):
            # 获取已分配的计算时间和网速
            allocated_timing = self.client_allocated_timings[client_id]
            forward_time = allocated_timing['forward_time']
            backward_time = allocated_timing['backward_time']
            upload_speed = allocated_timing['upload_speed']
            download_speed = allocated_timing['download_speed']
            
            compute_score = forward_time + backward_time
            
            # 根据已分配的网速计算通信耗时
            upload_speed_bytes_per_sec = upload_speed * 1_000_000 / 8
            download_speed_bytes_per_sec = download_speed * 1_000_000 / 8
            
            upload_time = lora_params_size / upload_speed_bytes_per_sec
            download_time = lora_params_size / download_speed_bytes_per_sec
            communication_score = upload_time + download_time
            
            # 总资源得分 = 计算耗时 + 通信耗时
            total_score = compute_score + communication_score
            client_scores.append((client_id, total_score, compute_score, communication_score))
        
        # 按资源得分排序（从小到大，得分越小资源越好）
        client_scores.sort(key=lambda x: x[1])
        
        num_clients = len(client_scores)
        tier1_size = num_clients // 3
        tier2_size = num_clients // 3
        tier3_size = num_clients - tier1_size - tier2_size
        
        self.client_ranks = {}
        
        # 第一档次（资源最优）：秩 = lora_rank
        for i in range(tier1_size):
            client_id, total_score, compute_score, communication_score = client_scores[i]
            self.client_ranks[client_id] = self.args.lora_rank
            print(f"客户端 {client_id} → 第一档次 (秩={self.args.lora_rank})")
            print(f"  计算耗时: {compute_score:.4f}s, 通信耗时: {communication_score:.4f}s, 总计: {total_score:.4f}s")
        
        # 第二档次（资源中等）：秩 = lora_rank / 4
        for i in range(tier1_size, tier1_size + tier2_size):
            client_id, total_score, compute_score, communication_score = client_scores[i]
            self.client_ranks[client_id] = self.args.lora_rank // 4
            print(f"客户端 {client_id} → 第二档次 (秩={self.args.lora_rank // 2})")
            print(f"  计算耗时: {compute_score:.4f}s, 通信耗时: {communication_score:.4f}s, 总计: {total_score:.4f}s")
        
        # 第三档次（资源受限）：秩 = lora_rank / 4
        for i in range(tier1_size + tier2_size, num_clients):
            client_id, total_score, compute_score, communication_score = client_scores[i]
            self.client_ranks[client_id] = self.args.lora_rank // 4
            print(f"客户端 {client_id} → 第三档次 (秩={self.args.lora_rank // 4})")
            print(f"  计算耗时: {compute_score:.4f}s, 通信耗时: {communication_score:.4f}s, 总计: {total_score:.4f}s")
        
        print("="*80 + "\n")
    
    def truncate_lora_rank(self, model, target_rank, client_id):
        """根据目标秩截断 LoRA 矩阵
        将超出秩的部分矩阵值设置为 0
        
        LoRA 矩阵维度：
        - B 矩阵: (in_features, rank)
        - A 矩阵: (rank, out_features)
        
        Args:
            model: LoRA 模型
            target_rank: 目标秩大小
            client_id: 客户端 ID
        """
        if target_rank >= self.args.lora_rank or self.args.method != 'HeLoRA':
            return model
        
        model_state = model.state_dict()
        for name in model_state:
            if 'lora_B' in name:
                # B 矩阵: (in_features, rank) -> 保留前 target_rank 列
                param = model_state[name]
                if param.shape[1] > target_rank:
                    model_state[name][:, target_rank:] = 0
            elif 'lora_A' in name:
                # A 矩阵: (rank, out_features) -> 保留前 target_rank 行
                param = model_state[name]
                if param.shape[0] > target_rank:
                    model_state[name][target_rank:, :] = 0
        
        model.load_state_dict(model_state)
        return model
    
    def fill_truncated_lora(self, model, client_id):
        """使用平均值填充被截断的 LoRA 矩阵
        
        LoRA 矩阵维度：
        - B 矩阵: (in_features, rank) -> 使用前 target_rank 列的平均值填充后续列
        - A 矩阵: (rank, out_features) -> 使用前 target_rank 行的平均值填充后续行
        
        Args:
            model: 训练后的 LoRA 模型
            client_id: 客户端 ID
        """
        target_rank = self.client_ranks.get(client_id, self.args.lora_rank)
        
        if target_rank >= self.args.lora_rank:
            return model
        
        model_state = model.state_dict()
        
        for name in model_state:
            if 'lora_B' in name:
                # B 矩阵: (in_features, rank) -> 使用前 target_rank 列的平均值填充后续列
                current_param = model_state[name]
                if target_rank > 0 and current_param.shape[1] > target_rank:
                    # 计算前 target_rank 列的平均值
                    col_mean = torch.mean(current_param[:, :target_rank], dim=1, keepdim=True)  # (in_features, 1)
                    # 用平均值填充后续列
                    for col_idx in range(target_rank, current_param.shape[1]):
                        model_state[name][:, col_idx] = col_mean.squeeze()
            elif 'lora_A' in name:
                # A 矩阵: (rank, out_features) -> 使用前 target_rank 行的平均值填充后续行
                current_param = model_state[name]
                if target_rank > 0 and current_param.shape[0] > target_rank:
                    # 计算前 target_rank 行的平均值
                    row_mean = torch.mean(current_param[:target_rank, :], dim=0, keepdim=True)  # (1, out_features)
                    # 用平均值填充后续行
                    for row_idx in range(target_rank, current_param.shape[0]):
                        model_state[name][row_idx, :] = row_mean.squeeze()
        
        model.load_state_dict(model_state)
        return model

    
    def allocate_client_timings(self, partition_map, lora_params_size):
        """为所有客户端预分配计算和通信时间
        
        Args:
            partition_map: 客户端数据分配映射
            lora_params_size: LoRA参数大小（字节）
        """
        print("\n" + "="*80)
        print("预分配客户端网速")
        print("="*80)
        
        for client_id in range(self.args.num_client):
            # 计算该客户端的计算耗时
            forward_time, backward_time = self.calculate_client_compute_time(partition_map[client_id])
            
            # 为该客户端分配网络速率
            upload_speed = np.random.uniform(5, 20)  # 上传速率 5-20 Mbps
            download_speed = 50.0  # 下载速率 50 Mbps
            
            # 存储预分配的速度和计算耗时
            self.client_allocated_timings[client_id] = {
                'forward_time': forward_time,
                'backward_time': backward_time,
                'upload_speed': upload_speed,
                'download_speed': download_speed
            }
            
            print(f"客户端 {client_id}:")
            print(f"  前向传播耗时: {forward_time:.4f}s")
            print(f"  反向传播耗时: {backward_time:.4f}s")
            print(f"  上传速率: {upload_speed:.2f}Mbps, 下载速率: {download_speed:.2f}Mbps")
        
        print("="*80 + "\n")

    
    def calculate_client_compute_time(self, data_indices):
        """计算单个客户端的计算耗时（秒）
        返回前向传播时间和反向传播时间
        公式：基于高斯分布抽样的平均耗时 * 批量数 * epoch数
        """
        if not self.timing_stats:
            self.load_timing_stats()
        
        # 计算批量数
        num_batches = len(data_indices) // self.args.batch_size
        if len(data_indices) % self.args.batch_size != 0:
            num_batches += 1
        
        # 从高斯分布中抽样前向和反向传播时间
        # 使用平均值作为均值，10%的平均值作为标准差
        avg_forward = self.timing_stats.get('avg_forward_time', 0.01)
        avg_backward = self.timing_stats.get('avg_backward_time', 0.01)
        # 计算前向和反向传播平均时间 = 平均时间 * 批量数 * epoch数
        avg_forward_time = avg_forward * num_batches * self.args.epochs
        avg_backward_time = avg_backward * num_batches * self.args.epochs
        std_forward_time = avg_forward_time  # 前向传播时间的标准差为平均值的10%
        std_backward_time = avg_backward_time  # 反向传播时间的标准差为平均值的10%
        
        forward_time = np.random.normal(avg_forward_time, std_forward_time)
        backward_time = np.random.normal(avg_backward_time, std_backward_time)
        
        # 确保采样值非负
        forward_time = max(forward_time, 0.001)
        backward_time = max(backward_time, 0.001)
        
        return forward_time, backward_time
    
    def estimate_transmitted_bytes(self, grad_tensor, default_bytes):
        """估算本轮上传梯度时需要传输的字节数。"""
        # if isinstance(grad_tensor, torch.Tensor):
        #     nonzero = torch.count_nonzero(grad_tensor).item()
        #     if nonzero > 0:
        #         return nonzero * 4
        #     return grad_tensor.numel() * 4
        if self.args.method == 'pq':
            return grad_tensor.numel() * self.args.bit_len // 8  # pq 传输原始梯度
        elif self.args.method == 'topk':
            return int(grad_tensor.numel() * self.args.proportion) * (33 + int(math.log2(grad_tensor.numel())))  # top-k 传输原始梯度
        return default_bytes
    
    def calculate_communication_time(self, lora_params_size):
        """计算单个客户端的通信耗时（秒）
        为每个客户端随机分配上传速率，下载速率为50Mbps
        公式：参数大小 / 网络速率
        """
        # 随机分配上传速率 (5-20 Mbps)
        upload_speed = np.random.uniform(5, 20)  # Mbps
        download_speed = 50.0  # Mbps
        
        # 转换为字节/秒：Mbps * 1024 * 1024 / 8
        upload_speed_bytes_per_sec = upload_speed * 1024 * 1024 / 8
        download_speed_bytes_per_sec = download_speed * 1024 * 1024 / 8
        
        # 通信时间 = 上传时间 + 下载时间
        upload_time = lora_params_size / upload_speed_bytes_per_sec
        download_time = lora_params_size / download_speed_bytes_per_sec
        
        return upload_time + download_time, upload_speed, download_speed
    
    def save_round_timings(self):
        """保存和打印每轮的耗时统计"""
        if not self.round_timings:
            print("没有时间统计数据")
            return
        
        # 打印显示
        print("\n" + "="*80)
        print("联邦学习每轮耗时统计")
        print("="*80)
        
        total_compute_time = 0
        total_communication_time = 0
        total_time = 0
        
        for timing in self.round_timings:
            print(f"\n轮次 {timing['round']}:")
            print(f"  最大前向耗时: {timing['max_forward_time']:.4f}s")
            print(f"  最大反向耗时: {timing['max_backward_time']:.4f}s")
            print(f"  最大通信耗时: {timing['max_communication_time']:.4f}s")
            print(f"  本轮总耗时: {timing['max_total_time']:.4f}s")
            print(f"  瓶颈客户端: {timing['bottleneck_client']}")
        
        print(f"\n总计统计:")
        total_forward_time = sum([t['max_forward_time'] for t in self.round_timings])
        total_backward_time = sum([t['max_backward_time'] for t in self.round_timings])
        total_communication_time = sum([t['max_communication_time'] for t in self.round_timings])
        total_time = sum([t['max_total_time'] for t in self.round_timings])
        print(f"  总前向耗时: {total_forward_time:.4f}s")
        print(f"  总反向耗时: {total_backward_time:.4f}s")
        print(f"  总通信耗时: {total_communication_time:.4f}s")
        print(f"  实际训练时间: {total_time:.4f}s")
        print(f"  平均每轮耗时: {total_time / len(self.round_timings):.4f}s")
        print(f"  通信比例: {total_communication_time / total_time * 100:.2f}%")
        print(f"  累计训练时间: {self.total_training_time:.4f}s")
        print("="*80)
        
        # 保存到文件
        output_file = f'fed_round_timings_{self.args.model}_{self.args.dataset}.txt'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("联邦学习每轮耗时统计\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"模型: {self.args.model}\n")
            f.write(f"数据集: {self.args.dataset}\n")
            f.write(f"批大小: {self.args.batch_size}\n")
            f.write(f"Epoch数: {self.args.epochs}\n")
            f.write(f"通信轮数: {self.args.comm_round}\n")
            f.write(f"采样比例: {self.args.sample_fraction}\n\n")
            
            for timing in self.round_timings:
                f.write(f"轮次 {timing['round']}:\n")
                f.write(f"  最大前向耗时: {timing['max_forward_time']:.4f}s\n")
                f.write(f"  最大反向耗时: {timing['max_backward_time']:.4f}s\n")
                f.write(f"  最大通信耗时: {timing['max_communication_time']:.4f}s\n")
                f.write(f"  本轮总耗时: {timing['max_total_time']:.4f}s\n")
                f.write(f"  瓶颈客户端: {timing['bottleneck_client']}\n\n")
            
            f.write(f"总计统计:\n")
            total_forward_time = sum([t['max_forward_time'] for t in self.round_timings])
            total_backward_time = sum([t['max_backward_time'] for t in self.round_timings])
            total_communication_time = sum([t['max_communication_time'] for t in self.round_timings])
            total_time = sum([t['max_total_time'] for t in self.round_timings])
            f.write(f"  总前向耗时: {total_forward_time:.4f}s\n")
            f.write(f"  总反向耗时: {total_backward_time:.4f}s\n")
            f.write(f"  总通信耗时: {total_communication_time:.4f}s\n")
            f.write(f"  实际训练时间: {total_time:.4f}s\n")
            f.write(f"  平均每轮耗时: {total_time / len(self.round_timings):.4f}s\n")
            f.write(f"  通信比例: {total_communication_time / total_time * 100:.2f}%\n")
            f.write(f"  累计训练时间: {self.total_training_time:.4f}s\n")
        
        print(f"\n时间统计已保存到: {output_file}")

def save_file_metric(name: str, metric_list: list):
    path = os.path.join('./newsave', name)
    with open(path, 'w') as file:
        for item in metric_list:
            file.write(str(item) + '\n')


def custom_deepcopy(module):
    """自定义深拷贝函数，处理权重归一化和剪枝"""
    if isinstance(module, nn.Module):
        # 创建新模块实例
        new_module = type(module)(*[getattr(module, param) for param in module.__constants__])
        new_module.load_state_dict(module.state_dict())
        
        # 处理子模块
        for name, child in module.named_children():
            setattr(new_module, name, custom_deepcopy(child))
        
        # 处理缓冲区（如掩码）
        for buffer_name, buffer in module.named_buffers():
            new_module.register_buffer(buffer_name, buffer.clone())
        
        return new_module
    return copy.deepcopy(module)
