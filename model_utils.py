from task import Task
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, \
    AutoModelForQuestionAnswering, AutoModelForCausalLM, LlamaForQuestionAnswering, AutoModel
from torch import nn
import torch
import random
import math
# ------------------- 兼容所有PEFT版本的核心自定义LoRA层 -------------------
class CustomLoRALinear(nn.Linear):
    def __init__(self, in_features, out_features, r=0, lora_alpha=1, lora_dropout=0., 
                 fan_in_fan_out=False, merge_weights=True, bias=True, zero_rows=None, **kwargs):
        super().__init__(in_features, out_features, bias=bias)
        
        # LoRA核心参数（对齐PEFT的Linear层）
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0. else nn.Identity()
        self.fan_in_fan_out = fan_in_fan_out
        self.merge_weights = merge_weights
        
        # 初始化LoRA矩阵A/B（对齐PEFT默认逻辑）
        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros((r, in_features)))
            self.lora_B = nn.Parameter(torch.zeros((out_features, r)))
            # 初始化权重（PEFT默认的kaiming_uniform）
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            # 标记是否是LoRA层
            self.merged = False
        
        # 核心功能：指定置零的行
        self.zero_rows = zero_rows if zero_rows is not None else []
        
        # 初始化时置零+注册梯度钩子
        if r > 0 and self.zero_rows:
            with torch.no_grad():
                self.lora_B.data[self.zero_rows] = 0.0
            
            # 注册梯度钩子，反向传播时清空指定行梯度
            def grad_hook(grad):
                grad_clone = grad.clone()
                grad_clone[self.zero_rows] = 0.0
                return grad_clone
            
            self.lora_B.register_hook(grad_hook)

    def forward(self, x: torch.Tensor):
        # 前向传播前强制置零
        if self.r > 0 and self.zero_rows:
            with torch.no_grad():
                self.lora_B.data[self.zero_rows] = 0.0
        
        # PEFT原生LoRA前向逻辑
        if self.r > 0 and not self.merged:
            result = super().forward(x)
            if self.fan_in_fan_out:
                x = x.transpose(0, 1)
            x = self.lora_dropout(x)
            x = x @ self.lora_A.T @ self.lora_B.T / self.lora_alpha
            if self.fan_in_fan_out:
                x = x.transpose(0, 1)
            result += x
            return result
        else:
            return super().forward(x)

# ------------------- 安全替换PEFT LoRA层 -------------------
def safe_replace_lora_layers(peft_model, proportion):
    with torch.no_grad():
        for name, module in peft_model.named_modules():
            if hasattr(module, 'lora_A') and module.lora_A is not None:
                lora_A_weight = module.lora_A['default'].weight
                row_num = lora_A_weight.shape[0]
                train_row = math.ceil(row_num * proportion)
                frozen_row = list(range(train_row,row_num))
                print(f"Froze row {frozen_row} of {name}.lora_A")
                def make_hook(row_idx):
                    def hook(grad):
                        if grad is not None:
                            grad.data[row_idx, :] = 0
                        return grad
                    return hook
                lora_A_weight.register_hook(make_hook(frozen_row))

                # lora_B_weight = module.lora_A['default'].weight
                # row_num = lora_B_weight.shape[0]
                # col_num = lora_B_weight.shape[1]
                # proportion_row = int((1-math.sqrt(1-proportion))*row_num) / row_num
                # proportion_col = 1-(1-proportion) / (1-proportion_row)
                # print("the froze proportion of row and col in A is ", proportion_row, proportion_col)
                # frozen_row_index = random.sample(list(range(row_num)), math.ceil(row_num * proportion_row))
                # frozen_col_index = random.sample(list(range(col_num)), math.ceil(col_num * proportion_col))
                # # lora_B_weight.data[frozen_row_index, :] = 0
                # print(f"Froze row {frozen_row_index} of {name}.lora_B")
                # print(f"Froze col {frozen_col_index} of {name}.lora_B")
                # def make_hook(row_idx, col_idx):
                #     def hook(grad):
                #         if grad is not None:
                #             grad.data[row_idx, :] = 0
                #             grad.data[:, col_idx] = 0
                #         return grad
                #     return hook
                # lora_B_weight.register_hook(make_hook(frozen_row_index, frozen_col_index))
            if hasattr(module, 'lora_B') and module.lora_B is not None:
                lora_B_weight = module.lora_B['default'].weight
                col_num = lora_B_weight.shape[1]
                train_col = math.ceil(col_num * proportion)
                frozen_col = list(range(train_col,row_num))
                print(f"Froze row {frozen_col} of {name}.lora_B")
                def make_hook(col_idx):
                    def hook(grad):
                        if grad is not None:
                            grad.data[:, col_idx] = 0
                        return grad
                    return hook
                lora_B_weight.register_hook(make_hook(frozen_col))
                
                # lora_A_weight = module.lora_B['default'].weight
                # column_num = lora_A_weight.shape[1]
                # row_num = lora_A_weight.shape[0]
                # proportion_col = int((1-math.sqrt(1-proportion))*column_num) / column_num
                # proportion_row = 1-(1-proportion) / (1-proportion_row)
                # print("the froze proportion of col and row in B is ", proportion_col, proportion_row)
                # frozen_column_index = random.sample(list(range(column_num)), math.ceil(column_num * proportion_col))
                # frozen_row_index = random.sample(list(range(row_num)), math.ceil(row_num * proportion_row))
                # # lora_A_weight.data[:, frozen_column_index] = 0
                # print(f"Froze column {frozen_column_index} of {name}.lora_A")
                # print(f"Froze row {frozen_row_index} of {name}.lora_A")
                # def make_hook(column_idx, row_idx):
                #     def hook(grad):
                #         if grad is not None:
                #             grad.data[:, column_idx] = 0
                #             grad.data[row_idx, :] = 0
                #         return grad
                #     return hook
                # lora_A_weight.register_hook(make_hook(frozen_column_index, frozen_row_index))
    return peft_model


    # for name, module in peft_model.named_modules():
    #     # 只处理PEFT的LoRA Linear层
    #     if 'lora_B' in name and isinstance(module, nn.Linear):
    #         for _, param in module.named_parameters():
    #             row_num = param.shape[0]
    #             zero_rows = random.sample(list(range(row_num)), int(row_num * proportion))
    #             # 提取原有LoRA参数
    #             lora_kwargs = {
    #                 'in_features': module.in_features,
    #                 'out_features': module.out_features,
    #                 'r': args.lora_rank,
    #                 'lora_alpha': args.lora_alpha,
    #                 'lora_dropout': 0.05,
    #                 'fan_in_fan_out': module.fan_in_fan_out,
    #                 'merge_weights': module.merge_weights,
    #                 'bias': module.bias is not None,
    #                 'zero_rows': zero_rows,
    #                 'device': module.weight.device,
    #                 'dtype': module.weight.dtype,
    #             }
                
    #             # 创建自定义层并复制权重
    #             custom_layer = CustomLoRALinear(**lora_kwargs)
    #             custom_layer.weight = module.weight
    #             if module.bias is not None:
    #                 custom_layer.bias = module.bias
    #             if module.r > 0:
    #                 custom_layer.lora_A = module.lora_A
    #                 custom_layer.lora_B = module.lora_B  # 触发置零和钩子注册
                
    #             # 替换模型中的层
    #             parent_name = '.'.join(name.split('.')[:-1])
    #             child_name = name.split('.')[-1]
    #             parent_module = peft_model.get_submodule(parent_name)
    #             setattr(parent_module, child_name, custom_layer)
    # return peft_model

class OneDCNNEncoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(OneDCNNEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_size),
            nn.ReLU()
            # nn.Conv1d(input_channels, hidden_channels, kernel_size=kernel_size, padding=padding),
            # nn.ReLU(True),
            # nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=kernel_size, padding=padding),
            # nn.ReLU(True),
            # nn.Conv1d(hidden_channels * 2, hidden_channels * 4, kernel_size=kernel_size, padding=padding),
            # nn.ReLU(True)
        )

    def forward(self, x):
        return self.encoder(x)


class OneDCNNDecoder(nn.Module):
    def __init__(self, output_size, input_size):
        super(OneDCNNDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(output_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, input_size),
            nn.Sigmoid()
            # nn.ConvTranspose1d(hidden_channels * 4, hidden_channels * 2, kernel_size=kernel_size,
            #                    stride=stride, padding=padding, output_padding=output_padding),
            # nn.ReLU(True),
            # nn.ConvTranspose1d(hidden_channels * 2, hidden_channels, kernel_size=kernel_size,
            #                    stride=stride, padding=padding, output_padding=output_padding),
            # nn.ReLU(True),
            # nn.ConvTranspose1d(hidden_channels, output_channels, kernel_size=kernel_size,
            #                    stride=stride, padding=padding, output_padding=output_padding),
            # nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(x)


class OneDCNNAutoencoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(OneDCNNAutoencoder, self).__init__()
        self.encoder = OneDCNNEncoder(input_size, output_size)
        self.decoder = OneDCNNDecoder(output_size, input_size)

    def forward(self, x):
        # encoded = self.encoder(x)
        # origin_shape = encoded.shape
        # x = nn.Flatten()(encoded)
        # input_size = x.shape[-1]
        # print(input_size)
        # exit(0)
        # x = nn.Linear(input_size, self.output_size)(x)
        # x = nn.Linear(self.output_size, input_size)(x)
        # x = x.reshape(origin_shape)
        # decoded = self.decoder(x)
        
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def get_model_path(model: str):
    if model == 'roberta-large':
        model_path = './model/models--roberta-large/'
    elif model == 'roberta-base':
        model_path = './model/models--roberta-base/'
    elif model == 'distilbert-base-multilingual-cased':
        model_path = './model/models--distilbert-base-multilingual-cased/'
    elif model == 'distilbert-base-uncased':
        model_path = './model/models--distilbert-base-uncased/'
    elif model == 'distilbert-base-cased':
        model_path = './model/models--distilbert-base-cased/'
    elif model == 'llama-2-7B':
        model_path = './model/models--llama-2-7b-hf/'
    elif model == 'llama-3.2-1B':
        model_path = './model/models--llama-3.2-1b/'
    else:
        model_path = './model/models--llama-3-8b/'
    return model_path


def get_model_tokenizer(model: str, max_length: int):
    model_path = get_model_path(model=model)
    if 'llama' in model:
        tokenizer = AutoTokenizer.from_pretrained(model_path, max_length=max_length, padding=True)
        tokenizer.pad_token = tokenizer.eos_token
    elif model == 'roberta-large':
        tokenizer = AutoTokenizer.from_pretrained(model_path, max_length=max_length, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, max_length=max_length)
    return tokenizer


def get_model_lora(model: str, lora_alpha: int, lora_rank: int, num_labels: list, task: Task, method):
    model_path = get_model_path(model=model)
    model_pre, task_type = get_automodel(model_path=model_path, num_labels=num_labels, task=task)
    if model == 'distilbert-base-multilingual-cased' or model == 'distilbert-base-cased':
        target_modules = ["q_lin", "v_lin"]
    elif model == 'roberta-large' or model == 'roberta-base':
        target_modules = ["query", "value"]
    elif model == 'llama-2-7B':
        target_modules = ["q_proj", "v_proj"]
        model_pre.config.pad_token_id = 2
    else:
        target_modules = ["q_proj", "v_proj"]
        model_pre.config.pad_token_id = 128001

    untrain_part = get_untrain_part(model, task)
    peft_config = LoraConfig(
        r=lora_rank,
        target_modules=target_modules,
        task_type=task_type,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        # init_lora_weights=False
        # use_rslora=True,
        # bias='lora_only'
    )
    peft_model = get_peft_model(model_pre, peft_config)
    trainable_parameters, _ = peft_model.get_nb_trainable_parameters()
    peft_model.print_trainable_parameters()
    set_requires_grad(untrain_part=untrain_part, model=peft_model)
    return peft_model, trainable_parameters, untrain_part


def get_automodel(model_path: str, num_labels: list, task: Task):
    if task == Task.SequenceClassification:
        model_pre = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels[0])
        task_type = TaskType.CAUSAL_LM
    elif task == Task.CausalLM:
        # model_pre = AutoModelForSeq2SeqLM.from_pretrained(model_path, is_decoder=True)
        model_pre = AutoModelForCausalLM.from_pretrained(model_path, is_decoder=True)
        task_type = TaskType.CAUSAL_LM
    elif task == Task.TokenClassification:
        if 'llama' in model_path:
            exit("Llama do not support task TokenClassification")
        model_pre = AutoModelForTokenClassification.from_pretrained(model_path, id2label=num_labels[0],
                                                                    label2id=num_labels[1])
        task_type = TaskType.CAUSAL_LM
    else:
        if 'llama' in model_path:
            model_pre = AutoModel.from_pretrained(model_path)
            model_pre.save_pretrained("./model/base_model")
            model_pre = AutoModelForQuestionAnswering.from_pretrained("./model/base_model")
            # model_pre = AutoModelForQuestionAnswering.from_pretrained(model_path)
        else:
            model_pre = AutoModelForQuestionAnswering.from_pretrained(model_path)
        task_type = TaskType.QUESTION_ANS
    return model_pre, task_type


def get_untrain_part(model: str, task: Task):
    if model == 'distilbert-base-multilingual-cased':
        task_train = {Task.SequenceClassification: ['classifier.bias', 'classifier.weight',
                                                    'pre_classifier.bias', 'pre_classifier.weight'],
                      Task.TokenClassification: ['classifier.bias', 'classifier.weight'],
                      Task.QuestionAnswering: ['qa_outputs.bias', 'qa_outputs.weight'],
                      Task.CausalLM: []}
    elif model == 'roberta-large' or model == 'roberta-base':
        task_train = {Task.SequenceClassification: ['classifier.dense.bias', 'classifier.dense.weight',
                                                    'classifier.out_proj.bias', 'classifier.out_proj.weight'],
                      Task.TokenClassification: ['classifier.bias', 'classifier.weight'],
                      Task.QuestionAnswering: ['qa_outputs.bias', 'qa_outputs.weight'],
                      Task.CausalLM: []}
    # elif 'llama' in model:
    else:
        task_train = {Task.SequenceClassification: ['score.weight'],
                      Task.QuestionAnswering: ['qa_outputs.bias', 'qa_outputs.weight']}
    return task_train[task]


def set_requires_grad(untrain_part: list, model):
    param_num = 0
    for name, param in model.named_parameters():
        if any(part in name for part in untrain_part):
            param.requires_grad = True
            param_num += param.numel()
        # for layer in untrain_part:
        #     if layer in name:
        #         param.requires_grad = True
        #         param_num += param.numel()
    return model, param_num
