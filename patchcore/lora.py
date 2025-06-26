"""
LoRA (Low-Rank Adaptation) implementation for continual learning in Vision Transformers
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALayer(nn.Module):
    """
    LoRA layer that can be applied to any Linear layer
    """
    def __init__(self, in_features, out_features, rank=4, alpha=1, dropout=0.1):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = self.alpha / self.rank
        
        # LoRA parameters
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x):
        # x: (batch_size, seq_len, in_features)
        # LoRA forward: x @ A^T @ B^T * scaling
        result = self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        return result

class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation
    """
    def __init__(self, linear_layer, rank=4, alpha=1, dropout=0.1, enable_lora=True):
        super().__init__()
        self.linear = linear_layer
        self.enable_lora = enable_lora
        
        if enable_lora:
            self.lora = LoRALayer(
                linear_layer.in_features, 
                linear_layer.out_features, 
                rank=rank, 
                alpha=alpha, 
                dropout=dropout
            )
        
        # Freeze the original linear layer
        for param in self.linear.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        result = self.linear(x)
        if self.enable_lora:
            result = result + self.lora(x)
        return result

class TaskSpecificLoRA(nn.Module):
    """
    Task-specific LoRA management for continual learning
    """
    def __init__(self, base_linear, num_tasks=10, rank=4, alpha=1, dropout=0.1):
        super().__init__()
        self.base_linear = base_linear
        self.num_tasks = num_tasks
        self.current_task = 0
        
        # Freeze base layer
        for param in self.base_linear.parameters():
            param.requires_grad = False
            
        # Create LoRA adapters for each task
        self.task_loras = nn.ModuleList([
            LoRALayer(
                base_linear.in_features,
                base_linear.out_features,
                rank=rank,
                alpha=alpha,
                dropout=dropout
            ) for _ in range(num_tasks)
        ])
        
        # Task selection mechanism
        self.task_keys = nn.Parameter(torch.randn(num_tasks, base_linear.in_features))
        nn.init.normal_(self.task_keys, std=0.02)
        
    def set_task(self, task_id):
        """Set current task for training"""
        self.current_task = task_id
        
        # Enable gradients only for current task LoRA
        for i, lora in enumerate(self.task_loras):
            for param in lora.parameters():
                param.requires_grad = (i == task_id)
                
    def get_task_similarity(self, x):
        """Compute similarity with task keys for inference"""
        # x: (B, seq_len, dim) -> use mean pooling
        x_mean = x.mean(dim=1)  # (B, dim)
        
        # Normalize
        x_norm = F.normalize(x_mean, dim=-1)  # (B, dim)
        keys_norm = F.normalize(self.task_keys, dim=-1)  # (num_tasks, dim)
        
        # Compute similarity
        similarity = torch.matmul(x_norm, keys_norm.T)  # (B, num_tasks)
        return similarity
        
    def forward(self, x, task_id=None, inference_mode=False):
        # Base forward pass
        base_output = self.base_linear(x)
        
        if inference_mode and task_id is None:
            # Automatic task selection based on similarity
            similarity = self.get_task_similarity(x)
            task_id = torch.argmax(similarity, dim=1)  # (B,)
            
            # Apply LoRA for each sample's selected task
            lora_output = torch.zeros_like(base_output)
            for i in range(len(self.task_loras)):
                mask = (task_id == i).unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
                if mask.any():
                    task_output = self.task_loras[i](x)
                    lora_output = lora_output + mask * task_output
                    
        else:
            # Training mode or specific task specified
            if task_id is None:
                task_id = self.current_task
            lora_output = self.task_loras[task_id](x)
            
        return base_output + lora_output

def apply_lora_to_vit(model, rank=4, alpha=1, dropout=0.1, target_modules=None):
    """
    Apply LoRA to specific modules in ViT
    
    Args:
        model: VisionTransformer model
        rank: LoRA rank
        alpha: LoRA alpha parameter
        dropout: LoRA dropout rate
        target_modules: List of module names to apply LoRA to
                       Default: ['qkv', 'proj', 'fc1', 'fc2']
    """
    if target_modules is None:
        target_modules = ['qkv', 'proj', 'fc1', 'fc2']
    
    lora_modules = {}
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module_name = name.split('.')[-1]
            if module_name in target_modules:
                # Replace with LoRA version
                parent_module = model
                for part in name.split('.')[:-1]:
                    parent_module = getattr(parent_module, part)
                
                lora_module = LoRALinear(
                    module, 
                    rank=rank, 
                    alpha=alpha, 
                    dropout=dropout
                )
                setattr(parent_module, module_name, lora_module)
                lora_modules[name] = lora_module
                
    return lora_modules

def apply_task_specific_lora_to_vit(model, num_tasks=10, rank=4, alpha=1, dropout=0.1, target_modules=None):
    """
    Apply task-specific LoRA to ViT for continual learning
    """
    if target_modules is None:
        target_modules = ['qkv', 'proj', 'fc1', 'fc2']
    
    task_lora_modules = {}
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module_name = name.split('.')[-1]
            if module_name in target_modules:
                # Replace with task-specific LoRA version
                parent_module = model
                for part in name.split('.')[:-1]:
                    parent_module = getattr(parent_module, part)
                
                task_lora_module = TaskSpecificLoRA(
                    module,
                    num_tasks=num_tasks,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout
                )
                setattr(parent_module, module_name, task_lora_module)
                task_lora_modules[name] = task_lora_module
                
    return task_lora_modules 