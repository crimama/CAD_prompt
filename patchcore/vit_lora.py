"""
ViT with LoRA for continual learning - replacing E-Prompt system
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import Mlp
from .vision_transformer import VisionTransformer, Block, Attention, _create_vision_transformer
from .lora import TaskSpecificLoRA, apply_task_specific_lora_to_vit

class LoRAAttention(nn.Module):
    """
    Attention module with LoRA adaptation
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 num_tasks=10, lora_rank=4, lora_alpha=1, lora_dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Original layers (will be frozen)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # LoRA adapters
        self.qkv_lora = TaskSpecificLoRA(
            self.qkv, num_tasks=num_tasks, rank=lora_rank,
            alpha=lora_alpha, dropout=lora_dropout
        )
        self.proj_lora = TaskSpecificLoRA(
            self.proj, num_tasks=num_tasks, rank=lora_rank,
            alpha=lora_alpha, dropout=lora_dropout
        )

    def set_task(self, task_id):
        """Set current task for LoRA modules"""
        self.qkv_lora.set_task(task_id)
        self.proj_lora.set_task(task_id)

    def forward(self, x, task_id=None, inference_mode=False):
        B, N, C = x.shape
        
        # QKV with LoRA
        qkv = self.qkv_lora(x, task_id=task_id, inference_mode=inference_mode)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        # Projection with LoRA
        x = self.proj_lora(x, task_id=task_id, inference_mode=inference_mode)
        x = self.proj_drop(x)
        return x

class LoRAMlp(nn.Module):
    """
    MLP module with LoRA adaptation
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 num_tasks=10, lora_rank=4, lora_alpha=1, lora_dropout=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # Original layers (will be frozen)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)
        
        # LoRA adapters
        self.fc1_lora = TaskSpecificLoRA(
            self.fc1, num_tasks=num_tasks, rank=lora_rank,
            alpha=lora_alpha, dropout=lora_dropout
        )
        self.fc2_lora = TaskSpecificLoRA(
            self.fc2, num_tasks=num_tasks, rank=lora_rank,
            alpha=lora_alpha, dropout=lora_dropout
        )

    def set_task(self, task_id):
        """Set current task for LoRA modules"""
        self.fc1_lora.set_task(task_id)
        self.fc2_lora.set_task(task_id)

    def forward(self, x, task_id=None, inference_mode=False):
        x = self.fc1_lora(x, task_id=task_id, inference_mode=inference_mode)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2_lora(x, task_id=task_id, inference_mode=inference_mode)
        x = self.drop2(x)
        return x

class LoRABlock(Block):
    """
    Transformer block with LoRA adaptation
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., 
                 init_values=None, drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 num_tasks=10, lora_rank=4, lora_alpha=1, lora_dropout=0.1):
        # Initialize parent without attn_layer to avoid conflicts
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        
        # LoRA Attention
        self.attn = LoRAAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            attn_drop=attn_drop, proj_drop=drop,
            num_tasks=num_tasks, lora_rank=lora_rank,
            lora_alpha=lora_alpha, lora_dropout=lora_dropout
        )
        
        # Layer scale and drop path - handle import gracefully
        try:
            from timm.models.layers import LayerScale, DropPath
            self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
            self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        except ImportError:
            # Fallback if LayerScale/DropPath not available
            self.ls1 = nn.Identity()
            self.drop_path1 = nn.Identity()

        self.norm2 = norm_layer(dim)
        
        # LoRA MLP
        self.mlp = LoRAMlp(
            in_features=dim, hidden_features=int(dim * mlp_ratio), 
            act_layer=act_layer, drop=drop,
            num_tasks=num_tasks, lora_rank=lora_rank,
            lora_alpha=lora_alpha, lora_dropout=lora_dropout
        )
        
        try:
            self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
            self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        except (ImportError, NameError):
            # Fallback if LayerScale/DropPath not available
            self.ls2 = nn.Identity()
            self.drop_path2 = nn.Identity()
        
    def set_task(self, task_id):
        """Set current task for all LoRA modules"""
        self.attn.set_task(task_id)
        self.mlp.set_task(task_id)
        
    def forward(self, x, task_id=None, inference_mode=False):
        # Attention with LoRA
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), task_id=task_id, inference_mode=inference_mode)))
        
        # MLP with LoRA
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x), task_id=task_id, inference_mode=inference_mode)))
        
        return x

class VisionTransformerLoRA(VisionTransformer):
    """
    Vision Transformer with LoRA for continual learning
    """
    def __init__(self, num_tasks=10, lora_rank=4, lora_alpha=1, lora_dropout=0.1,
                 lora_target_modules=None, **kwargs):
        # Remove prompt-related arguments to avoid conflicts
        prompt_args = [
            'use_e_prompt', 'use_g_prompt', 'prompt_length', 'prompt_pool', 
            'pool_size', 'top_k', 'e_prompt_layer_idx', 'g_prompt_layer_idx',
            'prompt_init', 'prompt_key', 'batchwise_prompt', 'prompt_key_init',
            'use_prompt_mask', 'use_prefix_tune_for_g_prompt', 
            'use_prefix_tune_for_e_prompt', 'same_key_value', 'embedding_key'
        ]
        for arg in prompt_args:
            kwargs.pop(arg, None)
        
        # Store LoRA parameters
        self.num_tasks = num_tasks
        self.current_task = 0
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        
        # Don't call parent init during re-initialization
        if not hasattr(self, 'patch_embed'):  # Only init if not already initialized
            # Set block function to LoRA block
            kwargs['block_fn'] = lambda *args, **kw: LoRABlock(
                *args, **kw, 
                num_tasks=num_tasks, lora_rank=lora_rank,
                lora_alpha=lora_alpha, lora_dropout=lora_dropout
            )
            
            super().__init__(**kwargs)
        else:
            # Already initialized, just replace blocks with LoRA blocks
            self.replace_blocks_with_lora(num_tasks, lora_rank, lora_alpha, lora_dropout)
        
        print(f"🔧 ViT-LoRA initialized: {num_tasks} tasks, rank={lora_rank}, alpha={lora_alpha}")
        
    def replace_blocks_with_lora(self, num_tasks, lora_rank, lora_alpha, lora_dropout):
        """Replace existing blocks with LoRA blocks"""
        new_blocks = []
        for i, block in enumerate(self.blocks):
            # Create new LoRA block with same dimensions
            lora_block = LoRABlock(
                dim=block.norm1.normalized_shape[0],  # Get dimension from norm layer
                num_heads=block.attn.num_heads,
                mlp_ratio=4.0,  # Standard ratio
                qkv_bias=hasattr(block.attn.qkv, 'bias') and block.attn.qkv.bias is not None,
                drop=0.0,
                attn_drop=0.0,
                num_tasks=num_tasks,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout
            )
            
            # Copy weights from original block to LoRA block
            # This preserves the pretrained weights
            lora_block.load_state_dict(block.state_dict(), strict=False)
            
            new_blocks.append(lora_block)
            
        # Replace blocks
        self.blocks = nn.ModuleList(new_blocks)
        
    def set_task(self, task_id):
        """Set current task for training"""
        assert 0 <= task_id < self.num_tasks, f"Task ID {task_id} out of range [0, {self.num_tasks})"
        self.current_task = task_id
        
        # Set task for all LoRA blocks
        for block in self.blocks:
            if hasattr(block, 'set_task'):
                block.set_task(task_id)
        
        print(f"📋 Current task set to: {task_id}")
            
    def get_task_similarity(self, x):
        """Get task similarity for inference (using first LoRA module)"""
        for block in self.blocks:
            if hasattr(block, 'attn') and hasattr(block.attn, 'qkv_lora'):
                return block.attn.qkv_lora.get_task_similarity(x)
        return None
        
    def forward_features(self, x, task_id=None, inference_mode=False, cls_features=None, train=False, query_task_id=None):
        x = self.patch_embed(x)
        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        
        x = self.pos_drop(x + self.pos_embed)
        
        res = dict()
        
        # Pass through transformer blocks
        for i, block in enumerate(self.blocks):
            if hasattr(block, 'set_task'):  # LoRA block
                x = block(x, task_id=task_id, inference_mode=inference_mode)
            else:  # Standard block
                x = block(x)
            
            # Feature extraction for segmentation (like original)
            if i == 5:  # Extract features from 6th layer
                res['seg_feat'] = [x[:, 1:, :]]  # Exclude CLS token
                
        x = self.norm(x)
        res['x'] = x
        return res
        
    def forward_head(self, res, pre_logits: bool = False, label=None, train=False, task_id=-1, image_path=None):
        x = res['x']
        
        # Use CLS token for classification
        if self.class_token:
            x = x[:, 0]  # CLS token
        else:
            x = x.mean(dim=1)  # Global average pooling
        
        res['pre_logits'] = x
        
        # No contrastive loss for LoRA version (simplified)
        res['loss'] = torch.tensor(0).float().cuda() if torch.cuda.is_available() else torch.tensor(0).float()
        
        return res
        
    def forward(self, x, task_id=None, inference_mode=False, cls_features=None, train=False, 
                pre_logits=False, label=None, image_path=None, query_task_id=None):
        
        if inference_mode and task_id is None:
            # Automatic task selection during inference
            with torch.no_grad():
                # First pass to get features for task selection
                temp_features = self.forward_features(x, task_id=0, inference_mode=False)
                task_similarity = self.get_task_similarity(temp_features['x'])
                if task_similarity is not None:
                    task_id = torch.argmax(task_similarity, dim=1)[0].item()  # Take first sample's task
                else:
                    task_id = 0  # Fallback
            
        res = self.forward_features(x, task_id=task_id, inference_mode=inference_mode, 
                                  cls_features=cls_features, train=train, query_task_id=query_task_id)
        res = self.forward_head(res, pre_logits, label, train, task_id=task_id, image_path=image_path)
        
        return res

def create_vit_lora_model(model_name='vit_base_patch16_224', num_tasks=10, lora_rank=4, 
                         lora_alpha=1, lora_dropout=0.1, pretrained=True, **kwargs):
    """
    Create ViT model with LoRA for continual learning
    """
    # Remove LoRA-specific arguments from kwargs before creating base model
    lora_kwargs = {
        'num_tasks': num_tasks,
        'lora_rank': lora_rank,
        'lora_alpha': lora_alpha,
        'lora_dropout': lora_dropout
    }
    
    # Filter out LoRA arguments from kwargs
    base_kwargs = {k: v for k, v in kwargs.items() 
                   if k not in ['num_tasks', 'lora_rank', 'lora_alpha', 'lora_dropout']}
    
    # Create base ViT model first
    base_model = _create_vision_transformer(
        model_name, 
        pretrained=pretrained,
        **base_kwargs
    )
    
    # Convert to LoRA version by changing class and re-initializing
    # Store original state dict
    original_state_dict = base_model.state_dict()
    
    # Change class to VisionTransformerLoRA
    base_model.__class__ = VisionTransformerLoRA
    
    # Re-initialize with LoRA parameters
    VisionTransformerLoRA.__init__(
        base_model, 
        **lora_kwargs,
        **base_kwargs
    )
    
    # Load back the original weights
    base_model.load_state_dict(original_state_dict, strict=False)
    
    return base_model 