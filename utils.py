import torch
import numpy as np
import timm


def load_weights_from_npz(model, model_name, default_cfgs=None):
    """
    이미 생성된 ViT 모델에 model_name을 통해 default_cfgs에서 NPZ URL을 찾아 가중치를 로드하는 함수
    
    Args:
        model (torch.nn.Module): 가중치를 로드할 모델 (이미 생성됨)
        model_name (str): 모델 이름 (예: 'vit_base_patch16_224')
        default_cfgs (dict, optional): 모델 설정 딕셔너리. None인 경우 patchcore.vision_transformer에서 import
    
    Returns:
        torch.nn.Module: 가중치가 로드된 모델
    """
    import torch
    import numpy as np
    from urllib.parse import urlparse
    import os
    
    # --- 1단계: default_cfgs에서 URL 찾기 ---
    if default_cfgs is None:
        try:
            from patchcore.vision_transformer import default_cfgs
        except ImportError:
            # 대체 방법: timm에서 가져오기
            import timm
            model_info = timm.models.model_entrypoint(model_name)
            if hasattr(model_info, 'default_cfg'):
                npz_url_or_path = model_info.default_cfg.get('url', '')
            else:
                raise ValueError(f"모델 '{model_name}'의 설정을 찾을 수 없습니다.")
    
    # default_cfgs가 있는 경우 (None이 아니거나 위에서 import된 경우)
    if model_name not in default_cfgs:
        raise ValueError(f"모델 '{model_name}'이 default_cfgs에 없습니다. 사용 가능한 모델: {list(default_cfgs.keys())}")
    
    cfg = default_cfgs[model_name]
    npz_url_or_path = cfg.get('url', '')
    
    if not npz_url_or_path:
        raise ValueError(f"모델 '{model_name}'에 대한 URL이 설정되지 않았습니다.")
    
    # NPZ 파일이 아닌 경우 (예: .pth 파일) 체크
    if npz_url_or_path and not npz_url_or_path.endswith('.npz'):
        raise ValueError(f"이 함수는 NPZ 파일만 지원합니다. '{model_name}'은 NPZ 형식이 아닙니다: {npz_url_or_path}")
    
    # --- 2단계: NPZ 파일 로드 (URL 또는 로컬 경로 처리) ---
    if urlparse(npz_url_or_path).scheme in ('http', 'https'):
        # URL인 경우 다운로드
        try:
            from timm.models.helpers import download_cached_file
            npz_path = download_cached_file(npz_url_or_path)
        except ImportError:
            import urllib.request
            import tempfile
            temp_dir = tempfile.gettempdir()
            filename = os.path.basename(urlparse(npz_url_or_path).path)
            npz_path = os.path.join(temp_dir, filename)
            urllib.request.urlretrieve(npz_url_or_path, npz_path)
    else:
        # 로컬 파일 경로인 경우
        npz_path = npz_url_or_path
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"NPZ 파일을 찾을 수 없습니다: {npz_path}")
    
    npz_weights = np.load(npz_path)
    
    # --- 3단계: 가중치 로드 및 매핑 ---
    new_state_dict = model.state_dict()
    
    def convert_npz_to_pytorch_key(pytorch_key):
        """PyTorch 키를 NPZ 키로 변환하는 규칙"""
        # Positional embedding
        if pytorch_key == 'pos_embed':
            return 'Transformer/posembed_input/pos_embedding'
        
        # Class token
        elif pytorch_key == 'cls_token':
            return 'cls'
        
        # Patch embedding
        elif pytorch_key == 'patch_embed.proj.weight':
            return 'embedding/kernel'
        elif pytorch_key == 'patch_embed.proj.bias':
            return 'embedding/bias'
        
        # Layer norm
        elif pytorch_key == 'norm.weight':
            return 'Transformer/encoder_norm/scale'
        elif pytorch_key == 'norm.bias':
            return 'Transformer/encoder_norm/bias'
        
        # Head
        elif pytorch_key == 'head.weight':
            return 'head/kernel'
        elif pytorch_key == 'head.bias':
            return 'head/bias'
        
        # Transformer blocks
        elif pytorch_key.startswith('blocks.'):
            # blocks.0.norm1.weight -> Transformer/encoderblock_0/LayerNorm_0/scale
            parts = pytorch_key.split('.')
            block_num = parts[1]
            
            if 'norm1.weight' in pytorch_key:
                return f'Transformer/encoderblock_{block_num}/LayerNorm_0/scale'
            elif 'norm1.bias' in pytorch_key:
                return f'Transformer/encoderblock_{block_num}/LayerNorm_0/bias'
            elif 'norm2.weight' in pytorch_key:
                return f'Transformer/encoderblock_{block_num}/LayerNorm_2/scale'
            elif 'norm2.bias' in pytorch_key:
                return f'Transformer/encoderblock_{block_num}/LayerNorm_2/bias'
            elif 'attn.qkv.weight' in pytorch_key:
                return f'Transformer/encoderblock_{block_num}/MultiHeadDotProductAttention_1/kernel'
            elif 'attn.qkv.bias' in pytorch_key:
                return f'Transformer/encoderblock_{block_num}/MultiHeadDotProductAttention_1/bias'
            elif 'attn.proj.weight' in pytorch_key:
                return f'Transformer/encoderblock_{block_num}/MultiHeadDotProductAttention_1/out/kernel'
            elif 'attn.proj.bias' in pytorch_key:
                return f'Transformer/encoderblock_{block_num}/MultiHeadDotProductAttention_1/out/bias'
            elif 'mlp.fc1.weight' in pytorch_key:
                return f'Transformer/encoderblock_{block_num}/MlpBlock_3/Dense_0/kernel'
            elif 'mlp.fc1.bias' in pytorch_key:
                return f'Transformer/encoderblock_{block_num}/MlpBlock_3/Dense_0/bias'
            elif 'mlp.fc2.weight' in pytorch_key:
                return f'Transformer/encoderblock_{block_num}/MlpBlock_3/Dense_1/kernel'
            elif 'mlp.fc2.bias' in pytorch_key:
                return f'Transformer/encoderblock_{block_num}/MlpBlock_3/Dense_1/bias'
        
        return None
    
    def convert_weight_shape(weight, pytorch_key, npz_key):
        """가중치 형태를 PyTorch 형식으로 변환"""
        weight = torch.from_numpy(weight)
        
        # Patch embedding: JAX (H, W, In, Out) -> PyTorch (Out, In, H, W)
        if pytorch_key == 'patch_embed.proj.weight':
            weight = weight.permute(3, 2, 0, 1)
        
        # Linear layers: JAX (In, Out) -> PyTorch (Out, In)
        elif any(x in pytorch_key for x in ['head.weight', 'attn.proj.weight', 'mlp.fc1.weight', 'mlp.fc2.weight']):
            if len(weight.shape) == 2:
                weight = weight.T
        
        # QKV weights: special handling for attention
        elif 'attn.qkv.weight' in pytorch_key:
            if len(weight.shape) == 3:  # (3, dim, dim) format
                weight = weight.reshape(-1, weight.shape[-1]).T
            elif len(weight.shape) == 2:
                weight = weight.T
        
        return weight
    
    # 자동으로 모든 키 매핑 시도
    matched_count = 0
    total_count = len(new_state_dict)
    
    for pytorch_key in list(new_state_dict.keys()):
        npz_key = convert_npz_to_pytorch_key(pytorch_key)
        
        if npz_key and npz_key in npz_weights:
            try:
                weight = convert_weight_shape(npz_weights[npz_key], pytorch_key, npz_key)
                
                # 형태가 맞는지 확인
                if weight.shape == new_state_dict[pytorch_key].shape:
                    new_state_dict[pytorch_key] = weight
                    matched_count += 1
            except Exception:
                continue
    
    # 최종적으로 변환된 가중치를 모델에 로드
    model.load_state_dict(new_state_dict, strict=False)
    
    # NPZ 파일 정리 (메모리 절약)
    npz_weights.close()
    
    return model

# 사용 예시:
# model = timm.create_model(
#     'vit_base_patch16_224',
#     pretrained=False,
#     num_classes=15,
#     drop_rate=0.0,
#     drop_path_rate=0.0
# )
# 
# # 방법 1: default_cfgs를 자동으로 import하여 사용
# model = load_weights_from_npz(
#     model=model,
#     model_name='vit_base_patch16_224'
# )
#
# # 방법 2: default_cfgs를 직접 전달
# from patchcore.vision_transformer import default_cfgs
# model = load_weights_from_npz(
#     model=model,
#     model_name='vit_base_patch16_224',
#     default_cfgs=default_cfgs
# )