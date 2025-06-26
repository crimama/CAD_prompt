"""
UCAD with LoRA implementation - replaces E-Prompt with LoRA for continual learning
"""
import contextlib
import logging
import os
import sys
import random
import pickle
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
import yaml
import pandas as pd
import json
import time
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.ndimage import label
from bisect import bisect

import patchcore.patchcore
import patchcore.utils
import patchcore.sampler  # Add sampler import
import patchcore.vit_lora  # Import LoRA implementation
from patchcore.datasets.mvtec import DatasetSplit  # Import DatasetSplit enum

LOGGER = logging.getLogger(__name__)

_DATASETS = {
    "mvtec": ["patchcore.datasets.mvtec", "MVTecDataset"]
}


def create_experiment_directories(args):
    """실험 디렉토리 구조 생성"""
    # results/{dataset}/exp_name/seed/* 구조
    if len(args.subdatasets) == 1:
        dataset_name = args.subdatasets[0]
    else:
        dataset_name = f"mvtec_{len(args.subdatasets)}classes_lora"
    
    exp_name = args.exp_name if hasattr(args, 'exp_name') and args.exp_name else f"lora_exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 기본 실험 디렉토리
    experiment_dir = os.path.join(args.results_path, dataset_name, exp_name, f"seed_{args.seed}")
    
    # 하위 디렉토리들 생성
    subdirs = {
        'logs': os.path.join(experiment_dir, 'logs'),
        'results': os.path.join(experiment_dir, 'results'), 
        'checkpoints': os.path.join(experiment_dir, 'checkpoints'),
        'configs': os.path.join(experiment_dir, 'configs')
    }
    
    for dir_path in subdirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return experiment_dir, subdirs


def save_experiment_config(args, config_dir):
    """실험 설정을 YAML 파일로 저장"""
    config_dict = vars(args).copy()
    
    # 저장하기 어려운 객체들 제거 또는 변환
    config_dict.pop('device', None)
    
    config_path = os.path.join(config_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=True)
    
    return config_path


def setup_experiment_logger(log_dir, exp_name):
    """실험 전용 로거 설정"""
    # 로그 파일 경로
    log_path = os.path.join(log_dir, f'{exp_name}_lora_training.log')
    
    # 실험 로거 생성
    exp_logger = logging.getLogger('lora_experiment_logger')
    exp_logger.setLevel(logging.INFO)
    
    # 기존 핸들러 제거 (중복 방지)
    for handler in exp_logger.handlers[:]:
        exp_logger.removeHandler(handler)
    
    # 파일 핸들러 추가
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 콘솔 핸들러 추가
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # 포맷터 설정
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                                datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    exp_logger.addHandler(file_handler)
    exp_logger.addHandler(console_handler)
    exp_logger.propagate = False  # 상위 로거로 전파 방지
    
    return exp_logger, log_path


class LoRAResultManager:
    """LoRA 결과 관리 클래스"""
    
    def __init__(self, results_dir):
        self.results_dir = results_dir
        self.result_csv_path = os.path.join(results_dir, 'result.csv')
        self.final_csv_path = os.path.join(results_dir, 'Final.csv')
        
        # CSV 컬럼 정의 - run_ucad_simple.py와 동일하게 수정
        self.result_columns = [
            'dataset_name', 'epoch', 'split_type', 'image_auroc', 'pixel_auroc', 
            'image_ap', 'pixel_ap', 'pixel_pro', 'time_cost'
        ]
        
        self.final_columns = [
            'test_task_id', 'test_task_name', 'predicted_task_id', 'predicted_task_name',
            'task_prediction_accuracy', 'image_auroc', 'pixel_auroc'
        ]
        
        # CSV 파일 초기화
        self._initialize_csv_files()
    
    def _initialize_csv_files(self):
        """CSV 파일 초기화"""
        # result.csv 초기화 - run_ucad_simple.py와 동일한 컬럼으로
        if not os.path.exists(self.result_csv_path):
            df_result = pd.DataFrame(columns=self.result_columns)
            df_result.to_csv(self.result_csv_path, index=False)
        
        # Final.csv 초기화  
        if not os.path.exists(self.final_csv_path):
            df_final = pd.DataFrame(columns=self.final_columns)
            df_final.to_csv(self.final_csv_path, index=False)
    
    def save_epoch_result(self, dataset_name, epoch, split_type, metrics):
        """에포크별 결과 저장 - run_ucad_simple.py와 동일한 형식"""
        new_row = {
            'dataset_name': dataset_name,
            'epoch': epoch,
            'split_type': split_type,  # 'limited_memory' or 'unlimited_memory'
            'image_auroc': metrics['auroc'],
            'pixel_auroc': metrics['full_pixel_auroc'],
            'image_ap': metrics['img_ap'],
            'pixel_ap': metrics['pixel_ap'],
            'pixel_pro': metrics['pixel_pro'],
            'time_cost': metrics['time_cost']
        }
        
        # CSV 파일에 추가
        df = pd.DataFrame([new_row])
        df.to_csv(self.result_csv_path, mode='a', header=False, index=False)
    
    def save_final_result(self, test_task_id, test_task_name, predicted_task_id, 
                         predicted_task_name, task_accuracy, image_auroc, pixel_auroc):
        """최종 inference 결과 저장"""
        new_row = {
            'test_task_id': test_task_id,
            'test_task_name': test_task_name,
            'predicted_task_id': predicted_task_id,
            'predicted_task_name': predicted_task_name,
            'task_prediction_accuracy': task_accuracy,
            'image_auroc': image_auroc,
            'pixel_auroc': pixel_auroc
        }
        
        # CSV 파일에 추가
        df = pd.DataFrame([new_row])
        df.to_csv(self.final_csv_path, mode='a', header=False, index=False)
    
    def get_best_results_summary(self):
        """최고 성능 결과 요약 반환 - run_ucad_simple.py와 동일"""
        if not os.path.exists(self.result_csv_path):
            return None
            
        df = pd.read_csv(self.result_csv_path)
        if df.empty:
            return None
        
        # 각 데이터셋별 최고 성능 추출
        best_results = []
        for dataset in df['dataset_name'].unique():
            dataset_df = df[df['dataset_name'] == dataset]
            
            # limited_memory와 unlimited_memory 별로 최고 성능
            for split_type in ['limited_memory', 'unlimited_memory']:
                split_df = dataset_df[dataset_df['split_type'] == split_type]
                if not split_df.empty:
                    best_idx = split_df['image_auroc'].idxmax()
                    best_row = split_df.loc[best_idx].to_dict()
                    best_results.append(best_row)
        
        return best_results


class LoRACheckpointManager:
    """LoRA 체크포인트 관리 클래스"""
    
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        self.task_models = {}
        self.key_features = {}
        
    def save_task_checkpoint(self, task_id, task_name, model, key_features, metrics):
        """태스크별 체크포인트 저장"""
        # LoRA 가중치 저장
        lora_path = os.path.join(self.checkpoint_dir, f'task_{task_id}_{task_name}_lora.pth')
        if hasattr(model, 'save_lora_weights'):
            model.save_lora_weights(lora_path)
        else:
            # 전체 모델 저장 (fallback)
            torch.save(model.state_dict(), lora_path)
        
        # 키 특징 저장
        key_features_path = os.path.join(self.checkpoint_dir, f'task_{task_id}_{task_name}_key_features.pkl')
        with open(key_features_path, 'wb') as f:
            pickle.dump(key_features, f)
        
        # 메트릭 정보 저장
        metrics_path = os.path.join(self.checkpoint_dir, f'task_{task_id}_{task_name}_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # 메모리에도 저장
        self.task_models[task_id] = lora_path
        self.key_features[task_id] = key_features
        
        return {
            'lora_path': lora_path,
            'key_features_path': key_features_path,
            'metrics_path': metrics_path
        }
    
    def load_task_checkpoint(self, task_id, task_name):
        """태스크별 체크포인트 로드"""
        lora_path = os.path.join(self.checkpoint_dir, f'task_{task_id}_{task_name}_lora.pth')
        key_features_path = os.path.join(self.checkpoint_dir, f'task_{task_id}_{task_name}_key_features.pkl')
        
        try:
            # LoRA 가중치 로드
            lora_weights = torch.load(lora_path, map_location='cpu')
            
            # 키 특징 로드
            with open(key_features_path, 'rb') as f:
                key_features = pickle.load(f)
                
            return lora_weights, key_features
        except FileNotFoundError:
            return None, None


def parse_arguments():
    parser = argparse.ArgumentParser(description='UCAD with LoRA for continual learning')
    
    # Basic arguments
    parser.add_argument("results_path", type=str, help="결과 저장 경로")
    parser.add_argument("--gpu", type=int, nargs="+", default=[0], help="사용할 GPU ID")
    parser.add_argument("--seed", type=int, default=0, help="랜덤 시드")
    parser.add_argument("--exp_name", type=str, default=None, help="실험 이름 (없으면 자동 생성)")
    
    # LoRA specific arguments
    parser.add_argument("--lora_rank", type=int, default=4, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=1, help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout rate")
    parser.add_argument("--num_tasks", type=int, default=15, help="Number of tasks for continual learning")
    
    # Continual learning arguments
    parser.add_argument("--memory_size", type=int, default=196, help="메모리 뱅크 크기")
    parser.add_argument("--epochs_num", type=int, default=25, help="훈련 에포크 수")
    parser.add_argument("--key_size", type=int, default=196, help="키 특징 크기")
    parser.add_argument("--basic_size", type=int, default=1960, help="기본 메모리 크기")
    
    # Logging arguments
    parser.add_argument("--log_group", type=str, default="lora_test", help="로그 그룹명")
    parser.add_argument("--log_project", type=str, default="MVTecAD_LoRA", help="로그 프로젝트명")
    
    # Model arguments
    parser.add_argument("--pretrain_embed_dimension", type=int, default=1024)
    parser.add_argument("--target_embed_dimension", type=int, default=1024)
    parser.add_argument("--anomaly_scorer_num_nn", type=int, default=5)
    parser.add_argument("--patchsize", type=int, default=3)
    
    # Data arguments
    parser.add_argument("--dataset_name", type=str, choices=list(_DATASETS.keys()), default="mvtec")
    parser.add_argument("--data_path", type=str, required=True, help="데이터셋 경로")
    parser.add_argument("--subdatasets", type=str, nargs="+", help="사용할 서브데이터셋")
    parser.add_argument("--train_val_split", type=float, default=1.0, help="훈련/검증 분할 비율")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--resize", type=int, default=224, help="리사이즈 크기")
    parser.add_argument("--image_size", type=int, default=224, help="이미지 크기")
    parser.add_argument("--imagesize", type=int, default=224, help="이미지 크기 (호환성)")
    parser.add_argument("--augment", action="store_true", help="데이터 증강 사용")
    
    # PatchCore arguments
    parser.add_argument("--sampler_name", type=str, default="approx_greedy_coreset")
    parser.add_argument("--percentage", type=float, default=0.1)
    
    # FAISS arguments
    parser.add_argument("--faiss_on_gpu", action="store_true", help="FAISS GPU 사용")
    parser.add_argument("--faiss_num_workers", type=int, default=8)
    
    # Save options
    parser.add_argument("--save_segmentation_images", action="store_true")
    parser.add_argument("--save_patchcore_model", action="store_true")
    
    return parser.parse_args()


class UCADLoRATrainer:
    """UCAD Trainer with LoRA for continual learning"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f"cuda:{args.gpu[0]}" if torch.cuda.is_available() else "cpu")
        
        # 실험 디렉토리 구조 생성
        self.experiment_dir, self.subdirs = create_experiment_directories(args)
        
        # 실험 설정 저장
        self.config_path = save_experiment_config(args, self.subdirs['configs'])
        
        # 로거 설정
        exp_name = os.path.basename(os.path.dirname(self.experiment_dir))
        self.exp_logger, self.log_path = setup_experiment_logger(self.subdirs['logs'], exp_name)
        
        # 결과 관리자 초기화
        self.result_manager = LoRAResultManager(self.subdirs['results'])
        
        # 체크포인트 관리자 초기화
        self.checkpoint_manager = LoRACheckpointManager(self.subdirs['checkpoints'])
        
        # 실험 시작 로그
        self.exp_logger.info("="*80)
        self.exp_logger.info(f"UCAD-LoRA Experiment Started")
        self.exp_logger.info(f"Experiment Directory: {self.experiment_dir}")
        self.exp_logger.info(f"Config saved to: {self.config_path}")
        self.exp_logger.info("="*80)
        
        # Task management
        self.task_id_to_name = {}
        self.current_task_id = 0
        
        # Memory management for continual learning
        self.memory_features = {}
        self.key_features = {}
        self.task_memories = {}
        
        # PatchCore models for each task
        self.patchcore_list = []
        
        print(f"🚀 UCAD-LoRA Trainer initialized")
        print(f"   Device: {self.device}")
        print(f"   LoRA Config: rank={args.lora_rank}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
        print(f"   Tasks: {args.num_tasks}")
        print(f"   Experiment Directory: {self.experiment_dir}")
        
    def create_model(self):
        """Create ViT model with LoRA"""
        from patchcore.vit_lora import create_vit_lora_model
        
        model = create_vit_lora_model(
            model_name='vit_base_patch16_224',
            num_tasks=self.args.num_tasks,
            lora_rank=self.args.lora_rank,
            lora_alpha=self.args.lora_alpha,
            lora_dropout=self.args.lora_dropout,
            pretrained=True
        )
        
        return model.to(self.device)
        
    def create_patchcore_model(self, task_name):
        """Create PatchCore model for a task"""
        
        # Create ViT model with LoRA
        model = self.create_model()
        
        # Create PatchCore with LoRA-enabled ViT
        patchcore_model = patchcore.patchcore.PatchCore(self.device)
        patchcore_model.load(
            backbone=None,  # No backbone needed for LoRA
            layers_to_extract_from=None,  # Not used with ViT
            device=self.device,
            input_shape=(3, self.args.image_size, self.args.image_size),
            pretrain_embed_dimension=self.args.pretrain_embed_dimension,
            target_embed_dimension=self.args.target_embed_dimension,
            patchsize=self.args.patchsize,
            anomaly_score_num_nn=self.args.anomaly_scorer_num_nn,
            featuresampler=patchcore.sampler.ApproximateGreedyCoresetSampler(self.args.percentage, self.device),
        )
        
        # Replace the model with our LoRA model
        patchcore_model.model = model
        patchcore_model.current_task_id = self.current_task_id  # Store current task ID
        
        # Override the _embed method to work with LoRA model
        def _embed_lora(self, images, detach=True, provide_patch_shapes=False):
            """Custom embed method for LoRA model"""
            def _detach(features):
                if detach:
                    return [x.detach().cpu().numpy() for x in features]
                return features

            with torch.no_grad():
                # Get features from LoRA model using current task ID
                task_id = getattr(self, 'current_task_id', 0)
                features_dict = self.model.forward_features(images, task_id=task_id, inference_mode=False)
                features = features_dict['seg_feat']
                for i in range(len(features)):
                    features[i] = features[i].reshape(-1, 14, 14, 768).permute(0, 3, 1, 2)

            features = [
                self.patch_maker.patchify(x, return_spatial_info=True) for x in features
            ]
            patch_shapes = [x[1] for x in features]
            features = [x[0] for x in features]
            
            # Process features through preprocessing and aggregator
            features = self.forward_modules["preprocessing"](features)
            features = self.forward_modules["preadapt_aggregator"](features)

            if provide_patch_shapes:
                return _detach(features), patch_shapes
            return _detach(features)
        
        # Bind the custom method
        import types
        patchcore_model._embed = types.MethodType(_embed_lora, patchcore_model)
        
        # Also override _predict method for inference
        def _predict_lora(self, images):
            """Custom predict method for LoRA model"""
            with torch.no_grad():
                # Ensure images are on the right device
                images = images.to(self.device)
                batchsize = images.shape[0]
                task_id = getattr(self, 'current_task_id', 0)
                features_dict = self.model.forward_features(images, task_id=task_id, inference_mode=False)
                features = features_dict['seg_feat']
                for i in range(len(features)):
                    features[i] = features[i].reshape(-1, 14, 14, 768).permute(0, 3, 1, 2)

            features = [
                self.patch_maker.patchify(x, return_spatial_info=True) for x in features
            ]
            patch_shapes = [x[1] for x in features]
            features = [x[0] for x in features]

            # Process features
            features = self.forward_modules["preprocessing"](features)
            features = self.forward_modules["preadapt_aggregator"](features)

            # Convert to right format for scoring - handle tensor lists properly
            if isinstance(features, list) and len(features) > 0:
                if isinstance(features[0], torch.Tensor):
                    features = [f.detach().cpu().numpy() for f in features]
                features = np.concatenate(features, axis=0)
            elif isinstance(features, torch.Tensor):
                features = features.detach().cpu().numpy()
            else:
                features = np.array(features)
            
            # Get anomaly scores
            patch_scores = self.anomaly_scorer.predict([features])[0]
            
            # Reshape patch scores correctly
            n_patches = patch_shapes[0][0] * patch_shapes[0][1]
            if len(patch_scores) == n_patches:
                scales = patch_shapes[0]
            else:
                # If size mismatch, use sqrt approximation
                import math
                side_length = int(math.sqrt(len(patch_scores)))
                scales = (side_length, side_length)
                
            patch_scores = patch_scores.reshape(scales[0], scales[1])
            
            # Generate segmentation maps - convert to proper format for interpolation
            # Add batch and channel dimensions for interpolation
            patch_scores_tensor = torch.from_numpy(patch_scores).unsqueeze(0).unsqueeze(0).float()
            if hasattr(self, 'device'):
                patch_scores_tensor = patch_scores_tensor.to(self.device)
            
            # Use F.interpolate directly instead of anomaly_segmentor
            anomaly_map = F.interpolate(
                patch_scores_tensor,
                size=(224, 224),  # Target image size
                mode='bilinear',
                align_corners=False
            )
            anomaly_map = anomaly_map.squeeze(0).squeeze(0).cpu().numpy()
            
            # Convert to image-level score (max of patch scores)
            anomaly_score = float(np.max(patch_scores))
            
            return [anomaly_score], [anomaly_map]
        
        patchcore_model._predict = types.MethodType(_predict_lora, patchcore_model)
        
        return patchcore_model
        
    def train_task(self, task_id, task_name, dataloaders):
        """Train on a specific task - updated to match run_ucad_simple.py process"""
        dataset_name = dataloaders["training"].name
        
        # 데이터셋 시작 로그
        self.log_dataset_start(dataset_name, task_id, len(self.args.subdatasets))
        
        # Set current task ID
        self.current_task_id = task_id
        
        # Create new PatchCore for this task
        patchcore_model = self.create_patchcore_model(task_name)
        
        # Set LoRA to current task
        if hasattr(patchcore_model.model, 'set_task'):
            patchcore_model.model.set_task(task_id)
        
        # 키 특징 추출 (run_ucad_simple.py와 동일한 방식)
        patchcore_model.set_dataloadercount(task_id)
        key_feature = patchcore_model.fit_with_limit_size(dataloaders["training"], self.args.key_size)
        self.key_features[task_id] = key_feature

        # 훈련 설정 (run_ucad_simple.py와 동일)
        args_dict = np.load('./args_dict.npy', allow_pickle=True).item()
        args_dict.lr = 0.0005
        args_dict.decay_epochs = 15
        args_dict.warmup_epochs = 3
        args_dict.cooldown_epochs = 5
        args_dict.patience_epochs = 5
        
        aggregator = {"scores": [], "segmentations": []}
        basic_aggregator = {"scores": [], "segmentations": []}
        start_time = time.time()
        pr_auroc = 0
        basic_pr_auroc = 0
        
        optimizer = create_optimizer(args_dict, patchcore_model.prompt_model)
        epochs = self.args.epochs_num
        patchcore_model.prompt_model.train()
        patchcore_model.prompt_model.train_contrastive = True
        
        if args_dict.sched != 'constant':
            lr_scheduler, _ = create_scheduler(args_dict, optimizer)
        else:
            lr_scheduler = None
            
        best_metrics = {
            'auroc': 0, 'full_pixel_auroc': 0, 'img_ap': 0,
            'pixel_ap': 0, 'pixel_pro': 0, 'time_cost': 0
        }
        best_basic_metrics = {
            'auroc': 0, 'full_pixel_auroc': 0, 'img_ap': 0,
            'pixel_ap': 0, 'pixel_pro': 0, 'time_cost': 0
        }

        # 에포크별 훈련 (run_ucad_simple.py와 동일한 구조)
        print(f"\n{'='*60}")
        print(f"Training Task {task_id}: {dataset_name}")
        print(f"Total Epochs: {epochs}")
        print(f"{'='*60}")
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            torch.cuda.empty_cache()
            
            # 훈련 단계
            patchcore_model.prompt_model.train()
            loss_list = []
                                    
            batch_count = 0
            total_batches = len(dataloaders["training"])
            
            for image in dataloaders["training"]:
                if isinstance(image, dict):
                    image_paths = image["image_path"]
                    image = image["image"].cuda()
                
                res = patchcore_model._embed_train_sam(image, provide_patch_shapes=True, image_path=image_paths)
                loss = res['loss']
                loss_list.append(loss.item())
                
                optimizer.zero_grad()
                if loss != 0:
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(patchcore_model.prompt_model.parameters(), args_dict.clip_grad)
                optimizer.step()
                
                batch_count += 1
                # 10% 단위로 진행률 표시
                if batch_count % max(1, total_batches // 10) == 0:
                    progress = int(batch_count / total_batches * 100)
                    print(f"{progress}%", end=" ", flush=True)
            
            epoch_train_time = time.time() - epoch_start_time
            avg_loss = np.mean(loss_list)
            print(f"Done! (Loss: {avg_loss:.4f}, Time: {epoch_train_time:.1f}s)")
            
            if lr_scheduler:
                lr_scheduler.step(epoch)
            
            # 평가 단계
            eval_start_time = time.time()
            patchcore_model.prompt_model.eval()
            
            print(f"Epoch {epoch+1}/{epochs} - Evaluating...", end=" ", flush=True)
            
            # 무제한 메모리 평가
            nolimimit_memory_feature = patchcore_model.fit_with_limit_size_prompt(dataloaders["training"], self.args.basic_size)
            patchcore_model.anomaly_scorer.fit(detection_features=[nolimimit_memory_feature])
            basic_scores, basic_segmentations, basic_labels_gt, basic_masks_gt = patchcore_model.predict_prompt(
                dataloaders["testing"]
            )
            basic_aggregator["scores"].append(basic_scores)
            basic_aggregator["segmentations"].append(basic_segmentations)
            basic_end_time = time.time()
            
            # 제한된 메모리 평가
            memory_feature = patchcore_model.fit_with_limit_size_prompt(dataloaders["training"], self.args.memory_size)
            patchcore_model.anomaly_scorer.fit(detection_features=[memory_feature])
            scores, segmentations, labels_gt, masks_gt = patchcore_model.predict_prompt(
                dataloaders["testing"]
            )
            aggregator["scores"].append(scores)
            aggregator["segmentations"].append(segmentations)
            end_time = time.time()
            
            eval_time = end_time - eval_start_time
            print(f"Done! (Time: {eval_time:.1f}s)")
            
            # 메트릭 계산 및 업데이트
            current_metrics = self.compute_metrics(aggregator, masks_gt, basic_end_time, end_time, dataloaders)
            current_basic_metrics = self.compute_basic_metrics(basic_aggregator, basic_masks_gt, start_time, basic_end_time, dataloaders)
            
            # 성능 변화 표시
            print(f"\n📊 Epoch {epoch+1} Results:")
            print(f"   Limited Memory   - AUROC: {current_metrics['auroc']:.4f}, Pixel AUROC: {current_metrics['full_pixel_auroc']:.4f}, PRO: {current_metrics['pixel_pro']:.4f}")
            print(f"   Unlimited Memory - AUROC: {current_basic_metrics['auroc']:.4f}, Pixel AUROC: {current_basic_metrics['full_pixel_auroc']:.4f}, PRO: {current_basic_metrics['pixel_pro']:.4f}")
            
            # 모든 에포크 결과 저장 (CSV에 누적)
            self.result_manager.save_epoch_result(dataset_name, epoch + 1, 'limited_memory', current_metrics)
            self.result_manager.save_epoch_result(dataset_name, epoch + 1, 'unlimited_memory', current_basic_metrics)
            
            # 최고 성능 업데이트 및 체크포인트 저장
            improved = False
            if current_metrics['auroc'] > pr_auroc:
                pr_auroc = current_metrics['auroc']
                best_metrics = current_metrics.copy()
                improved = True
                
                # 베스트 체크포인트 저장
                checkpoint_paths = self.checkpoint_manager.save_task_checkpoint(
                    task_id, 
                    dataset_name,
                    patchcore_model.model,
                    key_feature,
                    current_metrics
                )
                
                self.exp_logger.info(f"  🎯 Best checkpoint saved for Task {task_id}: {dataset_name}")
                self.exp_logger.info(f"     LoRA weights: {checkpoint_paths['lora_path']}")
                self.exp_logger.info(f"     Key features: {checkpoint_paths['key_features_path']}")
                self.exp_logger.info(f"     Metrics: {checkpoint_paths['metrics_path']}")
                
                if current_metrics['auroc'] == 1:
                    print(f"🎯 Perfect AUROC achieved! Early stopping.")
                    break
            
            if current_basic_metrics['auroc'] > basic_pr_auroc:
                basic_pr_auroc = current_basic_metrics['auroc']
                best_basic_metrics = current_basic_metrics.copy()
            
            # 성능 로그 기록
            self.log_epoch_performance(epoch, epochs, dataset_name, current_metrics, current_basic_metrics, improved)
            
            if improved:
                print(f"🚀 New best Limited Memory AUROC: {pr_auroc:.4f} (↑)")
            else:
                print(f"   Best Limited Memory AUROC remains: {pr_auroc:.4f}")
            
            print(f"   Total epoch time: {time.time() - epoch_start_time:.1f}s")
            print(f"{'-'*60}")
        
        # Store model and results
        self.patchcore_list.append(patchcore_model)
        self.task_id_to_name[task_id] = task_name
        
        # 최종 결과 출력 및 로그
        print(f"\n🏆 Final Results for Task {task_id}: {dataset_name}:")
        self.print_final_results(task_id, best_metrics, best_basic_metrics)
        self.log_final_dataset_results(dataset_name, best_metrics, best_basic_metrics)

    def inference(self):
        """Perform inference on all tasks"""
        print(f"\n{'='*80}")
        print("🔍 Starting LoRA inference phase...")
        print(f"{'='*80}")
        
        self.exp_logger.info("="*80)
        self.exp_logger.info("LORA CONTINUAL LEARNING INFERENCE STARTED")
        self.exp_logger.info("="*80)
        
        if not self.patchcore_list:
            print("❌ No trained models found!")
            self.exp_logger.error("No trained models found for inference!")
            return
            
        # Load dataset for inference
        dataset_info = _DATASETS[self.args.dataset_name]
        dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])
        
        total_results = {}
        
        for test_task_id, (patchcore_model, task_name) in enumerate(zip(self.patchcore_list, self.task_id_to_name.values())):
            print(f"\n📊 Testing on Task {test_task_id}: {task_name}")
            self.exp_logger.info(f"Testing on Task {test_task_id}: {task_name}")
            
            # Create test dataset
            dataset = dataset_library.__dict__[dataset_info[1]](
                source=self.args.data_path,
                classname=task_name,
                resize=self.args.image_size,
                imagesize=self.args.image_size,
                split=DatasetSplit.TEST,
                train_val_split=1.0,
                augment=False,
                seed=self.args.seed
            )
            
            dataloader_test = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.num_workers,
                pin_memory=True,
            )
            
            # Task selection and inference
            task_predictions = []
            task_similarities = []
            
            print("   🎯 Task prediction phase...", end=" ", flush=True)
            
            for batch_idx, batch_data in enumerate(dataloader_test):
                images = batch_data["image"].to(self.device)
                
                # Try to predict task using LoRA similarity
                with torch.no_grad():
                    # Get features for task selection
                    temp_features = patchcore_model.model.forward_features(images, task_id=0, inference_mode=False)
                    task_similarity = patchcore_model.model.get_task_similarity(temp_features['x'])
                    
                    if task_similarity is not None:
                        predicted_task = torch.argmax(task_similarity, dim=1)
                        task_predictions.extend(predicted_task.cpu().numpy())
                        task_similarities.extend(task_similarity.cpu().numpy())
                    else:
                        task_predictions.extend([test_task_id] * images.shape[0])
                        task_similarities.extend([[1.0] * self.args.num_tasks] * images.shape[0])
            
            print("Done!")
            
            # Calculate task prediction accuracy
            correct_predictions = sum(1 for pred in task_predictions if pred == test_task_id)
            task_accuracy = correct_predictions / len(task_predictions) if task_predictions else 0.0
            
            # 가장 많이 예측된 태스크 찾기
            if task_predictions:
                predicted_task_counts = {}
                for pred in task_predictions:
                    predicted_task_counts[pred] = predicted_task_counts.get(pred, 0) + 1
                most_predicted_task_id = max(predicted_task_counts, key=predicted_task_counts.get)
                most_predicted_task_name = self.task_id_to_name.get(most_predicted_task_id, f"task_{most_predicted_task_id}")
            else:
                most_predicted_task_id = test_task_id
                most_predicted_task_name = task_name
            
            print(f"   📈 Task Prediction Accuracy: {task_accuracy:.3f}")
            print(f"   🎯 Most Predicted Task: {most_predicted_task_name} (ID: {most_predicted_task_id})")
            
            # Perform PatchCore inference
            print("   🔮 Anomaly detection inference...", end=" ", flush=True)
            scores, segmentations, labels_gt, masks_gt = patchcore_model.predict(dataloader_test)
            print("Done!")
            
            # Calculate metrics
            print("   📊 Computing metrics...", end=" ", flush=True)
            from patchcore.metrics import compute_imagewise_retrieval_metrics, compute_pixelwise_retrieval_metrics
            
            auroc = compute_imagewise_retrieval_metrics(scores, labels_gt)["auroc"]
            
            if masks_gt is not None:
                auroc_px = compute_pixelwise_retrieval_metrics(segmentations, masks_gt)["auroc"]
            else:
                auroc_px = 0.0
            
            print("Done!")
                
            results = {
                "task_name": task_name,
                "task_id": test_task_id,
                "predicted_task_id": most_predicted_task_id,
                "predicted_task_name": most_predicted_task_name,
                "auroc": auroc,
                "auroc_px": auroc_px,
                "task_accuracy": task_accuracy
            }
            
            total_results[task_name] = results
            
            # Final.csv에 결과 저장
            self.result_manager.save_final_result(
                test_task_id, task_name, most_predicted_task_id, 
                most_predicted_task_name, task_accuracy, auroc, auroc_px
            )
            
            # 결과 출력
            print(f"   📊 Results for {task_name}:")
            print(f"      🎯 Predicted Task: {most_predicted_task_name}")
            print(f"      📈 Image AUROC: {auroc:.3f}")
            print(f"      📈 Pixel AUROC: {auroc_px:.3f}")
            print(f"      🎯 Task Accuracy: {task_accuracy:.3f}")
            
            # 로그 기록
            self.exp_logger.info(f"Inference results for {task_name}:")
            self.exp_logger.info(f"  Predicted Task: {most_predicted_task_name} (ID: {most_predicted_task_id})")
            self.exp_logger.info(f"  Image AUROC: {auroc:.3f}")
            self.exp_logger.info(f"  Pixel AUROC: {auroc_px:.3f}")
            self.exp_logger.info(f"  Task Prediction Accuracy: {task_accuracy:.3f}")
            
        # Print summary
        self.log_inference_summary(total_results)
        
        # Save results
        results_file = os.path.join(self.subdirs['results'], "lora_results.pkl")
        with open(results_file, "wb") as f:
            pickle.dump(total_results, f)
            
        self.exp_logger.info(f"Results saved to: {results_file}")
        print(f"\n💾 Results saved to: {results_file}")
        
        return total_results
    
    def log_inference_summary(self, total_results):
        """Inference 결과 요약 로그"""
        print("\n📋 Final Results Summary:")
        print("=" * 80)
        
        self.exp_logger.info("="*80)
        self.exp_logger.info("LORA INFERENCE SUMMARY")
        self.exp_logger.info("="*80)
        
        auroc_scores = [r["auroc"] for r in total_results.values()]
        auroc_px_scores = [r["auroc_px"] for r in total_results.values()]
        task_accuracies = [r["task_accuracy"] for r in total_results.values()]
        
        # 태스크 선택 통계
        task_selection_stats = {}
        for result in total_results.values():
            predicted = result["predicted_task_name"]
            task_selection_stats[predicted] = task_selection_stats.get(predicted, 0) + 1
        
        print("🎯 Task Selection Statistics:")
        self.exp_logger.info("Task Selection Statistics:")
        for task, count in task_selection_stats.items():
            print(f"   {task}: {count} times")
            self.exp_logger.info(f"  {task}: {count} times")
        
        print(f"\n📊 Performance Summary:")
        print(f"   Average Image AUROC: {np.mean(auroc_scores):.3f} ± {np.std(auroc_scores):.3f}")
        print(f"   Average Pixel AUROC: {np.mean(auroc_px_scores):.3f} ± {np.std(auroc_px_scores):.3f}")
        print(f"   Average Task Accuracy: {np.mean(task_accuracies):.3f} ± {np.std(task_accuracies):.3f}")
        
        # 개별 결과 출력
        print(f"\n📝 Detailed Results:")
        for task_name, result in total_results.items():
            print(f"   {task_name:15s} -> {result['predicted_task_name']:15s} | "
                  f"AUROC: {result['auroc']:.3f}, Pixel AUROC: {result['auroc_px']:.3f}, "
                  f"Task Acc: {result['task_accuracy']:.3f}")
        
        # 로그에도 기록
        self.exp_logger.info(f"Average Image AUROC: {np.mean(auroc_scores):.3f} ± {np.std(auroc_scores):.3f}")
        self.exp_logger.info(f"Average Pixel AUROC: {np.mean(auroc_px_scores):.3f} ± {np.std(auroc_px_scores):.3f}")
        self.exp_logger.info(f"Average Task Accuracy: {np.mean(task_accuracies):.3f} ± {np.std(task_accuracies):.3f}")
        
        self.exp_logger.info("Detailed Results:")
        for task_name, result in total_results.items():
            self.exp_logger.info(f"  {task_name:15s} -> {result['predicted_task_name']:15s} | "
                               f"AUROC: {result['auroc']:.3f}, Pixel AUROC: {result['auroc_px']:.3f}, "
                               f"Task Acc: {result['task_accuracy']:.3f}")
        
        self.exp_logger.info("="*80)

    def run(self):
        """Main training loop - updated to match run_ucad_simple.py structure"""
        print("🎯 Starting UCAD-LoRA training...")
        
        # 실험 로거 초기화 로그
        self.exp_logger.info("UCAD-LoRA Training Started")
        self.exp_logger.info(f"LoRA Config: rank={self.args.lora_rank}, alpha={self.args.lora_alpha}, dropout={self.args.lora_dropout}")
        self.exp_logger.info(f"Number of Tasks: {len(self.args.subdatasets)}")
        self.exp_logger.info(f"Subdatasets: {self.args.subdatasets}")
        self.exp_logger.info("")
        
        # 데이터로더 생성 (run_ucad_simple.py와 동일한 방식)
        list_of_dataloaders = self.create_dataloaders()
        total_datasets = len(list_of_dataloaders)
        print(f"📂 총 {total_datasets}개 데이터셋 로드됨: {self.args.subdatasets}")
        
        print(f"\n{'='*80}")
        print(f"🎯 LoRA 훈련 설정")
        print(f"   - 실험 디렉토리: {self.experiment_dir}")
        print(f"   - LoRA Rank: {self.args.lora_rank}")
        print(f"   - LoRA Alpha: {self.args.lora_alpha}")
        print(f"   - 총 태스크 수: {len(self.args.subdatasets)}")
        print(f"   - 메모리 크기: {self.args.memory_size}")
        print(f"   - 에포크 수: {self.args.epochs_num}")
        print(f"   - 배치 크기: {self.args.batch_size}")
        print(f"   - 설정 파일: {self.config_path}")
        print(f"   - 로그 파일: {self.log_path}")
        print(f"   - 결과 CSV: {self.result_manager.result_csv_path}")
        print(f"   - 체크포인트: {self.checkpoint_manager.checkpoint_dir}")
        print(f"{'='*80}\n")
        
        # 각 태스크에 대해 훈련 (run_ucad_simple.py와 동일한 구조)
        for task_id, dataloaders in enumerate(list_of_dataloaders):
            task_progress = f"({task_id + 1}/{total_datasets})"
            print(f"\n🔥 Task {task_progress}: {dataloaders['training'].name}")
            
            LOGGER.info(
                "Training Task [{}] ({}/{})...".format(
                    dataloaders["training"].name,
                    task_id + 1,
                    len(list_of_dataloaders),
                )
            )

            # 시드 고정
            import patchcore.utils
            patchcore.utils.fix_seeds(self.args.seed, self.device)
            
            # GPU 메모리 정리
            torch.cuda.empty_cache()
            
            # 단일 태스크 훈련 (run_ucad_simple.py와 동일한 방식)
            self.train_task(task_id, dataloaders["training"].name.split("_")[-1], dataloaders)
                
            print(f"\n✅ Task {task_progress} 완료!")
            LOGGER.info("\n\n-----\n")
        
        # 훈련 완료 로그
        print(f"\n{'='*80}")
        print("📊 모든 태스크 훈련 완료! 최종 결과 저장 중...")
        
        self.exp_logger.info("="*80)
        self.exp_logger.info("ALL TASKS TRAINING COMPLETED")
        self.exp_logger.info(f"Total tasks trained: {len(self.args.subdatasets)}")
        self.exp_logger.info("="*80)
        
        # Perform inference
        self.inference()
        
        # 최종 완료 로그
        print(f"\n🎉 실험 완료! 모든 결과가 저장되었습니다:")
        print(f"   📁 실험 디렉토리: {self.experiment_dir}")
        print(f"   📊 훈련 결과: {self.result_manager.result_csv_path}")
        print(f"   📊 최종 결과: {self.result_manager.final_csv_path}")
        print(f"   💾 체크포인트: {self.checkpoint_manager.checkpoint_dir}")
        print(f"   📝 로그 파일: {self.log_path}")
        print(f"   ⚙️ 설정 파일: {self.config_path}")
        print(f"{'='*80}")
        
        self.exp_logger.info("="*80)
        self.exp_logger.info("UCAD-LORA EXPERIMENT COMPLETED SUCCESSFULLY")
        self.exp_logger.info(f"Experiment Directory: {self.experiment_dir}")
        self.exp_logger.info(f"Training Results: {self.result_manager.result_csv_path}")
        self.exp_logger.info(f"Final Results: {self.result_manager.final_csv_path}")
        self.exp_logger.info(f"Checkpoints: {self.checkpoint_manager.checkpoint_dir}")
        self.exp_logger.info("="*80)

    def create_dataloaders(self):
        """데이터로더 생성 (run_ucad_simple.py와 동일한 방식)"""
        dataset_info = _DATASETS[self.args.dataset_name]
        dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

        dataloaders = []
        for subdataset in self.args.subdatasets:
            train_dataset = dataset_library.__dict__[dataset_info[1]](
                self.args.data_path,
                classname=subdataset,
                resize=self.args.image_size,
                train_val_split=1.0,
                imagesize=self.args.image_size,
                split=DatasetSplit.TRAIN,
                seed=self.args.seed,
                augment=self.args.augment,
            )

            test_dataset = dataset_library.__dict__[dataset_info[1]](
                self.args.data_path,
                classname=subdataset,
                resize=self.args.image_size,
                imagesize=self.args.image_size,
                split=DatasetSplit.TEST,
                seed=self.args.seed,
            )

            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=self.args.num_workers,
                pin_memory=True,
            )

            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.num_workers,
                pin_memory=True,
            )

            train_dataloader.name = self.args.dataset_name
            if subdataset is not None:
                train_dataloader.name += "_" + subdataset

            dataloader_dict = {
                "training": train_dataloader,
                "validation": None,
                "testing": test_dataloader,
            }

            dataloaders.append(dataloader_dict)
        
        return dataloaders

    def compute_metrics(self, aggregator, masks_gt, basic_end_time, end_time, dataloaders):
        """메트릭 계산 (run_ucad_simple.py와 동일)"""
        scores = np.array(aggregator["scores"])
        min_scores = scores.min(axis=-1).reshape(-1, 1)
        max_scores = scores.max(axis=-1).reshape(-1, 1)
        scores = (scores - min_scores) / (max_scores - min_scores)
        scores = np.mean(scores, axis=0)
        
        segmentations = np.array(aggregator["segmentations"])
        min_scores = (
            segmentations.reshape(len(segmentations), -1)
            .min(axis=-1)
            .reshape(-1, 1, 1, 1)
        )
        max_scores = (
            segmentations.reshape(len(segmentations), -1)
            .max(axis=-1)
            .reshape(-1, 1, 1, 1)
        )
        segmentations = (segmentations - min_scores) / (max_scores - min_scores)
        segmentations = np.mean(segmentations, axis=0)
        
        time_cost = (end_time - basic_end_time) / len(dataloaders["testing"])
        anomaly_labels = [
            x[1] != "good" for x in dataloaders["testing"].dataset.data_to_iterate
        ]
        
        ap_seg = np.asarray(segmentations)
        ap_seg = ap_seg.flatten()
        
        import patchcore.metrics
        auroc = patchcore.metrics.compute_imagewise_retrieval_metrics(
            scores, anomaly_labels
        )["auroc"]
        
        ap_mask = np.asarray(masks_gt)
        ap_mask = ap_mask.flatten().astype(np.int32)
        from sklearn.metrics import average_precision_score
        pixel_ap = average_precision_score(ap_mask, ap_seg)
        img_ap = average_precision_score(anomaly_labels, scores)
        
        # 픽셀 단위 메트릭 계산
        segmentations_reshaped = ap_seg.reshape(-1, 224, 224)
        pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(
            segmentations_reshaped, masks_gt
        )
        full_pixel_auroc = pixel_scores["auroc"]
        
        # PRO 스코어 계산
        for i, mask in enumerate(masks_gt):
            masks_gt[i] = np.array(mask[0])
        for i, seg in enumerate(segmentations_reshaped):
            segmentations_reshaped[i] = np.array(seg)
        pixel_pro, _ = calculate_au_pro(np.array(masks_gt), np.array(segmentations_reshaped))
        
        return {
            'auroc': auroc,
            'full_pixel_auroc': full_pixel_auroc,
            'img_ap': img_ap,
            'pixel_ap': pixel_ap,
            'pixel_pro': pixel_pro,
            'time_cost': time_cost
        }

    def compute_basic_metrics(self, basic_aggregator, basic_masks_gt, start_time, basic_end_time, dataloaders):
        """기본 메트릭 계산 (무제한 메모리) - run_ucad_simple.py와 동일"""
        basic_scores = np.array(basic_aggregator["scores"])
        basic_min_scores = basic_scores.min(axis=-1).reshape(-1, 1)
        basic_max_scores = basic_scores.max(axis=-1).reshape(-1, 1)
        basic_scores = (basic_scores - basic_min_scores) / (basic_max_scores - basic_min_scores)
        basic_scores = np.mean(basic_scores, axis=0)
        
        basic_segmentations = np.array(basic_aggregator["segmentations"])
        basic_min_scores = (
            basic_segmentations.reshape(len(basic_segmentations), -1)
            .min(axis=-1)
            .reshape(-1, 1, 1, 1)
        )
        basic_max_scores = (
            basic_segmentations.reshape(len(basic_segmentations), -1)
            .max(axis=-1)
            .reshape(-1, 1, 1, 1)
        )
        basic_segmentations = (basic_segmentations - basic_min_scores) / (basic_max_scores - basic_min_scores)
        basic_segmentations = np.mean(basic_segmentations, axis=0)
        
        basic_time_cost = (basic_end_time - start_time) / len(dataloaders["testing"])
        basic_anomaly_labels = [
            x[1] != "good" for x in dataloaders["testing"].dataset.data_to_iterate
        ]
        
        basic_ap_seg = np.asarray(basic_segmentations)
        basic_ap_seg = basic_ap_seg.flatten()
        
        import patchcore.metrics
        basic_auroc = patchcore.metrics.compute_imagewise_retrieval_metrics(
            basic_scores, basic_anomaly_labels
        )["auroc"]
        
        basic_ap_mask = np.asarray(basic_masks_gt)
        basic_ap_mask = basic_ap_mask.flatten().astype(np.int32)
        from sklearn.metrics import average_precision_score
        basic_pixel_ap = average_precision_score(basic_ap_mask, basic_ap_seg)
        basic_img_ap = average_precision_score(basic_anomaly_labels, basic_scores)
        
        # 기본 픽셀 단위 메트릭
        basic_segmentations_reshaped = basic_ap_seg.reshape(-1, 224, 224)
        basic_pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(
            basic_segmentations_reshaped, basic_masks_gt
        )
        basic_full_pixel_auroc = basic_pixel_scores["auroc"]
        
        # 기본 PRO 스코어
        for i, mask in enumerate(basic_masks_gt):
            basic_masks_gt[i] = np.array(mask[0])
        for i, seg in enumerate(basic_segmentations_reshaped):
            basic_segmentations_reshaped[i] = np.array(seg)
        basic_pixel_pro, _ = calculate_au_pro(np.array(basic_masks_gt), np.array(basic_segmentations_reshaped))
        
        return {
            'auroc': basic_auroc,
            'full_pixel_auroc': basic_full_pixel_auroc,
            'img_ap': basic_img_ap,
            'pixel_ap': basic_pixel_ap,
            'pixel_pro': basic_pixel_pro,
            'time_cost': basic_time_cost
        }

    def print_final_results(self, task_id, best_metrics, best_basic_metrics):
        """최종 결과 출력 (run_ucad_simple.py와 동일)"""
        print(f"   📊 Limited Memory Results:")
        print(f"      - Image AUROC: {best_metrics['auroc']:.4f}")
        print(f"      - Pixel AUROC: {best_metrics['full_pixel_auroc']:.4f}")
        print(f"      - Image AP: {best_metrics['img_ap']:.4f}")
        print(f"      - Pixel AP: {best_metrics['pixel_ap']:.4f}")
        print(f"      - Pixel PRO: {best_metrics['pixel_pro']:.4f}")
        print(f"      - Time Cost: {best_metrics['time_cost']:.4f}s")
        
        print(f"   📊 Unlimited Memory Results:")
        print(f"      - Image AUROC: {best_basic_metrics['auroc']:.4f}")
        print(f"      - Pixel AUROC: {best_basic_metrics['full_pixel_auroc']:.4f}")
        print(f"      - Image AP: {best_basic_metrics['img_ap']:.4f}")
        print(f"      - Pixel AP: {best_basic_metrics['pixel_ap']:.4f}")
        print(f"      - Pixel PRO: {best_basic_metrics['pixel_pro']:.4f}")
        print(f"      - Time Cost: {best_basic_metrics['time_cost']:.4f}s")

    def log_dataset_start(self, dataset_name, task_id, total_tasks):
        """데이터셋 시작 로그 (run_ucad_simple.py와 동일)"""
        self.exp_logger.info("="*80)
        self.exp_logger.info(f"Task ({task_id + 1}/{total_tasks}): {dataset_name}")
        self.exp_logger.info("="*80)

    def log_epoch_performance(self, epoch, total_epochs, dataset_name, current_metrics, current_basic_metrics, improved):
        """에포크별 성능 로그 (run_ucad_simple.py와 동일)"""
        self.exp_logger.info(f"Epoch {epoch+1}/{total_epochs} - {dataset_name}")
        self.exp_logger.info(f"  Limited Memory   - AUROC: {current_metrics['auroc']:.4f}, Pixel AUROC: {current_metrics['full_pixel_auroc']:.4f}, Image AP: {current_metrics['img_ap']:.4f}, Pixel AP: {current_metrics['pixel_ap']:.4f}, PRO: {current_metrics['pixel_pro']:.4f}")
        self.exp_logger.info(f"  Unlimited Memory - AUROC: {current_basic_metrics['auroc']:.4f}, Pixel AUROC: {current_basic_metrics['full_pixel_auroc']:.4f}, Image AP: {current_basic_metrics['img_ap']:.4f}, Pixel AP: {current_basic_metrics['pixel_ap']:.4f}, PRO: {current_basic_metrics['pixel_pro']:.4f}")
        if improved:
            self.exp_logger.info(f"  *** NEW BEST Limited Memory AUROC: {current_metrics['auroc']:.4f} ***")

    def log_final_dataset_results(self, dataset_name, best_metrics, best_basic_metrics):
        """데이터셋별 최종 결과 로그 (run_ucad_simple.py와 동일)"""
        self.exp_logger.info("-"*60)
        self.exp_logger.info(f"FINAL RESULTS - {dataset_name}")
        self.exp_logger.info("-"*60)
        self.exp_logger.info("Limited Memory Results:")
        self.exp_logger.info(f"  - Image AUROC: {best_metrics['auroc']:.4f}")
        self.exp_logger.info(f"  - Pixel AUROC: {best_metrics['full_pixel_auroc']:.4f}")
        self.exp_logger.info(f"  - Image AP: {best_metrics['img_ap']:.4f}")
        self.exp_logger.info(f"  - Pixel AP: {best_metrics['pixel_ap']:.4f}")
        self.exp_logger.info(f"  - Pixel PRO: {best_metrics['pixel_pro']:.4f}")
        self.exp_logger.info(f"  - Time Cost: {best_metrics['time_cost']:.4f}s")
        
        self.exp_logger.info("Unlimited Memory Results:")
        self.exp_logger.info(f"  - Image AUROC: {best_basic_metrics['auroc']:.4f}")
        self.exp_logger.info(f"  - Pixel AUROC: {best_basic_metrics['full_pixel_auroc']:.4f}")
        self.exp_logger.info(f"  - Image AP: {best_basic_metrics['img_ap']:.4f}")
        self.exp_logger.info(f"  - Pixel AP: {best_basic_metrics['pixel_ap']:.4f}")
        self.exp_logger.info(f"  - Pixel PRO: {best_basic_metrics['pixel_pro']:.4f}")
        self.exp_logger.info(f"  - Time Cost: {best_basic_metrics['time_cost']:.4f}s")
        self.exp_logger.info("")


# PRO 계산 관련 함수들 (run_ucad_simple.py에서 복사)
class GroundTruthComponent:
    def __init__(self, anomaly_scores):
        self.anomaly_scores = anomaly_scores.copy()
        self.anomaly_scores.sort()
        self.index = 0
        self.last_threshold = None

    def compute_overlap(self, threshold):
        if self.last_threshold is not None:
            assert self.last_threshold <= threshold

        while (self.index < len(self.anomaly_scores) and self.anomaly_scores[self.index] <= threshold):
            self.index += 1

        return 1.0 - self.index / len(self.anomaly_scores)


def trapezoid(x, y, x_max=None):
    x = np.array(x)
    y = np.array(y)
    finite_mask = np.logical_and(np.isfinite(x), np.isfinite(y))
    if not finite_mask.all():
        print("WARNING: Not all x and y values passed to trapezoid are finite. Will continue with only the finite values.")
    x = x[finite_mask]
    y = y[finite_mask]

    correction = 0.
    if x_max is not None:
        if x_max not in x:
            ins = bisect(x, x_max)
            assert 0 < ins < len(x)
            y_interp = y[ins - 1] + ((y[ins] - y[ins - 1]) * (x_max - x[ins - 1]) / (x[ins] - x[ins - 1]))
            correction = 0.5 * (y_interp + y[ins - 1]) * (x_max - x[ins - 1])

        mask = x <= x_max
        x = x[mask]
        y = y[mask]

    return np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])) + correction


def collect_anomaly_scores(anomaly_maps, ground_truth_maps):
    assert len(anomaly_maps) == len(ground_truth_maps)

    ground_truth_components = []
    anomaly_scores_ok_pixels = np.zeros(len(ground_truth_maps) * ground_truth_maps[0].size)

    structure = np.ones((3, 3), dtype=int)

    ok_index = 0
    for gt_map, prediction in zip(ground_truth_maps, anomaly_maps):
        labeled, n_components = label(gt_map, structure)

        num_ok_pixels = len(prediction[labeled == 0])
        anomaly_scores_ok_pixels[ok_index:ok_index + num_ok_pixels] = prediction[labeled == 0].copy()
        ok_index += num_ok_pixels

        for k in range(n_components):
            component_scores = prediction[labeled == (k + 1)]
            ground_truth_components.append(GroundTruthComponent(component_scores))

    anomaly_scores_ok_pixels = np.resize(anomaly_scores_ok_pixels, ok_index)
    anomaly_scores_ok_pixels.sort()

    return ground_truth_components, anomaly_scores_ok_pixels


def compute_pro(anomaly_maps, ground_truth_maps, num_thresholds):
    ground_truth_components, anomaly_scores_ok_pixels = collect_anomaly_scores(anomaly_maps, ground_truth_maps)

    threshold_positions = np.linspace(0, len(anomaly_scores_ok_pixels) - 1, num=num_thresholds, dtype=int)

    fprs = [1.0]
    pros = [1.0]
    for pos in threshold_positions:
        threshold = anomaly_scores_ok_pixels[pos]

        fpr = 1.0 - (pos + 1) / len(anomaly_scores_ok_pixels)

        pro = 0.0
        for component in ground_truth_components:
            pro += component.compute_overlap(threshold)
        pro /= len(ground_truth_components)

        fprs.append(fpr)
        pros.append(pro)

    fprs = fprs[::-1]
    pros = pros[::-1]

    return fprs, pros


def calculate_au_pro(gts, predictions, integration_limit=0.3, num_thresholds=100):
    pro_curve = compute_pro(anomaly_maps=predictions, ground_truth_maps=gts, num_thresholds=num_thresholds)

    au_pro = trapezoid(pro_curve[0], pro_curve[1], x_max=integration_limit)
    au_pro /= integration_limit

    return au_pro, pro_curve


def main():
    args = parse_arguments()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # Create trainer and run
    trainer = UCADLoRATrainer(args)
    trainer.run()

if __name__ == "__main__":
    main() 