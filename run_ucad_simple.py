import contextlib
import logging
import os
import sys
import argparse
from datetime import datetime

import numpy as np
import torch
import tqdm
import patchcore.common
import patchcore.metrics
import patchcore.patchcore
import patchcore.sampler
import patchcore.utils
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.ndimage import label
from bisect import bisect
import time
from metric_utils import find_optimal_threshold
import cv2
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
import yaml
import pandas as pd
import pickle
import json

LOGGER = logging.getLogger(__name__)

_DATASETS = {"mvtec": ["patchcore.datasets.mvtec", "MVTecDataset"]}


def create_experiment_directories(args):
    """실험 디렉토리 구조 생성"""
    # results/{dataset}/exp_name/seed/* 구조
    if len(args.subdatasets) == 1:
        dataset_name = args.subdatasets[0]
    else:
        dataset_name = f"mvtec_{len(args.subdatasets)}classes"
    
    exp_name = args.exp_name if hasattr(args, 'exp_name') and args.exp_name else f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
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
    config_dict.pop('device_context', None)
    
    config_path = os.path.join(config_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=True)
    
    return config_path


def setup_experiment_logger(log_dir, exp_name):
    """실험 전용 로거 설정"""
    # 로그 파일 경로
    log_path = os.path.join(log_dir, f'{exp_name}_training.log')
    
    # 실험 로거 생성
    exp_logger = logging.getLogger('experiment_logger')
    exp_logger.setLevel(logging.INFO)
    
    # 기존 핸들러 제거 (중복 방지)
    for handler in exp_logger.handlers[:]:
        exp_logger.removeHandler(handler)
    
    # 파일 핸들러 추가
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 콘솔 핸들러 추가 (선택적)
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


class ResultManager:
    """결과 관리 클래스"""
    
    def __init__(self, results_dir):
        self.results_dir = results_dir
        self.result_csv_path = os.path.join(results_dir, 'result.csv')
        self.final_csv_path = os.path.join(results_dir, 'Final.csv')
        
        # CSV 컬럼 정의
        self.result_columns = [
            'dataset_name', 'epoch', 'split_type', 'image_auroc', 'pixel_auroc', 
            'image_ap', 'pixel_ap', 'pixel_pro', 'time_cost'
        ]
        
        self.final_columns = [
            'test_dataset', 'selected_task', 'selected_task_id', 'image_auroc', 
            'pixel_auroc', 'image_ap', 'pixel_ap', 'pixel_pro', 'time_cost'
        ]
        
        # CSV 파일 초기화
        self._initialize_csv_files()
    
    def _initialize_csv_files(self):
        """CSV 파일 초기화"""
        # result.csv 초기화
        if not os.path.exists(self.result_csv_path):
            df_result = pd.DataFrame(columns=self.result_columns)
            df_result.to_csv(self.result_csv_path, index=False)
        
        # Final.csv 초기화  
        if not os.path.exists(self.final_csv_path):
            df_final = pd.DataFrame(columns=self.final_columns)
            df_final.to_csv(self.final_csv_path, index=False)
    
    def save_epoch_result(self, dataset_name, epoch, split_type, metrics):
        """에포크별 결과 저장"""
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
    
    def save_final_result(self, test_dataset, selected_task, selected_task_id, metrics):
        """최종 inference 결과 저장"""
        new_row = {
            'test_dataset': test_dataset,
            'selected_task': selected_task,
            'selected_task_id': selected_task_id,
            'image_auroc': metrics['image_auroc'],
            'pixel_auroc': metrics['pixel_auroc'],
            'image_ap': metrics['image_ap'],
            'pixel_ap': metrics['pixel_ap'],
            'pixel_pro': metrics['pixel_pro'],
            'time_cost': metrics['time_cost']
        }
        
        # CSV 파일에 추가
        df = pd.DataFrame([new_row])
        df.to_csv(self.final_csv_path, mode='a', header=False, index=False)
    
    def get_best_results_summary(self):
        """최고 성능 결과 요약 반환"""
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


class CheckpointManager:
    """체크포인트 관리 클래스"""
    
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        self.best_prompts = {}
        self.best_memory_features = {}
        self.best_key_features = {}
        
    def save_best_checkpoint(self, dataset_name, prompt, memory_feature, key_feature, metrics):
        """최고 성능 체크포인트 저장"""
        # 프롬프트 저장
        prompt_path = os.path.join(self.checkpoint_dir, f'{dataset_name}_best_prompt.pkl')
        with open(prompt_path, 'wb') as f:
            pickle.dump(prompt, f)
        
        # 메모리 특징 저장
        memory_path = os.path.join(self.checkpoint_dir, f'{dataset_name}_best_memory_feature.pkl')
        with open(memory_path, 'wb') as f:
            pickle.dump(memory_feature, f)
            
        # 키 특징 저장
        key_path = os.path.join(self.checkpoint_dir, f'{dataset_name}_best_key_feature.pkl')
        with open(key_path, 'wb') as f:
            pickle.dump(key_feature, f)
        
        # 메트릭 정보 저장
        metrics_path = os.path.join(self.checkpoint_dir, f'{dataset_name}_best_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # 메모리에도 저장
        self.best_prompts[dataset_name] = prompt
        self.best_memory_features[dataset_name] = memory_feature
        self.best_key_features[dataset_name] = key_feature
        
        return {
            'prompt_path': prompt_path,
            'memory_path': memory_path, 
            'key_path': key_path,
            'metrics_path': metrics_path
        }
    
    def load_best_checkpoint(self, dataset_name):
        """최고 성능 체크포인트 로드"""
        prompt_path = os.path.join(self.checkpoint_dir, f'{dataset_name}_best_prompt.pkl')
        memory_path = os.path.join(self.checkpoint_dir, f'{dataset_name}_best_memory_feature.pkl')
        key_path = os.path.join(self.checkpoint_dir, f'{dataset_name}_best_key_feature.pkl')
        
        try:
            with open(prompt_path, 'rb') as f:
                prompt = pickle.load(f)
            with open(memory_path, 'rb') as f:
                memory_feature = pickle.load(f)
            with open(key_path, 'rb') as f:
                key_feature = pickle.load(f)
                
            return prompt, memory_feature, key_feature
        except FileNotFoundError:
            return None, None, None


class UCADTrainer:
    """UCAD 훈련 및 평가를 위한 메인 클래스"""
    
    def __init__(self, args):
        self.args = args
        self.device = patchcore.utils.set_torch_device(args.gpu)
        self.device_context = (
            torch.cuda.device("cuda:{}".format(self.device.index))
            if "cuda" in self.device.type.lower()
            else contextlib.suppress()
        )
        
        # 실험 디렉토리 구조 생성
        self.experiment_dir, self.subdirs = create_experiment_directories(args)
        
        # 실험 설정 저장
        self.config_path = save_experiment_config(args, self.subdirs['configs'])
        
        # 로거 설정
        exp_name = os.path.basename(os.path.dirname(self.experiment_dir))
        self.exp_logger, self.log_path = setup_experiment_logger(self.subdirs['logs'], exp_name)
        
        # 결과 관리자 초기화
        self.result_manager = ResultManager(self.subdirs['results'])
        
        # 체크포인트 관리자 초기화
        self.checkpoint_manager = CheckpointManager(self.subdirs['checkpoints'])
        
        # 실험 시작 로그
        self.exp_logger.info("="*80)
        self.exp_logger.info(f"UCAD Experiment Started")
        self.exp_logger.info(f"Experiment Directory: {self.experiment_dir}")
        self.exp_logger.info(f"Config saved to: {self.config_path}")
        self.exp_logger.info("="*80)
        
        # 기존 변수들
        self.key_feature_list = [0] * 15
        self.memory_feature_list = [0] * 15
        self.prompt_list = [0] * 15

    def create_patchcore_model(self, input_shape, sampler):
        """PatchCore 모델 생성 (ViT 기반, backbone 사용 안함)"""
        nn_method = patchcore.common.FaissNN(self.args.faiss_on_gpu, self.args.faiss_num_workers)
        
        patchcore_instance = patchcore.patchcore.PatchCore(self.device)
        patchcore_instance.load(
            backbone=None,  # backbone 사용 안함
            layers_to_extract_from=[],  # 빈 리스트
            device=self.device,
            input_shape=input_shape,
            pretrain_embed_dimension=self.args.pretrain_embed_dimension,
            target_embed_dimension=self.args.target_embed_dimension,
            patchsize=self.args.patchsize,
            featuresampler=sampler,
            anomaly_scorer_num_nn=self.args.anomaly_scorer_num_nn,
            nn_method=nn_method,
        )
        
        return [patchcore_instance]

    def create_sampler(self):
        """샘플러 생성"""
        if self.args.sampler_name == "identity":
            return patchcore.sampler.IdentitySampler()
        elif self.args.sampler_name == "greedy_coreset":
            return patchcore.sampler.GreedyCoresetSampler(self.args.percentage, self.device)
        elif self.args.sampler_name == "approx_greedy_coreset":
            return patchcore.sampler.ApproximateGreedyCoresetSampler(self.args.percentage, self.device)

    def create_dataloaders(self):
        """데이터로더 생성"""
        dataset_info = _DATASETS[self.args.dataset_name]
        dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

        dataloaders = []
        for subdataset in self.args.subdatasets:
            train_dataset = dataset_library.__dict__[dataset_info[1]](
                self.args.data_path,
                classname=subdataset,
                resize=self.args.resize,
                train_val_split=self.args.train_val_split,
                imagesize=self.args.imagesize,
                split=dataset_library.DatasetSplit.TRAIN,
                seed=self.args.seed,
                augment=self.args.augment,
            )

            test_dataset = dataset_library.__dict__[dataset_info[1]](
                self.args.data_path,
                classname=subdataset,
                resize=self.args.resize,
                imagesize=self.args.imagesize,
                split=dataset_library.DatasetSplit.TEST,
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

            if self.args.train_val_split < 1:
                val_dataset = dataset_library.__dict__[dataset_info[1]](
                    self.args.data_path,
                    classname=subdataset,
                    resize=self.args.resize,
                    train_val_split=self.args.train_val_split,
                    imagesize=self.args.imagesize,
                    split=dataset_library.DatasetSplit.VAL,
                    seed=self.args.seed,
                )

                val_dataloader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=self.args.batch_size,
                    shuffle=False,
                    num_workers=self.args.num_workers,
                    pin_memory=True,
                )
            else:
                val_dataloader = None
                
            dataloader_dict = {
                "training": train_dataloader,
                "validation": val_dataloader,
                "testing": test_dataloader,
            }

            dataloaders.append(dataloader_dict)
        
        return dataloaders

    def train_single_dataset(self, dataloader_count, dataloaders, patchcore_list):
        """단일 데이터셋에 대한 훈련"""
        dataset_name = dataloaders["training"].name
        
        # 데이터셋 시작 로그
        self.log_dataset_start(dataset_name, dataloader_count, len(self.args.subdatasets))
        
        # PatchCore 모델은 현재 설정에서 1개만 있음
        patchcore = patchcore_list[0]  # 첫 번째(유일한) 모델 사용
        
        # 키 특징 추출
        patchcore.set_dataloadercount(dataloader_count)
        key_feature = patchcore.fit_with_limit_size(dataloaders["training"], self.args.key_size)
        self.key_feature_list[dataloader_count] = key_feature

        # 훈련 설정
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
        
        optimizer = create_optimizer(args_dict, patchcore.prompt_model)
        epochs = self.args.epochs_num
        patchcore.prompt_model.train()
        patchcore.prompt_model.train_contrastive = True
        
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

        # 에포크별 훈련
        print(f"\n{'='*60}")
        print(f"Training Dataset: {dataset_name}")
        print(f"Total Epochs: {epochs}")
        print(f"{'='*60}")
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            torch.cuda.empty_cache()
            
            # 훈련 단계
            patchcore.prompt_model.train()
            loss_list = []
                                    
            batch_count = 0
            total_batches = len(dataloaders["training"])
            
            for image in dataloaders["training"]:
                if isinstance(image, dict):
                    image_paths = image["image_path"]
                    image = image["image"].cuda()
                
                res = patchcore._embed_train_sam(image, provide_patch_shapes=True, image_path=image_paths)
                loss = res['loss']
                loss_list.append(loss.item())
                
                optimizer.zero_grad()
                if loss != 0:
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(patchcore.prompt_model.parameters(), args_dict.clip_grad)
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
            patchcore.prompt_model.eval()
            
            print(f"Epoch {epoch+1}/{epochs} - Evaluating...", end=" ", flush=True)
            
            # 무제한 메모리 평가
            nolimimit_memory_feature = patchcore.fit_with_limit_size_prompt(dataloaders["training"], self.args.basic_size)
            patchcore.anomaly_scorer.fit(detection_features=[nolimimit_memory_feature])
            basic_scores, basic_segmentations, basic_labels_gt, basic_masks_gt = patchcore.predict_prompt(
                dataloaders["testing"]
            )
            basic_aggregator["scores"].append(basic_scores)
            basic_aggregator["segmentations"].append(basic_segmentations)
            basic_end_time = time.time()
            
            # 제한된 메모리 평가
            memory_feature = patchcore.fit_with_limit_size_prompt(dataloaders["training"], self.args.memory_size)
            patchcore.anomaly_scorer.fit(detection_features=[memory_feature])
            scores, segmentations, labels_gt, masks_gt = patchcore.predict_prompt(
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
                # 메모리에 저장 (기존 방식 유지)
                self.memory_feature_list[dataloader_count] = memory_feature
                self.prompt_list[dataloader_count] = patchcore.prompt_model.get_cur_prompt()
                
                pr_auroc = current_metrics['auroc']
                best_metrics = current_metrics.copy()
                improved = True
                
                # 베스트 체크포인트 저장
                checkpoint_paths = self.checkpoint_manager.save_best_checkpoint(
                    dataset_name, 
                    patchcore.prompt_model.get_cur_prompt(),
                    memory_feature,
                    key_feature,
                    current_metrics
                )
                
                self.exp_logger.info(f"  🎯 Best checkpoint saved for {dataset_name}")
                self.exp_logger.info(f"     Prompt: {checkpoint_paths['prompt_path']}")
                self.exp_logger.info(f"     Memory: {checkpoint_paths['memory_path']}")
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
        
        # 최종 결과 출력 및 로그
        print(f"\n🏆 Final Results for {dataset_name}:")
        self.print_final_results(dataloader_count, best_metrics, best_basic_metrics)
        self.log_final_dataset_results(dataset_name, best_metrics, best_basic_metrics)

    def compute_metrics(self, aggregator, masks_gt, basic_end_time, end_time, dataloaders):
        """메트릭 계산"""
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
        
        auroc = patchcore.metrics.compute_imagewise_retrieval_metrics(
            scores, anomaly_labels
        )["auroc"]
        
        ap_mask = np.asarray(masks_gt)
        ap_mask = ap_mask.flatten().astype(np.int32)
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
        """기본 메트릭 계산 (무제한 메모리)"""
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
        
        basic_auroc = patchcore.metrics.compute_imagewise_retrieval_metrics(
            basic_scores, basic_anomaly_labels
        )["auroc"]
        
        basic_ap_mask = np.asarray(basic_masks_gt)
        basic_ap_mask = basic_ap_mask.flatten().astype(np.int32)
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

    def print_final_results(self, dataloader_count, best_metrics, best_basic_metrics):
        """최종 결과 출력"""
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

    def log_dataset_start(self, dataset_name, dataloader_count, total_datasets):
        """데이터셋 시작 로그"""
        self.exp_logger.info("="*80)
        self.exp_logger.info(f"Dataset ({dataloader_count + 1}/{total_datasets}): {dataset_name}")
        self.exp_logger.info("="*80)

    def log_epoch_performance(self, epoch, total_epochs, dataset_name, current_metrics, current_basic_metrics, improved):
        """에포크별 성능 로그"""
        self.exp_logger.info(f"Epoch {epoch+1}/{total_epochs} - {dataset_name}")
        self.exp_logger.info(f"  Limited Memory   - AUROC: {current_metrics['auroc']:.4f}, Pixel AUROC: {current_metrics['full_pixel_auroc']:.4f}, Image AP: {current_metrics['img_ap']:.4f}, Pixel AP: {current_metrics['pixel_ap']:.4f}, PRO: {current_metrics['pixel_pro']:.4f}")
        self.exp_logger.info(f"  Unlimited Memory - AUROC: {current_basic_metrics['auroc']:.4f}, Pixel AUROC: {current_basic_metrics['full_pixel_auroc']:.4f}, Image AP: {current_basic_metrics['img_ap']:.4f}, Pixel AP: {current_basic_metrics['pixel_ap']:.4f}, PRO: {current_basic_metrics['pixel_pro']:.4f}")
        if improved:
            self.exp_logger.info(f"  *** NEW BEST Limited Memory AUROC: {current_metrics['auroc']:.4f} ***")

    def log_final_dataset_results(self, dataset_name, best_metrics, best_basic_metrics):
        """데이터셋별 최종 결과 로그"""
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

    def log_overall_summary(self):
        """전체 결과 요약 로그"""
        self.exp_logger.info("="*80)
        self.exp_logger.info("OVERALL PERFORMANCE SUMMARY")
        self.exp_logger.info("="*80)
        
        best_results = self.result_manager.get_best_results_summary()
        if best_results:
            self.exp_logger.info("Best Results Summary:")
            for result in best_results:
                dataset_name = result["dataset_name"]
                auroc = result["image_auroc"]
                pixel_auroc = result["pixel_auroc"]
                img_ap = result["image_ap"]
                pixel_pro = result["pixel_pro"]
                self.exp_logger.info(f"  {dataset_name:15s} - AUROC: {auroc:.4f}, Pixel AUROC: {pixel_auroc:.4f}, Image AP: {img_ap:.4f}, PRO: {pixel_pro:.4f}")
            
            avg_auroc = sum(result["image_auroc"] for result in best_results) / len(best_results)
            avg_pixel_auroc = sum(result["pixel_auroc"] for result in best_results) / len(best_results)
            avg_img_ap = sum(result["image_ap"] for result in best_results) / len(best_results)
            avg_pixel_pro = sum(result["pixel_pro"] for result in best_results) / len(best_results)
            self.exp_logger.info(f"  {'AVERAGE':15s} - AUROC: {avg_auroc:.4f}, Pixel AUROC: {avg_pixel_auroc:.4f}, Image AP: {avg_img_ap:.4f}, PRO: {avg_pixel_pro:.4f}")
            self.exp_logger.info("")
        
        self.exp_logger.info("="*80)

    def inference(self):
        """모든 continual learning 완료 후 최종 inference 수행"""
        print(f"\n{'='*80}")
        print("🔍 Continual Learning Inference 시작...")
        print(f"{'='*80}")
        
        # 실험 로거에 inference 시작 로그
        self.exp_logger.info("="*80)
        self.exp_logger.info("CONTINUAL LEARNING INFERENCE STARTED")
        self.exp_logger.info("="*80)
        
        # 데이터로더 재생성
        list_of_dataloaders = self.create_dataloaders()
        inference_results = []
        
        # 각 데이터셋에 대해 inference 수행
        for dataloader_count, dataloaders in enumerate(list_of_dataloaders):
            dataset_name = dataloaders["training"].name
            
            print(f"\n🔎 Inference Dataset ({dataloader_count + 1}/{len(list_of_dataloaders)}): {dataset_name}")
            
            LOGGER.info(
                "Inference on dataset [{}] ({}/{})...".format(
                    dataset_name,
                    dataloader_count + 1,
                    len(list_of_dataloaders),
                )
            )
            
            # 시드 고정
            patchcore.utils.fix_seeds(self.args.seed, self.device)
            
            with self.device_context:
                torch.cuda.empty_cache()
                
                # PatchCore 모델 생성
                imagesize = dataloaders["training"].dataset.imagesize
                sampler = self.create_sampler()
                patchcore_list = self.create_patchcore_model(imagesize, sampler)
                
                if len(patchcore_list) > 1:
                    LOGGER.info(
                        "Utilizing PatchCore Ensemble (N={}).".format(len(patchcore_list))
                    )
                
                aggregator = {"scores": [], "segmentations": []}
                
                # 각 PatchCore 모델에 대해 inference
                for i, patchcore in enumerate(patchcore_list):
                    torch.cuda.empty_cache()
                    
                    LOGGER.info(
                        "Inference with model ({}/{})".format(i + 1, len(patchcore_list))
                    )
                    
                    start_time = time.time()
                    
                    # 쿼리 기반 태스크 선택: 각 태스크의 key feature로 현재 테스트 데이터와 유사도 계산
                    print("   🔍 태스크 선택을 위한 쿼리 계산 중...", end=" ", flush=True)
                    cur_query_list = []
                    
                    for key_count in range(len(list_of_dataloaders)):
                        # 유효한 key feature가 있는지 확인
                        if self.key_feature_list[key_count] == 0:
                            cur_query_list.append(float('inf'))  # 훈련되지 않은 태스크는 무한대
                            continue
                            
                        # 해당 태스크의 key feature로 anomaly scorer 설정
                        patchcore.anomaly_scorer.fit(detection_features=[self.key_feature_list[key_count]])
                        
                        # 현재 테스트 데이터에 대한 이상 점수 계산
                        query_scores, query_seg, labels_gt_query, masks_gt_query = patchcore.predict(
                            dataloaders["testing"]
                        )
                        
                        # 총 이상 점수 합계 (낮을수록 해당 태스크와 유사)
                        cur_query_list.append(np.sum(query_scores))
                    
                    print("Done!")
                    
                    # 가장 낮은 이상 점수를 가진 태스크 선택
                    print(f"   📊 Query scores: {[f'{score:.2f}' if score != float('inf') else 'N/A' for score in cur_query_list]}")
                    query_data_id = np.argmin(cur_query_list)
                    selected_task_name = self.args.subdatasets[query_data_id]
                    
                    print(f"   🎯 선택된 태스크: {selected_task_name} (ID: {query_data_id})")
                    
                    # 선택된 태스크의 설정 적용
                    patchcore.set_dataloadercount(query_data_id)
                    
                    # 선택된 태스크의 프롬프트 적용
                    if self.prompt_list[query_data_id] != 0:
                        patchcore.prompt_model.set_cur_prompt(self.prompt_list[query_data_id])
                    
                    # 평가 모드 설정
                    patchcore.prompt_model.eval()
                    
                    # 선택된 태스크의 메모리 특징으로 anomaly scorer 설정
                    if self.memory_feature_list[query_data_id] != 0:
                        patchcore.anomaly_scorer.fit(detection_features=[self.memory_feature_list[query_data_id]])
                    
                    print("   🔮 최종 예측 수행 중...", end=" ", flush=True)
                    
                    # 프롬프트 기반 최종 예측
                    scores, segmentations, labels_gt, masks_gt = patchcore.predict_prompt(
                        dataloaders["testing"]
                    )
                    
                    aggregator["scores"].append(scores)
                    aggregator["segmentations"].append(segmentations)
                    
                    print("Done!")
                
                # 앙상블 결과 처리
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
                
                end_time = time.time()
                time_cost = (end_time - start_time) / len(dataloaders["testing"])
                
                # 이상 라벨 추출
                anomaly_labels = [
                    x[1] != "good" for x in dataloaders["testing"].dataset.data_to_iterate
                ]
                
                # 메트릭 계산
                print("   📈 성능 메트릭 계산 중...", end=" ", flush=True)
                
                ap_seg = np.asarray(segmentations)
                ap_seg = ap_seg.flatten()
                
                LOGGER.info("Computing evaluation metrics.")
                
                # Image AUROC
                auroc = patchcore.metrics.compute_imagewise_retrieval_metrics(
                    scores, anomaly_labels
                )["auroc"]
                
                # Pixel 메트릭
                ap_mask = np.asarray(masks_gt)
                ap_mask = ap_mask.flatten().astype(np.int32)
                pixel_ap = average_precision_score(ap_mask, ap_seg)
                img_ap = average_precision_score(anomaly_labels, scores)
                
                # Pixel AUROC
                segmentations_reshaped = ap_seg.reshape(-1, 224, 224)
                pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(
                    segmentations_reshaped, masks_gt
                )
                full_pixel_auroc = pixel_scores["auroc"]
                
                # PRO 스코어
                for i, mask in enumerate(masks_gt):
                    masks_gt[i] = np.array(mask[0])
                for i, seg in enumerate(segmentations_reshaped):
                    segmentations_reshaped[i] = np.array(seg)
                pixel_pro, _ = calculate_au_pro(np.array(masks_gt), np.array(segmentations_reshaped))
                
                print("Done!")
                
                # 결과 저장
                inference_metrics = {
                    "image_auroc": auroc,
                    "pixel_auroc": full_pixel_auroc,
                    "image_ap": img_ap,
                    "pixel_ap": pixel_ap,
                    "pixel_pro": pixel_pro,
                    "time_cost": time_cost
                }
                
                # Final.csv에 저장
                self.result_manager.save_final_result(
                    dataset_name, 
                    selected_task_name, 
                    query_data_id, 
                    inference_metrics
                )
                
                # 결과 출력
                print(f"\n   📊 Inference Results for {dataset_name}:")
                print(f"      🎯 Selected Task: {selected_task_name}")
                print(f"      📈 Image AUROC: {auroc:.4f}")
                print(f"      📈 Pixel AUROC: {full_pixel_auroc:.4f}")
                print(f"      📈 Image AP: {img_ap:.4f}")
                print(f"      📈 Pixel AP: {pixel_ap:.4f}")
                print(f"      📈 Pixel PRO: {pixel_pro:.4f}")
                print(f"      ⏱️  Time Cost: {time_cost:.4f}s")
                
                # 성능 로그 기록
                self.exp_logger.info(f"Inference - {dataset_name}")
                self.exp_logger.info(f"  Selected Task: {selected_task_name} (ID: {query_data_id})")
                self.exp_logger.info(f"  Image AUROC: {auroc:.4f}")
                self.exp_logger.info(f"  Pixel AUROC: {full_pixel_auroc:.4f}")
                self.exp_logger.info(f"  Image AP: {img_ap:.4f}")
                self.exp_logger.info(f"  Pixel AP: {pixel_ap:.4f}")
                self.exp_logger.info(f"  Pixel PRO: {pixel_pro:.4f}")
                self.exp_logger.info(f"  Time Cost: {time_cost:.4f}s")
                self.exp_logger.info("")
                
                print(f'   📋 Test task: {dataloader_count+1}/{len(list_of_dataloaders)}, '
                      f'Selected task: {selected_task_name}, '
                      f'Image AUROC: {auroc:.4f}, '
                      f'Pixel AUROC: {full_pixel_auroc:.4f}, '
                      f'Image AP: {img_ap:.4f}, '
                      f'Pixel AP: {pixel_ap:.4f}, '
                      f'Pixel PRO: {pixel_pro:.4f}, '
                      f'Time cost: {time_cost:.4f}')
        
        # 최종 inference 결과 요약
        self.log_inference_summary()
        
        print(f"\n{'='*80}")
        print("✅ Continual Learning Inference 완료!")
        print(f"📝 Inference 결과 저장됨: {self.result_manager.final_csv_path}")
        print(f"{'='*80}")
        
        return inference_results

    def log_inference_summary(self):
        """Inference 결과 요약 로그"""
        self.exp_logger.info("="*80)
        self.exp_logger.info("CONTINUAL LEARNING INFERENCE SUMMARY")
        self.exp_logger.info("="*80)
        
        # Final.csv에서 결과 읽어오기
        if os.path.exists(self.result_manager.final_csv_path):
            df = pd.read_csv(self.result_manager.final_csv_path)
            if not df.empty:
                # 태스크 선택 통계
                task_selection_count = df['selected_task'].value_counts().to_dict()
                
                self.exp_logger.info("Task Selection Statistics:")
                for task, count in task_selection_count.items():
                    self.exp_logger.info(f"  {task}: {count} times")
                self.exp_logger.info("")
                
                self.exp_logger.info("Detailed Results:")
                for _, row in df.iterrows():
                    self.exp_logger.info(f"  {row['test_dataset']:15s} -> {row['selected_task']:15s} | "
                                       f"AUROC: {row['image_auroc']:.4f}, Pixel AUROC: {row['pixel_auroc']:.4f}, "
                                       f"Image AP: {row['image_ap']:.4f}, Pixel AP: {row['pixel_ap']:.4f}, "
                                       f"PRO: {row['pixel_pro']:.4f}, Time: {row['time_cost']:.4f}s")
                
                # 평균 계산
                avg_auroc = df['image_auroc'].mean()
                avg_pixel_auroc = df['pixel_auroc'].mean()
                avg_img_ap = df['image_ap'].mean()
                avg_pixel_ap = df['pixel_ap'].mean()
                avg_pixel_pro = df['pixel_pro'].mean()
                avg_time = df['time_cost'].mean()
                
                self.exp_logger.info("-"*80)
                self.exp_logger.info(f"  {'AVERAGE':15s} {'':18s} | "
                                   f"AUROC: {avg_auroc:.4f}, Pixel AUROC: {avg_pixel_auroc:.4f}, "
                                   f"Image AP: {avg_img_ap:.4f}, Pixel AP: {avg_pixel_ap:.4f}, "
                                   f"PRO: {avg_pixel_pro:.4f}, Time: {avg_time:.4f}s")
        
        self.exp_logger.info("="*80)

    def run(self):
        """메인 실행 함수"""
        print("🚀 UCAD 훈련 시작...")
        
        # 실험 로거 초기화 로그
        self.exp_logger.info("UCAD Training Started")
        self.exp_logger.info(f"Memory Size: {self.args.memory_size}")
        self.exp_logger.info(f"Epochs: {self.args.epochs_num}")
        self.exp_logger.info(f"Batch Size: {self.args.batch_size}")
        self.exp_logger.info(f"Subdatasets: {self.args.subdatasets}")
        self.exp_logger.info("")
        
        # 데이터로더 생성
        list_of_dataloaders = self.create_dataloaders()
        total_datasets = len(list_of_dataloaders)
        print(f"📂 총 {total_datasets}개 데이터셋 로드됨: {self.args.subdatasets}")
        
        print(f"\n{'='*80}")
        print(f"🎯 훈련 설정")
        print(f"   - 실험 디렉토리: {self.experiment_dir}")
        print(f"   - 메모리 크기: {self.args.memory_size}")
        print(f"   - 에포크 수: {self.args.epochs_num}")
        print(f"   - 배치 크기: {self.args.batch_size}")
        print(f"   - 설정 파일: {self.config_path}")
        print(f"   - 로그 파일: {self.log_path}")
        print(f"   - 결과 CSV: {self.result_manager.result_csv_path}")
        print(f"   - 체크포인트: {self.checkpoint_manager.checkpoint_dir}")
        print(f"{'='*80}\n")
        
        # 각 데이터셋에 대해 훈련
        for dataloader_count, dataloaders in enumerate(list_of_dataloaders):
            dataset_progress = f"({dataloader_count + 1}/{total_datasets})"
            print(f"\n🔥 Dataset {dataset_progress}: {dataloaders['training'].name}")
            
            LOGGER.info(
                "Evaluating dataset [{}] ({}/{})...".format(
                    dataloaders["training"].name,
                    dataloader_count + 1,
                    len(list_of_dataloaders),
                )
            )

            patchcore.utils.fix_seeds(self.args.seed, self.device)
            
            with self.device_context:
                torch.cuda.empty_cache()
                imagesize = dataloaders["training"].dataset.imagesize
                sampler = self.create_sampler()
                patchcore_list = self.create_patchcore_model(imagesize, sampler)                
                
                if len(patchcore_list) > 1:
                    LOGGER.info(
                        "Utilizing PatchCore Ensemble (N={}).".format(len(patchcore_list))
                    )
                
                # 단일 데이터셋 훈련
                self.train_single_dataset(dataloader_count, dataloaders, patchcore_list)
                
            print(f"\n✅ Dataset {dataset_progress} 완료!")
            LOGGER.info("\n\n-----\n")
        
        # 최종 결과 저장
        print(f"\n{'='*80}")
        print("📊 모든 데이터셋 훈련 완료! 최종 결과 저장 중...")
        self.save_final_results()
        
        print(f"\n🎉 실험 완료! 모든 결과가 저장되었습니다:")
        print(f"   📁 실험 디렉토리: {self.experiment_dir}")
        print(f"   📊 훈련 결과: {self.result_manager.result_csv_path}")
        print(f"   💾 체크포인트: {self.checkpoint_manager.checkpoint_dir}")
        print(f"   📝 로그 파일: {self.log_path}")
        print(f"   ⚙️ 설정 파일: {self.config_path}")
        print(f"{'='*80}")

        # 실험 완료 로그
        self.exp_logger.info("="*80)
        self.exp_logger.info("UCAD TRAINING COMPLETED SUCCESSFULLY")
        self.exp_logger.info(f"Experiment Directory: {self.experiment_dir}")
        self.exp_logger.info(f"Training Results: {self.result_manager.result_csv_path}")
        self.exp_logger.info(f"Checkpoints: {self.checkpoint_manager.checkpoint_dir}")
        self.exp_logger.info("="*80)

    def save_final_results(self):
        """최종 결과를 CSV로 저장"""
        print('\n📈 훈련 결과 요약:')
        
        # result.csv에서 최고 성능 결과 읽어오기
        if os.path.exists(self.result_manager.result_csv_path):
            df = pd.read_csv(self.result_manager.result_csv_path)
            if not df.empty:
                print("   에포크별 결과가 result.csv에 저장되었습니다.")
                
                # 각 데이터셋별 최고 성능 출력
                datasets = df['dataset_name'].unique()
                
                print('\n   📊 각 데이터셋별 최고 성능:')
                for dataset in datasets:
                    dataset_df = df[df['dataset_name'] == dataset]
                    
                    # Limited Memory 최고 성능
                    limited_df = dataset_df[dataset_df['split_type'] == 'limited_memory']
                    if not limited_df.empty:
                        best_limited = limited_df.loc[limited_df['image_auroc'].idxmax()]
                        print(f"      {dataset} (Limited)  : AUROC={best_limited['image_auroc']:.4f} (epoch {best_limited['epoch']})")
                    
                    # Unlimited Memory 최고 성능
                    unlimited_df = dataset_df[dataset_df['split_type'] == 'unlimited_memory']
                    if not unlimited_df.empty:
                        best_unlimited = unlimited_df.loc[unlimited_df['image_auroc'].idxmax()]
                        print(f"      {dataset} (Unlimited): AUROC={best_unlimited['image_auroc']:.4f} (epoch {best_unlimited['epoch']})")
                
                # 전체 평균 계산
                limited_avg = df[df['split_type'] == 'limited_memory'].groupby('dataset_name')['image_auroc'].max().mean()
                unlimited_avg = df[df['split_type'] == 'unlimited_memory'].groupby('dataset_name')['image_auroc'].max().mean()
                
                print(f'\n   📈 전체 평균 성능:')
                print(f"      Limited Memory Average AUROC  : {limited_avg:.4f}")
                print(f"      Unlimited Memory Average AUROC: {unlimited_avg:.4f}")
                
                # 실험 로거에도 기록
                self.exp_logger.info("Training Results Summary:")
                self.exp_logger.info(f"Limited Memory Average AUROC: {limited_avg:.4f}")
                self.exp_logger.info(f"Unlimited Memory Average AUROC: {unlimited_avg:.4f}")
                
                # 체크포인트 정보 출력
                print(f'\n   💾 체크포인트 저장 위치: {self.checkpoint_manager.checkpoint_dir}')
                checkpoint_files = [f for f in os.listdir(self.checkpoint_manager.checkpoint_dir) if f.endswith('.pkl') or f.endswith('.json')]
                if checkpoint_files:
                    print(f"      저장된 체크포인트 파일 수: {len(checkpoint_files)}")
                    self.exp_logger.info(f"Checkpoints saved: {len(checkpoint_files)} files in {self.checkpoint_manager.checkpoint_dir}")
        
        # 전체 요약 성능 로그 기록
        self.log_overall_summary()


def parse_arguments():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(description='UCAD 훈련 및 평가')
    
    # 기본 설정
    parser.add_argument('results_path', type=str, help='결과 저장 경로')
    parser.add_argument('--gpu', type=int, nargs='+', default=[0], help='사용할 GPU 번호')
    parser.add_argument('--seed', type=int, default=0, help='랜덤 시드')
    parser.add_argument('--log_group', type=str, default="group", help='로그 그룹명')
    parser.add_argument('--log_project', type=str, default="project", help='프로젝트명')
    parser.add_argument('--exp_name', type=str, default=None, help='실험 이름 (없으면 자동 생성)')
    parser.add_argument('--save_segmentation_images', action='store_true', help='세그멘테이션 이미지 저장')
    parser.add_argument('--save_patchcore_model', action='store_true', help='PatchCore 모델 저장')
    parser.add_argument('--memory_size', type=int, default=196, help='메모리 뱅크 크기')
    parser.add_argument('--epochs_num', type=int, default=25, help='훈련 에포크 수')
    parser.add_argument('--key_size', type=int, default=196, help='키 특징 크기')
    parser.add_argument('--basic_size', type=int, default=1960, help='기본 메모리 크기')
    
    # PatchCore 설정 (backbone 관련 제거)
    parser.add_argument('--pretrain_embed_dimension', type=int, default=1024, help='사전훈련 임베딩 차원')
    parser.add_argument('--target_embed_dimension', type=int, default=1024, help='타겟 임베딩 차원')
    parser.add_argument('--preprocessing', choices=["mean", "conv"], default="mean", help='전처리 방법')
    parser.add_argument('--aggregation', choices=["mean", "mlp"], default="mean", help='집계 방법')
    parser.add_argument('--anomaly_scorer_num_nn', type=int, default=5, help='이상 점수 계산용 NN 개수')
    parser.add_argument('--patchsize', type=int, default=3, help='패치 크기')
    parser.add_argument('--patchscore', type=str, default="max", help='패치 점수 계산 방법')
    parser.add_argument('--patchoverlap', type=float, default=0.0, help='패치 오버랩')
    parser.add_argument('--faiss_on_gpu', action='store_true', help='GPU에서 Faiss 실행')
    parser.add_argument('--faiss_num_workers', type=int, default=8, help='Faiss 워커 수')
    
    # 샘플러 설정
    parser.add_argument('--sampler_name', type=str, default="approx_greedy_coreset", help='샘플러 이름')
    parser.add_argument('--percentage', '-p', type=float, default=0.1, help='샘플링 비율')
    
    # 데이터셋 설정
    parser.add_argument('--dataset_name', type=str, default="mvtec", help='데이터셋 이름')
    parser.add_argument('--data_path', type=str, default="/Volume/VAD/UCAD/mvtec2d", help='데이터 경로')
    parser.add_argument('--subdatasets', '-d', type=str, nargs='+', 
                       default=['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 
                               'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper'],
                       help='서브 데이터셋 목록')
    parser.add_argument('--train_val_split', type=float, default=1, help='훈련/검증 분할 비율')
    parser.add_argument('--batch_size', type=int, default=8, help='배치 크기')
    parser.add_argument('--num_workers', type=int, default=8, help='데이터로더 워커 수')
    parser.add_argument('--resize', type=int, default=224, help='리사이즈 크기')
    parser.add_argument('--imagesize', type=int, default=224, help='이미지 크기')
    parser.add_argument('--augment', action='store_true', help='데이터 증강 사용')
    
    return parser.parse_args()


# PRO 계산 관련 함수들 (기존 코드와 동일)
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
    """메인 실행 함수"""
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
    
    # 인자 파싱
    args = parse_arguments()
    
    # UCAD 훈련기 생성 및 실행
    trainer = UCADTrainer(args)
    trainer.run()
    
    # Continual Learning 완료 후 최종 Inference 수행
    trainer.inference()


if __name__ == "__main__":
    main() 