#!/bin/bash

# =============================================================================
# UCAD (Unified Continual Anomaly Detection) 실행 스크립트
# - ViT 기반 PatchCore 사용 (backbone 제거됨)
# - Continual Learning 훈련 후 자동으로 inference 수행
# - 각 데이터셋별 성능 및 전체 요약 결과 생성
# =============================================================================

# MVTec AD 데이터셋 경로 설정
datapath="/Volume/VAD/UCAD/mvtec2d"

echo "🚀 UCAD 훈련 및 추론 시작..."
echo "📂 데이터 경로: $datapath"
echo "🔧 ViT 기반 PatchCore 사용"
echo ""

# UCAD 훈련 및 추론 실행 - ViT만 사용하는 간소화된 버전
CUDA_VISIBLE_DEVICES=0 python3 run_ucad_simple.py results \
    --gpu 0 \
    --seed 42 \
    --memory_size 196 \
    --epochs_num 25 \
    --key_size 196 \
    --basic_size 1960 \
    --log_group test_vit_only \
    --log_project MVTecAD_UCAD \
    --save_segmentation_images \
    --save_patchcore_model \
    --faiss_on_gpu \
    --faiss_num_workers 8 \
    --pretrain_embed_dimension 1024 \
    --target_embed_dimension 1024 \
    --anomaly_scorer_num_nn 5 \
    --patchsize 3 \
    --sampler_name approx_greedy_coreset \
    --percentage 0.1 \
    --dataset_name mvtec \
    --data_path "$datapath" \
    --subdatasets bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper \
    --batch_size 8 \
    --num_workers 8 \
    --resize 224 \
    --imagesize 224 \
    --train_val_split 1 \
    --exp_name Baseline

echo ""
echo "✅ UCAD 실행 완료!"
echo "📊 결과 확인:"
echo "   - 훈련 결과: results/ 폴더"
echo "   - 추론 결과: results_inference/ 폴더"
echo "   - 성능 로그: performance_results_*.log 파일"

# =============================================================================
# 주요 변경사항:
# 1. 제거된 인자들:
#    - --backbone_names (ViT만 사용)
#    - --layers_to_extract_from (backbone 미사용)
#
# 2. 추가된 인자들:
#    - --epochs_num: 각 태스크별 훈련 에포크 수
#    - --key_size: 태스크 선택용 키 특징 크기
#    - --basic_size: 무제한 메모리 크기 (성능 비교용)
#    - --batch_size, --num_workers: 데이터로더 설정
#    - --train_val_split: 훈련/검증 분할 비율
#
# 3. 자동 실행 기능:
#    - trainer.run(): 모든 데이터셋에 대한 continual learning 훈련
#    - trainer.inference(): 훈련 완료 후 자동으로 추론 수행
#    - 태스크 선택 기반 최종 성능 평가
# ============================================================================= 