#!/bin/bash

# =============================================================================
# UCAD with LoRA 실행 스크립트
# - E-Prompt 대신 LoRA 사용으로 메모리 효율적인 continual learning
# - Vision Transformer에 LoRA 적용하여 각 태스크별 적응
# - 자동 태스크 선택 및 성능 평가
# =============================================================================

# MVTec AD 데이터셋 경로 설정
datapath="/Volume/VAD/UCAD/mvtec2d"

echo "🚀 UCAD-LoRA 훈련 및 추론 시작..."
echo "📂 데이터 경로: $datapath"
echo "🔧 LoRA 기반 continual learning 사용"
echo ""

# UCAD LoRA 훈련 및 추론 실행
CUDA_VISIBLE_DEVICES=0 python3 run_ucad_lora.py results \
    --gpu 0 \
    --seed 42 \
    --lora_rank 4 \
    --lora_alpha 8 \
    --lora_dropout 0.1 \
    --num_tasks 15 \
    --memory_size 196 \
    --epochs_num 25 \
    --key_size 196 \
    --basic_size 1960 \
    --log_group lora_experiment \
    --log_project MVTecAD_LoRA \
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
    --batch_size 1 \
    --num_workers 8 \
    --image_size 224 \
    --save_segmentation_images \
    --save_patchcore_model \
    --exp_name Baseline

echo ""
echo "✅ UCAD-LoRA 실행 완료!"
echo "📊 결과는 results_lora/ 폴더에 저장됨"
echo "🔍 LoRA 설정:"
echo "   - Rank: 4"
echo "   - Alpha: 8"
echo "   - Dropout: 0.1"
echo "   - Tasks: 15" 