datapath=/Volume/VAD/UCAD/mvtec2d
datasets=('bottle' 'cable' 'capsule' 'carpet' 'grid' 'hazelnut' 'leather' 'metal_nut' 'pill' 'screw' 'tile' 'toothbrush' 'transistor' 'wood' 'zipper')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '$dataset; done))

CUDA_VISIBLE_DEVICES=0 python3 run_ucad.py --gpu 0 --seed 0 --memory_size 196 --log_group IM224_UCAD_L5_P01_D1024_M196 --save_segmentation_images --log_project MVTecAD_Results results ucad -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu --pretrain_embed_dimension 1024 --target_embed_dimension 1024 --anomaly_scorer_num_nn 1 --patchsize 1 sampler -p 0.1 approx_greedy_coreset dataset --resize 224 --imagesize 224 "${dataset_flags[@]}" mvtec $datapath