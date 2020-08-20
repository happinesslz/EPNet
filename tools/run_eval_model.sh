#! /bin/bash

# full_epnet_without_iou_branch
CUDA_VISIBLE_DEVICES=0 python eval_rcnn.py --cfg_file cfgs/LI_Fusion_with_attention_use_ce_loss.yaml --eval_mode rcnn_online    --output_dir ./log/Car/models/full_epnet_without_iou_branch/eval_results/  --ckpt ./log/Car/models/full_epnet_without_iou_branch/ckpt/checkpoint_epoch_45.pth --set  LI_FUSION.ENABLED True LI_FUSION.ADD_Image_Attention True RCNN.POOL_EXTRA_WIDTH 0.2  RPN.SCORE_THRESH 0.2 RCNN.SCORE_THRESH 0.2  USE_IOU_BRANCH False

# epnet_without_ce_loss
CUDA_VISIBLE_DEVICES=0 python eval_rcnn.py --cfg_file cfgs/LI_Fusion_with_attention_use_ce_loss.yaml --eval_mode rcnn_online    --output_dir ./log/Car/models/epnet_without_ce_loss/eval_results/  --ckpt ./log/Car/models/epnet_without_ce_loss/ckpt/checkpoint_epoch_44.pth --set  LI_FUSION.ENABLED True LI_FUSION.ADD_Image_Attention True RCNN.POOL_EXTRA_WIDTH 0.2  RPN.SCORE_THRESH 0.2 RCNN.SCORE_THRESH 0.2  USE_IOU_BRANCH False

# epnet_without_li_fusion
CUDA_VISIBLE_DEVICES=0 python eval_rcnn.py --cfg_file cfgs/LI_Fusion_with_attention_use_ce_loss.yaml --eval_mode rcnn_online   --output_dir ./log/Car/models/epnet_without_li_fusion/eval_results/  --ckpt ./log/Car/models/epnet_without_li_fusion/ckpt/checkpoint_epoch_44.pth --set  LI_FUSION.ENABLED False LI_FUSION.ADD_Image_Attention False RCNN.POOL_EXTRA_WIDTH 0.2  RPN.SCORE_THRESH 0.2 RCNN.SCORE_THRESH 0.2  USE_IOU_BRANCH False

# pointrcnn_ori
CUDA_VISIBLE_DEVICES=0 python eval_rcnn.py --cfg_file cfgs/LI_Fusion_with_attention_use_ce_loss.yaml --eval_mode rcnn_online   --output_dir ./log/Car/models/pointrcnn_ori/eval_results/  --ckpt ./log/Car/models/pointrcnn_ori/ckpt/checkpoint_epoch_47.pth --set  LI_FUSION.ENABLED False LI_FUSION.ADD_Image_Attention False RCNN.POOL_EXTRA_WIDTH 0.2  RPN.SCORE_THRESH 0.2 RCNN.SCORE_THRESH 0.2  USE_IOU_BRANCH False

# full_epnet_with_iou_branch
CUDA_VISIBLE_DEVICES=0 python eval_rcnn.py --cfg_file cfgs/LI_Fusion_with_attention_use_ce_loss.yaml --eval_mode rcnn_online    --output_dir ./log/Car/models/full_epnet_with_iou_branch/eval_results/  --ckpt ./log/Car/models/full_epnet_with_iou_branch/ckpt/checkpoint_epoch_46.pth --set  LI_FUSION.ENABLED True LI_FUSION.ADD_Image_Attention True RCNN.POOL_EXTRA_WIDTH 0.2  RPN.SCORE_THRESH 0.2 RCNN.SCORE_THRESH 0.2  USE_IOU_BRANCH True