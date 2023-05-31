ROOT=/private/home/xyang35/projects/Open-VCLIP
CKPT=/checkpoint/xyang35/experiments/patching_slowfast
cd $ROOT

PARTITION=${1:-"learnaccel"}

# You need to modify: 1. ROOT=  2.CKPT=/checkpoint/$USER/experiments/patching_slowfast
#                     3. DATA.PATH_TO_DATA_DIR $ROOT/label_db/full_splits \
#                     4. check DATA.PATH_LABEL_SEPARATOR and DATA.INDEX_LABEL_MAPPING_FILE
#                     5. DATA.PATH_PREFIX

python -W ignore -u run_with_submitit.py \
  --cfg configs/Kinetics/TemporalCLIP_vitb16_8x16_STAdapter.yaml \
  --job_dir $CKPT \
  --partition $PARTITION \
  --use_volta32 \
  --num_shards 1 \
  DATA.PATH_TO_DATA_DIR /private/home/xyang35/data/k400/gt_288px \
  NUM_GPUS 8 \
  DATA.PATH_PREFIX /datasets01/kinetics/092121/400 \
  DATA.PATH_LABEL_SEPARATOR , \
  DATA.INDEX_LABEL_MAPPING_FILE $ROOT/label_db/k400-index2cls.json \
  TRAIN.ENABLE True \
  TRAIN.BATCH_SIZE 64 \
  TRAIN.MIXED_PRECISION True \
  TRAIN.ZS_CONS False \
  TEST.BATCH_SIZE 128 \
  SOLVER.MAX_EPOCH 22 \
  SOLVER.WARMUP_EPOCHS 2.0 \
  SOLVER.BASE_LR 3.33e-6 \
  SOLVER.WARMUP_START_LR 3.33e-8 \
  SOLVER.COSINE_END_LR 3.33e-8 \
  DATA.DECODING_BACKEND "pyav" \
  MODEL.NUM_CLASSES 400 \
  MODEL.TEMPORAL_MODELING_TYPE 'expand_temporal_view' \
  MIXUP.ENABLE False \
  AUG.ENABLE False \
  AUG.NUM_SAMPLE 1 \
  TRAIN.EVAL_PERIOD 1 \
  TRAIN.CHECKPOINT_PERIOD 1 \
  MODEL.LOSS_FUNC soft_cross_entropy \
  TRAIN.LINEAR_CONNECT_CLIMB True \
  TRAIN.CLIP_ORI_PATH ~/.cache/clip/ViT-B-16.pt \
  TRAIN.LINEAR_CONNECT_LOSS_RATIO 0.5 \
  TRAIN.LINEAR_CONNECT_SAMPLE_L 0.0 \
  TRAIN.LINEAR_CONNECT_SAMPLE_R 0.6 \
# TEST.UPDATE_STATE can be removed, useless now.
