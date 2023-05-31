ROOT=/DDN_ROOT/ytcheng/code/Open-VCLIP
CKPT=/DDN_ROOT/ytcheng/code/patching_checkpoint

cd $ROOT

python -W ignore -u tools/run_net.py \
  --cfg configs/Kinetics/TemporalCLIP_vitl14_8x16_STAdapter.yaml \
  --opts DATA.PATH_TO_DATA_DIR $ROOT/label_db/weng_compress_full_splits \
  DATA.PATH_PREFIX /dev/shm/k400 \
  DATA.PATH_LABEL_SEPARATOR , \
  DATA.INDEX_LABEL_MAPPING_FILE $ROOT/label_db/k400-index2cls.json \
  TRAIN.ENABLE True \
  OUTPUT_DIR $CKPT/basetraining/temporalclip_vitl14_8x16_NOTHING \
  TRAIN.BATCH_SIZE 64 \
  TEST.BATCH_SIZE 240 \
  TEST.NUM_ENSEMBLE_VIEWS 3 \
  TEST.NUM_SPATIAL_CROPS 1 \
  NUM_GPUS 8 \
  SOLVER.MAX_EPOCH 22 \
  SOLVER.WARMUP_EPOCHS 2.0 \
  SOLVER.BASE_LR 3.33e-6 \
  SOLVER.WARMUP_START_LR 3.33e-8 \
  SOLVER.COSINE_END_LR 3.33e-8 \
  TRAIN.MIXED_PRECISION True \
  DATA.DECODING_BACKEND "pyav" \
  MODEL.NUM_CLASSES 400 \
  MODEL.TEMPORAL_MODELING_TYPE 'expand_temporal_view' \
  MIXUP.ENABLE False \
  AUG.ENABLE False \
  AUG.NUM_SAMPLE 1 \
  TRAIN.EVAL_PERIOD 1 \
  TRAIN.CHECKPOINT_PERIOD 1 \
  MODEL.LOSS_FUNC soft_cross_entropy \
  MODEL.USE_CHECKPOINT True \
  MODEL.STATIC_GRAPH True \
