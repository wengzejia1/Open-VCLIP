ROOT=/home/jia/workspace/Open-VCLIP
CKPT=/data/jia/checkpoint/OpenVCLIP
OUT_DIR=$CKPT/testing

# LOAD_CKPT_FILE=/DDN_ROOT/ytcheng/code/patching_checkpoint/basetraining/temporalclip_vitb16_8x16_interpolation_bugfix_0.5ratio_rand0.0_0.6sample/wa_checkpoints/swa_2_22.pth
LOAD_CKPT_FILE=/home/jia/openvclip-checkpoint/openvclip-b16/swa_2_22.pth
PATCHING_RATIO=0.5

cd $ROOT
# MODEL.TEMPORAL_MODELING_TYPE 'expand_temporal_view' \
python -W ignore -u tools/run_net.py \
    --cfg configs/Kinetics/TemporalCLIP_vitb16_8x16_STAdapter.yaml \
    --opts DATA.PATH_TO_DATA_DIR $ROOT/label_db/weng_compress_full_splits \
    DATA.PATH_PREFIX /data/jia/k400_compress/compress \
    DATA.PATH_LABEL_SEPARATOR , \
    DATA.INDEX_LABEL_MAPPING_FILE $ROOT/label_db/k400-index2cls.json \
    TRAIN.ENABLE False \
    OUTPUT_DIR $OUT_DIR \
    TEST.BATCH_SIZE 480 \
    NUM_GPUS 8 \
    DATA.DECODING_BACKEND "pyav" \
    MODEL.NUM_CLASSES 400 \
    TEST.CUSTOM_LOAD True \
    TEST.CUSTOM_LOAD_FILE $LOAD_CKPT_FILE \
    TEST.SAVE_RESULTS_PATH temp.pyth \
    TEST.NUM_ENSEMBLE_VIEWS 3 \
    TEST.NUM_SPATIAL_CROPS 1 \
    TEST.PATCHING_MODEL True \
    TEST.PATCHING_RATIO $PATCHING_RATIO \
    TEST.CLIP_ORI_PATH ~/.cache/clip/ViT-B-16.pt \
    DATA_LOADER.NUM_WORKERS 4 \


