ROOT=/DDN_ROOT/ytcheng/code/Open-VCLIP
CKPT=/DDN_ROOT/ytcheng/code/patching_checkpoint
OUT_DIR=$CKPT/testing

LOAD_CKPT_FILE=/DDN_ROOT/ytcheng/code/patching_checkpoint/basetraining/temporalclip_vitb16_8x16_interpolation_bugfix_0.5ratio_rand0.0_0.6sample/wa_checkpoints/swa_2_22.pth

K600_split=k600_split1
PATCHING_RATIO=0.5

cd $ROOT
# MODEL.TEMPORAL_MODELING_TYPE 'expand_temporal_view'
python -W ignore -u tools/run_net.py \
    --cfg configs/Kinetics/TemporalCLIP_vitb16_8x16_STAdapter.yaml \
    --opts DATA.PATH_TO_DATA_DIR $ROOT/zs_label_db/$K600_split \
    DATA.PATH_PREFIX /dev/shm/k600 \
    DATA.PATH_LABEL_SEPARATOR , \
    DATA.INDEX_LABEL_MAPPING_FILE $ROOT/zs_label_db/$K600_split/k600-openset-index2cls.json \
    TRAIN.ENABLE False \
    OUTPUT_DIR $OUT_DIR \
    TEST.BATCH_SIZE 480 \
    NUM_GPUS 8 \
    DATA.DECODING_BACKEND "pyav" \
    MODEL.NUM_CLASSES 160 \
    TEST.CUSTOM_LOAD True \
    TEST.CUSTOM_LOAD_FILE $LOAD_CKPT_FILE \
    TEST.SAVE_RESULTS_PATH temp.pyth \
    TEST.NUM_ENSEMBLE_VIEWS 3 \
    TEST.NUM_SPATIAL_CROPS 1 \
    TEST.PATCHING_MODEL True \
    TEST.PATCHING_RATIO $PATCHING_RATIO \
    TEST.CLIP_ORI_PATH /root/.cache/clip/ViT-B-16.pt \


