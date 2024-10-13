set -ex

API_KEY=$1
MODEL_NAME_OR_PATH=$2
OUTPUT_DIR=${MODEL_NAME_OR_PATH}/math_eval

SPLIT="test"
NUM_TEST_SAMPLE=-1

# English open datasets
DATA_NAME="math"
TOKENIZERS_PARALLELISM=false \
python3 -u eval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --api_key ${API_KEY} \
    --seed 0 \
    --temperature 0.7 \
    --n_sampling 1 \
    --top_p 0.8 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    --overwrite \