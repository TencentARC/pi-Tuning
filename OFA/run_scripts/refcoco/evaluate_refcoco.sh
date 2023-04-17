#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=6091
export GPUS_PER_NODE=8
user_dir=../../ofa_module
bpe_dir=../../utils/BPE
selected_cols=0,4,2,3


######################## Evaluate Refcoco ##########################

# data=../../dataset/refcoco_data/refcoco_val.tsv
# path=refcoco_checkpoints/large_interpolation_5_1e-4_480/checkpoint_best.pt
# result_path=../../results/refcoco
# split='refcoco_val'
# python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../evaluate.py \
#    ${data} \
#    --path=${path} \
#    --user-dir=${user_dir} \
#    --task=refcoco \
#    --batch-size=16 \
#    --log-format=simple --log-interval=10 \
#    --seed=7 \
#    --gen-subset=${split} \
#    --results-path=${result_path} \
#    --beam=5 \
#    --min-len=4 \
#    --max-len-a=0 \
#    --max-len-b=4 \
#    --no-repeat-ngram-size=3 \
#    --num-workers=0 \
#    --model-overr\ides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}"

#  data=../../dataset/refcoco_data/refcoco_testA.tsv
#  path=refcoco_checkpoints/large_interpolation_5_1e-4_480/checkpoint_best.pt
#  result_path=../../results/refcoco
#  split='refcoco_testA'
#  python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../evaluate.py \
#      ${data} \
#      --path=${path} \
#      --user-dir=${user_dir} \
#      --task=refcoco \
#      --batch-size=32 \
#      --log-format=simple --log-interval=10 \
#      --seed=7 \
#      --gen-subset=${split} \
#      --results-path=${result_path} \
#      --beam=5 \
#      --min-len=4 \
#      --max-len-a=0 \
#      --max-len-b=4 \
#      --no-repeat-ngram-size=3 \
#      --num-workers=0 \
#      --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}"

# data=../../dataset/refcoco_data/refcoco_testB.tsv
# path=refcoco_checkpoints/large_interpolation_5_1e-4_480/checkpoint_best.pt
# result_path=../../results/refcoco
# split='refcoco_testB'
# python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../evaluate.py \
#     ${data} \
#     --path=${path} \
#     --user-dir=${user_dir} \
#     --task=refcoco \
#     --batch-size=16 \
#     --log-format=simple --log-interval=10 \
#     --seed=7 \
#     --gen-subset=${split} \
#     --results-path=${result_path} \
#     --beam=5 \
#     --min-len=4 \
#     --max-len-a=0 \
#     --max-len-b=4 \
#     --no-repeat-ngram-size=3 \
#     --num-workers=0 \
#     --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}"



# # ####################### Evaluate Refcocoplus ##########################
# data=../../dataset/refcocoplus_data/refcocoplus_val.tsv
# path=refcocoplus_checkpoints/large_interpolation_5_1e-4_480/checkpoint_best.pt
# result_path=../../results/refcocoplus
# split='refcocoplus_val'
# python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../evaluate.py \
#     ${data} \
#     --path=${path} \
#     --user-dir=${user_dir} \
#     --task=refcoco \
#     --batch-size=16 \
#     --log-format=simple --log-interval=10 \
#     --seed=7 \
#     --gen-subset=${split} \
#     --results-path=${result_path} \
#     --beam=5 \
#     --min-len=4 \
#     --max-len-a=0 \
#     --max-len-b=4 \
#     --no-repeat-ngram-size=3 \
#     --num-workers=0 \
#     --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}"

# data=../../dataset/refcocoplus_data/refcocoplus_testA.tsv
# path=refcocoplus_checkpoints/large_interpolation_5_1e-4_480/checkpoint_best.pt
# result_path=../../results/refcocoplus
# split='refcocoplus_testA'
# python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../evaluate.py \
#     ${data} \
#     --path=${path} \
#     --user-dir=${user_dir} \
#     --task=refcoco \
#     --batch-size=16 \
#     --log-format=simple --log-interval=10 \
#     --seed=7 \
#     --gen-subset=${split} \
#     --results-path=${result_path} \
#     --beam=5 \
#     --min-len=4 \
#     --max-len-a=0 \
#     --max-len-b=4 \
#     --no-repeat-ngram-size=3 \
#     --num-workers=0 \
#     --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}"

# data=../../dataset/refcocoplus_data/refcocoplus_testB.tsv
# path=refcocoplus_checkpoints/large_interpolation_5_1e-4_480/checkpoint_best.pt
# result_path=../../results/refcocoplus
# split='refcocoplus_testB'
# python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../evaluate.py \
#     ${data} \
#     --path=${path} \
#     --user-dir=${user_dir} \
#     --task=refcoco \
#     --batch-size=16 \
#     --log-format=simple --log-interval=10 \
#     --seed=7 \
#     --gen-subset=${split} \
#     --results-path=${result_path} \
#     --beam=5 \
#     --min-len=4 \
#     --max-len-a=0 \
#     --max-len-b=4 \
#     --no-repeat-ngram-size=3 \
#     --num-workers=0 \
#     --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}"



######################## Evaluate Refcocog ##########################
data=../../dataset/refcocog_data/refcocog_val.tsv
path=refcocog_checkpoints/large_interpolation_5_1e-4_480/checkpoint.best_score_0.8650.pt
result_path=../../results/refcocog
split='refcocog_val'
python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../evaluate.py \
    ${data} \
    --path=${path} \
    --user-dir=${user_dir} \
    --task=refcoco \
    --batch-size=16 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --beam=5 \
    --min-len=4 \
    --max-len-a=0 \
    --max-len-b=4 \
    --no-repeat-ngram-size=3 \
    --num-workers=0 \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}"

data=../../dataset/refcocog_data/refcocog_test.tsv
path=refcocog_checkpoints/large_interpolation_5_1e-4_480/checkpoint.best_score_0.8650.pt
result_path=../../results/refcocog
split='refcocog_test'
python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../evaluate.py \
    ${data} \
    --path=${path} \
    --user-dir=${user_dir} \
    --task=refcoco \
    --batch-size=16 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --beam=5 \
    --min-len=4 \
    --max-len-a=0 \
    --max-len-b=4 \
    --no-repeat-ngram-size=3 \
    --num-workers=0 \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}"
