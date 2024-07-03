#!/bin/bash

# llm_id    model_max_length
# tiiuae/falcon-7b   2048
# tiiuae/falcon-7b-instruct  2048
# meta-llama/Llama-2-7b-hf  4096 
# meta-llama/Llama-2-7b-chat-hf 4096
# meta-llama/Meta-Llama-3-8B    8192
# meta-llama/Meta-Llama-3-8B-Instruct   8192
# mistralai/Mistral-7B-v0.1 8192
# mistralai/Mistral-7B-Instruct-v0.1    8192

# nq: max_new_tokens 15
# triviaqa: max_new_tokens 50

CUDA_VISIBLE_DEVICES=0 python src/generate_answers_llm_only_query.py \
    --output_dir data/gen_res \
    --dataset nq \
    --llm_id meta-llama/Llama-2-7b-chat-hf \
    --model_max_length 4096 \
    --use_test True \
    --max_new_tokens 15 \
    --batch_size 50 \
    --save_every 500

