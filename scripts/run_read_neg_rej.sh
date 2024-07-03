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


python src/read_negative_rejection.py \
    --output_dir data/gen_res \
    --dataset triviaqa \
    --llm_id meta-llama/Llama-2-7b-chat-hf \
    --use_model_chat_template False \
    --use_test True \
    --prompt_type retrieved \
    --use_no_rejection_prompt False \
    --num_retrieved_documents 10
