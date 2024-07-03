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


# nq: max_new_tokens 15, 200 if proof
# triviaqa: max_new_tokens 50, 200 if proof


python src/generate_answers_llm_mixed.py \
    --output_dir data/gen_res \
    --dataset triviaqa \
    --llm_id meta-llama/Meta-Llama-3-8B \
    --model_max_length 8192 \
    --use_model_chat_template False \
    --load_full_corpus False \
    --num_retrieved_documents 10 \
    --use_task_with_proof False \
    --use_no_rejection_prompt False \
    --use_test True \
    --max_new_tokens 50 \
    --batch_size 4 \
    --save_every 500


python src/generate_answers_llm_mixed.py \
    --output_dir data/gen_res \
    --dataset triviaqa \
    --llm_id meta-llama/Meta-Llama-3-8B-Instruct \
    --model_max_length 8192 \
    --use_model_chat_template False \
    --load_full_corpus False \
    --num_retrieved_documents 10 \
    --use_task_with_proof False \
    --use_no_rejection_prompt False \
    --use_test True \
    --max_new_tokens 50 \
    --batch_size 4 \
    --save_every 500
    

# with chat template
python src/generate_answers_llm_mixed.py \
    --output_dir data/gen_res \
    --dataset triviaqa \
    --llm_id meta-llama/Meta-Llama-3-8B-Instruct \
    --model_max_length 8192 \
    --use_model_chat_template True \
    --load_full_corpus False \
    --num_retrieved_documents 10 \
    --use_task_with_proof False \
    --use_no_rejection_prompt False \
    --use_test True \
    --max_new_tokens 50 \
    --batch_size 4 \
    --save_every 500