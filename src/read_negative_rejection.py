import os
import re
import json
import pickle
import argparse

import torch
import pandas as pd
from typing import List, Dict

from utils import str2bool
from normalize_answers import *



def compute_negative_rejection_accuracy(df: pd.DataFrame) -> float:
    num_no_ans = df[df['ans_in_documents'] == False].shape[0]
    num_neg_rej = df[(df['is_no_res'] == True) & (df['ans_in_documents'] == False)].shape[0]

    print(f"Number of samples with no answer in context: {num_no_ans}")
    print(f"Number of samples with negative rejection: {num_neg_rej}")

    if num_no_ans == 0:
        return 0.0

    return round(num_neg_rej / num_no_ans, 4) * 100


def compute_accuracy_correct_answer_not_in_context(df: pd.DataFrame) -> float:
    num_no_ans = df[df['ans_in_documents'] == False].shape[0]
    num_correct_ans_not_in_context = df[(df['ans_match_after_norm'] == True) & (df['ans_in_documents'] == False)].shape[0]

    print(f"Number of samples with no answer in context: {num_no_ans}")
    print(f"Number of samples with correct answer not in context: {num_correct_ans_not_in_context}")

    if num_no_ans == 0:
        return 0.0
    
    return round(num_correct_ans_not_in_context / num_no_ans, 4) * 100


def compute_number_no_proof(df: pd.DataFrame) -> float:
    num_no_proof = (df['proof'] == "NO-PROOF").sum()
    print(f"Number of samples with no proof: {num_no_proof}")

    return round(num_no_proof / len(df), 4) * 100


def check_negative_rejection(generated_answer):
    generated_answer = generated_answer.split('\n', 1)[0]
    ans_in_documents = is_answer_in_text(generated_answer, ['NO-RES'])
    return ans_in_documents


def get_retrieved_path(args):
    padding_str = f"_{args.padding_strategy}{args.model_max_length}" if args.padding_strategy != "longest" else "" 
    chat_template_str = "_template" if args.use_model_chat_template else ""

    filename_prefix = f"numdoc{args.num_doc}_retr{args.num_retrieved_documents}{padding_str}{chat_template_str}_info_"
    return filename_prefix


def get_only_query_path(args):
    chat_template_str = "_template" if args.use_model_chat_template else ""

    filename_prefix = f"only_query{chat_template_str}_info_"
    return filename_prefix


def parse_arguments():
    parser = argparse.ArgumentParser(description="Read Generation Results.")
    parser.add_argument('--output_dir', type=str, default='data/gen_res', help='Output directory')
    parser.add_argument('--dataset', type=str, default='nq', help='Dataset to use')
    parser.add_argument('--llm_id', type=str, default='meta-llama/Llama-2-7b-chat-hf', help='LLM model identifier')
    parser.add_argument('--model_max_length', type=int, help='Maximum input length for the LLM model', default=4096)
    parser.add_argument('--use_model_chat_template', type=str2bool, default=False, help='Whether to use the standard chat/instruct template of the model')
    parser.add_argument('--use_alternative_prompt', type=str2bool, help='Whether to use another prompt for the dataset', default=False)    
    parser.add_argument('--padding_strategy', type=str, help='Padding strategy for the LLM tokenizer', default='longest')
    parser.add_argument('--use_test', type=str2bool, help='Use the test set')
    parser.add_argument('--prompt_type', type=str, default='retrieved', help='Which type of prompt to use [retrieved, retrieved_proof, only_query]')
    parser.add_argument('--num_retrieved_documents', type=int, help='Number of retrieved documents in the context')
    
    args = parser.parse_args()

    if not args.prompt_type in ['retrieved', 'retrieved_proof', 'only_query']:
        parser.error("Invalid prompt type. Must be one of ['retrieved', 'retrieved_proof', 'only_query']")

    return args


info = {
    "nq": {
        "test": 'data/nq/nq-open/test_dataset.json',
    },
    "triviaqa": {
        "test": 'data/triviaqa/triviaqa-unfiltered/main_test.json',
    },
}

def main():
    args = parse_arguments()
    
    retriever_str = ""
    
    prompt_type = args.prompt_type
    if 'retrieved' in prompt_type:
        retriever_str = "contriever/"
        args.num_doc = args.num_retrieved_documents
        filename_prefix = get_retrieved_path(args)
    elif prompt_type == 'only_query':
        filename_prefix = get_only_query_path(args)
    else:
        raise ValueError("Invalid prompt type")


    llm_id = args.llm_id
    split = "test" if args.use_test else "train"
    llm_folder = llm_id.split("/")[1] if '/' in llm_id else llm_id
    doc_str = f"{args.num_doc}_doc" if 'only_query' not in prompt_type else ""
    directory = f'{args.output_dir}/{args.dataset}/{llm_folder}/{split}/{prompt_type}/{retriever_str}{doc_str}'
    print("Directory: ", directory)
    print("Filename Prefix: ", filename_prefix)
    
    df = pd.read_json(os.path.join(directory, f'{filename_prefix}all_extended.json'))        
    df['is_no_res'] = df['generated_answer'].apply(check_negative_rejection)

    neg_rej_accuracy = compute_negative_rejection_accuracy(df)
    print(f"Negative Rejection Accuracy: {neg_rej_accuracy}")
    print()
    correct_ans_not_in_context_accuracy = compute_accuracy_correct_answer_not_in_context(df)
    print(f"Correct Answer Not in Context Accuracy: {correct_ans_not_in_context_accuracy}")
    print()
    
    if 'proof' in directory:
        no_proof_accuracy = compute_number_no_proof(df)
        print(f"No Proof Accuracy: {no_proof_accuracy}")


if __name__ == "__main__":
    main()
