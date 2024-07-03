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
from read_negative_rejection import *



def are_answers_matching(prediction: str, ground_truths: List[str]) -> float:
    normalized_prediction = normalize_answer(prediction)

    for ground_truth in ground_truths:
        normalized_ground_truth = normalize_answer(ground_truth)
        if normalized_ground_truth in normalized_prediction:
            return True
    return False


def extract_proof_from_text(text: str) -> str:
    matches = list(re.finditer("Proof:", text))
    
    if matches:
        proof_end = matches[0].end()
        proof = text[proof_end:].strip()
        # Get the text until the first new line
        proof = proof.split('\n', 1)[0] 
    else:
        proof = "NO-PROOF"

    return proof


def compute_df_accuracy(df: pd.DataFrame, attribute: str) -> float:
    return round(df[attribute].sum() / len(df), 4) * 100


def read_generation_results(file_path: str, df: pd.DataFrame) -> List[Dict]:
    data = []
    with open(file_path, "r") as fin:
        file_data = json.load(fin)

        for example in file_data:
            example_ids = example['example_id']
            queries = example['query']
            prompts = example['prompt']
            document_indices = list(zip(*example['document_indices']))
            gold_document_indices = example['gold_document_idx']
            generated_answers = example['generated_answer']
            prompt_tokens_lens = example['prompt_tokens_len']

            for i in range(len(example_ids)):
                example_id = example_ids[i]
                query = queries[i]
                gold_document_idx = gold_document_indices[i]
                documents_idx = list(document_indices[i])
                # After the first new line, LLMs usually generate random text,
                # so it is skipped in the matching comparison
                generated_answer = generated_answers[i].split('\n', 1)[0]
                
                prompt = prompts[i]
                prompt_tokens_len = prompt_tokens_lens[i]

                answers = df[df['example_id'].astype(str) == str(example_id)].answers.iloc[0]
                gold_in_retrieved = False

                if int(gold_document_idx) in map(int, documents_idx):
                    gold_in_retrieved = True

                ans_match_after_norm: bool = are_answers_matching(generated_answer, answers)
                ans_in_documents: bool = is_answer_in_text(prompt, answers)
                data.append({
                    'example_id': str(example_id),
                    'query': query,
                    'prompt': prompt,
                    'document_indices': documents_idx,
                    'gold_document_idx': gold_document_idx,
                    'generated_answer': generated_answers[i],
                    'answers': answers,
                    'ans_match_after_norm': ans_match_after_norm,
                    'gold_in_retrieved': gold_in_retrieved,
                    'ans_in_documents': ans_in_documents,
                    "prompt_tokens_len": prompt_tokens_len,
                })

                if 'proof' in file_path:
                    proof = extract_proof_from_text(generated_answers[i]) 
                    data[-1]['proof'] = proof
                    data[-1]['ans_in_proof'] = is_answer_in_text(proof, [generated_answer])

    return data


def read_generation_results_only_query(file_path: str, df: pd.DataFrame) -> List[Dict]:
    data = []
    with open(file_path, "r") as fin:
        file_data = json.load(fin)

        for example in file_data:
            example_ids = example['example_id']
            queries = example['query']
            prompts = example['prompt']
            generated_answers = example['generated_answer']

            for i in range(len(example_ids)):
                example_id = example_ids[i]
                query = queries[i]
                # After the first new line, LLMs usually generate random text,
                # so it is skipped in the matching comparison
                generated_answer = generated_answers[i].split('\n', 1)[0]
                prompt = prompts[i]

                answers = df[df['example_id'].astype(str) == str(example_id)].answers.iloc[0]

                ans_match_after_norm: bool = are_answers_matching(generated_answer, answers)
                ans_in_documents: bool = is_answer_in_text(prompt, answers)
                data.append({
                    'example_id': str(example_id),
                    'query': query,
                    'prompt': prompt,
                    'generated_answer': generated_answers[i],
                    'answers': answers,
                    'ans_match_after_norm': ans_match_after_norm,
                    'ans_in_documents': ans_in_documents,
                })

    return data


def convert_tensors(cell):
    """ Converts tensors in the given cell to lists, if they are tensors. """
    if isinstance(cell, list):
        return [[t.tolist() if torch.is_tensor(t) else t for t in inner_list] for inner_list in cell]
    return cell


def extract_number_from_filename(filename: str, pattern: re.Pattern) -> int:
    """ Extracts the number from the filename based on the provided pattern. """
    match = pattern.search(filename)
    return int(match.group(1)) if match else 0


def load_pickle_files(directory: str, filename_prefix: str) -> pd.DataFrame:
    """ Loads and concatenates data from all pickle files in the directory with the given prefix. """
    pattern = re.compile(r'(\d+).pkl')
    files = [f for f in os.listdir(directory) if f.endswith('.pkl') and filename_prefix in f]
    files.sort(key=lambda f: extract_number_from_filename(f, pattern))
    print("I'm using the following files: ", files)

    data_list = []
    for file in files:
        with open(os.path.join(directory, file), 'rb') as f:
            data = pickle.load(f)
            data_list.extend(data)
    
    data_df = pd.DataFrame(data_list)
    if 'only_query' in directory:
        if data_df['example_id'].dtype != "O":
            data_df['example_id'] = data_df['example_id'].apply(lambda x: x.tolist())
    else:
        data_df['document_indices'] = data_df['document_indices'].apply(convert_tensors)

    if 'prompt_tokens_len' in data_df.columns:
        data_df['prompt_tokens_len'] = data_df['prompt_tokens_len'].apply(lambda x: x.tolist())
    return data_df


def save_data_to_json(data_df: pd.DataFrame, directory: str, filename_prefix: str):
    """ Saves the given DataFrame to a JSON file. """
    data_path = os.path.join(directory, f'{filename_prefix}all.json')
    # Check if the file already exists
    if os.path.exists(data_path):
        overwrite = input(f"File {data_path} already exists. Overwrite? (y/n): ")
        if overwrite.lower() != 'y':
            print("No overwrite.")

            results_df = pd.read_json(f'{directory}/{filename_prefix}all_extended.json')
            accuracy = compute_df_accuracy(results_df, 'ans_match_after_norm')
            print("ACCURACY: ", accuracy)

            if 'proof' in directory:
                accuracy_ans_in_proof = compute_df_accuracy(results_df, 'ans_in_proof')
                print("ACCURACY ANS IN PROOF", accuracy_ans_in_proof)

            correct_ans_not_in_context_accuracy = compute_accuracy_correct_answer_not_in_context(results_df)
            print(f"Correct Answer Not in Context Accuracy: {correct_ans_not_in_context_accuracy}")

            return None
        
    data_df.to_json(data_path, orient='records')
    return data_path


def get_retrieved_path(args):
    padding_str = f"_{args.padding_strategy}{args.model_max_length}" if args.padding_strategy != "longest" else "" 
    chat_template_str = "_template" if args.use_model_chat_template else ""
    prompt_configuration = "_no_rejection" if args.use_no_rejection_prompt else ""

    filename_prefix = f"numdoc{args.num_doc}_retr{args.num_retrieved_documents}{padding_str}{chat_template_str}{prompt_configuration}_info_"
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
    parser.add_argument('--padding_strategy', type=str, help='Padding strategy for the LLM tokenizer', default='longest')
    parser.add_argument('--use_test', type=str2bool, help='Use the test set')
    parser.add_argument('--prompt_type', type=str, default='retrieved', help='Which type of prompt to use [retrieved, retrieved_proof, only_query]')
    parser.add_argument('--use_no_rejection_prompt', type=str2bool, help='Whether to use the prompt without the NO-RES part', default=False)
    parser.add_argument('--gold_position', type=int, help='The (0-indexed) position of the gold document in the context')
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
    
    retriever_str = "contriever/"

    prompt_type = args.prompt_type
    if 'retrieved' in prompt_type:    
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

    df = pd.read_json(info[args.dataset][split], dtype={'example_id': str})

    data_df = load_pickle_files(directory, filename_prefix)
    data_path = save_data_to_json(data_df, directory, filename_prefix)
    if data_path is None:
        return
    
    if 'only_query' in directory:
        results = read_generation_results_only_query(data_path, df)
    else:
        results = read_generation_results(data_path, df)

    results_df = pd.DataFrame(results)
    accuracy = compute_df_accuracy(results_df, 'ans_match_after_norm')
    print("ACCURACY: ", accuracy)
    if 'proof' in directory:
        accuracy_ans_in_proof = compute_df_accuracy(results_df, 'ans_in_proof')
        print("ACCURACY ANS IN PROOF", accuracy_ans_in_proof)
        
    results_df.to_json(os.path.join(directory, f'{filename_prefix}all_extended.json'), orient='records')

    correct_ans_not_in_context_accuracy = compute_accuracy_correct_answer_not_in_context(results_df)
    print(f"Correct Answer Not in Context Accuracy: {correct_ans_not_in_context_accuracy}")

if __name__ == "__main__":
    main()
