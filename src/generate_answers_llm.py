import os
import re 
import argparse
import warnings
from tqdm import tqdm
from typing import Tuple, Dict, Optional

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from utils import *
from llm import LLM
from default_prompts import *
from prompt_dataset import PromptDataset


os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings('ignore')
SEED=10

info = {
    "nq": {
        "test": {
            "data_path": 'data/nq/nq-open/test_dataset.json',
            "contriever_search_results_path": "data/nq/search_results/contriever_IP_test_search_results_at150.pkl",
        }
    },
    "triviaqa": {
        "test": {
            "data_path": 'data/triviaqa/triviaqa-unfiltered/main_test.json',
            "contriever_search_results_path": "data/triviaqa/search_results/contriever_IP_test_search_results_at50.pkl",
        }
    },
}


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run LLM Generation with retrieved documents.")
    parser.add_argument('--output_dir', type=str, default='data/gen_res', help='Output directory')
    parser.add_argument('--dataset', type=str, default='nq', help='Dataset to use')
    parser.add_argument('--llm_id', type=str, default='meta-llama/Llama-2-7b-chat-hf', help='LLM model identifier')
    parser.add_argument('--model_max_length', type=int, help='Maximum input length for the LLM model', default=4096)
    parser.add_argument('--quantization_bits', type=int, help='If not 0, it is the number of bits to use for quantization', default=4)
    parser.add_argument('--use_model_chat_template', type=str2bool, help='Whether to use the standard chat/instruct template of the model', default=False)    
    parser.add_argument('--load_full_corpus', type=str2bool, help='Load the full corpus', default=True)    
    parser.add_argument('--gold_position', type=int, help='The (0-indexed) position of the gold document in the context', default=None)
    parser.add_argument('--num_retrieved_documents', type=int, help='Number of retrieved documents in the context')
    parser.add_argument('--use_test', type=str2bool, help='Use the test set', default=True)
    parser.add_argument('--padding_strategy', type=str, help='Padding strategy for the LLM tokenizer', default='longest')
    parser.add_argument('--max_new_tokens', type=int, help='Maximum number of tokens to generate', default=15)
    parser.add_argument('--use_task_with_proof', type=str2bool, help='Whether to consider the task where the LLM should also generate the proof containing the answer', default=False)
    parser.add_argument('--use_no_rejection_prompt', type=str2bool, help='Whether to use the prompt without the NO-RES part', default=False)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--save_every', type=int, default=250)

    args = parser.parse_args()
    args.split = "test" if args.use_test else "train"
    args.num_documents_in_context = args.num_retrieved_documents

    if args.num_retrieved_documents is None or args.num_retrieved_documents <= 0:
        parser.error("'num_retrieved_documents' must be specified")

    if args.use_model_chat_template and args.llm_id not in chat_task_templates:
        parser.error("The model does not have a chat template in the code.")

    return args


def load_corpus(
    args: argparse.Namespace
) -> Tuple[List[Dict], Optional[Dict[int, int]]]:
    # Load the corpus
    if args.load_full_corpus:
        if args.dataset == "nq":
            corpus_path = 'data/corpus/wiki_dump2018_nq_open.json'
        elif args.dataset == "triviaqa":
            corpus_path = 'data/corpus/wiki_dec_2018_no_duplicates.json'
        else:
            raise ValueError("Invalid dataset")
        
        corpus = read_corpus_json(corpus_path)
        return corpus, None

    corpus, full_to_subset_idx_map = read_subset_corpus(
        'data', args.dataset, args.split, 'contriever'
    )
    
    return corpus, full_to_subset_idx_map


def load_search_results(args: argparse.Namespace) -> List[Tuple[List[int], List[float]]]:
    search_results_path = info[args.dataset][args.split]['contriever_search_results_path']
    retriever_search_results = read_pickle(search_results_path)

    return retriever_search_results


def get_prompt_template(args: argparse.Namespace):
    prompt_configuration = args.dataset
    if args.use_no_rejection_prompt:
        prompt_configuration = f'{args.dataset}_no_rejection'

    if args.use_model_chat_template:
        chat_task_template_str = chat_task_templates[args.llm_id]['template']
        
        task_instruction = task_instructions[prompt_configuration]
        if args.use_task_with_proof:
            task_instruction = task_instructions['qa_proof'][prompt_configuration]

        prompt_template = apply_chat_task_template(chat_task_template_str, task_instruction)
    else:
        task_template = task_templates[prompt_configuration]

        if args.use_task_with_proof:
            task_template = task_templates['qa_proof'][prompt_configuration]

        prompt_template = task_template.create_prompt_template()

    return prompt_template


def initialize_dataset_and_loader(
    args: argparse.Namespace, 
    corpus: List[Dict], 
    full_to_subset_idx_map: Optional[Dict[int, int]], 
    retriever_search_results: List[Tuple[List[int], List[float]]], 
    tokenizer: PreTrainedTokenizer
) -> DataLoader:
    
    prompt_template = get_prompt_template(args)
    
    prompt_ds = PromptDataset(
        corpus=corpus, data_path=info[args.dataset][args.split]['data_path'], 
        tokenizer=tokenizer, 
        max_tokenized_length=args.model_max_length - 2, 
        search_results=retriever_search_results,
        prompt_template=prompt_template,
        full_to_subset_idx_map=full_to_subset_idx_map,
        do_normalize_query=True, 
        num_documents_in_context=args.num_retrieved_documents,
        gold_position=args.gold_position, # None in these experiments
    )
        
    prompt_dataloader = DataLoader(
        prompt_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    return prompt_dataloader


def print_info(args: argparse.Namespace):
    print("INFO:")    
    print(f"DATA: {info[args.dataset][args.split]['data_path']}")
    print(f"USE TEST: {args.use_test}")
    print(f"MODEL: {args.llm_id}")
    print(f"MODEL MAX LENGTH: {args.model_max_length}")
    print(f'MAX NEW TOKENS: {args.max_new_tokens}')
    print(f"USE MODEL CHAT TEMPLATE: {args.use_model_chat_template}")
    print(f"TASK WITH PROOF:", args.use_task_with_proof)
    print(f'NO REJECTION PROMPT: {args.use_no_rejection_prompt}')
    print(f"GOLD POSITION: {args.gold_position}")
    print(f"NUM DOCUMENTS IN CONTEXT: {args.num_documents_in_context}")
    print(f"BATCH SIZE: {args.batch_size}")
    print(f"SAVE EVERY: {args.save_every}")


def extract_generate_answers(
    args: argparse.Namespace, 
    generated_output: List[str]
) -> List[str]:
    answer_prefix = "Answer:"
    if args.use_model_chat_template:
        answer_prefix = re.escape(chat_task_templates[args.llm_id]['answer_prefix'])

    generated_answers = []
    for output in generated_output:
        matches = list(re.finditer(answer_prefix, output))
        match_idx = 0

        # When using the proof there is a one-shot example that already 
        # contains the string "Answer:". Thus, we should get the second (match_idx=1) match.
        if args.use_task_with_proof:
            match_idx = 1
            if args.use_model_chat_template and answer_prefix != "Answer:":
                match_idx = 0
 
        answer_end = matches[match_idx].end()
        response = output[answer_end:].strip()
        generated_answers.append(response)
    
    return generated_answers


def generate_and_save(
    args: argparse.Namespace, 
    llm: LLM, 
    prompt_dataloader: DataLoader
):
    # Info from arguments
    llm_id = args.llm_id
    num_doc = args.num_documents_in_context
    save_every = args.save_every
    retriever_str = "contriever"
    padding_str = f"_{args.padding_strategy}{args.model_max_length}" if args.padding_strategy != "longest" else "" 
    chat_template_str = "_template" if args.use_model_chat_template else ""
    prompt_type = "retrieved_proof" if args.use_task_with_proof else "retrieved"
    prompt_configuration = "_no_rejection" if args.use_no_rejection_prompt else ""

    # Create the saving directory
    llm_folder = llm_id.split("/")[1] if '/' in llm_id else llm_id
    saving_dir = f"{args.output_dir}/{args.dataset}/{llm_folder}/{args.split}/{prompt_type}/{retriever_str}/{num_doc}_doc"
    os.makedirs(saving_dir, exist_ok=True)

    all_info = []  
    for idx, prompt_batch in enumerate(tqdm(prompt_dataloader)):
        prompts = prompt_batch['prompt']
        generated_output = llm.generate(
            prompts,
            padding_strategy=args.padding_strategy, 
            max_new_tokens=args.max_new_tokens
        )

        generated_answers = extract_generate_answers(args, generated_output)
        prompt_batch['generated_answer'] = generated_answers
        all_info.append(prompt_batch)
        
        if (idx + 1) % save_every == 0 or (idx + 1) == len(prompt_dataloader):
            print(f"Saving at {idx + 1}...")
            file_name = f"{saving_dir}/numdoc{num_doc}_retr{args.num_retrieved_documents}{padding_str}{chat_template_str}{prompt_configuration}_info_{idx+1}.pkl"
            write_pickle(all_info, file_name)
            all_info = []


def main():
    args = parse_arguments()

    print("Loading LLM...")
    llm_id = args.llm_id
    llm = LLM(
        llm_id, device, 
        quantization_bits=args.quantization_bits, 
        model_max_length=args.model_max_length
    )
    tokenizer = llm.tokenizer
    print("LLM loaded")


    print("Loading corpus and search results...")
    corpus, full_to_subset_idx_map = load_corpus(args)
    retriever_search_results = load_search_results(args)
    print("Corpus and search results loaded")


    print("Loading prompt dataset...")
    prompt_dataloader = initialize_dataset_and_loader(
        args, corpus, full_to_subset_idx_map, 
        retriever_search_results, tokenizer
    )
    print("Prompt dataset loaded")

    print_info(args)
    generate_and_save(args, llm, prompt_dataloader)



if __name__ == "__main__":
    seed_everything(SEED)
    main()