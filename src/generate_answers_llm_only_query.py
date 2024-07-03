import os 
import re
import argparse
import warnings
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from utils import *
from llm import LLM
from default_prompts import *
from prompt_dataset import QueryDataset


os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings('ignore')
SEED=10

info = {
    "nq": {
        "test": {
            "data_path": 'data/nq/nq-open/test_dataset.json',
        }
    },
    "triviaqa": {
        "test": {
            "data_path": 'data/triviaqa/triviaqa-unfiltered/main_test.json',
        }
    },
}


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run LLM Closed-Book Generation (only query).")
    parser.add_argument('--output_dir', type=str, default='data/gen_res', help='Output directory')
    parser.add_argument('--dataset', type=str, default='nq', help='Dataset to use')
    parser.add_argument('--llm_id', type=str, default='meta-llama/Llama-2-7b-chat-hf', help='LLM model identifier')
    parser.add_argument('--model_max_length', type=int, help='Maximum input length for the LLM model', default=4096)
    parser.add_argument('--quantization_bits', type=int, help='If not None, it is the number of bits to use for quantization', default=4)
    parser.add_argument('--use_model_chat_template', type=str2bool, help='Whether to use the standard chat/instruct template of the model', default=False) 
    parser.add_argument('--use_test', type=str2bool, help='Use the test set', default=False)
    parser.add_argument('--max_new_tokens', type=int, help='Maximum number of tokens to generate', default=15)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--save_every', type=int, default=250)

    args = parser.parse_args()
    args.split = "test" if args.use_test else "train"

    if args.use_model_chat_template and args.llm_id not in chat_task_templates:
        parser.error("The model does not have a chat template in the code.")

    return args


def get_prompt_template(args: argparse.Namespace):
    if args.use_model_chat_template:
        chat_task_template_str = chat_task_templates[args.llm_id]['template']
        task_instruction = task_instructions['query_only']

        prompt_template = apply_chat_task_template(
            chat_task_template_str, 
            task_instruction,
            is_query_only_task=True
        )
    else:
        task_template = task_templates["query_only"]
        prompt_template = task_template.create_prompt_template()

    return prompt_template


def initialize_dataset_and_loader(
    args: argparse.Namespace, 
) -> DataLoader:
    
    prompt_template = get_prompt_template(args)

    prompt_ds = QueryDataset(
        data_path=info[args.dataset][args.split]['data_path'],
        prompt_template=prompt_template,
        do_normalize_query=True, 
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
    print("ONLY QUERY")
    print(f"DATA: {info[args.dataset][args.split]['data_path']}")
    print(f"MODEL: {args.llm_id}")
    print(f"MODEL MAX LENGTH: {args.model_max_length}")
    print(f"USE MODEL CHAT TEMPLATE: {args.use_model_chat_template}")
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
        answer_end = matches[0].end()
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
    save_every = args.save_every
    chat_template_str = "_template" if args.use_model_chat_template else ""

    # Create the saving directory
    llm_folder = llm_id.split("/")[1] if '/' in llm_id else llm_id
    saving_dir = f"{args.output_dir}/{args.dataset}/{llm_folder}/{args.split}/only_query"
    os.makedirs(saving_dir, exist_ok=True)

    all_info = []  
    for idx, prompt_batch in enumerate(tqdm(prompt_dataloader)):
        prompts = prompt_batch['prompt']
        generated_output = llm.generate(prompts, max_new_tokens=args.max_new_tokens)
        
        generated_answers = extract_generate_answers(args, generated_output)
        prompt_batch['generated_answer'] = generated_answers
        all_info.append(prompt_batch)
        
        if (idx + 1) % save_every == 0 or (idx + 1) == len(prompt_dataloader):
            print(f"Saving at {idx + 1}...")
            file_name = f"{saving_dir}/only_query{chat_template_str}_info_{idx+1}.pkl"
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
    print("LLM loaded")

    print("Loading prompt dataset...")
    prompt_dataloader = initialize_dataset_and_loader(args)
    print("Prompt dataset loaded")

    print_info(args)
    generate_and_save(args, llm, prompt_dataloader)



if __name__ == '__main__':
    seed_everything(SEED)
    main()