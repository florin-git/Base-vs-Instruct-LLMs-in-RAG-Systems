import os
import torch
import json
import ijson
import pickle
import random
import argparse
import numpy as np
from typing import Dict, List, Optional, Tuple

def seed_everything(seed=10):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

def read_pickle(file_path: str):
    with open(file_path, "rb") as reader:
        data = pickle.load(reader)
    return data


def write_pickle(data, file_path: str):
    with open(file_path, "wb") as writer:
        pickle.dump(data, writer)


def read_json(file_path: str):
    with open(file_path, "rb") as reader:
        data = json.load(reader)
    return data


def write_json(data, file_path: str):
    with open(file_path, "w") as writer:
        json.dump(data, writer)


def read_corpus_json(data_path: str, subset_to_full_idx_map: Optional[Dict[int, int]] = None) -> List[Dict]:
    """
    Reads documents from a JSON file, optionally applying a mapping to adjust the indices from a subset to those in the full corpus.

    Args:
        data_path (str): Path to the JSON file containing the corpus documents.
        subset_to_full_idx_map (Optional[Dict[int, int]]): A mapping from indices in a subset of the corpus
        to their corresponding indices in the full corpus. This is used to adjust document indices when the dataset 
        represents a subset of the original corpus. None implies direct reading without adjustments to indices.

    Returns:
        List[Dict]: A list of dictionaries, each representing a document in the corpus with an adjusted 'full_corpus_idx' key.
    """
    corpus = []
    with open(data_path, "rb") as f:
        for idx, record in enumerate(ijson.items(f, "item")):
            # 'full_corpus_idx' is the original position of the document in the corpus
            if subset_to_full_idx_map and len(subset_to_full_idx_map) != 0:
                record['full_corpus_idx'] = subset_to_full_idx_map[idx]
            else:
                record['full_corpus_idx'] = idx
            corpus.append(record)

    return corpus


def read_subset_corpus_with_map(
    full_to_subset_path: str,
    subset_to_full_path: str,
    corpus_path: str
) -> Tuple[List[Dict], Dict[int, int]]:
    full_to_subset_idx_map = read_pickle(full_to_subset_path)
    subset_to_full_idx_map = read_pickle(subset_to_full_path)
    corpus = read_corpus_json(corpus_path, subset_to_full_idx_map)
    return corpus, full_to_subset_idx_map


def read_subset_corpus(output_dir, dataset, split, file_name):
    full_to_subset_path = f"{output_dir}/{dataset}/mappings/full_to_subset_idx_map_{split}_{file_name}.pkl"
    subset_to_full_path = f"{output_dir}/{dataset}/mappings/subset_to_full_idx_map_{split}_{file_name}.pkl"
    corpus_path = f"{output_dir}/{dataset}/processed/{dataset}_corpus_{split}_{file_name}.json"
    return read_subset_corpus_with_map(
        full_to_subset_path,
        subset_to_full_path,
        corpus_path
    )