# A Tale of Trust and Accuracy: Base vs. Instruct LLMs in RAG Systems

This repository contains the code and data to reproduce the experiments from the paper [A Tale of Trust and Accuracy: Base vs. Instruct LLMs in RAG Systems](https://arxiv.org/abs/2406.14972).

## Installation

1. Set up a conda environment.

```
conda create -n base_vs_instruct python=3.9 --yes
conda activate base_vs_instruct
```

2. Install package and requirements.

```
pip install -r requirements.txt
```

## Data
We use two open-domain datasets for the experiments: Natural Questions (open) and TriviaQA-unfiltered. The test sets are located in the `data` folder, under the datasets' respective names. 


The corpus originates from the English Wikipedia (Dec. 20, 2018), where each document is segmented into non-overlapping passages of 100 words. 
After duplicate removal, the resulting corpus contains 20,970,784 entries, each consisting of a passage and the title of the originating Wikipedia page.

### TriviaQA

We employ the validation split of the TriviaQA-unfiltered dataset, which includes 11,313 entries. For the purposes of our experiments, this split is referred to as the test set since the actual test set lacks ground-truth answers.

The original dataset is downloaded from HuggingFace:
```
from datasets import load_dataset
dataset = load_dataset("mandarjoshi/trivia_qa", "unfiltered")
```
After processing, an entry is structured as follows:
```
Ex.
{
    'example_id': 'odql_3275',
    'question': "Who played the part of 'The Penguin' in the TV series 'Batman'?",
    'answers': ['Oliver Burgess Meredith', 'Burgess Meredith', ...],
}
```
 For this dataset we used the corpus described above that can be downloaded from HuggingFace:
```
from datasets import load_dataset
triviaqa_corpus = load_dataset('florin-hf/wiki_dump2018_no_duplicates')
```



### Natural Questions

The NQ dataset is processed as in ["The power of Noise"](https://arxiv.org/abs/2401.14887) paper. It includes a test set of 2,889 examples, available for download:
```
from datasets import load_dataset
dataset = load_dataset('florin-hf/nq_open_gold')
```

A sample in the dataset has the following format:
```
Ex.
{
    'example_id': -3440030035760311385,
    'question': 'who owned the millennium falcon before han solo',
    'answers': [Lando Calrissian],
    'text': "Han Solo won the Millennium Falcon from Lando Calrissian in the card game ' sabacc ' several years before the events of the film A New Hope..."
    'idx_gold_in_corpus': 20995349
}
```
Although the gold document is present, it is not used in these experiments. 
In this case, we use the Wikipedia corpus described above, also including the gold documents, as specified in ["The power of Noise"](https://github.com/florin-git/The-Power-of-Noise) repository. This corpus can be easily downloaded from HuggingFace:
```
from datasets import load_dataset
nq_corpus = load_dataset('florin-hf/wiki_dump2018_nq_open')
```


Data not present in this repository or not downloadable from HuggingFace is available in this [Google Drive](https://drive.google.com/drive/folders/131q2tiJBPBwE72aQ0YaTQNlKFEbwfnDW?usp=sharing).



#### Subsets of the Corpus 
Considering the substantial memory requirements (~25Gb) for loading the entire corpus, we provide subsets tailored to specific experiments, reducing the RAM footprint.

A subset contains only the documents present in the search results by the retriever. In this way, when running the generation, it is not needed to load the entire corpus in RAM, but only the documents that could possibly be included in the prompt of the LLMs. To load only the subset, set `load_full_corpus` to `False`, as can be seen in the examples of the **Generation** section. 

These subsets can be found in the Google Drive under the folder `data/processed`. 




## RAG Steps

### 1. Retrieval

In the first phase of a RAG system, a retriever is employed to search the top-ranked documents based on a given similarity metric. In these experiments we used Contriever. The search results of retriever are located in the `search_results` subfolder within the data directory (e.g., `data/nq/search_results/contriever_IP_test_search_results_at150.pkl`). Each result is a tuple containing in the first position the indices of the top-ranked documents in the corpus; and as second position their corresponding scores. In the case of dense retriever, an Inner Product (IP) search was adopted, thus the higher the score the closer the embeddings in the vector space.

The following three steps were used to compute the search results:
##### 1. Compute Corpus Embeddings
The `compute_corpus_embeddings.py` script computes embeddings in batches, storing them in the `output_dir`.

##### 2. Index Embeddings:
The `index_embeddings.py` script concatenates the piecewise stored embeddings and creates the index.
- Single Index: Leave `percentages_for_index_splitting` empty.
- Multiple Indices: Specify splitting percentages to create multiple indices that can be loaded into different GPUs.

##### 3. Retrieve Top-k Documents
The `compute_search_results.py` script retrieves the top-k documents for the given queries using the FAISS index/indices created earlier.

### 2. Generation

We test different prompt structures across 8 LLMs. The table below summarizes the LLMs used:
| Base LLM | Instruct LLM |
|-----------|---------|
| [Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) | [Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) |
| [Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)  | [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) |
| [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) | [Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) |
| [falcon-7b](https://huggingface.co/tiiuae/falcon-7b) | [falcon-7b-instruct](https://huggingface.co/tiiuae/falcon-7b-instruct) |


#### Closed-Book QA

In the closed-book QA configuration, the system generates answers based solely on the question, without external knowledge. The script `src/generate_answers_llm_only_query.py` allows to generate responses using only the task instruction and the query. A corresponding example script can be found in the file `scripts/run_generation_only_query.sh`.


#### Generate Answers under Task Instruction I

This setup aims to replicate the first table of the paper, where it is used Task Instruction I. In this case, the model is asked to extract the answer from the provided documents, or respond with *NO-RES* if no answer is present in the documents.

The script `src/generate_answers_llm.py` is used to manage this setup. For instance, to reproduce a scenario where we use the instruct version of Llama 3 without using its template, we can run the script below:
```
python src/generate_answers_llm_mixed.py \
    --output_dir data/gen_res \
    --dataset triviaqa \
    --llm_id meta-llama/Meta-Llama-3-8B-Instruct \
    --model_max_length 8192 \
    --use_model_chat_template False \
    --load_full_corpus False \
    --num_retrieved_documents 10 \
    --use_task_with_proof False \
    --use_test True \
    --max_new_tokens 50 \
    --batch_size 4 \
    --save_every 500
```

In particular, with the parameter `use_model_chat_template` set to `False` we are specifying to not use the model template. Here, we are using 10 retrieved documents as specified by the `num_retrieved_documents` parameter.

#### Generate Answers under Task Instruction II

In the second table of the paper, we employ Task Instruction II, which requires models to provide a *Proof* of their answers. To implement this, we use the `src/generate_answers_llm.py` script with the `use_task_with_proof` parameter set to `True`. This is the main distinction from the setup described above:
```
python src/generate_answers_llm_mixed.py \
    --output_dir data/gen_res \
    --dataset triviaqa \
    --llm_id meta-llama/Meta-Llama-3-8B-Instruct \
    --model_max_length 8192 \
    --use_model_chat_template True \
    --load_full_corpus False \
    --num_retrieved_documents 10 \
    --use_task_with_proof True \
    --use_test True \
    --max_new_tokens 200 \
    --batch_size 4 \
    --save_every 500
```
In this case, `use_model_chat_template` is set to `True` 
which means that the model's predefined template is used.


#### Experiments with No Rejection Prompt
We also test models without the *NO-RES* requirement in the prompt. In this case, we can use the parameter `use_no_rejection_prompt` set to `True`. An examples is as follows:
```
python src/generate_answers_llm_mixed.py \
    --output_dir data/gen_res \
    --dataset triviaqa \
    --llm_id meta-llama/Meta-Llama-3-8B-Instruct \
    --model_max_length 8192 \
    --use_model_chat_template True \
    --load_full_corpus False \
    --num_retrieved_documents 10 \
    --use_task_with_proof True \
    --use_no_rejection_prompt True \
    --use_test True \
    --max_new_tokens 50 \
    --batch_size 4 \
    --save_every 500
```

## 3. Evaluation
LLM responses are evaluated based on accuracy, which is defined as whether at least one of the predefined correct answers is contained within the response produced by the LLM. 

The following command is used to compute accuracy:
```
python src/read_generation_results.py \
    --output_dir data/gen_res \
    --dataset triviaqa \
    --llm_id meta-llama/Meta-Llama-3-8B-Instruct \
    --use_model_chat_template False \
    --use_test True \
    --prompt_type retrieved \
    --use_no_rejection_prompt False \
    --num_retrieved_documents 10
```

In addition to accuracy, we also compute the *recall from parametric memory* and the *negative rejection rates*. These metrics can be computed with the `src/read_negative_rejection.py` script. An example can be found in the `scripts` folder.



## References
If you find this repository useful, please consider giving a star and citing this work:
```
@misc{cuconasu2024taletrustaccuracybase,
      title={A Tale of Trust and Accuracy: Base vs. Instruct LLMs in RAG Systems}, 
      author={Florin Cuconasu and Giovanni Trappolini and Nicola Tonellotto and Fabrizio Silvestri},
      year={2024},
      eprint={2406.14972},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.14972}, 
}
```
