
from langchain_core.prompts import PromptTemplate

class TaskTemplate:
    def __init__(self, instruction, query_last=True, query_word="Question"):
        self.instruction = instruction
        self.query_last = query_last
        self.query_word = query_word
        self.prompt_template_str = f"{self.instruction}\n"
        self.setup_template()

    def setup_template(self):
        if self.query_last:
            prompt_str = "Documents:\n{context}\n" + self.query_word + ": {query}\nAnswer:"
        else:
            prompt_str = self.query_word + ": {query}\nDocuments:\n{context}\nAnswer:"
        self.prompt_template_str += prompt_str

    def create_prompt_template(self):
        return PromptTemplate.from_template(template=self.prompt_template_str)


class QueryOnlyTaskTemplate(TaskTemplate):
    # Override setup_template
    def setup_template(self):
        self.prompt_template_str = f"{self.instruction}\n"
        self.prompt_template_str += self.query_word + ": {query}\nAnswer:"


def apply_chat_task_template(
    chat_task_template_str: str, 
    task_instruction: str,
    is_query_only_task: bool = False    
):
    # Insert the task instruction in the chat template of the model
    chat_task_template = PromptTemplate.from_template(
        template=chat_task_template_str,
        partial_variables={"task_instruction": task_instruction}
    )

    # Create the template of the context with an empty instruction, since it was passed in the chat template 
    if is_query_only_task:
        context_template = QueryOnlyTaskTemplate("").create_prompt_template()
    else:
        context_template = TaskTemplate("").create_prompt_template()
    complete_task_template_str = chat_task_template.format(
        context_prompt=context_template.template
    )

    return PromptTemplate.from_template(template=complete_task_template_str)


task_instructions = {
    "query_only": "You are given a question and you MUST respond with a short answer based on your internal knowledge. If you do not know the answer, please respond with NO-RES.",
    "triviaqa": "You are given a question and you MUST respond by EXTRACTING the answer from one of the provided documents. If none of the documents contain the answer, respond with NO-RES.",
    "nq": "You are given a question and you MUST respond by EXTRACTING the answer (max 5 tokens) from one of the provided documents. If none of the documents contain the answer, respond with NO-RES.",
    "qa_proof": {
        "triviaqa": "You are given a question and you MUST respond by EXTRACTING the answer from one of the provided documents. If none of the documents contain the answer, respond with NO-RES. In addition, you must report the portion of the document (Proof) containing the answer.\nSTART example\nDocument [20970787](Title: Ancient Egyptian technology) Evidence indicates that Egyptians made use of potter 's wheels in the manufacturing of pottery from as early as the 4th Dynasty . Chariots , however , are only believed to have been introduced by the invasion of the Hyksos in the Second Intermediate period ; during the New Kingdom era , chariotry became central to Egypt 's military .\nQuestion: when was the potter's wheel first used in egypt\nAnswer: 4th Dynasty\nProof: Evidence indicates that Egyptians made use of potter 's wheels in the manufacturing of pottery from as early as the 4th Dynasty .\nEND example\n", 
        "nq": "You are given a question and you MUST respond by EXTRACTING the answer (max 5 tokens) from one of the provided documents. If none of the documents contain the answer, respond with NO-RES. In addition, you must report the portion of the document (Proof) containing the answer.\nSTART example\nDocument [20970787](Title: Ancient Egyptian technology) Evidence indicates that Egyptians made use of potter 's wheels in the manufacturing of pottery from as early as the 4th Dynasty . Chariots , however , are only believed to have been introduced by the invasion of the Hyksos in the Second Intermediate period ; during the New Kingdom era , chariotry became central to Egypt 's military .\nQuestion: when was the potter's wheel first used in egypt\nAnswer: 4th Dynasty\nProof: Evidence indicates that Egyptians made use of potter 's wheels in the manufacturing of pottery from as early as the 4th Dynasty .\nEND example\n",
        "triviaqa_no_rejection": "You are given a question and you MUST respond by EXTRACTING the answer from one of the provided documents. In addition, you must report the portion of the document (Proof) containing the answer.\nSTART example\nDocument [20970787](Title: Ancient Egyptian technology) Evidence indicates that Egyptians made use of potter 's wheels in the manufacturing of pottery from as early as the 4th Dynasty . Chariots , however , are only believed to have been introduced by the invasion of the Hyksos in the Second Intermediate period ; during the New Kingdom era , chariotry became central to Egypt 's military .\nQuestion: when was the wheel first used in egypt\nAnswer: 4th Dynasty\nProof: Evidence indicates that Egyptians made use of potter 's wheels in the manufacturing of pottery from as early as the 4th Dynasty .\nEND example\n", 
        "nq_no_rejection": "You are given a question and you MUST respond by EXTRACTING the answer (max 5 tokens) from one of the provided documents. In addition, you must report the portion of the document (Proof) containing the answer.\nSTART example\nDocument [20970787](Title: Ancient Egyptian technology) Evidence indicates that Egyptians made use of potter 's wheels in the manufacturing of pottery from as early as the 4th Dynasty . Chariots , however , are only believed to have been introduced by the invasion of the Hyksos in the Second Intermediate period ; during the New Kingdom era , chariotry became central to Egypt 's military .\nQuestion: when was the wheel first used in egypt\nAnswer: 4th Dynasty\nProof: Evidence indicates that Egyptians made use of potter 's wheels in the manufacturing of pottery from as early as the 4th Dynasty .\nEND example\n",

    },
    "triviaqa_no_rejection": "You are given a question and you MUST respond by EXTRACTING the answer from one of the provided documents.",
    "nq_no_rejection": "You are given a question and you MUST respond by EXTRACTING the answer (max 5 tokens) from one of the provided documents.",
}


task_templates = {
    "query_only": QueryOnlyTaskTemplate(task_instructions['query_only']),
    "triviaqa": TaskTemplate(task_instructions['triviaqa']),
    "nq": TaskTemplate(task_instructions['nq']),
    "qa_proof": {
        "triviaqa": TaskTemplate(task_instructions['qa_proof']['triviaqa']),
        "nq": TaskTemplate(task_instructions['qa_proof']['nq']),
        "triviaqa_no_rejection": TaskTemplate(task_instructions['qa_proof']['triviaqa_no_rejection']),
        "nq_no_rejection": TaskTemplate(task_instructions['qa_proof']['nq_no_rejection']),
    },
    "triviaqa_no_rejection": TaskTemplate(task_instructions['triviaqa_no_rejection']),
    "nq_no_rejection": TaskTemplate(task_instructions['nq_no_rejection']),
}


chat_task_templates = {
    'meta-llama/Llama-2-7b-chat-hf': {
        "template": "[INST] <<SYS>>\n{task_instruction}\n<</SYS>>\n{context_prompt} [/INST]",
        "answer_prefix": r"Answer: [/INST]"
    },
    'meta-llama/Meta-Llama-3-8B-Instruct': {
        "template": "<|start_header_id|>system<|end_header_id|>\n\n{task_instruction}<|eot_id|><|start_header_id|>user<|end_header_id|>\n{context_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "answer_prefix": "assistant\n\n"
    },
    'mistralai/Mistral-7B-Instruct-v0.1': {
        "template": "[INST] {task_instruction} {context_prompt} [/INST]",
        "answer_prefix": r"[/INST]"
    },
}