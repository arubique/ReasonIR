import json
import argparse
import time
import os
from datasets import load_dataset
from tqdm import tqdm
import functools
import sys
import pyarrow as pa
import pyarrow.parquet as pq
import re

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(
    0,
    ROOT_PATH
)
# from evaluation.bright.retrievers import (
#     get_scores,
#     calculate_retrieval_metrics
# )
from test_time_techniques.query_rewriting import (
    OpenAIModel,
    GeminiModel,
    HFModel,
    ClaudeModel,
    # format_chat,
    # call_api
)
from utility.utils import set_device_with_most_free_memory
sys.path.pop(0)


CACHE_DIR = os.path.join(ROOT_PATH, "evaluation", "bright", "cache")
DATASET_SOURCE = "xlangai/BRIGHT"
DOCUMENT_POSTFIX = ""


import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# def call_api(func):
#     count = 0
#     while True:
#         try:
#             count += 1
#             output = func()
#             break
#         except Exception as e:
#             logger.info(f"Exception while using api: {e}")
#             if "rate limit" in str(e).lower() or "rate_limit" in str(e).lower():
#                 logger.info("Rate limit exceeded, waiting 60 secs and retrying...")
#                 time.sleep(60)
#             # elif count < 5:
#             #     logger.info("Encountered error, retrying...")
#             #     time.sleep(5)
#             else:
#                 raise ValueError
#                 # logger.info("Skipping generation due to unknown error after 5 retries.")
#                 # output = None
#                 # break
#     return output


# def format_chat(message, include_system=True, system_message="You are a helpful assistant."):
#     if include_system:
#         chat = [{"role": "system", "content": system_message}, {"role": "user", "content": message}]
#     else:
#         chat = [{"role": "user", "content": message}]
#     print(chat)
#     return chat


# class ClaudeModel:

#     def __init__(self, version):
#         from anthropic import AnthropicVertex
#         PROJECT_ID = "xxx"  # @param
#         LOCATION = "xxx"  # @param
#         self.model = AnthropicVertex(region=LOCATION, project_id=PROJECT_ID)
#         self.version = version

#     def generate(self, prompt):
#         inputs = format_chat(prompt, include_system=False)
#         func = functools.partial(
#             self.model.messages.create,
#             max_tokens=2048,
#             messages=inputs,
#             model=self.version,
#             temperature=0.8,
#             top_p=0.8
#         )
#         message = call_api(func)
#         if message is not None:
#             response = json.loads(message.model_dump_json(indent=2))
#             return response['content'][0]['text']
#         return None


# class OpenAIModel:
#     def __init__(self, model_name, temperature=0.8, top_p=0.8, max_tokens=2048):
#         import openai
#         # if api_key is not None:
#         #     openai.api_key = api_key
#         if "azure" in model_name:
#             # env var: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, and OPENAI_API_VERSION
#             self.model = openai.AzureOpenAI()
#             model_name = model_name[model_name.index("/")+1:]
#         else:
#             # make sure to set the OPENAI_API_KEY environment variable
#             self.model = openai.OpenAI()
#         self.model_name = model_name
#         self.temperature = temperature
#         self.top_p = top_p
#         self.max_tokens = max_tokens

#     def generate(self, prompt, system_message="You are a helpful assistant", **kwargs):
#         # kwargs can be used to pass additional parameters to the model: max_tokens, stop, etc.
#         inputs = format_chat(prompt, system_message=system_message)
#         func = functools.partial(
#             self.model.chat.completions.create,
#             model=self.model_name,
#             messages=inputs,
#             max_tokens=self.max_tokens,
#             temperature=self.temperature,
#             top_p=self.top_p,
#             **kwargs,
#         )
#         output = call_api(func)
#         if output is not None:
#             return output.choices[0].message.content
#         return None


# class GeminiModel:
#     def __init__(self, model_name, temperature=0.8, top_p=0.8, max_tokens=2048):
#         import google.generativeai as genai
#         api_key=os.environ["GEMINI_API_KEY"]
#         genai.configure(api_key=api_key)
#         self.model = genai.GenerativeModel(model_name)
#         self.temperature = temperature
#         self.top_p = top_p
#         self.max_tokens = max_tokens
#         self.generation_config = genai.types.GenerationConfig(
#                 max_output_tokens=self.max_tokens,
#                 temperature=self.temperature,
#                 top_p=self.top_p
#             )

#     def generate(self, prompt, system_message="You are a helpful assistant", **kwargs):
#         # kwargs can be used to pass additional parameters to the model: max_tokens, stop, etc.
#         output = self.model.generate_content(
#                     prompt,
#                     generation_config=self.generation_config)
#         # time.sleep(1)
#         if output is not None:
#             try:
#                 return output.text
#             except:
#                 # import pdb; pdb.set_trace()
#                 return prompt.split("\n\nInstructions")[0]
#         return None


# class HFModel:
#     def __init__(self, model_name, temperature, top_p, max_tokens=2048):
#         import torch
#         from transformers import AutoModelForCausalLM, AutoTokenizer
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
#         self.temperature = temperature
#         self.top_p = top_p
#         self.max_tokens = max_tokens

#     def generate(self, message, **kwargs):
#         inputs = self.tokenizer([message], return_tensors="pt").to(self.device)
#         outputs = self.model.generate(
#             **inputs,
#             # max_length=1024,
#             max_new_tokens=512, # added for llama3.1-7B, because max_length runs into errors.
#             temperature=self.temperature,
#             top_p=self.top_p,
#             **kwargs,
#         )
#         text = self.tokenizer.decode(outputs[0, inputs.input_ids.size(1):], skip_special_tokens=True)
#         return text


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    # parser.add_argument('--example_file', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default="cache/reasoning")
    parser.add_argument('--model', type=str, default="gemini-1.5-flash")
    parser.add_argument('--output_token_limit', type=int, default=None)
    # parser.add_argument('--sweep_output_dir', type=str, default=None)
    # parser.add_argument('--api_key', type=str, default=None)
    parser.add_argument('--doc_id_range', type=str, default=None)
    args = parser.parse_args()

    if args.doc_id_range is not None:
        doc_id_range = args.doc_id_range.split(":")
        doc_id_range = [int(doc_id_range[0]), int(doc_id_range[1])]
    else:
        doc_id_range = None

    set_device_with_most_free_memory()

    # if args.api_key is not None:
    #     with open(args.api_key, 'r') as f:
    #         os.environ["OPENAI_API_KEY"] = f.read().strip()

    # if args.example_file is not None:
    #     # supports json and jsonl files
    #     examples = load_dataset("json", data_files=args.example_file)["train"]
    # else:
    #     examples = load_dataset('xlangai/BRIGHT', 'examples')[args.task]

    doc_pairs = load_dataset(DATASET_SOURCE, 'documents'+DOCUMENT_POSTFIX, cache_dir=CACHE_DIR)[args.task]

    os.makedirs(args.output_dir, exist_ok=True)
    model_name = args.model.split("/")[-1]
    output_file = os.path.join(args.output_dir, f"{args.task}_{model_name}_{args.output_token_limit}.parquet")

    if os.path.exists(output_file):
        print(f"{output_file} exists, skipping")

    else:

        if 'claude' in args.model:
            model = ClaudeModel(version=args.model)
        elif 'gpt' in args.model:
            model = OpenAIModel(model_name=args.model, max_tokens=args.output_token_limit)
        elif 'gemini' in args.model:
            model = GeminiModel(model_name=args.model, max_tokens=args.output_token_limit)
        else:
            logger.info(f"Assuming Hugging Face model: {args.model}")
            model = HFModel(model_name=args.model, temperature=1e-9, top_p=1e-9)

        # rewritten_examples = []
        augmented_docs = []
        total_input_len = 0
        total_output_len = 0
        for doc_id, doc_content in tqdm(zip(doc_pairs["id"], doc_pairs["content"])):
            # cur_post = e["query"].replace('\n', ' ')
            doc_id = int(doc_id)
            if doc_id_range is not None and (doc_id < doc_id_range[0] or doc_id >= doc_id_range[1]):
                continue
            cur_doc = doc_content.replace('\n', ' ')
            # prompt = (f'{cur_post}\n\n'
            #         f'Instructions:\n'
            #         f'1. Identify the essential problem.\n'
            #         f'2. Think step by step to reason and describe what information could be relevant and helpful to address the questions in detail.\n'
            #         f'3. Draft an answer with as many thoughts as you have.\n'
            #         )
            #
            if args.task == "theoremqa_theorems":
                # example_problem = (
                #     "A student needs to select 3 books from a shelf containing 10 different books. "
                #     "How many different possible combinations of books can they choose?"
                # )
                prompt = (
                    f"Theorem or definition: {cur_doc}"
                    f"\n\n"
                    f"Read the theorem or definition above and generate "
                    # f"numerous factual question-answer pairs "
                    # f"designed to resemble authentic user "
                    # f"search queries and natural language "
                    # f"variations. "
                    f"mathematical level problems"
                    f"designed to check university students "
                    f"ability to apply the theorem to solve the problem"
                    f"Each problem should "
                    f"accurately and semantically capture "
                    f"important aspects of the theorem or definition, with "
                    f"varying lengths and complexities that "
                    f"mirror patterns from university practice books. "
                    f"Include both shorter, keyword-focused "
                    f"questions such as 'what formula is used to compute number of combinations "
                    # f"Motors' "
                    f"and longer questions like 'A student needs to select 3 books from a shelf containing 10 different books. "
                    f"How many different possible combinations of books can they choose?'"
                    # f"Avoid having very similar questions twice."
                    f"Avoid having duplicate questions or very similar questions."
                    f"Avoid questions similar to the examples I showed. "
                    # f"Avoid questions about number of combinations if this concept is not mentioned in the theorem or definition. "
                    f"Avoid using concepts not mentioned in the theorem or definition, e.g. 'number of combinations' "
                    # f". Incorporate 'how' "
                    # f"and 'why' questions to reflect genuine "
                    # f"user curiosity.
                    f"Avoid using phrases like "
                    f"'according to the text' and abstain from "
                    f"pronouns by specifying names or entities. "
                    f"Ensure questions are not overly formal or "
                    f"artificial, maintaining a natural query "
                    f"style. "
                    # f"Immediately follow each question "
                    # f"with its precise answer on the same line, "
                    # f"formatted as 'Question? Answer', without "
                    # f"any additional formatting or commentary. "
                    f"Each problem should have a number and be on its own line. "
                )
            else:
                raise NotImplementedError(f"Task {args.task} not implemented")
            if args.output_token_limit is not None:
                prompt += f'Your answer must be written within {args.output_token_limit} tokens.'
            total_input_len += len(prompt)
            output = model.generate(prompt)
            total_output_len += len(output)
            # if output is not None:
                # e['query'] = output
                # doc_pairs['content'][int(doc_id)] = cur_doc + "\n\n" + output
            # rewritten_examples.append(e)

            # Manually add newlines before numbered items, as LLM forgets to add them sometimes even if asked
            # output = re.sub(r'(\d+\. )', r'\n\1', output)
            output = re.sub(r'(\d+\. )', r'\n', output)
            # Remove duplicate lines, keeping only the first occurrence
            output_lines = output.split('\n')
            output_lines = list(dict.fromkeys(output_lines))  # Preserves order while keeping first occurrence
            output = '\n\n'.join(output_lines)
            # Remove leading newline if present
            output = output.lstrip('\n')
            # use doc_content as the original document because it has \n
            augmented_docs.append((doc_id, doc_content + "\n\n" + output))
            # print("DEBUG below")
            # if doc_id > 10: # debug
            #     break

        logger.info(f"Total input length: {total_input_len}")
        logger.info(f"Total output length: {total_output_len}")

        logger.info(f"Saving augmented documents to {output_file}")
        # with open(output_file, 'w') as f:
        #     json.dump(augmented_docs, f, indent=2)

        # Convert to table
        table = pa.table([
            pa.array([doc_id for doc_id, _ in augmented_docs]),
            pa.array([doc for _, doc in augmented_docs])
        ], names=['doc_id', 'document'])

        # Write parquet file
        pq.write_table(table, output_file)

    # track successful completion of the run
    # if args.sweep_output_dir:
    #     with open(os.path.join(args.sweep_output_dir, 'done'), 'w') as f:
    #         f.write('done')
