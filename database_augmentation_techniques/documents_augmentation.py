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
    parser.add_argument('--doc_type', type=str, default="documents")
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

    if args.doc_type == "documents":
        doc_pairs = load_dataset(DATASET_SOURCE, 'documents'+DOCUMENT_POSTFIX, cache_dir=CACHE_DIR)[args.task]
    elif args.doc_type == "queries":
        examples = load_dataset(DATASET_SOURCE, 'examples', cache_dir=CACHE_DIR)[args.task]
    else:
        raise ValueError(f"Invalid doc_type: {args.doc_type}")

    os.makedirs(args.output_dir, exist_ok=True)
    model_name = args.model.split("/")[-1]
    if args.doc_type == "documents":
        output_file = os.path.join(
            args.output_dir,
            f"{args.task}_{model_name}_{args.output_token_limit}_{args.doc_id_range}.parquet"
        )
    else:
        output_file = os.path.join(
            args.output_dir,
            f"{args.task}_{model_name}_{args.output_token_limit}_{args.doc_id_range}.json"
        )

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

        if args.doc_type == "queries":
            rewritten_examples = []
            for i,e in tqdm(enumerate(examples)):
                if doc_id_range is not None and (i < doc_id_range[0] or i >= doc_id_range[1]):
                    continue
                cur_post = e["query"].replace('\n', ' ')
                prompt = (f'{cur_post}\n\n'
                        f'Instructions:\n'
                        f'1. Identify the essential problem.\n'
                        f'2. Think step by step to reason and describe what information could be relevant and helpful to address the questions in detail.\n'
                        f'3. Draft an answer with as many thoughts as you have.\n'
                        )
                if args.output_token_limit is not None:
                    prompt += f'Your answer must be written within {args.output_token_limit} tokens.'
                output = model.generate(prompt)
                if output is not None:
                    e['query'] = output
                rewritten_examples.append(e)
            logger.info(f"Saving rewritten examples to {output_file}")
            with open(output_file, 'w') as f:
                json.dump(rewritten_examples, f, indent=2)
        elif args.doc_type == "documents":
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
