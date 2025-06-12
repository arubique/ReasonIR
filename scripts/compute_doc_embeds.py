import os
import json
import torch
from transformers import AutoTokenizer, AutoModel
import pyarrow.parquet as pq
import sys
import argparse

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# local imports
sys.path.insert(
    0,
    ROOT_PATH
)
from utility.utils_for_notebooks import compute_doc_embeds
sys.path.pop(0)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="theoremqa_theorems")
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--doc_ranges", type=str, required=True) # 0:5000,5000:10000,10000:15000,15000:20000,20000:25000
    parser.add_argument("--model", type=str, default="Llama-3.1-8B-Instruct")
    args = parser.parse_args()

    augmented_docs_path_prefix = os.path.join(
        ROOT_PATH,
        "output",
        "documents_augmentation_llama3"
    )

    # print(augmented_docs)
    doc_ranges = args.doc_ranges.split(",")
    # theoremqa_theorems_Llama-3.1-8B-Instruct_None_0:3.parquet
    augmented_docs_path_list = [
        os.path.join(
            augmented_docs_path_prefix,
            f"{args.task}_{args.model}_None_{doc_range}.parquet" # theoremqa_theorems_Llama-3.1-8B-Instruct_None_0:5000.parquet
        )
            for doc_range in doc_ranges
    ]
    # augmented_docs_path_list = [
    #     os.path.join(augmented_docs_path_prefix, path)
    #         for path in augmented_docs_path_list
    # ]

    doc_ids_augmented = []
    documents_augmented = []
    # for dp in augmented_docs:
    for augmented_docs_path in augmented_docs_path_list:
        # Read the parquet file
        augmented_docs = pq.read_table(augmented_docs_path)
        augmented_docs = augmented_docs.to_pandas()
        for _, dp in augmented_docs.iterrows():
            # print(dp)
            # print(dp['doc_id'])
            # print(dp['document'])
            doc_ids_augmented.append(str(dp['doc_id']))
            documents_augmented.append(dp['document'])

    doc_emb = compute_doc_embeds(
        documents=documents_augmented,
        task="theoremqa_theorems",
        save_path="/home/oh/arubinstein17/github/ReasonIR/evaluation/bright/theoremqa_theorems/augmented_doc_emb.pkl"
    )


if __name__ == "__main__":
    main()
