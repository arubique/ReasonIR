import os
import json
import torch
from transformers import AutoTokenizer, AutoModel
import pyarrow.parquet as pq
import sys
import argparse

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# local imports
sys.path.insert(0, ROOT_PATH)
from utility.utils_for_notebooks import compute_doc_embeds, get_queries

sys.path.pop(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="theoremqa_theorems")
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument(
        "--doc_ranges", type=str, required=True
    )  # 0:5000,5000:10000,10000:15000,15000:20000,20000:25000
    parser.add_argument("--model", type=str, default="Llama-3.1-8B-Instruct")
    parser.add_argument("--prefix", type=str, required=True)
    parser.add_argument(
        "--base_folder",
        type=str,
        default=os.path.join(
            ROOT_PATH, "output", "documents_augmentation_llama3"
        ),
    )
    args = parser.parse_args()

    # print(augmented_docs)
    doc_ranges = args.doc_ranges.split(",")
    # theoremqa_theorems_Llama-3.1-8B-Instruct_None_0:3.parquet
    if "documents" in args.prefix:
        doc_type = "documents"
    else:
        assert "queries" in args.prefix
        doc_type = "queries"

    if doc_type == "documents":
        augmented_docs_path_prefix = args.base_folder
        augmented_docs_path_list = [
            os.path.join(
                augmented_docs_path_prefix,
                f"{args.prefix}_{args.task}_{args.model}_{doc_range}.parquet",  # theoremqa_theorems_Llama-3.1-8B-Instruct_None_0:5000.parquet
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
                doc_ids_augmented.append(str(dp["doc_id"]))
                documents_augmented.append(dp["document"])
        to_embed = documents_augmented
    else:
        assert len(doc_ranges) == 1
        augmented_examples = json.load(
            open(
                os.path.join(
                    args.base_folder,
                    f"{args.prefix}_{args.task}_{args.model}_{doc_ranges[0]}.json",
                )
            )
        )
        assert doc_type == "queries"
        queries, query_ids = get_queries(augmented_examples)
        to_embed = queries

    doc_emb = compute_doc_embeds(
        documents=to_embed,
        task=args.task,
        save_path=args.save_path,
        doc_type=doc_type,
    )


if __name__ == "__main__":
    main()
