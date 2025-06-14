import copy
import os
import json
from tqdm import tqdm
import argparse
from vllm import LLM, SamplingParams
from datasets import load_dataset
import torch
import prompts
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from utility.utils import get_device_with_most_free_memory
sys.path.pop(0)

class Reranker:
    def __init__(self, task, device_count=None, device=None):
        model_name = "Qwen/Qwen2.5-32B-Instruct"
        if device_count is None:
            device_count = torch.cuda.device_count()
        if device is None:
            device = get_device_with_most_free_memory()
        self.sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=32, logprobs=5)
        self.model = LLM(
            model=model_name,
            dtype="bfloat16",
            tensor_parallel_size=device_count,
            max_model_len=16384,
            device=device
        )

        retrieval_dict = {
                "aops": prompts.bright_aops,
                "theoremqa_questions": prompts.bright_theoremqa_questions,
                "leetcode": prompts.bright_leetcode,
                "pony": prompts.bright_pony,
                "theoremqa_theorems": prompts.bright_theoremqa_theorems
            }
        if args.task in retrieval_dict:
            self.prompt = retrieval_dict[task]
        else:
            self.prompt = prompts.bright_general

        # set random seed
        torch.manual_seed(42)

    def rerank(self, docs, query, topk, batch_size=50):
        scores = []

        for i in tqdm(range(0, len(docs), batch_size)):
            list_docs = docs[i:i+batch_size]

            doc_prompts = [self.prompt.format(query, doc["text"]) for doc in list_docs]

            output = self.model.generate(doc_prompts, self.sampling_params)
            doc_prompt_outputs = [o.outputs[0].text for o in output]

            for j in range(len(doc_prompt_outputs)):
                pos_score = doc_prompt_outputs[j].rfind("Relevance score:")
                if pos_score != -1:
                    try:
                        # get the score from the prompt output
                        score = float(doc_prompt_outputs[j][pos_score+16:pos_score+18])/5
                        scores.append(score)
                    except:
                        print("In exception!! {}".format(doc_prompt_outputs[j]))
                        scores.append(-1)
                else:
                    scores.append(0)


        ranking = {doc["id"]: score for doc, score in zip(docs, scores)}
        ranking = dict(sorted(ranking.items(), key=lambda item: item[1], reverse=True)[:topk])
        return ranking


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True,
                        choices=['biology','earth_science','economics','pony','psychology','robotics','theoremqa_questions', "theoremqa_theorems",
                                 'stackoverflow','sustainable_living','aops','leetcode'])
    parser.add_argument('--long_context', action='store_true')
    parser.add_argument('--retriever_score_file', type=str, default=None)
    parser.add_argument('--input_k', type=int)
    parser.add_argument('--k', type=int)
    parser.add_argument('--cache_dir', type=str, default='cache')
    parser.add_argument('--reasoning', type=str, default=None)
    parser.add_argument('--bm25_score_file', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=50)
    args = parser.parse_args()

    if args.reasoning is not None:
        raw_examples = load_dataset('xlangai/bright', f"{args.reasoning}_reason", cache_dir=args.cache_dir)[args.task]
    else:
        raw_examples = load_dataset('xlangai/bright', 'examples',cache_dir=args.cache_dir)[args.task]

    # raw_examples = load_dataset('xlangai/bright', 'examples', cache_dir=args.cache_dir)[args.task]
    examples = {}
    for e in raw_examples:
        examples[e['id']] = e
    if args.long_context:
        doc_pairs = load_dataset('xlangai/bright', 'long_documents', cache_dir=args.cache_dir)[args.task]
    else:
        doc_pairs = load_dataset('xlangai/bright', 'documents', cache_dir=args.cache_dir)[args.task]
    documents = {}
    for d in doc_pairs:
        documents[d['id']] = d['content']

    with open(args.retriever_score_file) as f:
        all_scores = json.load(f)

    outputs_path = args.output_dir
    score_file_path = os.path.join(outputs_path, f"{args.reasoning}_score.json")

    if not os.path.isfile(score_file_path):
        new_scores = copy.deepcopy(all_scores)

        model = Reranker(args.task, device_count=1, device=get_device_with_most_free_memory())

        for qid,scores in tqdm(all_scores.items()):
            docs = []
            sorted_scores = sorted(scores.items(),key=lambda x:x[1],reverse=True)[:args.input_k]
            for did, _ in sorted_scores:
                docs.append([did, documents[did]])

            ctxs = [{'id': did, 'text': documents[did]} for did, _ in sorted_scores]

            cur_score = model.rerank(
                query=examples[qid]['query'],
                docs=ctxs,
                topk=args.k,
                batch_size=args.batch_size
            )

            assert len(cur_score) == len(sorted_scores)

            new_scores[qid] = cur_score

        os.makedirs(outputs_path, exist_ok=True)
        with open(score_file_path, 'w') as f:
            json.dump(new_scores, f, indent=2)
    else:
        with open(score_file_path) as f:
            new_scores = json.load(f)
        print(score_file_path,'exists')

    if args.long_context:
        key = 'gold_ids_long'
    else:
        key = 'gold_ids'
    ground_truth = {}
    for e in raw_examples:
        ground_truth[e['id']] = {}
        for gid in e[key]:
            ground_truth[e['id']][gid] = 1
        for i in e["excluded_ids"]:
            if i in documents:
                ground_truth[e['id']][i] = 0

    # The import is here to prevent environment issues
    from retrievers import calculate_retrieval_metrics

    results = calculate_retrieval_metrics(results=new_scores, qrels=ground_truth)
    with open(os.path.join(outputs_path, "reranker_results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    # break ties by interpolating with the retriever scores
    retriever_interpolated_scores = {}
    for qid in new_scores:
        retriever_interpolated_scores[qid] = {}
        for did in new_scores[qid]:
            retriever_interpolated_scores[qid][did] = (0.5 * new_scores[qid][did]) + (0.5 * all_scores[qid][did])
    results = calculate_retrieval_metrics(results=retriever_interpolated_scores, qrels=ground_truth)
    with open(os.path.join(outputs_path, f"reranker_retriever_results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    # break ties by combining with the bm25 scores
    if args.bm25_score_file is not None:
        bm25_interpolated_scores = {}
        with open(args.bm25_score_file) as f:
            bm25_scores = json.load(f)
        for qid in new_scores:
            bm25_interpolated_scores[qid] = {}
            for did in new_scores[qid]:
                bm25_interpolated_scores[qid][did] = (100 * new_scores[qid][did]) + bm25_scores[qid][did]
        results = calculate_retrieval_metrics(results=bm25_interpolated_scores, qrels=ground_truth)
        with open(os.path.join(outputs_path, f"reranker_bm25_results.json"), 'w') as f:
            json.dump(results, f, indent=2)
