import os
import json
import torch
from transformers import AutoTokenizer, AutoModel


ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def compute_doc_embeds(documents, task, save_path, doc_type="documents"):
    config_dir = os.path.join(ROOT_PATH, "evaluation", "bright", "configs")
    model_arg = "reasonir"
    with open(
        os.path.join(
            config_dir,
            model_arg.split("_ckpt")[0].split("_bilevel")[0],
            f"{task}.json",
        )
    ) as f:
        config = json.load(f)
    instructions = config["instructions"]
    customized_checkpoint = "reasonir/ReasonIR-8B"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        customized_checkpoint, torch_dtype="auto", trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        customized_checkpoint, torch_dtype="auto", trust_remote_code=True
    )
    model.eval()
    model.to(device)
    query_instruction = instructions["query"].format(task=task)
    doc_instruction = instructions["document"]
    # query_max_length = kwargs.get('query_max_length',32768)
    # doc_max_length = kwargs.get('doc_max_length',32768)
    query_max_length = 32768
    doc_max_length = 32768
    # print("doc max length:",doc_max_length)
    # print("query max length:", query_max_length)
    batch_size = 1
    # Override CUDA device count to 1 to control batch processing
    torch.cuda.device_count = lambda: 1
    if doc_type == "documents":
        instruction = doc_instruction
    else:
        assert doc_type == "queries"
        instruction = query_instruction
    doc_emb = model.encode(
        documents,
        instruction=instruction,
        batch_size=batch_size,
        max_length=doc_max_length,
    )
    if save_path is not None:
        dirname = os.path.dirname(save_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
        torch.save(doc_emb, save_path)
    return doc_emb


def get_queries(examples):
    queries = []
    query_ids = []
    excluded_ids = {}
    separate_reasoning = False
    reasoning_examples = []
    for qid, e in enumerate(examples):
        if separate_reasoning:
            new_query = (
                f"{e['query']}\n<REASON>\n{reasoning_examples[qid]['query']}"
            )
            queries.append(new_query)
        else:
            queries.append(e["query"])
        query_ids.append(e["id"])
        excluded_ids[e["id"]] = e["excluded_ids"]
        overlap = set(e["excluded_ids"]).intersection(set(e["gold_ids"]))
        assert len(overlap) == 0
    assert len(queries) == len(query_ids), f"{len(queries)}, {len(query_ids)}"
    return queries, query_ids
