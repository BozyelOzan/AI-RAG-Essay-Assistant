"""Evaluation runner for the RAG pipeline.

Metrics
- Retrieval (no LLM calls):
    A retrieved chunk is "relevant" when it comes from the expected paper AND
    contains one of the expected keywords. Reported as Hit@5 and MRR, both for
    the raw vector search and after reranking.
- Answer quality (LLM-as-judge, same LLM as the app):
    faithful: every claim in the answer is supported by the retrieved context.
    relevant: the answer addresses the question.
- Answer correctness (keyword check): at least one expected keyword is in the answer.
- Refusal (out-of-scope questions): the answer says the documents do not contain it.
- Latency: retrieval, reranking and end-to-end time per question.

Usage
    python tests/run_eval.py
    python tests/run_eval.py --only en_resnet_idea,tr_unet_expanding   (merged into results.json)
"""

import sys
import os
import json
import re
import time
import statistics
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llama_index.core import Settings
from llama_index.core.postprocessor import SentenceTransformerRerank

from src.data_ingestion import load_data
from src.engine import create_chat_engine
from src.retriever import get_postprocessors
from config.settings import (
    LLM_MODEL_NAME,
    LLM_PROVIDER,
    RERANK_TOP_N,
    RERANKER_MODEL_NAME,
    SIMILARITY_TOP_K,
)

HERE = os.path.dirname(os.path.abspath(__file__))
QUESTIONS_FILE = os.path.join(HERE, "test_questions.json")
RESULTS_FILE = os.path.join(HERE, "results.json")

REFUSAL_PHRASES = [
    "don't have that information",
    "do not have that information",
    "bilgi yok",
    "bilgi bulunmamaktadır",
    "bilgi bulunmuyor",
    "bilgi yer almıyor",
    "bilgiye sahip değilim",
    "belgelerde yer almıyor",
    "belgelerde bulunmuyor",
    "belgelerde yok",
    "don't have",
]

JUDGE_PROMPT = """You are grading an answer produced by a RAG system.

Question:
{question}

Retrieved context:
{context}

Answer:
{answer}

Return ONLY a JSON object with these keys:
- "faithful": 1 if every claim in the answer is supported by the retrieved context, otherwise 0
- "relevant": 1 if the answer addresses the question, otherwise 0
"""


def is_refusal(answer):
    text = answer.lower()
    return any(phrase.lower() in text for phrase in REFUSAL_PHRASES)


def is_relevant(node, expected_source, keywords):
    file_name = (node.metadata.get("file_name") or "").lower()
    text = node.get_content().lower()
    return expected_source in file_name and any(k.lower() in text for k in keywords)


def first_rank(nodes, predicate):
    for rank, node in enumerate(nodes, start=1):
        if predicate(node):
            return rank
    return None


def reciprocal(rank):
    return 1.0 / rank if rank else 0.0


def judge(question, response):
    context = "\n\n---\n\n".join(n.node.get_content() for n in response.source_nodes)
    prompt = JUDGE_PROMPT.format(
        question=question, context=context, answer=response.response or ""
    )
    try:
        raw = Settings.llm.complete(prompt).text
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        scores = json.loads(match.group(0))
        return {
            "faithful": int(scores.get("faithful", 0)) == 1,
            "relevant": int(scores.get("relevant", 0)) == 1,
            "judge_error": None,
        }
    except Exception as e:
        return {"faithful": None, "relevant": None, "judge_error": str(e)}


def retrieval_metrics(index, reranker, case):
    query = case["question"]
    source = case["expected_source"]
    keywords = case["expected_keywords"]
    rel = lambda n: is_relevant(n, source, keywords)

    t0 = time.time()
    raw_nodes = index.as_retriever(similarity_top_k=SIMILARITY_TOP_K).retrieve(query)
    retrieve_s = time.time() - t0

    t1 = time.time()
    reranked = reranker.postprocess_nodes(raw_nodes, query_str=query)
    rerank_s = time.time() - t1

    raw_rank = first_rank(raw_nodes, rel)
    rerank_rank = first_rank(reranked, rel)
    return {
        "hit5_vector": raw_rank is not None and raw_rank <= 5,
        "hit5_rerank": rerank_rank is not None and rerank_rank <= 5,
        "hit60_vector": raw_rank is not None,
        "mrr_vector": reciprocal(raw_rank),
        "mrr_rerank": reciprocal(rerank_rank),
        "retrieve_s": round(retrieve_s, 3),
        "rerank_s": round(rerank_s, 3),
    }


def summarize(records):
    in_scope = [r for r in records if r.get("expected_source") is not None]
    out_scope = [r for r in records if r.get("should_refuse")]
    answered = [r for r in in_scope if r.get("error") is None]
    judged = [r for r in answered if r.get("faithful") is not None]

    def mean(values):
        values = [v for v in values if v is not None]
        return round(statistics.mean(values), 3) if values else None

    def pct(flags):
        flags = [f for f in flags if f is not None]
        return round(sum(flags) / len(flags), 3) if flags else None

    def p90(values):
        values = sorted(v for v in values if v is not None)
        if not values:
            return None
        return round(values[min(len(values) - 1, int(0.9 * len(values)))], 2)

    return {
        "date": datetime.now().isoformat(timespec="seconds"),
        "llm_provider": LLM_PROVIDER,
        "llm_model": LLM_MODEL_NAME,
        "reranker_model": RERANKER_MODEL_NAME,
        "total_questions": len(records),
        "passed": sum(bool(r.get("passed")) for r in records),
        "in_scope_questions": len(in_scope),
        "errors": sum(r.get("error") is not None for r in records),
        "retrieval": {
            "hit5_vector": pct([r.get("hit5_vector") for r in in_scope]),
            "hit5_rerank": pct([r.get("hit5_rerank") for r in in_scope]),
            "hit60_vector": pct([r.get("hit60_vector") for r in in_scope]),
            "mrr_vector": mean([r.get("mrr_vector") for r in in_scope]),
            "mrr_rerank": mean([r.get("mrr_rerank") for r in in_scope]),
        },
        "answer": {
            "keyword_hit_rate": pct([r.get("keyword_hit") for r in answered]),
            "faithfulness_rate": pct([r.get("faithful") for r in judged]),
            "relevance_rate": pct([r.get("relevant") for r in judged]),
            "judged_questions": len(judged),
        },
        "refusal": {
            "out_of_scope_questions": len(out_scope),
            "refusal_rate": pct([r.get("refusal_ok") for r in out_scope]),
        },
        "latency_s": {
            "retrieve_mean": mean([r.get("retrieve_s") for r in in_scope]),
            "rerank_mean": mean([r.get("rerank_s") for r in in_scope]),
            "end_to_end_mean": mean([r.get("e2e_s") for r in records]),
            "end_to_end_median": round(
                statistics.median([r["e2e_s"] for r in records if r.get("e2e_s") is not None]), 2
            ),
            "end_to_end_p90": p90([r.get("e2e_s") for r in records]),
        },
    }


def run_case(engine, index, reranker, case):
    record = {
        "id": case["id"],
        "lang": case["lang"],
        "question": case["question"],
        "expected_source": case["expected_source"],
        "should_refuse": bool(case.get("should_refuse")),
        "answer": None,
        "error": None,
        "sources": [],
    }

    if case["expected_source"] is not None:
        record.update(retrieval_metrics(index, reranker, case))

    start = time.time()
    try:
        response = engine.chat(case["question"])
    except Exception as e:
        record["error"] = f"{type(e).__name__}: {e}"
        record["e2e_s"] = round(time.time() - start, 2)
        record["passed"] = False
        return record
    record["e2e_s"] = round(time.time() - start, 2)

    answer = response.response or ""
    record["answer"] = answer
    record["sources"] = [n.metadata.get("file_name") for n in response.source_nodes]

    if case.get("should_refuse"):
        record["refusal_ok"] = is_refusal(answer)
        record["passed"] = record["refusal_ok"]
        return record

    keywords = case.get("expected_keywords", [])
    record["keyword_hit"] = any(k.lower() in answer.lower() for k in keywords)
    record["passed"] = bool(record["hit5_rerank"]) and record["keyword_hit"]
    record.update(judge(case["question"], response))
    return record


def main():
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        cases = json.load(f)

    only = None
    if "--only" in sys.argv:
        only = set(sys.argv[sys.argv.index("--only") + 1].split(","))
        cases = [c for c in cases if c["id"] in only]

    print("Loading index and models...")
    index = load_data()
    if index is None:
        print("No index available. Add PDFs to data/raw and run ingestion first.")
        return

    reranker = SentenceTransformerRerank(model=RERANKER_MODEL_NAME, top_n=SIMILARITY_TOP_K)
    engine = create_chat_engine(index, get_postprocessors(top_k=RERANK_TOP_N))

    records = []
    for case in cases:
        print(f"\n[{case['id']}] {case['question']}")
        record = run_case(engine, index, reranker, case)
        status = "PASS" if record["passed"] else "FAIL"
        print(f"  -> {status} ({record['e2e_s']}s){'  ERROR' if record['error'] else ''}")
        records.append(record)

    if only and os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, encoding="utf-8") as f:
            previous = {r["id"]: r for r in json.load(f)["records"]}
        new_by_id = {r["id"]: r for r in records}
        all_ids = [c["id"] for c in json.load(open(QUESTIONS_FILE, encoding="utf-8"))]
        merged = [new_by_id.get(i, previous.get(i)) for i in all_ids]
        records = [r for r in merged if r is not None]

    summary = summarize(records)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records}, f, ensure_ascii=False, indent=2)

    print("\n==== SUMMARY ====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nDetailed results: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
