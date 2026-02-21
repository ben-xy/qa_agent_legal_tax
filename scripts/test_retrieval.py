import sys
import json
import logging
import numpy as np
from pathlib import Path, PurePath

sys.path.append(str(PurePath(__file__).parents[1]))  # noqa: E402
from lib.chatgpt.function_calls import retrieve_sg_acts
from config import openai_chat, openai_gpt_model, lc_cll_in_use
from lib.logger import logger
from config import get_config
from src.retrievers.hybrid_retriever import HybridRetriever


if __name__ == "__main__":
    query = "What is Central Provident fund in singapore?"
    key_words = ["CPF", "Central Provident Fund"]

    results = retrieve_sg_acts(
        key_words=key_words,
        query=query,
        lc_vs=lc_cll_in_use,
        no_filter=False,
        chat_client=openai_chat,
        model=openai_gpt_model,
    )
    logger.info(results)

    # --- Hybrid retrieval example (BM25 + Vector) ---
    cfg = get_config()
    config_dict = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)
    config_dict.update({
        "use_bm25": True,
        "use_vector": True,
        "hybrid_alpha": 0.5,
        "acts_chunked_dir": "data/acts_chunked",
    })

    retriever = HybridRetriever(config_dict)
    hybrid_docs = retriever.retrieve(query=query, top_k=5)

    logger.info("Hybrid retrieval results:")
    logger.info(json.dumps([
        {
            "title": d.get("title") or d.get("act_title"),
            "section": d.get("section"),
            "source": d.get("_source_file"),
        } for d in hybrid_docs
    ], indent=2))

    # docs = lc_cll_in_use.similarity_search_with_score(
    #     ",".join(key_words), k=10)
    # results = [
    #     {"score": score,  **d.metadata} for d, score in docs[:10]
    # ]
    # logger.info(json.dumps(results, indent=2))
    # logger.info(min(s['score'] for s in results))
