import sys
import json
import logging
from pathlib import Path, PurePath

sys.path.append(str(PurePath(__file__).parents[1]))  # noqa: E402
from lib.chatgpt.function_calls import retrieve_sg_acts
from config import openai_chat, openai_gpt_model, lc_cll_in_use
from lib.logger import logger


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

    # docs = lc_cll_in_use.similarity_search_with_score(
    #     ",".join(key_words), k=10)
    # results = [
    #     {"score": score,  **d.metadata} for d, score in docs[:10]
    # ]
    # logger.info(json.dumps(results, indent=2))
    # logger.info(min(s['score'] for s in results))
