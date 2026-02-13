import json
import logging

from lib.chatgpt.function_calls import functions
from lib.chatgpt.minions import MnLawyer, MnRelevanceChecker, MnKeywordFinder
from lib.chatgpt.utils import get_asst, EventHandler
from config import openai_chat, openai_gpt_model, lc_cll_in_use, exit, tool_choice, asst_lawyer
from lib.logger import logger


available_functions = functions['declaration']
tools = functions['description']

my_kw_finder = MnKeywordFinder(
    "my Kw Finder", model=openai_gpt_model, chat_client=openai_chat)
my_relv_checker = MnRelevanceChecker(
    "my Relv Checker", model=openai_gpt_model, chat_client=openai_chat)
my_lawyer = MnLawyer(
    "Singapore Lawyer", model=openai_gpt_model, chat_client=openai_chat)

if __name__ == "__main__":

    asst_lawyer = get_asst(
        my_lawyer.chat_client,
        asst_id=asst_lawyer,
        name=my_lawyer.name,
        description=my_lawyer.description,
        instructions=my_lawyer.prompt.instructions,
        model=my_lawyer.model,
        tools=tools)

    thread = openai_chat.beta.threads.create()
    logger.info('Assistant ready for conversation.')

    while True:
        user_input = input()
        if user_input == exit:
            break

        message = openai_chat.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=user_input
        )

        with openai_chat.beta.threads.runs.stream(
            thread_id=thread.id,
            assistant_id=asst_lawyer.id,
            event_handler=EventHandler(
                openai_chat,
                functions=available_functions,
                extra_function_call_args={
                    "lc_vs": lc_cll_in_use,
                    "relv_checker": my_relv_checker,
                    "kw_finder": my_kw_finder}
            ),
        ) as stream:
            stream.until_done()
            print("\n")
