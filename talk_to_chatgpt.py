import json
import logging

from lib.chatgpt.function_calls import functions
from lib.chatgpt.minions import MnLawyer, MnRelevanceChecker, MnKeywordFinder
from lib.chatgpt.utils import pprint_conversation
from config import openai_chat, openai_gpt_model, lc_cll_in_use, exit, tool_choice
from lib.logger import logger


available_functions = functions['declaration']
tools = functions['description']

my_lawyer = MnLawyer(
    "Singapore Lawyer", model=openai_gpt_model, chat_client=openai_chat)
my_kw_finder = MnKeywordFinder(
    "my Kw Finder", model=openai_gpt_model, chat_client=openai_chat)
my_relv_checker = MnRelevanceChecker(
    "my Relv Checker", model=openai_gpt_model, chat_client=openai_chat)

if __name__ == "__main__":
    # Kick start conversation
    messages = my_lawyer.prompt.messages

    while True:
        # prune messages before sending for completion
        # To be implemented

        # Ask ASSISTANT to reply
        chat_response = my_lawyer.help_me(
            messages, tool_choice=tool_choice, tools=tools,
        )
        assistant_message = chat_response.choices[0].message
        messages.append({"role": assistant_message.role,
                         "content": assistant_message.content})

        # USER/TOOL to reply
        if not assistant_message.tool_calls:
            pprint_conversation(messages[-1:])
            logger.info(
                "=Assistant no need function call, Ask USER to reply")
            # Assistant no need function call
            # Ask USER to reply
            user_input = input()
            if user_input.lower() == exit:
                break
            messages.append({"role": "user", "content": user_input})

        else:
            messages[-1]['tool_calls'] = assistant_message.tool_calls
            logger.info(
                "=Assistant requests function call, Get TOOL to reply with function response")
            # Assistant requests function call
            # Get TOOL to reply with function response
            for tool_call in assistant_message.tool_calls:
                logger.info("==func name: " + tool_call.function.name)
                # Send the info for each function call and function response to the model
                function_name = tool_call.function.name
                function_to_call = available_functions[function_name]
                function_args = json.loads(tool_call.function.arguments)
                logger.info("==func args: \n" + str(function_args))
                function_response = function_to_call(
                    lc_vs=lc_cll_in_use,
                    relv_checker=my_relv_checker,
                    kw_finder=my_kw_finder,
                    **function_args)
                logger.info("==func resp: \n" + str(function_response[:200]))
                # extend conversation with function response
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                    }
                )
