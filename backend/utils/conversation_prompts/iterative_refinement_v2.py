import re
from functools import cache
import copy
from datetime import datetime
import concurrent.futures

from ..data_structs import Language

# conversation prompt interfaces have to implement at least the following functions and dictionaries:
#
# chat_gpt_parameter_dict: dict["temperature" | "presence_penalty", | "frequency_penalty", float]
#
# get_conversation_prompt_chat_gpt(translation_model, input_text, language, user_name, memory_buffer, remembered_message_count, mood, logger, additional_parameters: dict | None)
# -> chatgpt_prompt: list, mood_style: str | "undefined", message_length: str | "undefined"
#
# _get_base_prompt(language, translation_model, logger, *args)
# -> prompt: list[dict["role" | "content", str]]
#
# _extract_prompt_answers(full_answer: str, *args)
# -> answer_dict: dict["raw" | "clean" | "style", str | None]


def cache_notify(func):
    func = cache(func)

    def notify_wrapper(*args, **kwargs):
        stats = func.cache_info()
        hits = stats.hits
        results = func(*args, **kwargs)
        stats = func.cache_info()
        if stats.hits > hits:
            return True, results
        else:
            return False, results

    return notify_wrapper


chat_gpt_parameter_dict = {
    "temperature": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
}


def get_conversation_prompt_chat_gpt(
    translation_model,
    text,
    language,
    name,
    memory_buffer,
    remembered_message_count,
    mood,
    logger,
    additional_parameters=None,
):
    print("DEBUG MEMORY", remembered_message_count, len(memory_buffer))
    if additional_parameters is None:
        additional_parameters = {}

    # select the last N messages based on memory buffer size and configuration
    relevant_message_pairs = []
    if remembered_message_count > len(memory_buffer):
        remembered_message_count = len(memory_buffer)
    for i in range(remembered_message_count):
        relevant_message_pairs.append(memory_buffer[-1 - i])

    involved_users: set[str] = set()
    involved_users.add(name)
    for message_pair in relevant_message_pairs:
        involved_users.add(message_pair[language].name_user)
    group_conversation: bool = len(involved_users) > 1

    print(
        "DEBUG",
        relevant_message_pairs,
        text,
        language,
        name,
        memory_buffer,
        remembered_message_count,
        additional_parameters,
        f"group conversation={group_conversation}",
    )

    mood_style, _ = mood.get_style(translation_model, language)

    prompt = copy.deepcopy(
        _get_base_prompt(
            language, translation_model, logger, name, mood_style, group_conversation
        )
    )
    print("DEBUG base prompt:", prompt)

    if (
        "memory-database" in additional_parameters
        and additional_parameters["memory-database"] is not None
    ):
        additional_parameters["memory-excerpt"] = _get_memory_from_database(
            additional_parameters["memory-database"],
            relevant_message_pairs,
            name,
            text,
            language,
            context_length=2,
            memories_per_context=1,
        )
        print("DEBUG retrieved memories:", additional_parameters["memory-excerpt"])

    if (
        "situation-description" in additional_parameters
        or "memory-excerpt" in additional_parameters
    ):
        prompt = _enrich_base_prompt(
            language, translation_model, prompt, logger, additional_parameters
        )
    print("DEBUG enriched prompt:", prompt)

    message_query = 'With all the information above, this is the current excerpt:\n"'
    if len(relevant_message_pairs) > 0:
        reversed_relevant_message_pairs = list(reversed(relevant_message_pairs))
        for message_pair in reversed_relevant_message_pairs:
            message_query += (
                f"{message_pair[language].name_user}: {message_pair[language].msg_user}"
                + "\n"
                + f"Charlie: {message_pair[language].msg_charlie}"
                + "\n"
            )

    if text.startswith("("):
        message_query = message_query[:-1] + '"\n'
        # TODO: Yet untested on how to properly implement this
        # message_query += (
        #     _localize_logged(
        #         language,
        #         translation_model,
        #         f"You want to give your own input to the conversation according to the following instruction:",
        #         logger,
        #     )
        #     + "\n"
        # )
        # message_query += f'"{text}"\n'
        # message_query += f"Now, take date, situation, memories, excerpt and your instruction and give reply options A,B,C,D according to the instruction:"
        assert False
    else:
        clipped_message = " ".join(text.split(" ")[:5])
        if len(clipped_message) < len(text):
            clipped_message += "..."
        message_query += f'{name}: {text}"\n'
        message_query += (
            _localize_logged(
                language,
                translation_model,
                f"With date, situation, memories, and excerpt go ahead and replace the parts in parentheses for A, B, C, and D!",
                logger,
            )
            + "\n"
        )
        message_query += (
            f'"A. {name}: {clipped_message} - Charlie: ('
            + _localize_logged(language, translation_model, f"neutral answer", logger)
            + ")\n<or>\n"
        )
        message_query += (
            f'"A. {name}: {clipped_message} - Charlie: ('
            + _localize_logged(
                language, translation_model, f"incorporates the style a little", logger
            )
            + ")\n<or>\n"
        )
        message_query += (
            f'"A. {name}: {clipped_message} - Charlie: ('
            + _localize_logged(
                language,
                translation_model,
                f"goes as far as your guidelines allow",
                logger,
            )
            + ")\n<or>\n"
        )
        message_query += (
            f'"A. {name}: {clipped_message} - Charlie: ('
            + _localize_logged(
                language, translation_model, f'100% "{mood_style}"', logger
            )
            + ")\n"
        )
        message_query += _localize_logged(
            language,
            translation_model,
            f"Give all four reply options and don't repeat what was already mentioned in the current excerpt. Repeating what you already said is unnatural! Cut out {name} line from your answer!",
            logger,
        )
    print("DEBUG message_query", message_query)

    prompt = [
        {"role": "user", "content": prompt[0]["content"] + "\n\n" + message_query}
    ]

    print("DEBUG", prompt, mood_style)
    logger.debug_log(str(prompt))
    return prompt, mood_style, None


def _get_base_prompt(
    language, translation_model, logger, name, mood_style, group_conversation
):
    if group_conversation:
        prompt = [
            {
                "role": "user",
                "content": _localize_logged(
                    language,
                    translation_model,
                    f"From now on you pretend to be Charlie having a group conversation. The user inputs will either be messages including the username, after which you will reply as Charlie or instructions for special messages by Charlie.",
                    logger,
                ),
            }
        ]
    else:
        extension = ""
        if name is not None:
            extension = f" with {name}"
        prompt = [
            {
                "role": "user",
                "content": _localize_logged(
                    language,
                    translation_model,
                    f"From now on you pretend to be Charlie having a conversation{extension}. The user inputs will either be messages by {name}, after which you will reply as Charlie or instructions for special messages by Charlie.",
                    logger,
                ),
            }
        ]

    return prompt


def _enrich_base_prompt(
    language, translation_model, prompt, logger, additional_parameters
):
    enriched_content = ""
    if "situation-description" in additional_parameters:
        enriched_content += (
            _localize_logged(
                language,
                translation_model,
                "To help you answer more like Charlie, here is the current date and situation they're in:",
                logger,
            )
            + "\n"
        )
        date = datetime.now().strftime("%Y-%m-%d")
        weekday = datetime.now().strftime("%A")
        enriched_content += (
            f'"Current date: {weekday}, {date} - Current situation: '
            + additional_parameters["situation-description"]
            + '"'
        )
        if (
            "memory-excerpt" in additional_parameters
            and len(additional_parameters["memory-excerpt"]) > 0
        ):
            enriched_content += "\n"
    if (
        "memory-excerpt" in additional_parameters
        and len(additional_parameters["memory-excerpt"]) > 0
    ):
        enriched_content += (
            _localize_logged(
                language,
                translation_model,
                "Here are some memories of Charlie that you can reference, they include events, old conversations, and other details:",
                logger,
            )
            + "\n"
        )
        for memory_excerpt in additional_parameters["memory-excerpt"][:-1]:
            enriched_content += '"' + memory_excerpt + '"\n'
        enriched_content += '"' + additional_parameters["memory-excerpt"][-1] + '"'

    prompt = [
        {"role": "user", "content": prompt[0]["content"] + "\n\n" + enriched_content},
    ]

    return prompt


def extract_prompt_answers(full_answer: str):
    print("DEBUG", full_answer)
    full_answer = full_answer.replace("<br>", "<or>").replace("\n", "_N_")
    answer_1 = list(re.finditer("(?:A.*?Charlie:\s*)(.*?)(?:_N_|$)", full_answer))
    answer_2 = list(re.finditer("(?:B.*?Charlie:\s*)(.*?)(?:_N_|$)", full_answer))
    answer_3 = list(re.finditer("(?:C.*?Charlie:\s*)(.*?)(?:_N_|$)", full_answer))
    answer_4 = list(re.finditer("(?:D.*?Charlie:\s*)(.*?)(?:_N_|$)", full_answer))
    if len(answer_4) > 0 and not _contains_bad_text(answer_4[-1].group(1)):
        answer = answer_4[-1].group(1)
    elif len(answer_3) > 0 and not _contains_bad_text(answer_3[-1].group(1)):
        answer = answer_3[-1].group(1)
    elif len(answer_4) > 0:
        answer = answer_4[-1].group(1)
    elif len(answer_3) > 0:
        answer = answer_3[-1].group(1)
    elif len(answer_2) > 0 and not _contains_bad_text(answer_2[-1].group(1)):
        answer = answer_2[-1].group(1)
    elif len(answer_1) > 0:
        answer = answer_1[-1].group(1)
    else:
        answer = full_answer
    answer = re.sub("\(.*?\)", "", answer).strip()
    if answer[0] == '"' or answer[0] == "'":
        answer = answer.strip('"').strip("'")
    answer_dict = {}
    answer_dict["none"] = re.sub(
        "[^\u0000-\uD7FF\uE000-\uFFFF]", "", answer, flags=re.UNICODE
    )
    answer_dict["raw"] = None
    answer_dict["clean"] = None
    answer_dict["style"] = None
    return answer_dict


def _contains_bad_text(message):
    msg_low = message.lower()
    if "how can i help" in msg_low:
        return True
    elif "how can i make" in msg_low:
        return True
    elif "what can i do for you" in msg_low:
        return True
    return False


def _get_memory_from_database(
    memory_database,
    relevant_message_pairs: list,
    name: str,
    text: str,
    language: Language,
    context_length: int = 1,
    memories_per_context: int = 1,
) -> list:
    if len(memory_database.memory_summary_db) < 1:
        return []
    date = datetime.now().strftime("%Y-%m-%d")
    weekday = datetime.now().strftime("%A")

    if context_length > len(relevant_message_pairs) + 1:
        context_length = len(relevant_message_pairs) + 1

    context_list = [f"Current date: {weekday}, {date}\n{name}: {text}"]

    for message_pair in list(reversed(relevant_message_pairs))[: context_length - 1]:
        context_list.append(
            f"Current date: {weekday}, {date}\n"
            + f"{name}: {message_pair[language].msg_user}"
            + "\n"
            + f"Charlie: {message_pair[language].msg_charlie}"
        )

    from ..helper_functions import get_text_embedding

    with concurrent.futures.ThreadPoolExecutor(max_workers=context_length) as executor:
        embeddings = list(executor.map(get_text_embedding, context_list))

    print(f"DEBUG: Num memory embeddings:", len(embeddings))

    memories = []
    for embedding in embeddings:
        memories += memory_database.retrieve_memory(embedding, memories_per_context)

    return list(dict.fromkeys(memories[::-1]))


def _localize_logged(
    language, translation_model, message, logger, source_language=Language.ENGLISH
):
    was_cached, localization = _localize(
        language, translation_model, message, source_language
    )
    if was_cached:
        logger.track_stats("deepl", message)
    return localization


@cache_notify
def _localize(language, translation_model, message, source_language=Language.ENGLISH):
    if language == source_language:
        return message

    from ..helper_functions import translate_transcript

    message_localized = translate_transcript(
        translation_model, message, source_language, language
    ).text

    return message_localized
