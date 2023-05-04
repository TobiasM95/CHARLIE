import re
from functools import cache
import copy
from datetime import datetime
import multiprocessing

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

    print(
        "DEBUG",
        relevant_message_pairs,
        text,
        language,
        name,
        memory_buffer,
        remembered_message_count,
        additional_parameters,
    )

    mood_style, _ = mood.get_style(translation_model, language)

    prompt = copy.deepcopy(
        _get_base_prompt(language, translation_model, logger, name, mood_style)
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
            context_length=1,
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

    message_query = ""
    if len(relevant_message_pairs) > 0:
        message_query += (
            'With all the information above, this is the current excerpt:\n"'
        )
        reversed_relevant_message_pairs = list(reversed(relevant_message_pairs))
        for message_pair in reversed_relevant_message_pairs[:-1]:
            message_query += (
                f"{name}: {message_pair[language].msg_user}"
                + "\n"
                + f"Charlie: {message_pair[language].msg_charlie}"
                + "\n"
            )
        message_query += (
            f"{name}: {reversed_relevant_message_pairs[-1][language].msg_user}"
            + "\n"
            + f"Charlie: {reversed_relevant_message_pairs[-1][language].msg_charlie}"
            + '"\n'
        )

    message_query += (
        _localize_logged(
            language,
            translation_model,
            "Respond to the last message/input once with only the four options A,B,C,D:",
            logger,
        )
        + "\n"
    )
    if text.startswith("("):
        message_query += f'"{text}"'
    else:
        message_query += f'"{name}: {text}"'
    print("DEBUG message_query", message_query)

    prompt = [
        {"role": "user", "content": prompt[0]["content"] + "\n\n" + message_query}
    ]

    print("DEBUG", prompt, mood_style)
    return prompt, mood_style, None


def _get_base_prompt(language, translation_model, logger, name, mood_style):
    prompt = [
        {
            "role": "user",
            "content": _localize_logged(
                language,
                translation_model,
                f'From now on you pretend to be Charlie having a conversation with {name}. The user inputs will either be messages by {name}, after which you will reply as Charlie or instructions for special messages by Charlie.\nAll your answers will be given in 4 independent parts, separated by an "<or>" separator. You will give a neutral response and then gradually transform the response to the style "{mood_style}" but each message is independent. The message "A. ..." will be neutral, "B. ..." incorporates the style a little bit, "C. ..." is as far as you can go within your guidelines, and "D. ..." is 100% "{mood_style}". Don\'t give an introduction and don\'t continue the conversation. Always answer with exactly 4 messages in this format and don\'t miss the "<or>" separators in between:\n"A. ...\n<or>\nB. ...\n<or>\nC. ...\n<or>\nD. ..."\nlike described earlier.',
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
        if "memory-excerpt" in additional_parameters:
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
    full_answer = full_answer.replace("<br>", "<or>")
    answer_1 = list(re.finditer("(?:A\.\s*)(.*)", full_answer))[-1]
    answer_2 = list(re.finditer("(?:B\.\s*)(.*)", full_answer))[-1]
    answer_3 = list(re.finditer("(?:C\.\s*)(.*)", full_answer))[-1]
    answer_4 = list(re.finditer("(?:D\.\s*)(.*)", full_answer))[-1]
    if answer_4 is not None and not _contains_bad_text(answer_4.group(1)):
        answer = answer_4.group(1)
    elif answer_3 is not None and not _contains_bad_text(answer_3.group(1)):
        answer = answer_3.group(1)
    elif answer_4 is not None:
        answer = answer_4.group(1)
    elif answer_3 is not None:
        answer = answer_3.group(1)
    elif answer_2 is not None and not _contains_bad_text(answer_2.group(1)):
        answer = answer_2.group(1)
    elif answer_1 is not None:
        answer = answer_1.group(1)
    else:
        answer = full_answer
    answer = answer.strip()
    if answer[0] == '"' or answer[0] == "'":
        answer = answer.strip('"').strip("'")
    answer_dict = {}
    answer_dict["none"] = answer
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
) -> list:
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

    with multiprocessing.Pool(context_length) as pool:
        embeddings = pool.map(get_text_embedding, context_list)

    memories = []
    for embedding in embeddings:
        memories += memory_database.retrieve_memory(embedding, 1)

    return memories[::-1]


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
