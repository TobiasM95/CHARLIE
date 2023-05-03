import re
from functools import cache
import copy

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
    "temperature": 0.3,
    "presence_penalty": 0.67,
    "frequency_penalty": 1.03,
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

    # select the last N messages based on memory buffer size and configuration
    relevant_message_pairs = []
    if remembered_message_count > len(memory_buffer):
        remembered_message_count = len(memory_buffer)
    for i in range(remembered_message_count):
        relevant_message_pairs.append(memory_buffer[-1 - i])

    prompt = []
    print(
        "DEBUG",
        relevant_message_pairs,
        text,
        language,
        name,
        memory_buffer,
        remembered_message_count,
    )

    mood_style, style_example = mood.get_style(translation_model, language)
    message_length = mood.get_message_length(text)

    prompt = copy.deepcopy(
        _get_base_prompt(
            language, translation_model, logger, name, mood_style, style_example
        )
    )

    for message_pair in reversed(relevant_message_pairs):
        prompt += [
            {
                "role": "user",
                "content": f'{name}: {message_pair[language].msg_user}\nCharlie: <msg style="{message_pair[language].reply_style}" length="{message_pair[language].reply_length}"></msg>',
            },
            {
                "role": "assistant",
                "content": f"Charlie: {message_pair[language].msg_charlie_style}",
            },
        ]
    prompt.append(
        {
            "role": "user",
            "content": f'{name}: {text}\nCharlie: <msg style="{mood_style}" length="{message_length}"></msg>',
        }
    )

    print("DEBUG", prompt, mood_style, style_example)
    return prompt, mood_style, message_length


def _get_base_prompt(
    language, translation_model, logger, name, mood_style, style_example
):
    prompt += [
        {
            "role": "system",
            "content": "You are a dialog and style completion engine.",
        },
        {
            "role": "user",
            "content": f'The following is a meta description of a dialog. I will provide you with a dialog excerpt and you fill in the message inside the <msg style="..." length="..."></msg> tag according to the "style" and "length" parameters (length="short" means a few words, length="very long" means at least 3 to 4 long sentences or even a paragraph). Emphasize the character traits. You never ask how you can help {name} or what you can do for {name}. Here is an example:\nI give you the input:\n{name}: Yo, wanna hang out?\nCharlie: <msg style="{mood_style}" length="short"></msg>\n\nAnd you give me the output without any extra information, i.e. follow this format and be concise and never ask how you can help {name} or what you can do for {name}:\nCharlie: {style_example}\n\nDo you understand?',
        },
        {
            "role": "assistant",
            "content": f'Yes, I understand. Please provide me with the dialog excerpt and the desired <msg style="..." length="..."></msg> tag for me to fill in.',
        },
    ]
    if language != Language.ENGLISH:
        prompt_translated = []
        for part in prompt:
            prompt_translated.append(
                {
                    "role": "system",
                    "content": _localize_logged(
                        language, translation_model, part["content"], logger
                    ),
                }
            )
        return prompt_translated
    return prompt


def extract_prompt_answers(full_answer):
    print("DEBUG", full_answer)
    answer_dict = {}
    answer_dict["raw"] = None
    answer_dict["clean"] = None
    extracted_answer = re.search("(?<=Charlie:)[\w\W\s]*", full_answer)
    answer_dict["style"] = (
        extracted_answer[0] if extracted_answer is not None else full_answer
    ).strip()
    return answer_dict


def _localize_logged(
    language, translation_model, message, logger, source_language=Language.ENGLISH
):
    was_cached, localization = _localize(
        language, translation_model, message, logger, source_language
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
