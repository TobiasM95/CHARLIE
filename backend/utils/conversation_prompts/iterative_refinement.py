import re
from functools import cache
from ..data_structs import Language

# conversation prompt interfaces have to implement at least the following functions:
#
# get_conversation_prompt_chat_gpt(translation_model, input_text, language, user_name, memory_buffer, remembered_message_count, mood, additional_parameters: dict | None)
# -> chatgpt_prompt: list, mood_style: str | "undefined", message_length: str | "undefined"
#
# from functools import cache
# @cache
# _get_base_prompt(language, translation_model, *args)
# -> prompt: list[dict["role" | "content", str]]
#
# _extract_prompt_answers(full_answer: str, *args)
# -> answer_dict: dict["raw" | "clean" | "style", str | None]


def get_conversation_prompt_chat_gpt(
    translation_model,
    text,
    language,
    name,
    memory_buffer,
    remembered_message_count,
    mood,
    additional_parameters=None,
):
    print("DEBUG MEMORY", remembered_message_count, len(memory_buffer))

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
    )

    if language not in mood.style or language not in mood.style_example:
        mood.translate_style(translation_model, language)
    mood_style = mood.style[language]

    prompt = _get_base_prompt(language, translation_model, name, mood_style)

    message_query = ""
    for message_pair in reversed(relevant_message_pairs):
        message_query += (
            f"{name}: {message_pair[language].msg_user}"
            + "\n"
            + f"{name}: {message_pair[language].msg_charlie}"
            + "\n"
        )
    if text.startswith("("):
        message_query += text
    else:
        message_query += f"{name}: {text}"

    prompt.append({"role": "user", "content": message_query})

    print("DEBUG", prompt, mood_style)
    return prompt, mood_style, None


@cache
def _get_base_prompt(language, translation_model, name, mood_style):
    prompt = [
        {
            "role": "user",
            "content": f'From now on you pretend to be Charlie having a conversation with {name}. The user inputs will either be messages by {name}, after which you will reply as Charlie or instructions for special messages by Charlie.\nAll your answers will be given in 4 independent parts. You will give a neutral response and then gradually transform the response to the style "{mood_style}" but each message is independent. The message "A. ..." will be neutral, "B. ..." incorporates the style a little bit, "C. ..." is as far as you can go within your guidelines, and "D. ..." is 100% "{mood_style}". Don\'t give an introduction and don\'t continue the conversation. Always answer with exactly 4 messages "\nA. ...\nor\nB. ...\nor\nC. ...\nor\nD. ...\n" like described earlier. If you understand the instructions answer with yes.',
        },
        {"role": "assistant", "content": "Yes."},
    ]
    if language != Language.ENGLISH:
        from ..helper_functions import translate_transcript

        for part in prompt:
            part["content"] = translate_transcript(
                translation_model, part["content"], Language.ENGLISH, language
            ).text

    return prompt


def extract_prompt_answers(full_answer):
    print("DEBUG", full_answer)
    answer_1 = re.search("(?:A\.\s*)(.*)", full_answer)
    answer_2 = re.search("(?:B\.\s*)(.*)", full_answer)
    answer_3 = re.search("(?:C\.\s*)(.*)", full_answer)
    answer_4 = re.search("(?:D\.\s*)(.*)", full_answer)
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
    return False
