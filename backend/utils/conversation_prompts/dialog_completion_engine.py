import re

from ..data_structs import Language

# conversation prompt interfaces have to implement the following functions:
#
# get_conversation_prompt_chat_gpt(translation_model, input_text, language, user_name, memory_buffer, remembered_message_count, mood)
# -> chatgpt_prompt: list, mood_style: str | "undefined", message_length: str | "undefined"
#
# extract_prompt_answers(full_answer: str)
# -> answer_dict: dict["raw" | "clean" | "style", str | None]


def get_conversation_prompt_chat_gpt(
    translation_model,
    text,
    language,
    name,
    memory_buffer,
    remembered_message_count,
    mood,
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

    if language not in mood.style or language not in mood.style_example:
        mood.translate_style(translation_model, language)
    mood_style = mood.style[language]
    style_example = mood.style_example[language]
    message_length = mood.get_message_length(text)

    if language == Language.ENGLISH:
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
    elif language == Language.GERMAN:
        prompt += [
            {
                "role": "system",
                "content": "Du bist eine Engine zur Vervollständigung von Dialogen und Stilen.",
            },
            {
                "role": "user",
                "content": f'Das Folgende ist eine Meta-Beschreibung eines Dialogs. Ich stelle dir einen Dialogauszug zur Verfügung, und du füllst die Nachricht innerhalb des Tags <msg style="..." length="..."></msg> entsprechend den "style"- und "length"-Parametern aus (length="short" bedeutet ein paar Wörter, length="very long" bedeutet mindestens 3 bis 4 lange Sätze oder sogar einen Absatz). Hebe die Charaktereigenschaften hervor. Du fragst nie, wie du {name} helfen könntest oder was du für {name} tun könntest. Hier ist ein Beispiel:\nIch gebe dir den Input:\n{name}: Hey, willst du abhängen?\nCharlie: <msg style="{style_example}" length="short"></msg>\n\nUnd du gibst mir die Antwort ohne zusätzliche Informationen, d.h. folge diesem Format und sei kurz und bündig und frage nie, wie du {name} helfen kannst oder was du für {name} tun kannst:\nCharlie: {style_example}\n\nVerstehst du das?',
            },
            {
                "role": "assistant",
                "content": f'Ja, ich verstehe. Bitte stelle mir den Dialogauszug und das gewünschte <msg style="..." length="..."></msg>-Tag zur Verfügung, damit ich es ausfüllen kann.',
            },
        ]
    else:
        assert False

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
