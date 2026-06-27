import os

import json

from typing import List, Dict

from yandex_gpt import YandexGPT, YandexGPTConfigManagerForAPIKey
from config.variables import YC_API_KEY, YC_FOLDER_ID

CONFIG_MANAGER = YandexGPTConfigManagerForAPIKey(
    model_type="yandexgpt-lite",
    catalog_id=YC_FOLDER_ID,
    api_key=YC_API_KEY
)
GPT = YandexGPT(config_manager=CONFIG_MANAGER)


SYSTEM_PROMPT = """Ты — точный AI-аналитик, специализирующийся на извлечении фактов из деловых новостей.
Твоя задача — извлекать именованные сущности из текста без искажений и без выдумывания данных.
Если какой-то сущности нет в тексте, верни null для нее. Никогда не придумывай данные, которых нет в тексте."""

USER_PROMPT_TEMPLATE = """Извлеки из текста новости следующие сущности:
- companies: список названий компаний, упомянутых в тексте
- persons: список имен людей (с должностями, если указаны)
- deal_amount: сумма сделки с указанием валюты (строка, например "150 млн долларов") или null
- deal_date: дата сделки или новости (строка) или null
- location: упомянутая локация (город/страна) или null

Верни ответ ТОЛЬКО в формате валидного JSON-объекта, без markdown-разметки и 
без сопроводительного текста.

Пример формата ответа:
{{
  "companies": ["Компания А", "Компания Б"],
  "persons": ["Иван Иванов, генеральный директор Компании А"],
  "deal_amount": "50 млн рублей",
  "deal_date": "12 марта 2026",
  "location": "Москва"
}}

Текст новости:
{text}"""


def build_messages(text: str) -> List[Dict[str, str]]:
    """
    Builds a list of messages for the GPT model based on the provided text.
    """
    return [
        {"role": "system", "text": SYSTEM_PROMPT},
        {"role": "user", "text": USER_PROMPT_TEMPLATE.format(text=text)},
    ]

def clean_json_response(raw_text: str) -> str:
    """
    Cleans the raw text response from the GPT model to ensure it is a valid JSON string.
    Removes any leading/trailing whitespace and markdown formatting.
    """
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    return text.strip()

def extract_entities_from_text(text: str) -> str:
    """
    Extracts named entities from the provided text using the GPT model.
    Returns the response as a JSON string.
    """
    messages = build_messages(text)

    response = GPT.get_sync_completion(
        messages=messages,
        temperature=0.2,
        max_tokens=1000,    
    )
    return response

def save_response_to_file(response: str, filename: str) -> None:
    """
    Saves the GPT response to a specified file.
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write(response)


if __name__ == "__main__":
    sample_text = (
        "Компания ABC.ai объявила о заключении сделки с Компанией Б на сумму 50 млн долларов. Сделка была подписана 12 марта 2026 года в Москве. Иван Иванов, генеральный директор Компании ABC.ai, выразил удовлетворение результатом.",
        "Сделка была подписана 12 марта 2026 года в Москве. ",
        "Иван Иванов, генеральный директор Компании ABC.ai, выразил удовлетворение результатом."
    )

    for i, new_text in enumerate(sample_text):
        print(f"Processing text: {new_text}")
        response = extract_entities_from_text(new_text)
        cleaned_response = clean_json_response(response)
        print(json.dumps(json.loads(cleaned_response), ensure_ascii=False, indent=2))
        save_response_to_file(cleaned_response, os.path.join("results", f"ner_{i}.json"))