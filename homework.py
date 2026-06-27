import os

import json
import csv
import re

from typing import List, Dict

from yandex_gpt import YandexGPT, YandexGPTConfigManagerForAPIKey
from config.variables import YC_API_KEY, YC_FOLDER_ID

CONFIG_MANAGER = YandexGPTConfigManagerForAPIKey(
    model_type="yandexgpt-lite",
    catalog_id=YC_FOLDER_ID,
    api_key=YC_API_KEY
)
GPT = YandexGPT(config_manager=CONFIG_MANAGER)


SYSTEM_PROMPT = """Ты - строгий и точный AI-помощник юриста.
Твоя задача - извлекать факты из текста без искажений и без выдумывания данных.
Если какой-то сущности нет в тексте, верни null для нее. Никогда не придумывай данные, которых нет в тексте."""

USER_PROMPT_TEMPLATE = """Извлеки из текста юридического документа следующие сущности:
- companies: список названий компаний, ИНН, КПП, упомянутых в тексте
- date_signing: дата подписания сделки или null
- validity_period: срок действия или null
- deal_amount: сумма сделки с указанием валюты (строка, например "150 млн долларов") или null
- commitment_period: cроки обязательств (в днях/месяцах или конкретных датах) или null

Верни ответ ТОЛЬКО в формате валидного JSON-объекта, без markdown-разметки и 
без сопроводительного текста.

Пример формата ответа:
{{
  "companies": ["Компания А ИНН 770401001 КПП 771401001", "Компания Б ИНН 780501007 КПП 781701007"],
  "date_signing": "30 декабря 2017 года",
  "validity_period": "7 лет 11 месяцев",
  "deal_amount": "50 млн рублей",
  "commitment_period": "10 лет"
}}

Текст юридического документа:
{text}"""


def read_csv_file(file_path):
    """
    Читает CSV файл и возвращает содержимое
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        data = [row for row in reader]
    return data

def preprocess_text(text):
    """
    Очистка текста от лишних пробелов и форматирование
    """
    # Удаление лишних пробелов
    text = re.sub(r'\s+', ' ', text)
    # Удаление пустых строк
    text = '\n'.join([line.strip() for line in text.split('\n') if line.strip()])
    # Замена кавычек
    text = text.replace('«', '"').replace('»', '"')
    return text

def split_decisions(text):
    """
    Разделяет текст на отдельные судебные решения
    """
    # Разделение по заголовкам решений
    decisions = re.split(r'Решение по (?:административному|гражданскому) делу', text)
    # decisions = re.split(r'\n\s*(?:Решение по (?:административному|гражданскому) делу)\s*\n', text)
    # Очистка от пустых элементов
    decisions = [preprocess_text(d) for d in decisions if d.strip()]
    return decisions


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
    # Читаем CSV файл
    csv_data = read_csv_file('./RuLegalNER/test2.csv')
    
    # Объединяем все строки в один текст
    full_text = '\n'.join([item[0] for item in csv_data])
    
    # Обрабатываем текст
    processed_text = preprocess_text(full_text)
    
    # Разделяем на решения
    decisions = split_decisions(processed_text)
    # print(len(decisions))

    for i, new_text in enumerate(decisions):
        print(f"Processing text: {new_text}")
        response = extract_entities_from_text(new_text)
        cleaned_response = clean_json_response(response)
        print(json.dumps(json.loads(cleaned_response), ensure_ascii=False, indent=2))
        save_response_to_file(cleaned_response, os.path.join("hw_results", f"ner_{i}.json"))