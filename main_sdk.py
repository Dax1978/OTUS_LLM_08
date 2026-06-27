from yandex_gpt import YandexGPT, YandexGPTConfigManagerForAPIKey
from config.variables import YC_API_KEY, YC_FOLDER_ID


messages = [
    {
        "role": "system",
        "text": "Ты ассистент дроид, способный помочь в галактических приключениях."
    },
    {
        "role": "user",
        "text": "Привет, Дроид! Мне нужна твоя помощь, чтобы узнать больше о Силе. Как я могу научиться ее использовать?"
    },
    {
        "role": "assistant",
        "text": "Привет! Чтобы овладеть Силой, тебе нужно понять ее природу. Сила находится вокруг нас и соединяет всю галактику. Начнем с основ медитации."
    },
    {
        "role": "user",
        "text": "Хорошо, а как насчет строения светового меча? Это важная часть тренировки джедая. Как мне создать его?"
    }
]

config_manager = YandexGPTConfigManagerForAPIKey(
    model_type="yandexgpt-lite",
    catalog_id=YC_FOLDER_ID,
    api_key=YC_API_KEY
)
gpt = YandexGPT(config_manager=config_manager)

response = gpt.get_sync_completion(
    messages=messages,
    temperature=0.6,
    max_tokens=2000,
)

with open('result_sdk.json', 'w', encoding='utf-8') as f:
    f.write(response)

print(response)