import os

from dotenv import load_dotenv

load_dotenv()
YC_API_KEY_ID = os.getenv('YC_API_KEY_ID')
YC_API_KEY = os.getenv('YC_API_KEY')
YC_FOLDER_ID = os.getenv('YC_FOLDER_ID')