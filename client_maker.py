from google import genai
import dotenv

def get_client():
    api_key = dotenv.find_dotenv(".env","gemini_api_key")
    client = genai.Client(api_key=api_key)
    return client

