import os

from dotenv import load_dotenv
from google import genai

# .env dosyasını oku
load_dotenv()

# Anahtarı al (senin .env dosyasındaki ismine göre)
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_STUDIO_API") or os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

print("\n=== SENİN KULLANABİLECEĞİN MODELLER ===")
try:
    for model in client.models.list():
        supported_actions = getattr(model, "supported_actions", None)
        supported_generation_methods = getattr(model, "supported_generation_methods", None)
        supported = supported_actions or supported_generation_methods or []
        supports_generate_content = any(
            str(action).lower() in {"generatecontent", "generate_content"}
            for action in supported
        )
        if supports_generate_content or not supported:
            print(model.name)
except Exception as e:
    print("Hata:", e)
