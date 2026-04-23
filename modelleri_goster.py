import google.generativeai as genai
import os
from dotenv import load_dotenv

# .env dosyasını oku
load_dotenv()

# Anahtarı al (senin .env dosyasındaki ismine göre)
api_key = os.getenv("GOOGLE_STUDIO_API") or os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

print("\n=== SENİN KULLANABİLECEĞİN MODELLER ===")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)
except Exception as e:
    print("Hata:", e)