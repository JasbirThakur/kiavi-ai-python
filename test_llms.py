from groq import Groq
from openai import OpenAI
from config import GROQ_API_KEY, NVIDIA_API_KEY, NVIDIA_BASE_URL

print("=== 1. TESTING GROQ DYNAMIC ===")
try:
    g_client = Groq(api_key=GROQ_API_KEY)
    models_data = g_client.models.list().data
    chat_models = [m.id for m in models_data if "whisper" not in m.id and "guard" not in m.id]
    
    print(f"Found {len(chat_models)} active Groq models.")
    resp = g_client.chat.completions.create(
        model=chat_models[0],
        messages=[{"role": "user", "content": "Say 'Groq is ready'"}],
        max_tokens=15
    )
    print(f"✅ GROQ WORKING ({chat_models[0]}): {resp.choices[0].message.content.strip()}")
except Exception as e:
    print(f"❌ Groq Failed: {e}")

print("\n=== 2. TESTING NVIDIA CLUSTER ===")
try:
    nv_client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)
    resp = nv_client.chat.completions.create(
        model="nvidia/llama-3.1-nemotron-70b-instruct",
        messages=[{"role": "user", "content": "Say 'NVIDIA is ready'"}],
        max_tokens=15
    )
    print(f"✅ NVIDIA WORKING: {resp.choices[0].message.content.strip()}")
except Exception as e:
    print(f"⚠️ NVIDIA: {e}")