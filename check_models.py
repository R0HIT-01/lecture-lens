import google.generativeai as genai
import os

# --- PASTE YOUR API KEY HERE ---
API_KEY = ""

genai.configure(api_key=API_KEY)

print("🔍 Scanning for available models...")
try:
    # 1. List all models
    models = list(genai.list_models())
    
    found_any = False
    for m in models:
        # We only care about models that can generate content
        if 'generateContent' in m.supported_generation_methods:
            print(f"✅ FOUND: {m.name}")
            found_any = True
            
    if not found_any:
        print("❌ No 'generateContent' models found. Check your API Key permissions.")
        
except Exception as e:
    print(f"❌ Error: {e}")
