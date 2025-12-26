import google.generativeai as genai
import time
import os
import yt_dlp
import json
# --- CONFIGURATION ---
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found. Check your .env file.")

genai.configure(api_key=API_KEY)

# --- 1. DOWNLOADER ---
def download_video(url):
    print(f"⬇️ Downloading {url}...")
    if os.path.exists("test_video.mp4"):
        os.remove("test_video.mp4")
        
    ydl_opts = {
        'format': 'best[ext=mp4]',
        'outtmpl': 'test_video.mp4',
        'cookiefile': 'cookies.txt',
        'quiet': True,
        'overwrites': True 
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return "test_video.mp4"

# --- 2. ANALYZER ---
def analyze_video(video_path):
    print("🚀 Uploading to Gemini 3 Flash...")
    
    # Upload video
    video_file = genai.upload_file(path=video_path)
    
    # Wait for processing
    print(f"⏳ Processing {video_file.name}", end="", flush=True)
    while video_file.state.name == "PROCESSING":
        print(".", end="", flush=True)
        time.sleep(1) # Gemini 3 is faster, check more often
        video_file = genai.get_file(video_file.name)

    if video_file.state.name == "FAILED":
        raise ValueError("Video processing failed.")

    print("\n✅ Video Ready! Asking Gemini...")
    
    # --- KEY CHANGE: USING YOUR SPECIFIC MODEL ---
    model = genai.GenerativeModel(model_name="gemini-3-flash-preview")
    
    prompt = """
    Analyze this video. Return a JSON list of key visual moments.
    Each item must have: "timestamp" (MM:SS), "topic", and "summary".
    
    Example Output:
    [
        {"timestamp": "01:00", "topic": "Intro", "summary": "The title card."}
    ]
    
    Output strict JSON.
    """
    
    # Gemini 3 often requires you to be explicit about JSON mime type for best results
    response = model.generate_content(
        [video_file, prompt],
        generation_config={"response_mime_type": "application/json"}
    )
    return response.text

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    try:
        # Run the pipeline
        vid_path = download_video("https://youtu.be/p_MZraK3w2s?si=BiNqGB7dq4Ldbnfi")
        json_output = analyze_video(vid_path)
        
        print("\n--- GEMINI 3 RESPONSE ---")
        print(json_output)
        print("-----------------------")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")