import streamlit as st
import google.generativeai as genai
import yt_dlp
import cv2
import time
import os
import json
import re
from dotenv import load_dotenv
load_dotenv()  # Load variables from .env

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found. Check your .env file.")

genai.configure(api_key=API_KEY)


st.set_page_config(
    page_title="Lecture-Lens",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- UTILS: CACHING & PROCESSING ---

@st.cache_resource(show_spinner=False)
def download_video(url):
    """
    Downloads video. Cached so it only runs once per URL.
    """
    # Create a unique filename based on the URL hash to avoid conflicts
    safe_filename = "current_lecture.mp4"
    
    # Clean up previous
    if os.path.exists(safe_filename):
        try:
            os.remove(safe_filename)
        except:
            pass # File might be in use, we overwrite anyway

    ydl_opts = {
        'format': 'best[ext=mp4]',
        'outtmpl': safe_filename,
        'cookiefile': 'cookies.txt',
        'quiet': True,
        'overwrites': True
        
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return safe_filename
    except Exception as e:
        st.error(f"Download failed: {str(e)}")
        return None

def extract_json_from_text(text):
    """
    Helper to find JSON object inside Gemini's response if it includes extra text.
    """
    try:
        # Try finding the first '[' and last ']'
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(text) # Try direct parse
    except:
        return []

def get_frame(video_path, timestamp_str):
    """Extracts frame at MM:SS"""
    try:
        parts = list(map(int, timestamp_str.split(':')))
        if len(parts) == 2: seconds = parts[0] * 60 + parts[1]
        elif len(parts) == 3: seconds = parts[0] * 3600 + parts[1] * 60 + parts[2]
        else: return None

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(seconds * fps))
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None
    except:
        return None

# --- MAIN UI ---

st.title("🎓 Lecture-Lens")
st.markdown("""
<style>
    .big-font { font-size:20px !important; font-weight: 500; }
    .stButton>button { width: 100%; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# Sidebar for controls
with st.sidebar:
    st.header("Input")
    url = st.text_input("YouTube URL", placeholder="https://youtu.be/...")
    process_btn = st.button("Analyze Video", type="primary")
    st.info("💡 Tip: Try math, physics, or coding tutorials.")

# Session State to hold results
if 'analysis' not in st.session_state:
    st.session_state.analysis = None
if 'video_file' not in st.session_state:
    st.session_state.video_file = None

if process_btn and url:
    # 1. Download
    with st.status("⬇️ Downloading Video...", expanded=True) as status:
        video_path = download_video(url)
        st.session_state.video_file = video_path
        status.write("✅ Download Complete.")
        
        # 2. Upload to Gemini
        status.write("☁️ Uploading to AI Vision Model...")
        myfile = genai.upload_file(video_path)
        
        while myfile.state.name == "PROCESSING":
            time.sleep(1)
            myfile = genai.get_file(myfile.name)
        status.write("✅ AI Processing Ready.")

        # 3. Analyze
        status.write("🧠 Extracting Visual Knowledge Graph...")
        model = genai.GenerativeModel("gemini-3-flash-preview")
        
        prompt = """
        You are an expert visual note-taker. Analyze this video timeline.
        Identify distinct chapters where the visual content (slides, whiteboard, code) changes significantly.
        
        Return a valid JSON list. Each object must have:
        - "timestamp": (MM:SS) The start time of the topic.
        - "topic": (Title Case) A short title.
        - "summary": (Sentence) What exactly is shown on the screen?
        """
        
        response = model.generate_content(
            [myfile, prompt], 
            generation_config={"response_mime_type": "application/json"}
        )
        
        # 4. Parse & Store
        data = extract_json_from_text(response.text)
        st.session_state.analysis = data
        status.update(label="✅ Analysis Complete!", state="complete", expanded=False)

# --- RESULTS DASHBOARD ---
if st.session_state.analysis and st.session_state.video_file:
    st.divider()
    
    # Iterate through the analysis
    for item in st.session_state.analysis:
        col_img, col_text = st.columns([1, 1.5])
        
        with col_img:
            # Show screenshot
            img = get_frame(st.session_state.video_file, item['timestamp'])
            if img is not None:
                st.image(img, use_container_width=True, caption=f"Snapshot at {item['timestamp']}")
            else:
                st.video(st.session_state.video_file) # Fallback
        
        with col_text:
            st.markdown(f"### {item['topic']}")
            st.caption(f"⏱️ Timestamp: {item['timestamp']}")
            st.write(item['summary'])
            st.markdown("---")

elif not url:
    st.write("👈 Paste a URL on the left to start.")