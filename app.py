import streamlit as st
import google.generativeai as genai
import concurrent.futures
import yt_dlp
import cv2
import time
import os
from dotenv import load_dotenv
from PIL import Image

# ================== ENV SETUP ==================
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in .env")

genai.configure(api_key=API_KEY)
model_name = "gemini-2.0-flash"

# ================== STREAMLIT CONFIG ==================
st.set_page_config(
    page_title="Lecture-Lens",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== UTILS ==================

@st.cache_resource(show_spinner=False)
def download_video(url):
    # Create a unique filename to avoid locking issues
    timestamp = int(time.time())
    filename = f"lecture_{timestamp}.mp4"

    # Clean up old mp4 files (optional, but good for hygiene)
    for f in os.listdir():
        if f.startswith("lecture_") and f.endswith(".mp4"):
            try:
                os.remove(f)
            except OSError:
                pass  # If locked, skip it

    ydl_opts = {
        "format": "best[height<=480][ext=mp4]",  # faster download
        "outtmpl": filename,
        "quiet": True,
        "overwrites": True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return filename


def extract_keyframes(
    video_path,
    max_frames=9,
    min_gap_sec=25,
    diff_threshold=25
):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    keyframes = []
    prev_gray = None
    last_saved_time = -min_gap_sec
    frame_idx = 0

    try:
        while cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            time_sec = frame_idx / fps

            if time_sec - last_saved_time < min_gap_sec:
                frame_idx = int((last_saved_time + min_gap_sec) * fps)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff_score = 100 if prev_gray is None else cv2.absdiff(prev_gray, gray).mean()

            if diff_score > diff_threshold:
                keyframes.append({
                    "timestamp": time.strftime('%M:%S', time.gmtime(time_sec)),
                    "frame": cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                })
                prev_gray = gray
                last_saved_time = time_sec

                if len(keyframes) >= max_frames:
                    break

            frame_idx += int(fps)

    finally:
        cap.release()
        
    return keyframes


def summarize_frame(image, timestamp):
    prompt = f"""
    This image is a snapshot from a lecture at timestamp {timestamp}.
    In 1–2 sentences, describe:
    - What topic is being discussed
    - What is visually shown (slides, board, code, diagram)
    """

    # 🔥 CONVERT numpy array → PIL Image
    pil_image = Image.fromarray(image)
    
    # Convert PIL Image to bytes
    import io
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()

    model = genai.GenerativeModel(model_name)
    response = model.generate_content([
        prompt,
        {"mime_type": "image/jpeg", "data": img_bytes}
    ])
    return response.text.strip()


# ================== UI ==================

st.title("🎓 Lecture-Lens")
st.caption("AI-powered visual lecture breakdown")

with st.sidebar:
    st.header("Input")
    url = st.text_input("YouTube URL", placeholder="https://youtu.be/...")
    analyze_btn = st.button("Analyze Video", type="primary")
    st.info("Works best for lectures, tutorials, and classes")

# ================== SESSION STATE ==================

if "analysis" not in st.session_state:
    st.session_state.analysis = None

# ================== MAIN PIPELINE ==================

if analyze_btn and url:
    with st.status("Processing lecture...", expanded=True) as status:

        status.write("⬇️ Downloading video")
        video_path = download_video(url)
        status.write("✅ Download complete")

        status.write("🎞️ Detecting key scenes")
        keyframes = extract_keyframes(video_path)
        status.write(f"✅ {len(keyframes)} scenes detected")

        status.write("🧠 Generating AI summaries")
        
        def process_frame(kf):
            summary = summarize_frame(kf["frame"], kf["timestamp"])
            return {
                "timestamp": kf["timestamp"],
                "frame": kf["frame"],
                "summary": summary
            }

        with concurrent.futures.ThreadPoolExecutor() as executor:
            summaries = list(executor.map(process_frame, keyframes))

        st.session_state.analysis = summaries
        status.update(label="✅ Analysis Complete", state="complete")

# ================== RESULTS ==================

if st.session_state.analysis:
    st.subheader("📌 Lecture Breakdown")

    for item in st.session_state.analysis:
        col_img, col_text = st.columns([1, 3])

        with col_img:
            st.image(
                item["frame"],
                width=220,
                caption=item["timestamp"]
            )

        with col_text:
            st.markdown(f"**⏱ {item['timestamp']}**")
            st.write(item["summary"])

        st.divider()

elif not url:
    st.write("👈 Paste a YouTube lecture link to begin")
