import os
import sys
import re
import urllib.request
import feedparser
from youtube_transcript_api import YouTubeTranscriptApi
from deep_translator import GoogleTranslator
from gtts import gTTS
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips
from duckduckgo_search import DDGS
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from PIL import Image
import nltk
import subprocess
import glob

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def initialize_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')

def get_latest_video_from_channel(channel_id):
    feed_url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
    feed = feedparser.parse(feed_url)
    if not feed.entries:
        return None
    return feed.entries[0].yt_videoid

def check_if_processed(video_id):
    if not os.path.exists("last_processed.txt"):
        return False
    with open("last_processed.txt", "r") as f:
        last_id = f.read().strip()
    return last_id == video_id

def mark_as_processed(video_id):
    with open("last_processed.txt", "w") as f:
        f.write(video_id)

def extract_video_id(url):
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    return match.group(1) if match else None

def get_transcript_ytdlp(video_id):
    try:
        out_tmpl = os.path.join(OUTPUT_DIR, f"{video_id}.%(ext)s")
        cmd = [
            "python", "-m", "yt_dlp",
            "--write-auto-sub",
            "--skip-download",
            "--sub-langs", "en.*",
            "--sub-format", "vtt",
            "--output", out_tmpl,
            f"https://www.youtube.com/watch?v={video_id}"
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        files = glob.glob(os.path.join(OUTPUT_DIR, f"{video_id}*.vtt"))
        if not files:
            return None
            
        with open(files[0], "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        text_blocks = []
        for line in lines:
            line = line.strip()
            if not line or "-->" in line or re.match(r'^(WEBVTT|Kind:|Language:|Style|Region)', line) or re.match(r'^[0-9]+$', line):
                continue
            clean_text = re.sub(r'<[^>]+>', '', line).strip()
            if clean_text:
                text_blocks.append(clean_text)
                
        final_text = []
        for tb in text_blocks:
            if not final_text or final_text[-1] != tb:
                final_text.append(tb)
                
        return [{'text': tb} for tb in final_text]
    except Exception as e:
        print(f"yt-dlp fallback failed: {e}")
        return None

def get_transcript(video_id):
    try:
        # Also tries getting auto-generated English captions if manual is missing
        if hasattr(YouTubeTranscriptApi, 'get_transcript'):
            return YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'en-US', 'en-GB'])
        else:
            api = YouTubeTranscriptApi()
            return api.fetch(video_id, languages=['en', 'en-US', 'en-GB'])
    except Exception as e:
        print(f"Primary transcript fetch failed: {e}")
        print("Attempting to use yt-dlp fallback to bypass IP block...")
        return get_transcript_ytdlp(video_id)

def chunk_transcript(transcript, limit=400):
    chunks = []
    current_chunk = []
    current_length = 0
    for entry in transcript:
        text = (entry.text if hasattr(entry, 'text') else entry['text']).replace('\n', ' ')
        if current_length + len(text) > limit:
            chunks.append(" ".join(current_chunk))
            current_chunk = [text]
            current_length = len(text)
        else:
            current_chunk.append(text)
            current_length += len(text)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def summarize_text(text, sentences_count=5):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentences_count)
    return " ".join([str(sentence) for sentence in summary])

def extract_keyword(text):
    words = re.findall(r'\b\w+\b', text.lower())
    stop_words = set(["the", "a", "an", "is", "and", "or", "to", "in", "it", "of", "for", "on", "with", "as", "by", "that", "this", "they", "we", "are", "you", "so", "be", "at", "not", "but", "from", "have", "has", "do", "does", "was", "were"])
    meaningful = [w for w in words if w not in stop_words and len(w) > 3]
    if meaningful:
        counts = {w: meaningful.count(w) for w in set(meaningful)}
        return max(counts, key=counts.get)
    return "technology" # fallback keyword
    
def fetch_image_and_resize(keyword, index, target_size=(1280, 720)):
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.images(keyword, max_results=5, safesearch="on")]
            if results:
                image_url = results[0]['image']
                ext = image_url.split('.')[-1].split('?')[0]
                if ext.lower() not in ['jpg', 'jpeg', 'png', 'webp']:
                    ext = 'jpg'
                filepath = os.path.join(OUTPUT_DIR, f"image_{index}.{ext}")
                
                req = urllib.request.Request(image_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=10) as response, open(filepath, 'wb') as out_file:
                    out_file.write(response.read())
                
                # Resize using Pillow for perfect fit and no stretching errors in moviepy
                with Image.open(filepath) as img:
                    img = img.convert("RGB")
                    img = img.resize(target_size, Image.Resampling.LANCZOS)
                    jpg_filepath = os.path.join(OUTPUT_DIR, f"safe_image_{index}.jpg")
                    img.save(jpg_filepath, format="JPEG")
                    
                return jpg_filepath
    except Exception as e:
        print(f"Error fetching image for keyword {keyword}: {e}")
    return None

def main():
    initialize_nltk()
    
    manual_video_url = os.environ.get("MANUAL_VIDEO_URL", "").strip()
    channel_id = os.environ.get("CHANNEL_ID", "").strip()

    video_id = None
    if manual_video_url:
        video_id = extract_video_id(manual_video_url)
        print(f"Using manual video ID: {video_id}")
    elif channel_id:
        video_id = get_latest_video_from_channel(channel_id)
        print(f"Latest video ID from channel: {video_id}")
        if not video_id:
            print("No videos found in channel.")
            return
        if check_if_processed(video_id):
            print("Video already processed. Exiting.")
            return
    else:
        print("Please provide Manual Video URL or Channel ID as environment variables.")
        return
        
    print("Fetching transcript...")
    transcript = get_transcript(video_id)
    if not transcript:
        print("Could not get transcript. Video might not have closed captions.")
        return
        
    full_text = " ".join([t.text if hasattr(t, 'text') else t['text'] for t in transcript]).replace('\n', ' ')
    
    print("Summarizing...")
    summary_en = summarize_text(full_text, sentences_count=4)
    translator = GoogleTranslator(source='auto', target='hi')
    
    # Safely chunk for translation if summary is too long (though 4 sentences usually fine)
    summary_hi = translator.translate(summary_en)
    
    with open(os.path.join(OUTPUT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("--- English Summary ---\n" + summary_en + "\n\n--- Hindi Summary ---\n" + summary_hi)
        
    print("Chunking and Translating transcript for Voiceover...")
    # Using length around 350 to ensure translation limits aren't hit and visual changes happen periodically
    chunks_en = chunk_transcript(transcript, limit=350)
    
    clips = []
    full_hindi_transcript = ""
    
    for i, chunk in enumerate(chunks_en):
        if not chunk.strip():
            continue
        try:
            print(f"Processing chunk {i+1}/{len(chunks_en)}...")
            
            # Translate text
            chunk_hi = translator.translate(chunk)
            if not chunk_hi:
                continue
            full_hindi_transcript += chunk_hi + " "
            
            # Generate Audio
            audio_path = os.path.join(OUTPUT_DIR, f"audio_{i}.mp3")
            tts = gTTS(text=chunk_hi, lang='hi', slow=False)
            tts.save(audio_path)
            
            # Keyword & Image
            keyword = extract_keyword(chunk)
            print(f"  > Keyword: {keyword}")
            image_path = fetch_image_and_resize(keyword, i)
            
            # Build Video Clip
            audio_clip = AudioFileClip(audio_path)
            
            if image_path and os.path.exists(image_path):
                img_clip = ImageClip(image_path)
            else:
                # create a dummy color clip if image fetching fails
                from moviepy.editor import ColorClip
                img_clip = ColorClip(size=(1280, 720), color=(0,0,0))
                
            img_clip = img_clip.set_duration(audio_clip.duration)
            img_clip = img_clip.set_audio(audio_clip)
            clips.append(img_clip)
        except Exception as e:
            print(f"Error processing chunk {i}: {e}")
            continue
            
    with open(os.path.join(OUTPUT_DIR, "hindi_transcript.txt"), "w", encoding="utf-8") as f:
        f.write(full_hindi_transcript)

    print("Concatenating video clips...")
    if clips:
        final_video = concatenate_videoclips(clips, method="compose")
        output_file = os.path.join(OUTPUT_DIR, "final_hindi_video.mp4")
        final_video.write_videofile(output_file, fps=24, codec="libx264", audio_codec="aac")
        print("Video generated successfully!")
    else:
        print("No valid clips were generated.")
        
    mark_as_processed(video_id)

if __name__ == "__main__":
    main()
