import os
import sys
import re
import urllib.request
import feedparser
from youtube_transcript_api import YouTubeTranscriptApi
from deep_translator import GoogleTranslator
import edge_tts
import asyncio
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips
from duckduckgo_search import DDGS
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from PIL import Image
import nltk
import subprocess
import glob
import time
import json

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
        # Expanded rotation strategy with specialized clients
        clients = ["android", "ios", "tv", "android_music", "web_embedded", "tv_embedded", "mweb", "android_vr"]
        urls = [
            f"https://www.youtube.com/watch?v={video_id}",
            f"https://www.youtube.com/v/{video_id}",
            f"https://youtu.be/{video_id}"
        ]
        
        for client in clients:
            for url in urls:
                cmd = [
                    "python", "-m", "yt_dlp",
                    "--write-auto-sub",
                    "--write-sub",
                    "--skip-download",
                    "--sub-langs", "en.*",
                    "--sub-format", "vtt",
                    "--client-name", client,
                    "--output", out_tmpl,
                    url
                ]
                subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                files = glob.glob(os.path.join(OUTPUT_DIR, f"{video_id}*.vtt"))
                if files:
                    print(f"  [ytdlp] Success with client: {client}")
                    break
            if glob.glob(os.path.join(OUTPUT_DIR, f"{video_id}*.vtt")):
                break
        
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

def get_transcript_playwright(video_id):
    try:
        from playwright.sync_api import sync_playwright
        import xml.etree.ElementTree as ET
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--disable-blink-features=AutomationControlled"])
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            page = context.new_page()
            page.goto(f"https://www.youtube.com/watch?v={video_id}")
            
            # Optionally dismiss consent popup if it blocks
            try:
                page.click('button[aria-label="Accept all"]', timeout=3000)
            except:
                pass
                
            # Hide automation features and wait for the precise variable rather than a visual element
            try:
                page.wait_for_function("typeof window.ytInitialPlayerResponse !== 'undefined'", timeout=15000)
            except:
                pass
            
            player_response = page.evaluate("window.ytInitialPlayerResponse")
            if not player_response:
                browser.close()
                return None
                
            captions = player_response.get('captions', {})
            track_list = captions.get('playerCaptionsTracklistRenderer', {}).get('captionTracks', [])
            
            if not track_list:
                browser.close()
                return None
                
            en_track = next((t for t in track_list if t.get('languageCode', '').startswith('en')), track_list[0])
            url = en_track['baseUrl']
            
            # Fetch the actual transcript XML using the browser context to maintain IP trust
            xml_text = page.evaluate("(url) => fetch(url).then(r => r.text())", url)
            browser.close()
            
            root = ET.fromstring(xml_text)
            results = []
            for child in root:
                if child.text:
                    import html
                    clean_text = html.unescape(child.text.replace('\n', ' ')).strip()
                    if clean_text:
                        results.append({'text': clean_text})
            return results
    except Exception as e:
        print(f"Playwright fallback failed: {e}")
        return None

def get_transcript_audio(video_id):
    try:
        import whisper
        import yt_dlp
        import nltk
        
        audio_path = os.path.join(OUTPUT_DIR, f"{video_id}.m4a")
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': audio_path,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'm4a',
            }],
            'quiet': True,
            'extractor_args': {'youtube': {'client': ['ios']}},
        }
        
        print("Downloading audio for Whisper STT (rotating deep clients)...")
        # Rotated specialized clients for audio
        clients = ["android", "ios", "tv", "android_music", "web_embedded", "tv_embedded"]
        success = False
        
        for client in clients:
            try:
                ydl_opts = {
                    'format': 'bestaudio/best',
                    'outtmpl': audio_path,
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'm4a',
                    }],
                    'quiet': True,
                    'extractor_args': {'youtube': {'client': [client]}},
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
                
                if os.path.exists(audio_path):
                    print(f"  [whisper] Audio success with client: {client}")
                    success = True
                    break
            except Exception as e:
                continue
                
        if not success:
            return None
        model = whisper.load_model("tiny.en")
        result = model.transcribe(audio_path)
        
        text = result.get('text', '').strip()
        if not text:
            return None
            
        sentences = nltk.tokenize.sent_tokenize(text)
        return [{'text': s} for s in sentences]
    except Exception as e:
        print(f"Whisper fallback failed: {e}")
        return None

def get_transcript_piped(video_id):
    """Fetches transcript/captions from a Piped proxy instance to bypass YouTube IP blocks."""
    # List of reliable Piped instances
    instances = ["https://pipedapi.kavin.rocks", "https://api.piped.victr.me", "https://pipedapi.leptons.xyz"]
    
    for instance in instances:
        try:
            print(f"  Attempting Piped instance: {instance}")
            url = f"{instance}/streams/{video_id}"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode('utf-8'))
                
            # Piped provides subtitles in the 'subtitles' key
            subtitles = data.get('subtitles', [])
            if not subtitles:
                continue
                
            # Find English subtitles
            en_sub = next((s for s in subtitles if s.get('code', '').startswith('en')), subtitles[0])
            sub_url = en_sub['url']
            
            # Fetch the actual vtt/srt content
            with urllib.request.urlopen(sub_url, timeout=10) as sub_response:
                lines = sub_response.read().decode('utf-8').splitlines()
                
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
            
            if final_text:
                return [{'text': tb} for tb in final_text]
        except Exception as e:
            print(f"    Failed instance {instance}: {e}")
            continue
    return None

def get_transcript(video_id):
    try:
        # 1. Primary Method: Piped API (Best for Cloud IPs)
        print("Using Piped Proxy Fallback (Primary)...")
        res = get_transcript_piped(video_id)
        if res: return res
        
        # 2. Secondary Method: standard API
        print("Attempting YouTube Transcript API...")
        if hasattr(YouTubeTranscriptApi, 'get_transcript'):
            return YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'en-US', 'en-GB'])
        else:
            api = YouTubeTranscriptApi()
            return api.fetch(video_id, languages=['en', 'en-US', 'en-GB'])
    except Exception as e:
        print(f"Transcript fetch failed: {e}")
        
        print("Attempting to use yt-dlp fallback (Deep rotation)...")
        res = get_transcript_ytdlp(video_id)
        if res: return res
        
        print("Attempting to use Playwright (Chromium) fallback...")
        return get_transcript_playwright(video_id)

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
    stop_words = set(["the", "a", "an", "is", "and", "or", "to", "in", "it", "of", "for", "on", "with", "as", "by", "that", "this", "they", "we", "are", "you", "so", "be", "at", "not", "but", "from", "have", "has", "do", "does", "was", "were", "okay", "yeah", "yes", "well", "just", "like", "what", "how", "when", "where", "why", "who", "which", "will", "would", "can", "could", "should", "shall", "might", "must", "if", "then", "else", "because", "about", "into", "through", "during"])
    meaningful = [w for w in words if w not in stop_words and len(w) > 4]
    if meaningful:
        counts = {w: meaningful.count(w) for w in set(meaningful)}
        return max(counts, key=counts.get)
    return "technology"
    
def fetch_image_and_resize(keyword, index, target_size=(1280, 720)):
    try:
        time.sleep(2) # Avoid rapid DDOS on DuckDuckGo
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
        print(f"Error fetching image from DDG for keyword '{keyword}': {e}")
        print("Falling back to Wikimedia Commons API...")
        try:
            url = "https://en.wikipedia.org/w/api.php"
            import urllib.parse
            params = {
                "action": "query",
                "format": "json",
                "prop": "imageinfo",
                "iiprop": "url",
                "generator": "search",
                "gsrsearch": f"filetype:bitmap {keyword}",
                "gsrnamespace": "6",
                "gsrlimit": "1"
            }
            req = urllib.request.Request(f"{url}?{urllib.parse.urlencode(params)}", headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode('utf-8'))
                pages = data.get("query", {}).get("pages", {})
                if pages:
                    page = list(pages.values())[0]
                    image_url = page.get("imageinfo", [{}])[0].get("url")
                    if image_url:
                        filepath = os.path.join(OUTPUT_DIR, f"image_{index}_wiki.jpg")
                        req_img = urllib.request.Request(image_url, headers={'User-Agent': 'Mozilla/5.0'})
                        with urllib.request.urlopen(req_img, timeout=10) as img_resp, open(filepath, 'wb') as out_file:
                            out_file.write(img_resp.read())
                        with Image.open(filepath) as img:
                            img = img.convert("RGB")
                            img = img.resize(target_size, Image.Resampling.LANCZOS)
                            jpg_filepath = os.path.join(OUTPUT_DIR, f"safe_image_{index}.jpg")
                            img.save(jpg_filepath, format="JPEG")
                        return jpg_filepath
        except Exception as fallback_e:
            print(f"Wikimedia fallback failed for '{keyword}': {fallback_e}")
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
        print("Could not get transcript from YouTube. Falling back to downloading audio and using Whisper STT...")
        transcript = get_transcript_audio(video_id)
        
    if not transcript:
        print("Could not get transcript. Video might not have closed captions and STT failed.")
        return
        
    full_text = " ".join([t.text if hasattr(t, 'text') else t['text'] for t in transcript]).replace('\n', ' ')
    
    print("Summarizing...")
    # Generate ~15 sentences to guarantee the video spans roughly 50s-1minute in length
    summary_en = summarize_text(full_text, sentences_count=15)
    translator = GoogleTranslator(source='auto', target='hi')
    
    # Safely chunk for translation if summary is too long (though 4 sentences usually fine)
    summary_hi = translator.translate(summary_en)
    
    with open(os.path.join(OUTPUT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("--- English Summary ---\n" + summary_en + "\n\n--- Hindi Summary ---\n" + summary_hi)
        
    print("Breaking summary into sentences for Voiceover...")
    from nltk.tokenize import sent_tokenize
    sentences_en = sent_tokenize(summary_en)
    
    clips = []
    full_hindi_transcript = ""
    
    for i, sentence in enumerate(sentences_en):
        if not sentence.strip():
            continue
        try:
            print(f"Processing chunk {i+1}/{len(sentences_en)}...")
            
            # Translate text
            sentence_hi = translator.translate(sentence)
            if not sentence_hi:
                continue
            full_hindi_transcript += sentence_hi + " "
            
            # Generate Audio using edge-tts
            audio_path = os.path.join(OUTPUT_DIR, f"audio_{i}.mp3")
            communicate = edge_tts.Communicate(sentence_hi, "hi-IN-SwaraNeural")
            asyncio.run(communicate.save(audio_path))
            
            # Keyword & Image
            keyword = extract_keyword(sentence)
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
            
            # Animate the transition to make it feel like a dynamic short instead of a static slideshow
            img_clip = img_clip.fadein(0.5).fadeout(0.5)
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
