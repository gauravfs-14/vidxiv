import os
import tempfile
import arxiv
import pymupdf as fitz  # PyMuPDF
import streamlit as st
from gtts import gTTS
from moviepy import (
    TextClip, AudioFileClip, ImageClip, ColorClip,
    concatenate_videoclips, CompositeVideoClip, CompositeAudioClip
)
from langchain_google_genai import ChatGoogleGenerativeAI
from PIL import Image
from dotenv import load_dotenv
import requests
import re
from textwrap import fill
import numpy as np
from pydub import AudioSegment

# Load environment variables
load_dotenv(override=True)

# Compact logging system for better UI
class CompactLogger:
    def __init__(self):
        self.logs = []
        self.container = None
    
    def init_container(self):
        """Initialize the streamlit container for logs"""
        if not self.container:
            with st.expander("📝 Processing Logs (click to expand)", expanded=False):
                self.container = st.container()
    
    def log(self, message, level="info"):
        """Add a log message"""
        self.logs.append(f"[{level.upper()}] {message}")
        if self.container:
            with self.container:
                # Show only last 10 logs in a compact format
                recent_logs = self.logs[-10:]
                log_text = "\n".join(recent_logs)
                st.text_area("", value=log_text, height=200, disabled=True, key=f"logs_{len(self.logs)}")
    
    def info(self, message):
        self.log(message, "info")
    
    def warning(self, message):
        self.log(message, "warn")
    
    def error(self, message):
        self.log(message, "error")
    
    def success(self, message):
        self.log(message, "success")

# Global compact logger
compact_logger = CompactLogger()

model = ChatGoogleGenerativeAI(
    model=os.getenv("LLM_MODEL", "gemini-1.5-flash"),
    temperature=float(os.getenv("LLM_TEMPERATURE", 0.7)),
    max_tokens=int(os.getenv("LLM_MAX_TOKENS", 2048)),
)

# --- Streamlit UI ---
st.set_page_config(page_title="ArXiv to Video Generator", layout="centered")
st.title("🎥 ArXiv Paper to Video (YouTube or Shorts)")
arxiv_id = st.text_input("Enter arXiv ID (e.g., 2401.06015):")
vertical = st.checkbox("Make vertical video for Shorts/Reels (9:16)?", value=False)
background_music_file = st.file_uploader("Upload background music (MP3, optional)", type="mp3")

if st.button("Generate Video") and arxiv_id:
    # Initialize compact logger
    compact_logger.init_container()
    
    with st.spinner("Fetching paper from arXiv..."):
        search = arxiv.Search(id_list=[arxiv_id])
        paper = next(search.results())
        title, abstract = paper.title.strip(), paper.summary.strip()
        pdf_url = paper.pdf_url

    st.success(f"Paper fetched: {title}")
    compact_logger.info(f"Fetched paper: {title[:50]}...")
    st.markdown("---")
    st.markdown("**Generating multi-scene script using Gemini...**")

    prompt = f"""
You are a science explainer. Break down the following paper into a 4-6 scene YouTube video script.
Each scene should be a SHORT paragraph (maximum 80 words) suitable for a narrated slide.
Also provide a SHORT caption for each scene as a title (maximum 6 words) to use on the slide.
Try to match each scene with a possible figure from the paper (figure number or topic).

IMPORTANT: Keep text concise as it will be displayed on video slides with limited space.
- Title: Maximum 6 words
- Text: Maximum 80 words (about 2-3 sentences)

Title: {title}

Abstract: {abstract}

Your output should be exactly in the following format without any additional text or formatting:
Scene 1:
Title: ...
Text: ...
Figure Hint: ...
Scene 2:
Title: ...
Text: ...
Figure Hint: ...
...
"""
    response = model.invoke(prompt)
    script_text = str(response.content).strip()
    st.code(script_text)

    scene_blocks = script_text.split("Scene")
    scenes = []
    for block in scene_blocks:
        if ":" in block:
            lines = block.strip().splitlines()
            title_line = next((l for l in lines if l.startswith("Title:")), "")
            text_line = next((l for l in lines if l.startswith("Text:")), "")
            figure_hint = next((l for l in lines if l.startswith("Figure Hint:")), "")
            title = title_line.replace("Title:", "").strip()
            text = text_line.replace("Text:", "").strip()
            hint = figure_hint.replace("Figure Hint:", "").strip()
            if title and text:
                scenes.append((title, text, hint))

    with tempfile.TemporaryDirectory() as tmpdir:
        st.info("🎨 Preparing beautiful, varied layouts for the video...")
        
        # Define beautiful color schemes for varied layouts with improved contrast
        color_schemes = [
            {
                'name': 'Modern Blue',
                'bg': (15, 32, 60),  # Dark navy for better contrast
                'primary': (255, 255, 255),  # White text for contrast
                'secondary': (70, 130, 180),  # Steel blue
                'text': (240, 248, 255),  # Light blue text
                'accent': (30, 144, 255),  # Bright blue accent
                'shadow': (0, 0, 0)  # Black shadow
            },
            {
                'name': 'Elegant Purple',
                'bg': (45, 20, 70),  # Deep purple background
                'primary': (255, 255, 255),  # White title
                'secondary': (147, 112, 219),  # Medium purple
                'text': (230, 220, 255),  # Light purple text
                'accent': (186, 85, 211),  # Medium orchid
                'shadow': (0, 0, 0)
            },
            {
                'name': 'Professional Green',
                'bg': (20, 50, 30),  # Dark forest green
                'primary': (255, 255, 255),  # White title
                'secondary': (60, 179, 113),  # Medium sea green
                'text': (220, 255, 220),  # Light green text
                'accent': (50, 205, 50),  # Lime green
                'shadow': (0, 0, 0)
            },
            {
                'name': 'Warm Orange',
                'bg': (80, 40, 20),  # Dark brown/orange
                'primary': (255, 255, 255),  # White title
                'secondary': (255, 165, 0),  # Orange
                'text': (255, 248, 220),  # Cornsilk text
                'accent': (255, 140, 0),  # Dark orange
                'shadow': (0, 0, 0)
            },
            {
                'name': 'Academic Navy',
                'bg': (20, 25, 50),  # Deep navy
                'primary': (255, 255, 255),  # White title
                'secondary': (106, 90, 205),  # Slate blue
                'text': (220, 230, 255),  # Light blue text
                'accent': (72, 61, 139),  # Dark slate blue
                'shadow': (0, 0, 0)
            }
        ]

        video_clips = []
        screen_size = (720, 1280) if vertical else (1280, 720)

        from textwrap import fill

        # Create an intro slide
        try:
            st.markdown("**Creating intro slide...**")
            intro_background = ColorClip(size=screen_size, color=(44, 62, 80))  # Dark blue background
            try:
                intro_background = intro_background.with_duration(3)  # 3 second intro
            except AttributeError:
                intro_background = intro_background.duration(3)
            
            # Create intro slide text with proper word wrapping
            intro_title_text = fill(title, width=max(15, (screen_size[0] - 100) // 35), 
                                   break_long_words=False, break_on_hyphens=True)
            intro_subtitle_text = fill("ArXiv Paper Video Summary", width=max(10, (screen_size[0] - 100) // 25), 
                                      break_long_words=False, break_on_hyphens=True)
            
            intro_title = TextClip(
                text=intro_title_text,
                font_size=max(32, min(48 if not vertical else 36, len(title) < 50 and 48 or 32)),
                color="white",
                size=(screen_size[0] - 100, screen_size[1] // 3),
                method="caption"  # Better text wrapping
            )
            
            intro_subtitle = TextClip(
                text=intro_subtitle_text,
                font_size=24 if not vertical else 18,
                color="#ecf0f1",
                size=(screen_size[0] - 100, screen_size[1] // 6),
                method="caption"
            )
            
            try:
                intro_title = intro_title.with_duration(3).with_position(('center', int(screen_size[1] * 0.35)))
                intro_subtitle = intro_subtitle.with_duration(3).with_position(('center', int(screen_size[1] * 0.55)))
            except AttributeError:
                intro_title = intro_title.duration(3).set_position(('center', int(screen_size[1] * 0.35)))
                intro_subtitle = intro_subtitle.duration(3).set_position(('center', int(screen_size[1] * 0.55)))
            
            intro_scene = CompositeVideoClip([intro_background, intro_title, intro_subtitle], size=screen_size)
            try:
                intro_scene = intro_scene.with_fps(24)
            except AttributeError:
                intro_scene = intro_scene.with_fps(24)
            video_clips.append(intro_scene)
        except Exception as e:
            st.warning(f"Could not create intro slide: {e}")



        def calculate_optimal_font_size(text, base_font_size, max_width, max_height, vertical=False):
            """Calculate optimal font size to maximize space utilization"""
            if not text:
                return base_font_size
            
            # Even more aggressive sizing to really fill available space
            min_font_size = 24 if not vertical else 20  # Increased minimum
            max_font_size = min(base_font_size * 6, 160)  # Allow much larger fonts
            
            best_font_size = min_font_size
            
            # Use binary search for more efficient optimization
            low, high = min_font_size, max_font_size
            
            while low <= high:
                test_size = (low + high) // 2
                
                # Test with wrapped text at this font size - proper word wrapping
                chars_per_line = max(8, int(max_width / (test_size * 0.45)))  # Character width estimate
                test_wrapped = fill(text, width=chars_per_line, break_long_words=False, break_on_hyphens=True)
                
                # Calculate actual dimensions with more precise estimates
                lines = test_wrapped.count('\n') + 1
                longest_line = max(test_wrapped.split('\n'), key=len) if test_wrapped.split('\n') else text
                estimated_width = len(longest_line) * test_size * 0.45  # Character width estimate
                estimated_height = lines * test_size * 1.1  # Line height estimate
                
                # Use even more of available space for maximum utilization
                if estimated_width <= max_width * 0.98 and estimated_height <= max_height * 0.96:
                    best_font_size = test_size
                    low = test_size + 1  # Try larger
                else:
                    high = test_size - 1  # Try smaller
            
            return max(best_font_size, min_font_size)
        
        def truncate_text_if_needed(text, max_words=80):
            """Truncate text if it's too long"""
            words = text.split()
            if len(words) > max_words:
                return ' '.join(words[:max_words]) + '...'
            return text
        
        def truncate_title_if_needed(title, max_words=6):
            """Truncate title if it's too long"""
            words = title.split()
            if len(words) > max_words:
                return ' '.join(words[:max_words])
            return title

        def calculate_text_dimensions(text, font_size):
            """Estimate text dimensions accurately"""
            if not text:
                return 0, 0
            
            lines = text.count('\n') + 1
            longest_line = max(text.split('\n'), key=len) if '\n' in text else text
            
            # Accurate width calculation
            width = len(longest_line) * font_size * 0.55
            height = lines * font_size * 1.3  # Include line spacing
            
            return width, height

        def create_animated_text(text, font_size, color, position, size, duration, animation_type="slide_in"):
            """Create animated text clips with various animation effects"""
            try:
                # Ensure text is not empty and size is valid
                if not text or not text.strip():
                    return None
                
                # Validate and fix size parameter
                if not size or len(size) != 2:
                    size = (screen_size[0] - 100, None)  # Default size with padding
                else:
                    width, height = size
                    if width <= 0:
                        width = screen_size[0] - 100
                    if height is not None and height <= 0:
                        height = None  # Let TextClip auto-calculate height
                    size = (width, height)
                
                # Helper function to handle 'center' positions in animations
                def get_numeric_position(pos, screen_size):
                    if isinstance(pos, tuple):
                        x, y = pos
                        if x == 'center':
                            x = screen_size[0] // 2
                        if isinstance(y, str) and y == 'center':
                            y = screen_size[1] // 2
                        return (int(x), int(y))
                    return pos
                
                # Create the base text clip with more robust size handling
                try:
                    # Ensure text is properly formatted and not empty
                    if not text or not text.strip():
                        st.warning("Empty text provided to create_animated_text")
                        return None
                    
                    # Ensure size is valid
                    safe_width = max(size[0] if size[0] else 300, 100)
                    safe_size = (safe_width, size[1])
                    
                    # Pre-wrap text to ensure word boundaries are respected
                    wrapped_text = fill(text.strip(), width=max(40, safe_width // (max(font_size, 12) // 2)), 
                                       break_long_words=False, break_on_hyphens=True)
                    
                    text_clip = TextClip(
                        text=wrapped_text,
                        font_size=max(font_size, 12),
                        color=f"rgb{color}",
                        size=safe_size,
                        method="caption",
                        text_align="center",
                        vertical_align="center",
                        margin=(20, 15),
                        bg_color=None
                    )
                    
                    # Validate the text clip size immediately and thoroughly
                    if not hasattr(text_clip, 'size') or not text_clip.size or len(text_clip.size) != 2:
                        raise ValueError(f"TextClip size attribute is invalid: {getattr(text_clip, 'size', 'None')}")
                    
                    clip_width, clip_height = text_clip.size
                    if clip_width <= 0 or clip_height <= 0:
                        st.warning(f"TextClip has invalid dimensions: {clip_width}x{clip_height}, recreating with safer parameters")
                        # If size is invalid, recreate with safer parameters
                        wrapped_text = fill(text.strip(), width=max(30, safe_width // 20), 
                                           break_long_words=False, break_on_hyphens=True)
                        text_clip = TextClip(
                            text=wrapped_text,
                            font_size=max(font_size, 16),
                            color=f"rgb{color}",
                            size=(max(safe_width, 300), None),  # Force minimum width
                            method="label"
                        )
                        
                        # Validate again
                        if not hasattr(text_clip, 'size') or not text_clip.size or text_clip.size[0] <= 0 or text_clip.size[1] <= 0:
                            st.error(f"Failed to create valid TextClip even with fallback parameters")
                            return None
                    
                except Exception as text_create_error:
                    st.warning(f"Caption method failed: {text_create_error}, trying label method")
                    # If caption method fails, try label method
                    try:
                        safe_width = max(size[0] if size[0] else 300, 100)
                        wrapped_text = fill(text.strip(), width=max(25, safe_width // 25), 
                                           break_long_words=False, break_on_hyphens=True)
                        text_clip = TextClip(
                            text=wrapped_text,
                            font_size=max(font_size, 16),
                            color=f"rgb{color}",
                            size=(safe_width, None),
                            method="label"
                        )
                        
                        # Final validation
                        if not hasattr(text_clip, 'size') or not text_clip.size or text_clip.size[0] <= 0 or text_clip.size[1] <= 0:
                            st.error(f"Failed to create valid TextClip with label method")
                            return None
                            
                    except Exception as label_error:
                        st.error(f"Both caption and label methods failed: {label_error}")
                        return None
                
                # Apply duration
                try:
                    text_clip = text_clip.with_duration(duration)
                except AttributeError:
                    text_clip = text_clip.duration(duration)
                
                # Convert position to numeric for animations
                numeric_position = get_numeric_position(position, screen_size)
                
                # Apply animations based on type - simplified to avoid zero-width frames
                if animation_type == "slide_in":
                    # Use simple static positioning instead of complex animation to avoid frame size issues
                    try:
                        text_clip = text_clip.with_position(position)
                    except AttributeError:
                        text_clip = text_clip.set_position(position)
                        
                elif animation_type == "fade_in_up":
                    # Use simple static positioning
                    try:
                        text_clip = text_clip.with_position(position)
                    except AttributeError:
                        text_clip = text_clip.set_position(position)
                        
                elif animation_type == "scale_in":
                    # Use simple static positioning
                    try:
                        text_clip = text_clip.with_position(position)
                    except AttributeError:
                        text_clip = text_clip.set_position(position)
                        
                else:  # Default: simple positioning
                    try:
                        text_clip = text_clip.with_position(position)
                    except AttributeError:
                        text_clip = text_clip.set_position(position)
                
                return text_clip
                
            except Exception as e:
                # Fallback to simple text clip with safe dimensions
                try:
                    # Ensure safe fallback size
                    safe_width = max(screen_size[0] - 100, 300)  # Minimum 300px width
                    
                    text_clip = TextClip(
                        text=text,
                        font_size=max(font_size, 16),  # Minimum font size
                        color=f"rgb{color}",
                        size=(safe_width, None),
                        method="label"
                    )
                    try:
                        return text_clip.with_duration(duration).with_position(position)
                    except AttributeError:
                        return text_clip.duration(duration).set_position(position)
                except Exception as fallback_error:
                    # If even the fallback fails, return None
                    return None

        def create_gradient_background(size, color1, color2, duration):
            """Create a gradient-like background using multiple color strips"""
            strips = []
            strip_count = 20
            strip_width = size[0] // strip_count
            
            for i in range(strip_count):
                # Interpolate between colors
                progress = i / (strip_count - 1)
                r = int(color1[0] + (color2[0] - color1[0]) * progress)
                g = int(color1[1] + (color2[1] - color1[1]) * progress)
                b = int(color1[2] + (color2[2] - color1[2]) * progress)
                
                strip_color = (r, g, b)
                strip = ColorClip(size=(strip_width + 2, size[1]), color=strip_color)  # +2 to avoid gaps
                
                try:
                    strip = strip.with_duration(duration).with_position((i * strip_width, 0))
                except AttributeError:
                    strip = strip.duration(duration).set_position((i * strip_width, 0))
                    
                strips.append(strip)
            
            return strips

        for i, (scene_title, scene_text, figure_hint) in enumerate(scenes):
            try:
                st.markdown(f"**Processing Scene {i+1}...**")
                audio_path = os.path.join(tmpdir, f"scene_{i}.mp3")
                gTTS(text=scene_text, lang='en').save(audio_path)
                
                # Speed up the audio to 1.5x using pydub
                try:
                    # Load with pydub and speed up
                    audio_segment = AudioSegment.from_mp3(audio_path)
                    sped_up_audio = audio_segment.speedup(playback_speed=1.15)
                    
                    # Save the sped up version
                    fast_audio_path = os.path.join(tmpdir, f"scene_{i}_fast.mp3")
                    sped_up_audio.export(fast_audio_path, format="mp3")
                    
                    # Load with MoviePy
                    audio_clip = AudioFileClip(fast_audio_path)
                    st.info(f"🔊 Sped up audio by 1.5x (duration: {audio_clip.duration:.1f}s)")
                except Exception as e:
                    # Fallback to normal speed
                    audio_clip = AudioFileClip(audio_path)
                    st.warning(f"Could not speed up audio for scene {i+1}: {e}")

                # Truncate text if needed for professional layout
                scene_title = truncate_title_if_needed(scene_title, 6)
                scene_text = truncate_text_if_needed(scene_text, 80)

                # Get color scheme for this scene
                scheme = color_schemes[i % len(color_schemes)]
                st.info(f"🎨 Using {scheme['name']} color scheme for scene {i+1}")
                compact_logger.info(f"Scene {i+1}: Using {scheme['name']} theme")

                # Calculate responsive layout dimensions
                if vertical:  # 9:16 - Vertical layout
                    content_width = screen_size[0] - 120  # More padding
                    title_area_height = 200
                    text_area_height = 400
                    
                    title_y_pos = 150
                    text_y_pos = 380
                    
                else:  # 16:9 - Horizontal layout
                    content_width = screen_size[0] - 160  # More padding
                    title_area_height = 150
                    text_area_height = 300
                    
                    title_y_pos = 120
                    text_y_pos = 300
                
                # Use larger, more readable font sizes
                title_font_size = 56 if not vertical else 42  # Even larger for impact
                text_font_size = 32 if not vertical else 26   # Better readability

                # Create beautiful gradient background
                gradient_strips = create_gradient_background(
                    screen_size, 
                    scheme["bg"], 
                    tuple(max(0, c - 40) for c in scheme["bg"]),  # Darker gradient
                    audio_clip.duration
                )
                
                # Start with gradient background
                clips = gradient_strips.copy()
                
                # Add simplified geometric elements for visual interest
                try:
                    if scheme["name"] in ["Modern Blue", "Academic Navy"]:
                        # Static side panel (no animation to avoid frame issues)
                        panel_width = 100
                        panel_height = screen_size[1]
                        
                        left_panel = ColorClip(size=(panel_width, panel_height), color=scheme["secondary"])
                        try:
                            left_panel = left_panel.with_duration(audio_clip.duration).with_position((0, 0))
                        except AttributeError:
                            left_panel = left_panel.duration(audio_clip.duration).set_position((0, 0))
                        clips.append(left_panel)
                        
                        # Static accent strip
                        strip_height = 25
                        strip_width = screen_size[0]
                        accent_strip = ColorClip(size=(strip_width, strip_height), color=scheme["accent"])
                        
                        try:
                            accent_strip = accent_strip.with_duration(audio_clip.duration).with_position((0, screen_size[1] - strip_height - 30))
                        except AttributeError:
                            accent_strip = accent_strip.duration(audio_clip.duration).set_position((0, screen_size[1] - strip_height - 30))
                        clips.append(accent_strip)
                        
                    elif scheme["name"] in ["Elegant Purple", "Professional Green"]:
                        # Static corner element
                        corner_size = 150
                        corner_element = ColorClip(size=(corner_size, corner_size), color=scheme["accent"])
                        
                        try:
                            corner_element = corner_element.with_duration(audio_clip.duration).with_position((screen_size[0] - corner_size, screen_size[1] - corner_size))
                        except AttributeError:
                            corner_element = corner_element.duration(audio_clip.duration).set_position((screen_size[0] - corner_size, screen_size[1] - corner_size))
                        clips.append(corner_element)
                        
                        # Static accent circles
                        for j in range(3):
                            circle_size = 40 + j * 20
                            circle = ColorClip(size=(circle_size, circle_size), color=scheme["secondary"])
                            
                            x = 100 + j * 150
                            y = 100 + j * 100
                            
                            try:
                                circle = circle.with_duration(audio_clip.duration).with_position((x, y))
                            except AttributeError:
                                circle = circle.duration(audio_clip.duration).set_position((x, y))
                            clips.append(circle)
                        
                    elif scheme["name"] == "Warm Orange":
                        # Static wave-like strips
                        for k in range(5):
                            strip_width = screen_size[0] // 8
                            strip_height = screen_size[1]
                            
                            wave_color = tuple(int(c * (0.9 - k * 0.1)) for c in scheme["accent"])
                            wave_strip = ColorClip(size=(strip_width, strip_height), color=wave_color)
                            
                            base_x = screen_size[0] - strip_width * (k + 1)
                            
                            try:
                                wave_strip = wave_strip.with_duration(audio_clip.duration).with_position((base_x, 0))
                            except AttributeError:
                                wave_strip = wave_strip.duration(audio_clip.duration).set_position((base_x, 0))
                            clips.append(wave_strip)
                    
                except Exception as e:
                    st.warning(f"Could not add geometric elements for scene {i+1}: {e}")

                # Create decorative line - static positioning
                line_height = 8
                line_width = content_width
                line_color = scheme["accent"]
                
                # Static line positioned above title
                line_clip = ColorClip(size=(line_width, line_height), color=line_color)
                line_y = title_y_pos - 50
                line_x = (screen_size[0] - line_width) // 2  # Center the line
                
                try:
                    line_clip = line_clip.with_duration(audio_clip.duration).with_position((line_x, line_y))
                except AttributeError:
                    line_clip = line_clip.duration(audio_clip.duration).set_position((line_x, line_y))
                clips.append(line_clip)

                # Create animated text with shadows and effects
                try:
                    # Validate text content before creating clips
                    if not scene_title or not scene_title.strip():
                        scene_title = f"Scene {i+1}"
                        st.warning(f"Empty title for scene {i+1}, using fallback")
                    
                    if not scene_text or not scene_text.strip():
                        scene_text = "Content not available"
                        st.warning(f"Empty text for scene {i+1}, using fallback")
                    
                    # Title with shadow effect and slide-in animation
                    title_clip = create_animated_text(
                        text=scene_title.strip(),
                        font_size=title_font_size,
                        color=scheme['primary'],
                        position=('center', title_y_pos),
                        size=(content_width, title_area_height),
                        duration=audio_clip.duration,
                        animation_type="slide_in"
                    )
                    
                    # Body text with fade-in-up animation (delayed)
                    text_clip = create_animated_text(
                        text=scene_text.strip(),
                        font_size=text_font_size,
                        color=scheme['text'],
                        position=('center', text_y_pos),
                        size=(content_width, text_area_height),
                        duration=audio_clip.duration,
                        animation_type="fade_in_up"
                    )
                    
                    # Validate clips more thoroughly before adding
                    if (title_clip is not None and 
                        hasattr(title_clip, 'size') and 
                        title_clip.size and 
                        len(title_clip.size) == 2 and
                        title_clip.size[0] > 0 and 
                        title_clip.size[1] > 0):
                        clips.append(title_clip)
                        compact_logger.info(f"Scene {i+1}: Added title clip {title_clip.size}")
                    else:
                        compact_logger.warning(f"Scene {i+1}: Title clip invalid")
                        
                    if (text_clip is not None and 
                        hasattr(text_clip, 'size') and 
                        text_clip.size and 
                        len(text_clip.size) == 2 and
                        text_clip.size[0] > 0 and 
                        text_clip.size[1] > 0):
                        clips.append(text_clip)
                        compact_logger.info(f"Scene {i+1}: Added text clip {text_clip.size}")
                    else:
                        compact_logger.warning(f"Scene {i+1}: Text clip invalid")
                    
                except Exception as e:
                    st.warning(f"⚠️ Using fallback text method for Scene {i+1}: {e}")
                    # Simple fallback with validation
                    try:
                        # Ensure safe dimensions for fallback clips
                        safe_width = max(content_width - 40, 300)
                        
                        # Apply word wrapping to fallback text
                        wrapped_title = fill(scene_title or "Scene Title", width=max(20, safe_width // 30), 
                                           break_long_words=False, break_on_hyphens=True)
                        wrapped_text = fill(scene_text or "Scene content", width=max(25, safe_width // 25), 
                                          break_long_words=False, break_on_hyphens=True)
                        
                        title_clip = TextClip(
                            text=wrapped_title,
                            font_size=max(24, title_font_size - 6),
                            color=f"rgb{scheme['primary']}",
                            size=(safe_width, None),
                            method="label"  # Use label method as more reliable fallback
                        )
                        text_clip = TextClip(
                            text=wrapped_text,
                            font_size=max(20, text_font_size - 4),
                            color=f"rgb{scheme['text']}",
                            size=(safe_width, None),
                            method="label"  # Use label method as more reliable fallback
                        )
                        
                        # Check if clips have valid dimensions before positioning
                        if hasattr(title_clip, 'size') and title_clip.size[1] > 0:
                            try:
                                title_clip = title_clip.with_duration(audio_clip.duration).with_position(('center', title_y_pos))
                            except AttributeError:
                                title_clip = title_clip.duration(audio_clip.duration).set_position(('center', title_y_pos))
                            clips.append(title_clip)
                        else:
                            st.warning(f"Fallback title clip has invalid dimensions for scene {i+1}")
                        
                        if hasattr(text_clip, 'size') and text_clip.size[1] > 0:
                            try:
                                text_clip = text_clip.with_duration(audio_clip.duration).with_position(('center', text_y_pos))
                            except AttributeError:
                                text_clip = text_clip.duration(audio_clip.duration).set_position(('center', text_y_pos))
                            clips.append(text_clip)
                        else:
                            st.warning(f"Fallback text clip has invalid dimensions for scene {i+1}")
                            
                    except Exception as fallback_error:
                        st.warning(f"Even fallback text creation failed for scene {i+1}: {fallback_error}")
                        # Skip text clips for this scene

                # Add animated scene counter with simple animation
                try:
                    scene_counter_text = f"{i+1}/{len(scenes)}"
                    if scene_counter_text and len(scene_counter_text.strip()) > 0:
                        scene_counter = TextClip(
                            text=scene_counter_text,
                            font_size=22,
                            color=f"rgb{scheme['accent']}",
                            bg_color=None,
                            method="label"  # Use label method for better reliability
                        )
                        
                        # Validate scene counter thoroughly before adding animation
                        if (hasattr(scene_counter, 'size') and 
                            scene_counter.size and 
                            len(scene_counter.size) == 2 and
                            scene_counter.size[0] > 0 and 
                            scene_counter.size[1] > 0):
                            # Simple fade in animation for scene counter
                            def counter_pos_func(t):
                                if t < 0.5:
                                    # Slide in from right
                                    progress = t / 0.5
                                    eased = 1 - (1 - progress) ** 3
                                    start_x = screen_size[0]  # Start from right edge
                                    end_x = screen_size[0] - 80  # End position
                                    x = start_x + (end_x - start_x) * eased
                                    return (int(x), 40)
                                else:
                                    return (screen_size[0] - 80, 40)
                            
                            try:
                                scene_counter = scene_counter.with_duration(audio_clip.duration).with_position(counter_pos_func)
                            except AttributeError:
                                scene_counter = scene_counter.duration(audio_clip.duration).set_position(counter_pos_func)
                            clips.append(scene_counter)
                        else:
                            st.warning(f"Scene counter has invalid dimensions for scene {i+1}: {scene_counter.size if hasattr(scene_counter, 'size') else 'no size'}")
                    else:
                        st.warning(f"Scene counter text is empty for scene {i+1}")
                        
                except Exception as counter_error:
                    st.warning(f"Could not create scene counter for scene {i+1}: {counter_error}")



                try:
                    # Validate all clips before creating the scene
                    valid_scene_clips = []
                    compact_logger.info(f"Scene {i+1}: Validating {len(clips)} clips")
                    
                    for j, clip in enumerate(clips):
                        try:
                            if clip is None:
                                compact_logger.warning(f"Scene {i+1}: Skipping None clip {j+1}")
                                continue
                                
                            if not hasattr(clip, 'size') or not hasattr(clip, 'duration'):
                                compact_logger.warning(f"Scene {i+1}: Clip {j+1} missing attributes")
                                continue
                            
                            clip_size = getattr(clip, 'size', None)
                            clip_duration = getattr(clip, 'duration', None)
                            
                            if not clip_size or not clip_duration:
                                compact_logger.warning(f"Scene {i+1}: Clip {j+1} None size/duration")
                                continue
                                
                            if len(clip_size) != 2 or clip_size[0] <= 0 or clip_size[1] <= 0:
                                compact_logger.warning(f"Scene {i+1}: Clip {j+1} invalid size {clip_size}")
                                continue
                                
                            if clip_duration <= 0:
                                compact_logger.warning(f"Scene {i+1}: Clip {j+1} invalid duration")
                                continue
                            
                            # Test a frame from this clip
                            try:
                                test_frame = clip.get_frame(min(0.1, clip_duration/2))
                                if test_frame is None or not hasattr(test_frame, 'shape'):
                                    compact_logger.warning(f"Scene {i+1}: Clip {j+1} invalid frame")
                                    continue
                                if test_frame.shape[0] == 0 or test_frame.shape[1] == 0:
                                    compact_logger.warning(f"Scene {i+1}: Clip {j+1} zero-dim frame")
                                    continue
                            except Exception as frame_test_error:
                                compact_logger.warning(f"Scene {i+1}: Clip {j+1} frame test failed")
                                continue
                            
                            valid_scene_clips.append(clip)
                            
                        except Exception as clip_validation_error:
                            compact_logger.warning(f"Scene {i+1}: Clip {j+1} validation error")
                            continue
                    
                    if len(valid_scene_clips) == 0:
                        st.error(f"❌ No valid clips found for scene {i+1}")
                        continue
                    
                    compact_logger.success(f"Scene {i+1}: Using {len(valid_scene_clips)}/{len(clips)} clips")
                    
                    # Create the scene with validated clips
                    scene = CompositeVideoClip(valid_scene_clips, size=screen_size)
                    
                    # Add audio
                    try:
                        scene = scene.with_audio(audio_clip).with_fps(24)
                    except AttributeError:
                        scene = scene.with_audio(audio_clip).set_fps(24)
                    
                    # Skip fade transitions for compatibility
                    # Custom fade effects are not compatible with this MoviePy version
                    
                    video_clips.append(scene)
                    st.success(f"✅ Created animated scene {i+1} with {scheme['name']} theme")

                except Exception as e:
                    st.warning(f"⚠️ Error creating scene {i+1}: {e}")
                    # Create simple fallback scene with safer text clip creation
                    try:
                        simple_bg = ColorClip(size=screen_size, color=scheme["bg"])
                        
                        # Create safer text clips for fallback
                        safe_width = screen_size[0] - 100
                        
                        # Apply word wrapping to simple fallback text
                        wrapped_title = fill(scene_title or "Scene Title", width=max(15, safe_width // 40), 
                                           break_long_words=False, break_on_hyphens=True)
                        wrapped_text = fill(scene_text or "Scene content", width=max(20, safe_width // 30), 
                                          break_long_words=False, break_on_hyphens=True)
                        
                        simple_title = TextClip(
                            text=wrapped_title,
                            font_size=36, 
                            color="white",
                            size=(safe_width, None),
                            method="label"
                        )
                        simple_text = TextClip(
                            text=wrapped_text,
                            font_size=24, 
                            color="white",
                            size=(safe_width, None),
                            method="label"
                        )
                        
                        # Validate fallback clips before using
                        if hasattr(simple_title, 'size') and simple_title.size[1] > 0 and \
                           hasattr(simple_text, 'size') and simple_text.size[1] > 0:
                            
                            try:
                                simple_bg = simple_bg.with_duration(audio_clip.duration)
                                simple_title = simple_title.with_duration(audio_clip.duration).with_position(('center', 200))
                                simple_text = simple_text.with_duration(audio_clip.duration).with_position(('center', 350))
                                simple_scene = CompositeVideoClip([simple_bg, simple_title, simple_text], size=screen_size).with_audio(audio_clip).with_fps(24)
                            except AttributeError:
                                simple_bg = simple_bg.duration(audio_clip.duration)
                                simple_title = simple_title.duration(audio_clip.duration).set_position(('center', 200))
                                simple_text = simple_text.duration(audio_clip.duration).set_position(('center', 350))
                                simple_scene = CompositeVideoClip([simple_bg, simple_title, simple_text], size=screen_size).with_audio(audio_clip).set_fps(24)
                            
                            video_clips.append(simple_scene)
                            st.info(f"✅ Created fallback scene {i+1}")
                        else:
                            st.warning(f"❌ Even fallback scene creation failed for scene {i+1} - skipping")
                    except Exception as fallback_scene_error:
                        st.warning(f"❌ Fallback scene creation failed for scene {i+1}: {fallback_scene_error}")
                        # Skip this scene entirely if fallback fails

            except Exception as e:
                st.warning(f"❌ Skipped scene {i+1} due to error: {e}")


        if not video_clips:
            st.error("No scenes were successfully created. Please check your inputs or retry.")
        else:
            # Create an outro slide
            try:
                st.markdown("**Creating outro slide...**")
                outro_background = ColorClip(size=screen_size, color=(39, 174, 96))  # Green background
                try:
                    outro_background = outro_background.with_duration(3)  # 3 second outro
                except AttributeError:
                    outro_background = outro_background.duration(3)
                
                # Create outro slide text with proper word wrapping
                outro_title_text = fill("Thank You for Watching!", width=max(10, (screen_size[0] - 100) // 40), 
                                       break_long_words=False, break_on_hyphens=True)
                outro_subtitle_text = fill("Generated by VidXiv", width=max(8, (screen_size[0] - 100) // 35), 
                                          break_long_words=False, break_on_hyphens=True)
                
                outro_title = TextClip(
                    text=outro_title_text,
                    font_size=40 if not vertical else 32,
                    color="white",
                    size=(screen_size[0] - 100, screen_size[1] // 4),
                    method="caption"
                )
                
                outro_subtitle = TextClip(
                    text=outro_subtitle_text,
                    font_size=28 if not vertical else 20,
                    color="#ecf0f1",
                    size=(screen_size[0] - 100, screen_size[1] // 6),
                    method="caption"
                )
                
                # Check if text clips have valid dimensions
                if hasattr(outro_title, 'size') and outro_title.size[1] > 0:
                    st.info(f"Outro title size: {outro_title.size}")
                else:
                    st.warning("Outro title has invalid dimensions, using fallback")
                    outro_title = TextClip("Thank You!", font_size=40, color="white")
                
                if hasattr(outro_subtitle, 'size') and outro_subtitle.size[1] > 0:
                    st.info(f"Outro subtitle size: {outro_subtitle.size}")
                else:
                    st.warning("Outro subtitle has invalid dimensions, using fallback")
                    outro_subtitle = TextClip("VidXiv", font_size=28, color="#ecf0f1")
                
                try:
                    outro_title = outro_title.with_duration(3).with_position(('center', int(screen_size[1] * 0.4)))
                    outro_subtitle = outro_subtitle.with_duration(3).with_position(('center', int(screen_size[1] * 0.55)))
                except AttributeError:
                    outro_title = outro_title.duration(3).set_position(('center', int(screen_size[1] * 0.4)))
                    outro_subtitle = outro_subtitle.duration(3).set_position(('center', int(screen_size[1] * 0.55)))
                
                outro_scene = CompositeVideoClip([outro_background, outro_title, outro_subtitle], size=screen_size)
                try:
                    outro_scene = outro_scene.with_fps(24)
                except AttributeError:
                    outro_scene = outro_scene.with_fps(24)
                video_clips.append(outro_scene)
            except Exception as e:
                st.warning(f"Could not create outro slide: {e}")
                
            # Ensure all video clips have the same properties before concatenation
            if video_clips:
                st.info(f"📹 Validating and concatenating {len(video_clips)} video clips...")
                
                # Validate and filter video clips before concatenation
                valid_clips = []
                compact_logger.info(f"Validating {len(video_clips)} video clips for concatenation")
                
                for i, clip in enumerate(video_clips):
                    try:
                        # Basic attribute validation
                        if not hasattr(clip, 'size') or not hasattr(clip, 'duration'):
                            compact_logger.warning(f"Clip {i+1}: Missing attributes")
                            continue
                            
                        # Get clip properties safely
                        size = getattr(clip, 'size', None)
                        duration = getattr(clip, 'duration', None)
                        
                        if size is None or duration is None:
                            compact_logger.warning(f"Clip {i+1}: None size/duration")
                            continue
                            
                        if not isinstance(size, (tuple, list)) or len(size) != 2:
                            compact_logger.warning(f"Clip {i+1}: Invalid size format")
                            continue
                            
                        width, height = size
                        
                        # Validate dimensions - this is the critical check for broadcasting errors
                        if width <= 0 or height <= 0:
                            compact_logger.warning(f"Clip {i+1}: Invalid dimensions ({width}x{height})")
                            continue
                            
                        if duration <= 0:
                            compact_logger.warning(f"Clip {i+1}: Invalid duration")
                            continue
                            
                        # Test multiple frame times to catch dynamic zero-dimension issues
                        all_frames_valid = True
                        test_times = [0.0, 0.1, duration * 0.25, duration * 0.5, duration * 0.75]
                        if duration > 0.2:
                            test_times.append(duration - 0.1)
                        
                        for test_time in test_times:
                            if test_time >= 0 and test_time < duration:
                                try:
                                    test_frame = clip.get_frame(test_time)
                                    if test_frame is None:
                                        compact_logger.warning(f"Clip {i+1}: None frame at {test_time:.1f}s")
                                        all_frames_valid = False
                                        break
                                    if not hasattr(test_frame, 'shape') or len(test_frame.shape) < 2:
                                        compact_logger.warning(f"Clip {i+1}: Invalid frame shape")
                                        all_frames_valid = False
                                        break
                                    if test_frame.shape[0] == 0 or test_frame.shape[1] == 0:
                                        compact_logger.warning(f"Clip {i+1}: Zero-dim frame at {test_time:.1f}s")
                                        all_frames_valid = False
                                        break
                                except Exception as frame_error:
                                    compact_logger.warning(f"Clip {i+1}: Frame test failed at {test_time:.1f}s")
                                    all_frames_valid = False
                                    break
                        
                        if all_frames_valid:
                            # Ensure clip has the correct screen size
                            if clip.size != screen_size:
                                compact_logger.info(f"Clip {i+1}: Resizing from {clip.size} to {screen_size}")
                                try:
                                    clip = clip.resized(screen_size)
                                except AttributeError:
                                    clip = clip.resize(screen_size)
                                
                                # Validate resized clip
                                if not hasattr(clip, 'size') or clip.size[0] <= 0 or clip.size[1] <= 0:
                                    compact_logger.warning(f"Clip {i+1}: Invalid after resizing")
                                    continue
                            
                            # Final validation: test one more frame after potential resizing
                            try:
                                final_test_frame = clip.get_frame(0.1 if duration > 0.2 else 0.0)
                                if final_test_frame is None or final_test_frame.shape[0] == 0 or final_test_frame.shape[1] == 0:
                                    compact_logger.warning(f"Clip {i+1}: Final validation failed")
                                    continue
                            except Exception as final_test_error:
                                compact_logger.warning(f"Clip {i+1}: Final test failed")
                                continue
                            
                            valid_clips.append(clip)
                            compact_logger.success(f"Clip {i+1}: Valid ({width}x{height}, {duration:.1f}s)")
                        else:
                            compact_logger.warning(f"Clip {i+1}: Contains invalid frames")
                        
                    except Exception as validation_error:
                        compact_logger.warning(f"Clip {i+1}: Validation error")
                
                if len(valid_clips) == 0:
                    st.error("No valid video clips found for concatenation")
                    st.stop()
                elif len(valid_clips) < len(video_clips):
                    st.warning(f"Using {len(valid_clips)} out of {len(video_clips)} clips (some were invalid)")
                
                try:
                    final_video = concatenate_videoclips(valid_clips, method="compose")
                    st.success(f"✅ Successfully concatenated {len(valid_clips)} video clips")
                except Exception as concat_error:
                    st.error(f"Error concatenating video clips: {concat_error}")
                    
                    # Try simpler concatenation method as fallback
                    try:
                        st.info("Trying alternative concatenation method...")
                        final_video = concatenate_videoclips(valid_clips)
                        st.success("✅ Alternative concatenation successful")
                    except Exception as concat_error2:
                        st.error(f"All concatenation methods failed: {concat_error2}")
                        st.stop()
            else:
                st.error("No video clips to concatenate")
                st.stop()

            if background_music_file is not None:
                try:
                    music_path = os.path.join(tmpdir, "bg_music.mp3")
                    with open(music_path, "wb") as f:
                        f.write(background_music_file.read())
                    
                    combined_audio = final_video.audio  # Default to original audio
                    try:
                        # Simple approach - just load the audio and set duration
                        bg_music = AudioFileClip(music_path)
                        # Try to set duration to match video
                        try:
                            bg_music = bg_music.with_duration(final_video.duration)
                        except AttributeError:
                            try:
                                bg_music = bg_music.duration(final_video.duration)
                            except AttributeError:
                                # Use as-is if we can't set duration
                                pass
                        
                        # Try to combine audio
                        combined_audio = CompositeAudioClip([final_video.audio, bg_music])
                            
                    except Exception as bg_error:
                        # If anything fails with background music, skip it
                        st.warning(f"Could not add background music: {bg_error}")
                        combined_audio = final_video.audio
                    try:
                        # Try to set the combined audio
                        final_video = final_video.with_audio(combined_audio)
                    except Exception as audio_error:
                        # If audio combination fails, keep original
                        st.warning(f"Could not combine audio: {audio_error}")
                        pass
                except Exception as e:
                    st.warning(f"Could not add background music: {e}")

            final_path = os.path.join(tmpdir, f"{arxiv_id}_video.mp4")
            try:
                # Additional validation before writing
                st.info(f"📹 Final video properties: duration={final_video.duration:.1f}s, size={final_video.size}")
                
                # Test getting multiple frames from different time points to catch issues early
                test_times = [0.1, final_video.duration * 0.25, final_video.duration * 0.5, final_video.duration * 0.75]
                for test_time in test_times:
                    if test_time < final_video.duration:
                        try:
                            test_frame = final_video.get_frame(test_time)
                            if test_frame is None or not hasattr(test_frame, 'shape') or test_frame.shape[1] == 0:
                                st.error(f"❌ Invalid frame at time {test_time:.1f}s: shape={test_frame.shape if test_frame is not None else 'None'}")
                                st.error("The final video has frames with zero dimensions. This will cause broadcasting errors.")
                                st.stop()
                        except Exception as frame_test_error:
                            st.error(f"❌ Frame test failed at time {test_time:.1f}s: {frame_test_error}")
                            st.stop()
                
                st.info("✅ All frame tests passed")
                
                # Write video with error handling and progress tracking
                st.info("🎬 Writing final video file...")
                
                # Additional debugging: log final video composition
                compact_logger.info(f"Final video: {len(valid_clips)} clips, {final_video.size}, {final_video.duration:.1f}s")
                
                # One final frame test across the entire video duration
                try:
                    # Test a few strategic frames across the entire video
                    test_points = [0.0, final_video.duration * 0.1, final_video.duration * 0.5, final_video.duration * 0.9]
                    for point in test_points:
                        if point < final_video.duration:
                            test_frame = final_video.get_frame(point)
                            if test_frame is None:
                                st.error(f"❌ None frame detected at time {point:.1f}s")
                                st.stop()
                            elif not hasattr(test_frame, 'shape'):
                                st.error(f"❌ Frame without shape attribute at time {point:.1f}s")
                                st.stop()
                            elif test_frame.shape[0] == 0 or test_frame.shape[1] == 0:
                                st.error(f"❌ Zero-dimension frame detected at time {point:.1f}s: {test_frame.shape}")
                                st.error("This indicates a clip with zero dimensions made it through validation.")
                                st.stop()
                            compact_logger.success(f"Frame at {point:.1f}s: {test_frame.shape}")
                except Exception as pre_write_error:
                    st.error(f"❌ Pre-write validation failed: {pre_write_error}")
                    st.stop()
                
                # Use the most basic, compatible settings for video writing
                final_video.write_videofile(
                    final_path, 
                    fps=24,
                    codec='libx264',
                    audio_codec='aac'
                )

                # Verify the output file was created successfully
                if os.path.exists(final_path) and os.path.getsize(final_path) > 0:
                    with open(final_path, "rb") as f:
                        video_data = f.read()
                        st.video(video_data)
                        st.download_button("📥 Download Video", video_data, file_name=f"{arxiv_id}_video.mp4", mime="video/mp4")
                    
                    st.success("✅ Video generation complete!")
                else:
                    st.error("❌ Video file was not created or is empty")
                    
            except Exception as e:
                st.error(f"Error generating video: {e}")
                
                # Try alternative video writing approach with even simpler settings
                try:
                    st.info("🔄 Attempting alternative video export method...")
                    alt_final_path = os.path.join(tmpdir, f"{arxiv_id}_video_alt.mp4")
                    
                    # Try to recreate the video with a fresh composite to avoid any cached issues
                    st.info("Recreating video composite...")
                    fresh_video = concatenate_videoclips(valid_clips, method="compose")
                    
                    # Write with the most basic settings possible
                    fresh_video.write_videofile(
                        alt_final_path,
                        fps=24
                    )
                    
                    if os.path.exists(alt_final_path) and os.path.getsize(alt_final_path) > 0:
                        with open(alt_final_path, "rb") as f:
                            video_data = f.read()
                            st.video(video_data)
                            st.download_button("📥 Download Video", video_data, file_name=f"{arxiv_id}_video.mp4", mime="video/mp4")
                        
                        st.success("✅ Video generation complete (using alternative method)!")
                    else:
                        st.error("❌ Alternative video export also failed")
                        
                except Exception as alt_error:
                    st.error(f"Alternative video export failed: {alt_error}")
                    
                    # Final fallback: try writing individual scenes and manually combining them
                    try:
                        st.info("🔄 Attempting manual scene-by-scene export...")
                        scene_paths = []
                        
                        for i, clip in enumerate(valid_clips):
                            scene_path = os.path.join(tmpdir, f"scene_{i}.mp4")
                            try:
                                clip.write_videofile(scene_path, fps=24)
                                if os.path.exists(scene_path) and os.path.getsize(scene_path) > 0:
                                    scene_paths.append(scene_path)
                                    st.info(f"✅ Exported scene {i+1}")
                                else:
                                    st.warning(f"❌ Failed to export scene {i+1}")
                            except Exception as scene_error:
                                st.warning(f"❌ Scene {i+1} export failed: {scene_error}")
                        
                        if scene_paths:
                            st.info(f"Successfully exported {len(scene_paths)} individual scenes")
                            st.info("You can download the individual scene files, though automatic concatenation failed")
                            
                            # Offer the first scene as a sample
                            if scene_paths:
                                with open(scene_paths[0], "rb") as f:
                                    sample_data = f.read()
                                    st.download_button("📥 Download Sample Scene", sample_data, file_name=f"{arxiv_id}_scene_1.mp4", mime="video/mp4")
                        else:
                            st.error("❌ All export methods failed")
                            
                    except Exception as manual_error:
                        st.error(f"Manual export also failed: {manual_error}")
                        st.info("Please try again or check if all dependencies are properly installed.")
