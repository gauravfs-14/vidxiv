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

# Load environment variables
load_dotenv(override=True)

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
    with st.spinner("Fetching paper from arXiv..."):
        search = arxiv.Search(id_list=[arxiv_id])
        paper = next(search.results())
        title, abstract = paper.title.strip(), paper.summary.strip()
        pdf_url = paper.pdf_url

    st.success(f"Paper fetched: {title}")
    st.markdown("---")
    st.markdown("**Generating multi-scene script using Gemini...**")

    prompt = f"""
You are a science explainer. Break down the following paper into a 4-6 scene YouTube video script.
Each scene should be a short paragraph suitable for a narrated slide.
Also provide a short caption for each scene as a title to use on the slide.
Try to match each scene with a possible figure from the paper (figure number or topic).

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
        pdf_path = os.path.join(tmpdir, f"{arxiv_id}.pdf")
        if pdf_url:
            response = requests.get(pdf_url)
            with open(pdf_path, 'wb') as f:
                f.write(response.content)
        doc = fitz.open(pdf_path)
        figure_paths = []

        for i in range(len(doc)):
            page = doc[i]
            images = page.get_images(full=True)
            for j, img in enumerate(images):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_path = os.path.join(tmpdir, f"figure_{i}_{j}.{image_ext}")
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                figure_paths.append(image_path)

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
            
            intro_title = TextClip(
                text=title,
                font_size=48 if not vertical else 36,
                color="white",
                size=(screen_size[0] - 100, None),
                method="caption"
            )
            
            intro_subtitle = TextClip(
                text="ArXiv Paper Video Summary",
                font_size=28 if not vertical else 22,
                color="#ecf0f1",
                size=(screen_size[0] - 100, None),
                method="caption"
            )
            
            try:
                intro_title = intro_title.with_duration(3).with_position(('center', screen_size[1] * 0.35))
                intro_subtitle = intro_subtitle.with_duration(3).with_position(('center', screen_size[1] * 0.55))
            except AttributeError:
                intro_title = intro_title.duration(3).set_position(('center', screen_size[1] * 0.35))
                intro_subtitle = intro_subtitle.duration(3).set_position(('center', screen_size[1] * 0.55))
            
            intro_scene = CompositeVideoClip([intro_background, intro_title, intro_subtitle], size=screen_size)
            try:
                intro_scene = intro_scene.with_fps(24)
            except AttributeError:
                intro_scene = intro_scene.with_fps(24)
            video_clips.append(intro_scene)
        except Exception as e:
            st.warning(f"Could not create intro slide: {e}")

        # Match figures to scenes based on figure hints
        def match_figure_to_scene(scene_hint, figure_paths):
            """Match a figure to a scene based on the hint and figure ordering"""
            if not scene_hint or not figure_paths:
                return None
            
            # Try to extract figure numbers from hint
            figure_numbers = re.findall(r'[Ff]igure?\s*(\d+)', scene_hint)
            if figure_numbers:
                # Convert to 0-based index
                fig_num = int(figure_numbers[0]) - 1
                if 0 <= fig_num < len(figure_paths):
                    return figure_paths[fig_num]
            
            # Try keyword matching
            hint_lower = scene_hint.lower()
            keywords = ['graph', 'chart', 'plot', 'diagram', 'image', 'photo', 'table']
            if any(keyword in hint_lower for keyword in keywords):
                # Return first available figure if hint suggests visual content
                return figure_paths[0] if figure_paths else None
            
            return None

        def calculate_optimal_font_size(text, base_font_size, max_width, max_height, vertical=False):
            """Calculate optimal font size to fit text within given dimensions"""
            # Estimate character dimensions (rough approximation)
            char_width_ratio = 0.6  # Characters are roughly 60% as wide as they are tall
            char_height_pixels = base_font_size
            char_width_pixels = char_height_pixels * char_width_ratio
            
            # Calculate how many characters fit per line
            chars_per_line = int(max_width / char_width_pixels)
            
            # Estimate number of lines needed
            lines_needed = len(text) / chars_per_line
            total_height_needed = lines_needed * char_height_pixels * 1.5  # 1.5 for line spacing
            
            # Reduce font size if text doesn't fit
            if total_height_needed > max_height:
                scale_factor = max_height / total_height_needed
                new_font_size = int(base_font_size * scale_factor)
                # Don't go below minimum readable size
                return max(new_font_size, 16 if not vertical else 14)
            
            return base_font_size

        for i, (scene_title, scene_text, figure_hint) in enumerate(scenes):
            try:
                st.markdown(f"**Processing Scene {i+1}...**")
                audio_path = os.path.join(tmpdir, f"scene_{i}.mp3")
                gTTS(text=scene_text, lang='en').save(audio_path)
                audio_clip = AudioFileClip(audio_path)

                # Calculate optimal font sizes based on content length
                title_font_size = calculate_optimal_font_size(
                    scene_title, 
                    72 if not vertical else 58, 
                    screen_size[0] - 120, 
                    screen_size[1] * 0.15,  # Title area is 15% of screen
                    vertical
                )
                
                text_font_size = calculate_optimal_font_size(
                    scene_text,
                    36 if not vertical else 28,
                    screen_size[0] - 140,
                    screen_size[1] * 0.25,  # Text area is 25% of screen
                    vertical
                )

                # Dynamic text wrapping based on font size
                chars_per_line = max(40, int((screen_size[0] - 140) / (text_font_size * 0.6)))
                title_chars_per_line = max(30, int((screen_size[0] - 120) / (title_font_size * 0.6)))
                
                wrapped_text = fill(scene_text, width=chars_per_line)
                wrapped_title = fill(scene_title, width=title_chars_per_line)

                # Create a light background clip (PowerPoint style)
                background = ColorClip(size=screen_size, color=(248, 249, 250))
                try:
                    background = background.with_duration(audio_clip.duration)
                except AttributeError:
                    background = background.duration(audio_clip.duration)
                
                try:
                    # Create title with dynamic font sizing
                    title_clip = TextClip(
                        text=wrapped_title,
                        font_size=title_font_size,
                        color="#1a1a1a",  # Dark text
                        size=(screen_size[0] - 120, None),  # Leave margins
                        method="caption",
                        bg_color=None,
                        stroke_width=0,
                        interline=max(6, int(title_font_size * 0.15))
                    )
                    
                    # Create main text with dynamic font sizing
                    text_clip = TextClip(
                        text=wrapped_text,
                        font_size=text_font_size,
                        color="#2c3e50",  # Slightly lighter dark text
                        size=(screen_size[0] - 140, None),  # Leave margins
                        method="caption",
                        bg_color=None,
                        stroke_width=0,
                        interline=max(8, int(text_font_size * 0.25))
                    )
                    
                    # Position elements hierarchically
                    title_y_pos = screen_size[1] * 0.15  # 15% from top
                    text_y_pos = screen_size[1] * 0.35   # 35% from top
                    
                    # Try MoviePy v2 methods first
                    try:
                        title_clip = title_clip.with_duration(audio_clip.duration).with_position(('center', title_y_pos))
                        text_clip = text_clip.with_duration(audio_clip.duration).with_position(('center', text_y_pos))
                    except AttributeError:
                        # Fallback to MoviePy v1 methods
                        title_clip = title_clip.duration(audio_clip.duration).set_position(('center', title_y_pos))
                        text_clip = text_clip.duration(audio_clip.duration).set_position(('center', text_y_pos))
                        
                except Exception as e:
                    st.warning(f"⚠️ Using fallback font method for Scene {i+1}: {e}")
                    title_y_pos = screen_size[1] * 0.15  # Define here too for fallback
                    text_y_pos = screen_size[1] * 0.35
                    
                    # Simplified fallback with dynamic sizing
                    title_clip = TextClip(
                        text=wrapped_title,
                        font_size=max(title_font_size - 8, 32),  # Slightly smaller for compatibility
                        color="#1a1a1a",
                        size=(screen_size[0] - 100, None),
                        method="label"
                    )
                    text_clip = TextClip(
                        text=wrapped_text,
                        font_size=max(text_font_size - 4, 20),  # Slightly smaller for compatibility
                        color="#2c3e50",
                        size=(screen_size[0] - 120, None),
                        method="label"
                    )
                    
                    try:
                        title_clip = title_clip.with_duration(audio_clip.duration).with_position(('center', title_y_pos))
                        text_clip = text_clip.with_duration(audio_clip.duration).with_position(('center', text_y_pos))
                    except AttributeError:
                        title_clip = title_clip.duration(audio_clip.duration).set_position(('center', title_y_pos))
                        text_clip = text_clip.duration(audio_clip.duration).set_position(('center', text_y_pos))

                # Start with background and text
                clips = [background, title_clip, text_clip]

                # Add intelligently matched figure/image
                matched_figure = match_figure_to_scene(figure_hint, figure_paths)
                if matched_figure:
                    try:
                        st.info(f"📊 Using figure for scene {i+1}: {os.path.basename(matched_figure)} (hint: {figure_hint})")
                        
                        # Load and resize image intelligently
                        from PIL import Image
                        with Image.open(matched_figure) as img:
                            img_width, img_height = img.size
                            aspect_ratio = img_width / img_height
                        
                        # Calculate optimal figure size (max 30% of screen height, maintain aspect ratio)
                        max_figure_height = int(screen_size[1] * 0.3)
                        max_figure_width = int(screen_size[0] * 0.8)  # Max 80% of screen width
                        
                        if aspect_ratio > (max_figure_width / max_figure_height):
                            # Width is limiting factor
                            figure_width = max_figure_width
                            figure_height = int(figure_width / aspect_ratio)
                        else:
                            # Height is limiting factor
                            figure_height = max_figure_height
                            figure_width = int(figure_height * aspect_ratio)
                        
                        try:
                            figure = ImageClip(matched_figure).resized(width=figure_width, height=figure_height)
                        except AttributeError:
                            # Fallback for older MoviePy versions
                            figure = ImageClip(matched_figure)
                            figure = figure.resized(width=figure_width, height=figure_height)
                        
                        # Position at bottom with appropriate spacing
                        figure_y_pos = screen_size[1] - figure_height - 50  # 50px margin from bottom
                        
                        try:
                            figure = figure.with_duration(audio_clip.duration).with_position(('center', figure_y_pos))
                        except AttributeError:
                            figure = figure.with_duration(audio_clip.duration).set_position(('center', figure_y_pos))
                        
                        clips.append(figure)
                        
                        # Adjust text position if figure takes up space
                        if figure_y_pos < (screen_size[1] * 0.6):
                            # Move text up if figure is large
                            new_text_y = screen_size[1] * 0.25
                            try:
                                text_clip = text_clip.with_position(('center', new_text_y))
                            except AttributeError:
                                text_clip = text_clip.set_position(('center', new_text_y))
                            clips[2] = text_clip  # Update text clip in list
                            
                    except Exception as e:
                        st.warning(f"Could not add matched figure for scene {i+1}: {e}")
                elif figure_paths:
                    # Fallback: use the first available figure if no specific match
                    try:
                        st.info(f"📊 Using fallback figure for scene {i+1}: {os.path.basename(figure_paths[0])}")
                        figure_height = int(screen_size[1] * 0.25)  # Smaller for fallback
                        try:
                            figure = ImageClip(figure_paths[0]).resized(height=figure_height)
                        except AttributeError:
                            # Fallback for older MoviePy versions
                            figure = ImageClip(figure_paths[0])
                            figure = figure.resized(height=figure_height)
                        
                        figure_y_pos = screen_size[1] * 0.7  # Position lower
                        
                        try:
                            figure = figure.with_duration(audio_clip.duration).with_position(('center', figure_y_pos))
                        except AttributeError:
                            figure = figure.with_duration(audio_clip.duration).set_position(('center', figure_y_pos))
                        clips.append(figure)
                    except Exception as e:
                        st.warning(f"Could not add fallback figure for scene {i+1}: {e}")

                # Add scene number indicator (top-right corner)
                scene_number = TextClip(
                    text=f"Scene {i+1}/{len(scenes)}",
                    font_size=24,
                    color="#6c757d",  # Gray text
                    bg_color=None
                )
                try:
                    scene_number = scene_number.with_duration(audio_clip.duration).with_position((screen_size[0] - 150, 30))
                except AttributeError:
                    scene_number = scene_number.duration(audio_clip.duration).set_position((screen_size[0] - 150, 30))
                clips.append(scene_number)

                # Create decorative header line using ColorClip
                line_height = 4
                line_width = int(screen_size[0] * 0.8)
                line_clip = ColorClip(size=(line_width, line_height), color=(52, 152, 219))
                try:
                    line_clip = line_clip.with_duration(audio_clip.duration).with_position(('center', title_y_pos - 40))
                except AttributeError:
                    line_clip = line_clip.duration(audio_clip.duration).set_position(('center', title_y_pos - 40))
                clips.append(line_clip)

                try:
                    # Try MoviePy v2 methods
                    scene = CompositeVideoClip(clips, size=screen_size).with_audio(audio_clip).with_fps(24)
                except AttributeError:
                    # Fallback to MoviePy v1 methods
                    scene = CompositeVideoClip(clips, size=screen_size).with_audio(audio_clip).set_fps(24)
                
                # Add subtle fade in/out effects for professional look
                try:
                    from moviepy.video.fx import FadeIn
                    from moviepy.video.fx import FadeOut
                    fadein = FadeIn(0.5)  # 0.5 second fade
                    fadeout = FadeOut(0.5)  # 0.5 second fade
                    scene = scene.fx(fadein, 0.5).fx(fadeout, 0.5)
                except:
                    # Skip fade effects if not available
                    pass
                    
                video_clips.append(scene)

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
                
                outro_title = TextClip(
                    text="Thank You for Watching!",
                    font_size=52 if not vertical else 40,
                    color="white",
                    size=(screen_size[0] - 100, None),
                    method="caption"
                )
                
                outro_subtitle = TextClip(
                    text="Generated by VidXiv",
                    font_size=32 if not vertical else 24,
                    color="#ecf0f1",
                    size=(screen_size[0] - 100, None),
                    method="caption"
                )
                
                try:
                    outro_title = outro_title.with_duration(3).with_position(('center', screen_size[1] * 0.4))
                    outro_subtitle = outro_subtitle.with_duration(3).with_position(('center', screen_size[1] * 0.55))
                except AttributeError:
                    outro_title = outro_title.duration(3).set_position(('center', screen_size[1] * 0.4))
                    outro_subtitle = outro_subtitle.duration(3).set_position(('center', screen_size[1] * 0.55))
                
                outro_scene = CompositeVideoClip([outro_background, outro_title, outro_subtitle], size=screen_size)
                try:
                    outro_scene = outro_scene.with_fps(24)
                except AttributeError:
                    outro_scene = outro_scene.with_fps(24)
                video_clips.append(outro_scene)
            except Exception as e:
                st.warning(f"Could not create outro slide: {e}")
            
            final_video = concatenate_videoclips(video_clips, method="compose")

            if background_music_file is not None:
                try:
                    music_path = os.path.join(tmpdir, "bg_music.mp3")
                    with open(music_path, "wb") as f:
                        f.write(background_music_file.read())
                    
                    try:
                        # Try MoviePy v2 methods first
                        bg_music = AudioFileClip(music_path).with_volume_scaled(0.2).with_duration(final_video.duration)
                    except AttributeError:
                        # Fallback to MoviePy v1 methods
                        bg_music = AudioFileClip(music_path).with_volume_scaled(0.2).with_duration(final_video.duration)
                    
                    combined_audio = CompositeAudioClip([final_video.audio, bg_music])
                    try:
                        # Try MoviePy v2 methods
                        final_video = final_video.with_audio(combined_audio)
                    except AttributeError:
                        # Fallback to MoviePy v1 methods
                        final_video = final_video.with_audio(combined_audio)
                except Exception as e:
                    st.warning(f"Could not add background music: {e}")

            final_path = os.path.join(tmpdir, f"{arxiv_id}_video.mp4")
            try:
                final_video.write_videofile(final_path, fps=24)

                with open(final_path, "rb") as f:
                    video_data = f.read()
                    st.video(video_data)
                    st.download_button("📥 Download Video", video_data, file_name=f"{arxiv_id}_video.mp4", mime="video/mp4")

                st.success("✅ Video generation complete!")
            except Exception as e:
                st.error(f"Error generating video: {e}")
                st.info("Please try again or check if all dependencies are properly installed.")
