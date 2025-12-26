"""
Story Video Engine - Create cinematic multi-scene story videos
Uses Stock Videos (Pexels/Pixabay) + AI Images for dynamic video content
"""
import random
import base64
import time
import re
import requests
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple
from urllib.parse import quote
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_settings
from story_engine.generate_story import Story, StoryScene


@dataclass
class StoryVideoResult:
    """Result of story video generation"""
    file_path: Path
    thumbnail_path: Path
    duration: float
    scene_count: int


class StockVideoFetcher:
    """Fetch stock videos from Pexels and Pixabay APIs"""
    
    # Keywords mapping for different story types
    STORY_TYPE_KEYWORDS = {
        "moral": ["family", "nature", "sunrise", "peaceful", "village", "countryside", "helping", "kindness"],
        "funny": ["comedy", "funny", "laughing", "celebration", "party", "colorful", "happy", "dance"],
        "anime": ["japan", "city night", "neon", "rain", "cyberpunk", "sunset anime", "cherry blossom", "tokyo"],
        "horror": ["dark forest", "fog", "night", "abandoned", "creepy", "shadows", "moonlight", "mystery"],
        "inspirational": ["success", "mountains", "sunrise", "running", "achievement", "city skyline", "motivation", "climbing"]
    }
    
    def __init__(self):
        self.settings = get_settings()
        self.video_dir = self.settings.temp_dir / "story_videos"
        self.video_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_keywords(self, scene: StoryScene, story_type: str) -> List[str]:
        """Extract search keywords from scene description"""
        text = scene.image_prompt.lower()
        
        # Common visual keywords to look for
        visual_words = [
            "forest", "mountain", "ocean", "city", "village", "house", "tree", "river",
            "sunset", "sunrise", "night", "rain", "snow", "fire", "sky", "clouds",
            "person", "child", "old man", "woman", "family", "crowd", "alone",
            "running", "walking", "sitting", "crying", "laughing", "thinking",
            "car", "road", "path", "door", "window", "light", "dark", "shadow"
        ]
        
        found_keywords = []
        for word in visual_words:
            if word in text:
                found_keywords.append(word)
        
        # Add story type specific keywords
        type_keywords = self.STORY_TYPE_KEYWORDS.get(story_type, [])
        
        # Combine and prioritize
        if found_keywords:
            return found_keywords[:2] + type_keywords[:1]
        else:
            return type_keywords[:3]
    
    def fetch_video(self, scene: StoryScene, story_type: str, filename: str, duration: float = 10) -> Optional[Path]:
        """Fetch a stock video clip for the scene"""
        keywords = self.extract_keywords(scene, story_type)
        
        # Try Pexels first
        if self.settings.pexels_api_key:
            for keyword in keywords:
                try:
                    video_path = self._fetch_from_pexels(keyword, filename, duration)
                    if video_path:
                        return video_path
                except Exception as e:
                    logger.warning(f"Pexels failed for '{keyword}': {e}")
        
        # Try Pixabay
        if self.settings.pixabay_api_key:
            for keyword in keywords:
                try:
                    video_path = self._fetch_from_pixabay(keyword, filename, duration)
                    if video_path:
                        return video_path
                except Exception as e:
                    logger.warning(f"Pixabay failed for '{keyword}': {e}")
        
        return None
    
    def _fetch_from_pexels(self, keyword: str, filename: str, min_duration: float) -> Optional[Path]:
        """Fetch video from Pexels API"""
        url = "https://api.pexels.com/videos/search"
        headers = {"Authorization": self.settings.pexels_api_key}
        params = {
            "query": keyword,
            "orientation": "portrait",
            "size": "medium",
            "per_page": 10
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            videos = data.get("videos", [])
            
            # Filter for appropriate duration and quality
            for video in videos:
                if video.get("duration", 0) >= min_duration:
                    video_files = video.get("video_files", [])
                    # Prefer HD quality, portrait orientation
                    for vf in video_files:
                        if vf.get("height", 0) >= 720:
                            video_url = vf.get("link")
                            if video_url:
                                return self._download_video(video_url, filename)
            
            # If no perfect match, take first available
            if videos:
                video_files = videos[0].get("video_files", [])
                if video_files:
                    video_url = video_files[0].get("link")
                    if video_url:
                        return self._download_video(video_url, filename)
        
        return None
    
    def _fetch_from_pixabay(self, keyword: str, filename: str, min_duration: float) -> Optional[Path]:
        """Fetch video from Pixabay API"""
        url = "https://pixabay.com/api/videos/"
        params = {
            "key": self.settings.pixabay_api_key,
            "q": keyword,
            "video_type": "film",
            "per_page": 10
        }
        
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            hits = data.get("hits", [])
            
            for hit in hits:
                if hit.get("duration", 0) >= min_duration:
                    videos = hit.get("videos", {})
                    # Prefer medium or large quality
                    for quality in ["medium", "large", "small"]:
                        if quality in videos:
                            video_url = videos[quality].get("url")
                            if video_url:
                                return self._download_video(video_url, filename)
            
            # If no match, take first available
            if hits and hits[0].get("videos"):
                videos = hits[0]["videos"]
                for quality in ["medium", "large", "small"]:
                    if quality in videos:
                        video_url = videos[quality].get("url")
                        if video_url:
                            return self._download_video(video_url, filename)
        
        return None
    
    def _download_video(self, url: str, filename: str) -> Path:
        """Download video from URL"""
        output_path = self.video_dir / f"{filename}.mp4"
        
        response = requests.get(url, stream=True, timeout=120)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"Downloaded stock video: {output_path.name}")
            return output_path
        
        raise Exception(f"Failed to download video: {response.status_code}")


class AIImageGenerator:
    """Generate AI images with multiple fallback sources"""
    
    POLLINATIONS_MODELS = ["flux", "flux-pro", "flux-realism", "turbo"]
    
    def __init__(self):
        self.settings = get_settings()
        self.output_dir = self.settings.temp_dir / "story_images"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            from google import genai
            self.gemini_client = genai.Client(api_key=self.settings.google_api_key)
            self.has_gemini = True
        except:
            self.gemini_client = None
            self.has_gemini = False
    
    def generate_image(self, scene: StoryScene, story_type: str, filename: str) -> Path:
        """Generate image with multiple retry attempts and fallbacks"""
        output_path = self.output_dir / f"{filename}.png"
        prompt = self._build_prompt(scene, story_type)
        
        # Try Pollinations with multiple models
        for model in self.POLLINATIONS_MODELS:
            for attempt in range(2):
                try:
                    result = self._generate_pollinations(prompt, output_path, model)
                    if result:
                        return result
                except Exception as e:
                    logger.warning(f"Pollinations {model} attempt {attempt+1} failed: {e}")
                    time.sleep(2)
        
        # Try Gemini
        if self.has_gemini:
            try:
                return self._generate_gemini(prompt, output_path)
            except Exception as e:
                logger.warning(f"Gemini failed: {e}")
        
        # Final fallback
        return self._generate_fallback(scene, story_type, output_path)
    
    def _build_prompt(self, scene: StoryScene, story_type: str) -> str:
        """Build enhanced cinematic prompt"""
        styles = {
            "moral": "cinematic Pixar Disney style, warm lighting, emotional, family friendly, storybook illustration",
            "funny": "vibrant cartoon style, Dreamworks quality, colorful, comedic, expressive characters",
            "anime": "high quality anime art, Studio Ghibli style, dramatic lighting, detailed manga illustration",
            "horror": "dark cinematic, atmospheric fog, moonlit shadows, creepy mood, horror movie style",
            "inspirational": "epic cinematic photography, golden hour, dramatic sky, inspirational mood"
        }
        
        style = styles.get(story_type, styles["moral"])
        return f"{style}, {scene.image_prompt}, vertical 9:16 aspect ratio, no text, masterpiece, 8k quality"
    
    def _generate_pollinations(self, prompt: str, output_path: Path, model: str) -> Optional[Path]:
        """Generate with Pollinations.ai"""
        url = f"https://image.pollinations.ai/prompt/{quote(prompt)}"
        params = {
            "width": self.settings.video_width,
            "height": self.settings.video_height,
            "model": model,
            "seed": random.randint(1, 999999),
            "nologo": "true"
        }
        
        logger.info(f"Trying Pollinations ({model})...")
        response = requests.get(url, params=params, timeout=120)
        
        if response.status_code == 200 and len(response.content) > 5000:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            self._enhance_image(output_path)
            logger.info(f"Generated AI image: {output_path.name}")
            return output_path
        
        return None
    
    def _generate_gemini(self, prompt: str, output_path: Path) -> Path:
        """Generate with Gemini"""
        from google.genai import types
        
        response = self.gemini_client.models.generate_content(
            model="gemini-2.0-flash-preview-image-generation",
            contents=prompt,
            config=types.GenerateContentConfig(response_modalities=["image", "text"])
        )
        
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'inline_data') and part.inline_data:
                image_data = base64.b64decode(part.inline_data.data)
                with open(output_path, 'wb') as f:
                    f.write(image_data)
                self._enhance_image(output_path)
                return output_path
        
        raise Exception("No image in response")
    
    def _generate_fallback(self, scene: StoryScene, story_type: str, output_path: Path) -> Path:
        """Generate stylized gradient fallback"""
        colors = {
            "moral": [(255, 220, 180), (180, 100, 60)],
            "funny": [(255, 240, 150), (255, 160, 40)],
            "anime": [(180, 120, 220), (80, 40, 140)],
            "horror": [(60, 50, 70), (15, 12, 20)],
            "inspirational": [(150, 180, 220), (60, 100, 150)]
        }
        
        c = colors.get(story_type, colors["moral"])
        w, h = self.settings.video_width, self.settings.video_height
        
        img = Image.new('RGB', (w, h))
        draw = ImageDraw.Draw(img)
        
        for y in range(h):
            r = y / h
            color = tuple(int(c[0][i] * (1-r) + c[1][i] * r) for i in range(3))
            draw.line([(0, y), (w, y)], fill=color)
        
        img.save(output_path, "PNG")
        logger.info(f"Generated fallback gradient: {output_path.name}")
        return output_path
    
    def _enhance_image(self, path: Path):
        """Resize and enhance image"""
        try:
            with Image.open(path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize to target dimensions
                target_w, target_h = self.settings.video_width, self.settings.video_height
                img_ratio = img.width / img.height
                target_ratio = target_w / target_h
                
                if img_ratio > target_ratio:
                    new_w = int(img.height * target_ratio)
                    left = (img.width - new_w) // 2
                    img = img.crop((left, 0, left + new_w, img.height))
                else:
                    new_h = int(img.width / target_ratio)
                    top = (img.height - new_h) // 2
                    img = img.crop((0, top, img.width, top + new_h))
                
                img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
                
                # Enhance
                img = ImageEnhance.Contrast(img).enhance(1.05)
                img = ImageEnhance.Color(img).enhance(1.05)
                
                img.save(path, "PNG")
        except Exception as e:
            logger.warning(f"Enhancement failed: {e}")


class SubtitleRenderer:
    """Render cinematic subtitles"""
    
    def __init__(self):
        self.settings = get_settings()
    
    def render_on_frame(self, frame, text: str, width: int, height: int):
        """Render subtitle on a video frame (numpy array)"""
        from PIL import Image
        import numpy as np
        
        # Convert frame to PIL Image
        img = Image.fromarray(frame)
        img = self._add_subtitle(img, text)
        
        return np.array(img)
    
    def render_on_image(self, image_path: Path, text: str, output_path: Path) -> Path:
        """Render subtitle on image file"""
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            img = self._add_subtitle(img, text)
            img.save(output_path, "PNG")
        return output_path
    
    def _add_subtitle(self, img: Image.Image, text: str) -> Image.Image:
        """Add subtitle overlay to image"""
        img = img.convert('RGBA')
        width, height = img.size
        
        # Create gradient overlay at bottom
        overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw_overlay = ImageDraw.Draw(overlay)
        
        gradient_height = int(height * 0.35)
        for y in range(gradient_height):
            alpha = int(180 * (y / gradient_height))
            y_pos = height - gradient_height + y
            draw_overlay.line([(0, y_pos), (width, y_pos)], fill=(0, 0, 0, alpha))
        
        img = Image.alpha_composite(img, overlay)
        
        # Draw text
        draw = ImageDraw.Draw(img)
        font = self._get_font(42)
        wrapped = self._wrap_text(text, font, width - 100)
        
        bbox = draw.multiline_textbbox((0, 0), wrapped, font=font)
        text_h = bbox[3] - bbox[1]
        text_y = height - text_h - 100
        x = width // 2
        
        # Shadow
        for off in [(2, 2), (3, 3)]:
            draw.multiline_text((x + off[0], text_y + off[1]), wrapped,
                               font=font, fill=(0, 0, 0, 200), anchor="ma", align="center")
        
        # Main text
        draw.multiline_text((x, text_y), wrapped, font=font, fill="#FFFFFF", anchor="ma", align="center")
        
        return img.convert('RGB')
    
    def _get_font(self, size: int):
        for name in ["arial.ttf", "Arial.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"]:
            try:
                return ImageFont.truetype(name, size)
            except:
                continue
        return ImageFont.load_default()
    
    def _wrap_text(self, text: str, font, max_width: int) -> str:
        words = text.split()
        lines, current = [], []
        
        for word in words:
            current.append(word)
            try:
                line_width = font.getbbox(' '.join(current))[2]
            except:
                line_width = len(' '.join(current)) * 20
            
            if line_width > max_width:
                if len(current) > 1:
                    current.pop()
                    lines.append(' '.join(current))
                    current = [word]
                else:
                    lines.append(word)
                    current = []
        
        if current:
            lines.append(' '.join(current))
        
        return '\n'.join(lines)


class CinematicStoryComposer:
    """Compose cinematic story videos using stock videos + AI images"""
    
    def __init__(self):
        self.settings = get_settings()
        self.video_fetcher = StockVideoFetcher()
        self.image_gen = AIImageGenerator()
        self.subtitle_renderer = SubtitleRenderer()
    
    def create_story_reel(self, story: Story, audio_path: Path, 
                          audio_duration: float, filename: str) -> StoryVideoResult:
        """Create cinematic story video with stock videos and AI images"""
        from moviepy import VideoFileClip, ImageClip, AudioFileClip, CompositeVideoClip, concatenate_videoclips
        from moviepy.video.fx import CrossFadeIn, CrossFadeOut
        
        output_type = "anime" if story.story_type == "anime" else "stories"
        output_dir = self.settings.output_dir / output_type
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            num_scenes = len(story.scenes)
            transition_time = 0.6
            scene_duration = (audio_duration + 2.0) / num_scenes
            
            logger.info(f"Creating {num_scenes} scenes, ~{scene_duration:.1f}s each")
            
            clips = []
            
            for i, scene in enumerate(story.scenes):
                logger.info(f"Processing scene {scene.scene_number}/{num_scenes}...")
                
                # Try to get stock video first
                video_path = self.video_fetcher.fetch_video(
                    scene, story.story_type, 
                    f"{filename}_scene{scene.scene_number}",
                    duration=scene_duration
                )
                
                if video_path and video_path.exists():
                    # Use stock video
                    clip = self._process_video_clip(video_path, scene, scene_duration)
                else:
                    # Fall back to AI image
                    logger.info(f"No stock video, generating AI image for scene {scene.scene_number}...")
                    image_path = self.image_gen.generate_image(
                        scene, story.story_type,
                        f"{filename}_scene{scene.scene_number}"
                    )
                    clip = self._process_image_clip(image_path, scene, scene_duration)
                
                clips.append(clip)
                time.sleep(0.5)
            
            # Combine clips with transitions
            final_clips = []
            current_time = 0
            
            for i, clip in enumerate(clips):
                if i > 0:
                    clip = clip.with_effects([CrossFadeIn(transition_time)])
                if i < len(clips) - 1:
                    clip = clip.with_effects([CrossFadeOut(transition_time)])
                
                clip = clip.with_start(current_time)
                final_clips.append(clip)
                
                if i < len(clips) - 1:
                    current_time += clip.duration - transition_time
                else:
                    current_time += clip.duration
            
            # Composite
            final_video = CompositeVideoClip(
                final_clips, 
                size=(self.settings.video_width, self.settings.video_height)
            ).with_duration(current_time)
            
            # Add audio
            audio = AudioFileClip(str(audio_path))
            if audio.duration > final_video.duration:
                audio = audio.subclipped(0, final_video.duration)
            final_video = final_video.with_audio(audio)
            
            # Export
            output_path = output_dir / f"{filename}.mp4"
            final_video.write_videofile(
                str(output_path),
                fps=self.settings.video_fps,
                codec="libx264",
                audio_codec="aac",
                preset="medium",
                bitrate="8000k",
                logger=None
            )
            
            # Thumbnail
            thumb_path = self._create_thumbnail(clips[0], story.title, output_dir, filename)
            
            # Cleanup
            final_video.close()
            audio.close()
            for c in clips:
                c.close()
            
            logger.info(f"Created cinematic story: {output_path}")
            return StoryVideoResult(
                file_path=output_path,
                thumbnail_path=thumb_path,
                duration=current_time,
                scene_count=num_scenes
            )
            
        except Exception as e:
            logger.error(f"Story creation failed: {e}")
            raise
    
    def _process_video_clip(self, video_path: Path, scene: StoryScene, target_duration: float):
        """Process stock video clip with subtitle"""
        from moviepy import VideoFileClip
        import numpy as np
        
        clip = VideoFileClip(str(video_path))
        
        # Resize to target dimensions
        target_w, target_h = self.settings.video_width, self.settings.video_height
        
        # Crop to vertical aspect ratio
        clip_ratio = clip.w / clip.h
        target_ratio = target_w / target_h
        
        if clip_ratio > target_ratio:
            # Wider than needed, crop sides
            new_w = int(clip.h * target_ratio)
            x_center = clip.w // 2
            clip = clip.cropped(x1=x_center - new_w//2, x2=x_center + new_w//2)
        else:
            # Taller than needed, crop top/bottom
            new_h = int(clip.w / target_ratio)
            y_center = clip.h // 2
            clip = clip.cropped(y1=y_center - new_h//2, y2=y_center + new_h//2)
        
        clip = clip.resized((target_w, target_h))
        
        # Adjust duration
        if clip.duration > target_duration:
            clip = clip.subclipped(0, target_duration)
        elif clip.duration < target_duration:
            # Loop the video
            n_loops = int(target_duration / clip.duration) + 1
            from moviepy import concatenate_videoclips
            clip = concatenate_videoclips([clip] * n_loops).subclipped(0, target_duration)
        
        # Add subtitle overlay
        def add_subtitle(get_frame, t):
            frame = get_frame(t)
            return self.subtitle_renderer.render_on_frame(frame, scene.narration, target_w, target_h)
        
        clip = clip.transform(add_subtitle)
        
        return clip
    
    def _process_image_clip(self, image_path: Path, scene: StoryScene, duration: float):
        """Process AI image as video clip with Ken Burns effect and subtitle"""
        from moviepy import ImageClip
        
        # Add subtitle to image
        subtitle_path = image_path.parent / f"{image_path.stem}_sub.png"
        self.subtitle_renderer.render_on_image(image_path, scene.narration, subtitle_path)
        
        clip = ImageClip(str(subtitle_path)).with_duration(duration)
        clip = clip.resized((self.settings.video_width, self.settings.video_height))
        
        return clip
    
    def _create_thumbnail(self, first_clip, title: str, output_dir: Path, filename: str) -> Path:
        """Create thumbnail from first clip"""
        thumb_dir = output_dir / "thumbnails"
        thumb_dir.mkdir(exist_ok=True)
        thumb_path = thumb_dir / f"{filename}_thumb.jpg"
        
        try:
            # Get frame from first clip
            frame = first_clip.get_frame(0)
            img = Image.fromarray(frame)
            
            draw = ImageDraw.Draw(img)
            font = self._get_font(72)
            
            display_title = title[:25] + "..." if len(title) > 25 else title
            x, y = img.width // 2, img.height // 2
            draw.text((x, y), display_title, font=font, fill="#FFFFFF",
                     anchor="mm", stroke_width=4, stroke_fill="#000000")
            
            img.thumbnail((540, 960))
            img.save(thumb_path, "JPEG", quality=90)
        except:
            # Fallback
            img = Image.new('RGB', (540, 960), (50, 50, 100))
            img.save(thumb_path, "JPEG")
        
        return thumb_path
    
    def _get_font(self, size: int):
        for name in ["arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"]:
            try:
                return ImageFont.truetype(name, size)
            except:
                continue
        return ImageFont.load_default()


# Backward compatibility
StoryReelComposer = CinematicStoryComposer
