"""
Video Engine for creating YouTube Shorts / Reels
Uses Gemini for image generation with gradient fallback
"""
import random
import base64
from pathlib import Path
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from google import genai
from google.genai import types
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_settings, VISUAL_THEMES


@dataclass
class VideoResult:
    """Result of video generation"""
    file_path: Path
    thumbnail_path: Path
    duration: float


class ImageGenerator:
    """Generate background images using Gemini or gradients"""
    
    def __init__(self):
        self.settings = get_settings()
        self.client = genai.Client(api_key=self.settings.google_api_key)
        self.output_dir = self.settings.temp_dir / "images"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_background(self, theme: dict, filename: str) -> Path:
        """Generate background image"""
        output_path = self.output_dir / f"{filename}.png"
        
        try:
            # Try Gemini image generation
            return self._generate_with_gemini(theme, output_path)
        except Exception as e:
            logger.warning(f"Gemini image gen failed: {e}, using gradient")
            return self._generate_gradient(theme, output_path)
    
    def _generate_with_gemini(self, theme: dict, output_path: Path) -> Path:
        """Generate image using Gemini 2.0 Flash"""
        prompt = f"""Create a beautiful vertical background image for a motivational quote.
Style: {theme['prompt']}
Requirements: 
- Vertical format (9:16 aspect ratio)
- Visually stunning but not too busy
- Good for text overlay
- No text or words in the image
- Professional quality"""

        response = self.client.models.generate_content(
            model="gemini-2.0-flash-preview-image-generation",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["image", "text"],
            )
        )
        
        # Extract image from response
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'inline_data') and part.inline_data:
                image_data = base64.b64decode(part.inline_data.data)
                with open(output_path, 'wb') as f:
                    f.write(image_data)
                self._resize_image(output_path)
                logger.info(f"Generated Gemini image: {output_path}")
                return output_path
        
        raise Exception("No image in response")
    
    def _generate_gradient(self, theme: dict, output_path: Path) -> Path:
        """Generate gradient background as fallback"""
        width, height = self.settings.video_width, self.settings.video_height
        
        gradients = {
            "sunset": [(255, 150, 50), (255, 50, 100)],
            "ocean": [(0, 100, 200), (0, 50, 100)],
            "mountains": [(100, 130, 160), (40, 60, 80)],
            "forest": [(20, 100, 50), (10, 50, 30)],
            "space": [(10, 10, 50), (50, 20, 100)],
            "minimalist": [(240, 240, 250), (200, 200, 220)],
        }
        
        colors = gradients.get(theme.get("name", "sunset"), gradients["sunset"])
        
        image = Image.new('RGB', (width, height))
        draw = ImageDraw.Draw(image)
        
        for y in range(height):
            ratio = y / height
            r = int(colors[0][0] * (1 - ratio) + colors[1][0] * ratio)
            g = int(colors[0][1] * (1 - ratio) + colors[1][1] * ratio)
            b = int(colors[0][2] * (1 - ratio) + colors[1][2] * ratio)
            draw.line([(0, y), (width, y)], fill=(r, g, b))
        
        image.save(output_path, "PNG")
        logger.info(f"Generated gradient: {output_path}")
        return output_path
    
    def _resize_image(self, path: Path):
        """Resize to exact reel dimensions"""
        target_w, target_h = self.settings.video_width, self.settings.video_height
        
        with Image.open(path) as img:
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
            img.save(path, "PNG")


class TextRenderer:
    """Render text on images"""
    
    def __init__(self):
        self.settings = get_settings()
        self.fonts_dir = Path("./assets/fonts")
    
    def render_quote(self, image_path: Path, quote: str, author: str, theme: dict) -> Path:
        """Render quote on image"""
        output_path = image_path.parent / f"quote_{image_path.name}"
        
        with Image.open(image_path) as img:
            # Add dark overlay
            overlay = Image.new('RGBA', img.size, (0, 0, 0, 100))
            img = img.convert('RGBA')
            img = Image.alpha_composite(img, overlay).convert('RGB')
            
            draw = ImageDraw.Draw(img)
            
            # Fonts
            quote_font = self._get_font(56)
            author_font = self._get_font(36)
            
            # Wrap and draw quote
            wrapped = self._wrap_text(quote, quote_font, img.width - 100)
            bbox = draw.multiline_textbbox((0, 0), wrapped, font=quote_font)
            quote_h = bbox[3] - bbox[1]
            
            quote_y = (img.height - quote_h - 80) // 2
            x = img.width // 2
            
            # Shadow + text
            for offset in [(3, 3)]:
                draw.multiline_text((x + offset[0], quote_y + offset[1]), wrapped,
                                   font=quote_font, fill="#000000", anchor="ma", align="center")
            draw.multiline_text((x, quote_y), wrapped,
                               font=quote_font, fill="#FFFFFF", anchor="ma", align="center")
            
            # Author (only if provided)
            if author and author.strip():
                author_y = quote_y + quote_h + 40
                draw.text((x + 2, author_y + 2), f"- {author}",
                         font=author_font, fill="#000000", anchor="ma")
                draw.text((x, author_y), f"- {author}",
                         font=author_font, fill="#FFFFFF", anchor="ma")
            
            img.save(output_path, "PNG")
            return output_path
    
    def _get_font(self, size: int):
        """Get font with fallback"""
        font_files = ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf"]
        
        for name in font_files:
            try:
                return ImageFont.truetype(name, size)
            except:
                continue
        
        return ImageFont.load_default()
    
    def _wrap_text(self, text: str, font, max_width: int) -> str:
        """Wrap text to fit width"""
        words = text.split()
        lines, current = [], []
        
        for word in words:
            current.append(word)
            if font.getbbox(' '.join(current))[2] > max_width:
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


class ReelComposer:
    """Compose final reel video"""
    
    def __init__(self):
        self.settings = get_settings()
        self.image_gen = ImageGenerator()
        self.text_renderer = TextRenderer()
        self.output_dir = self.settings.output_dir / "reels"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_reel(self, quote: str, author: str, audio_path: Path, 
                    audio_duration: float, theme: dict = None, filename: str = "reel") -> VideoResult:
        """Create complete reel video"""
        from moviepy import ImageClip, AudioFileClip
        
        if theme is None:
            theme = random.choice(VISUAL_THEMES)
        
        try:
            # Generate background
            bg_path = self.image_gen.generate_background(theme, f"{filename}_bg")
            
            # Render quote on image
            quote_img = self.text_renderer.render_quote(bg_path, quote, author, theme)
            
            # Create video
            duration = audio_duration + 1.5
            video = ImageClip(str(quote_img)).with_duration(duration)
            video = video.resized((self.settings.video_width, self.settings.video_height))
            
            # Add audio
            audio = AudioFileClip(str(audio_path))
            video = video.with_audio(audio)
            
            # Export
            output_path = self.output_dir / f"{filename}.mp4"
            video.write_videofile(
                str(output_path),
                fps=self.settings.video_fps,
                codec="libx264",
                audio_codec="aac",
                logger=None
            )
            
            # Thumbnail
            thumb_path = self._create_thumbnail(quote_img, filename)
            
            video.close()
            audio.close()
            
            logger.info(f"Created reel: {output_path}")
            return VideoResult(file_path=output_path, thumbnail_path=thumb_path, duration=duration)
            
        except Exception as e:
            logger.error(f"Reel creation failed: {e}")
            raise
    
    def _create_thumbnail(self, image_path: Path, filename: str) -> Path:
        """Create thumbnail"""
        thumb_dir = self.settings.output_dir / "thumbnails"
        thumb_dir.mkdir(exist_ok=True)
        
        with Image.open(image_path) as img:
            img.thumbnail((540, 960))
            thumb_path = thumb_dir / f"{filename}_thumb.jpg"
            img.save(thumb_path, "JPEG", quality=85)
        
        return thumb_path


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    print("Testing Image Generator...")
    
    gen = ImageGenerator()
    theme = VISUAL_THEMES[0]
    
    try:
        path = gen.generate_background(theme, "test_bg")
        print(f"\n[SUCCESS] Image: {path}")
    except Exception as e:
        print(f"\n[FAILED]: {e}")
