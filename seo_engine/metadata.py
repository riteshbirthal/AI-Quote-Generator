"""
SEO Engine using Google Gemini
Generates optimized metadata for YouTube with auto model switching
"""
import json
import random
import time
from dataclasses import dataclass
from typing import Optional
from google import genai
from google.genai import types
from google.genai.errors import ClientError
from loguru import logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_settings


@dataclass
class VideoMetadata:
    """YouTube video metadata"""
    title: str
    description: str
    tags: list[str]
    hashtags: list[str]


class SEOOptimizer:
    """Generate SEO-optimized metadata using Gemini with auto model fallback"""
    
    # Models to try in order (when quota exhausted)
    MODELS = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-2.5-pro",
        "gemini-3-flash-preview",
        "gemini-3-pro-preview",
        "gemini-flash-latest",
        "gemini-exp-1206",
    ]
    
    ENGLISH_HASHTAGS = [
        "#shorts", "#motivation", "#quotes", "#inspirational",
        "#success", "#mindset", "#growth", "#viral", "#fyp"
    ]
    
    HINDI_HASHTAGS = [
        "#shorts", "#hindiquotes", "#motivation", "#anmolvachan",
        "#suvichar", "#success", "#viral", "#trending", "#fyp"
    ]
    
    def __init__(self):
        self.settings = get_settings()
        self.client = genai.Client(api_key=self.settings.google_api_key)
        self.current_model_index = 0
    
    @property
    def model(self):
        return self.MODELS[self.current_model_index]
    
    def _switch_model(self):
        """Switch to next available model"""
        if self.current_model_index < len(self.MODELS) - 1:
            self.current_model_index += 1
            logger.warning(f"SEO switching to model: {self.model}")
            return True
        return False
    
    def generate_metadata(self, quote: str, category: str, hook: str, language: str = "english") -> VideoMetadata:
        """Generate metadata with auto model fallback"""
        
        if language.lower() == "hindi":
            prompt = self._hindi_seo_prompt(quote, category, hook)
            fallback_hashtags = self.HINDI_HASHTAGS
        else:
            prompt = self._english_seo_prompt(quote, category, hook)
            fallback_hashtags = self.ENGLISH_HASHTAGS

        attempts = 0
        last_error = None
        
        while attempts < len(self.MODELS):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.7,
                        max_output_tokens=1024,
                    )
                )
                
                text = response.text.strip()
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0]
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0]
                
                data = json.loads(text.strip())
                
                hashtags = [w for w in data['description'].split() if w.startswith('#')]
                if not hashtags:
                    hashtags = random.sample(fallback_hashtags, 5)
                
                logger.info(f"SEO generated using {self.model}")
                
                return VideoMetadata(
                    title=data['title'][:100],
                    description=data['description'],
                    tags=data['tags'][:15],
                    hashtags=hashtags[:10]
                )
                
            except ClientError as e:
                last_error = e
                error_str = str(e)
                
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    logger.warning(f"SEO quota exhausted for {self.model}")
                    
                    if self._switch_model():
                        attempts += 1
                        time.sleep(1)
                        continue
                    else:
                        break
                else:
                    if self._switch_model():
                        attempts += 1
                        continue
                    break
                    
            except Exception as e:
                last_error = e
                logger.warning(f"SEO generation error: {e}, using fallback")
                return self._fallback_metadata(quote, category, hook, language)
        
        # All models failed, use fallback
        logger.warning("All models exhausted, using fallback metadata")
        return self._fallback_metadata(quote, category, hook, language)
    
    def _english_seo_prompt(self, quote: str, category: str, hook: str) -> str:
        return f"""Create viral YouTube Shorts metadata for this motivational quote video.

Quote: "{quote}"
Category: {category}
Hook: {hook}

Generate:
1. Viral title (max 60 chars) with emoji - must grab attention
2. Engaging description (150 words) with hashtags for discoverability
3. 15 SEO-optimized tags for YouTube search

JSON format only:
{{"title": "...", "description": "...", "tags": ["...", "..."]}}"""

    def _hindi_seo_prompt(self, quote: str, category: str, hook: str) -> str:
        return f"""Create viral YouTube Shorts metadata for this Hindi motivational quote video.

Quote: "{quote}"
Category: {category}
Hook: {hook}

Generate:
1. Viral Hindi title (max 60 chars) with emoji - attention grabbing
2. Engaging Hindi description (150 words) with hashtags
3. 15 SEO tags mixing Hindi and English for maximum reach

JSON format only:
{{"title": "...", "description": "...", "tags": ["...", "..."]}}"""

    def _fallback_metadata(self, quote: str, category: str, hook: str, language: str) -> VideoMetadata:
        """Fallback metadata when all models fail"""
        if language.lower() == "hindi":
            title = f"{hook} | {category} #shorts"[:100]
            hashtags = self.HINDI_HASHTAGS[:6]
            tags = ["shorts", "hindi quotes", "motivation", category, "suvichar", "anmol vachan"]
        else:
            title = f"{hook} | {category.title()} Quote #shorts"[:100]
            hashtags = self.ENGLISH_HASHTAGS[:6]
            tags = ["shorts", "motivation", "quotes", category, "inspirational", "viral"]
        
        description = f'''"{quote}"

Subscribe for daily motivation!

{' '.join(hashtags)}'''
        
        return VideoMetadata(
            title=title,
            description=description,
            tags=tags,
            hashtags=hashtags
        )


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    print("Testing SEO Optimizer with auto model switching...")
    seo = SEOOptimizer()
    
    metadata = seo.generate_metadata(
        quote="Success is built on daily discipline.",
        category="discipline",
        hook="Listen carefully",
        language="english"
    )
    
    print(f"Using model: {seo.model}")
    print(f"Title: {metadata.title}")
