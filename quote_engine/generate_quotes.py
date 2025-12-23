"""
Quote Generation using Google Gemini
Supports English and Hindi quotes with auto model switching
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
from config import get_settings, QUOTE_CATEGORIES


@dataclass
class Quote:
    """Generated quote data"""
    text: str
    category: str
    hook: str
    hashtags: list[str]
    language: str


class QuoteGenerator:
    """Generate motivational quotes using Gemini with auto model fallback"""
    
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
            logger.warning(f"Switching to model: {self.model}")
            return True
        return False
    
    def _reset_models(self):
        """Reset to first model"""
        self.current_model_index = 0
    
    def generate_quote(self, category: Optional[str] = None, language: str = "english") -> Quote:
        """Generate a single quote with auto model fallback"""
        if category is None:
            category = random.choice(QUOTE_CATEGORIES)
        
        if language.lower() == "hindi":
            prompt = self._hindi_prompt(category)
        else:
            prompt = self._english_prompt(category)

        # Try each model until one works
        last_error = None
        attempts = 0
        
        while attempts < len(self.MODELS):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.8,
                        max_output_tokens=2048,
                    )
                )
                
                text = response.text.strip()
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0]
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0]
                
                data = json.loads(text.strip())
                
                logger.info(f"Quote generated using {self.model}")
                
                return Quote(
                    text=data["quote"],
                    category=category,
                    hook=data["hook"],
                    hashtags=data["hashtags"],
                    language=language.lower()
                )
                
            except ClientError as e:
                last_error = e
                error_str = str(e)
                
                # Check if quota exhausted
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    logger.warning(f"Quota exhausted for {self.model}")
                    
                    if self._switch_model():
                        attempts += 1
                        time.sleep(1)  # Brief pause before retry
                        continue
                    else:
                        logger.error("All models exhausted")
                        break
                else:
                    # Other error, try next model
                    logger.error(f"Error with {self.model}: {e}")
                    if self._switch_model():
                        attempts += 1
                        continue
                    break
                    
            except Exception as e:
                last_error = e
                logger.error(f"Quote generation failed: {e}")
                if self._switch_model():
                    attempts += 1
                    continue
                break
        
        raise last_error or Exception("All models failed")
    
    def _english_prompt(self, category: str) -> str:
        return f"""Generate a powerful, original {category} quote for social media reels.

RULES:
1. Must be ORIGINAL - not an existing famous quote
2. 15-30 words for maximum impact
3. Deep, thought-provoking, and emotionally powerful
4. Should hit hard and make people think
5. No cliches, no generic motivation
6. Write like a philosopher or wise mentor

Respond in JSON only:
{{
    "quote": "your powerful original quote",
    "hook": "3-5 word attention grabber",
    "hashtags": ["motivation", "mindset", "success", "quotes", "viral"]
}}"""

    def _hindi_prompt(self, category: str) -> str:
        return f"""Generate a powerful, original {category} quote in HINDI (Devanagari script) for social media reels.

RULES:
1. Must be ORIGINAL - not an existing famous quote
2. 15-30 words in Hindi for maximum impact
3. Deep, thought-provoking, emotionally powerful
4. Should hit hard and make people think deeply
5. Use pure Hindi with Devanagari script
6. Write like a philosopher or wise mentor
7. No English words mixed in

Respond in JSON only:
{{
    "quote": "आपका शक्तिशाली मूल हिंदी उद्धरण यहां",
    "hook": "3-5 word Hindi attention grabber",
    "hashtags": ["hindiquotes", "motivation", "success", "mindset", "viral"]
}}"""

    def generate_batch(self, count: int = 5, language: str = "english") -> list[Quote]:
        """Generate multiple quotes"""
        quotes = []
        for i in range(count):
            category = QUOTE_CATEGORIES[i % len(QUOTE_CATEGORIES)]
            try:
                quote = self.generate_quote(category, language)
                quotes.append(quote)
                logger.info(f"Generated {i+1}/{count}: {quote.text[:40]}...")
            except Exception as e:
                logger.error(f"Failed quote {i+1}: {e}")
        return quotes


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    print("Testing Quote Generator with auto model switching...")
    generator = QuoteGenerator()
    
    try:
        quote = generator.generate_quote("motivation", "english")
        print(f"\n[SUCCESS] Using: {generator.model}")
        print(f"Quote: {quote.text}")
        print(f"Hook: {quote.hook}")
    except Exception as e:
        print(f"\n[FAILED]: {e}")
