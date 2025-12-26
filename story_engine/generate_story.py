"""
Story Generation using Google Gemini
Supports moral stories, funny stories, and anime-style stories
"""
import json
import random
import time
from dataclasses import dataclass
from typing import Optional, List
from google import genai
from google.genai import types
from google.genai.errors import ClientError
from loguru import logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_settings, STORY_CATEGORIES


@dataclass
class StoryScene:
    """A single scene in a story"""
    scene_number: int
    narration: str
    image_prompt: str
    duration: float = 5.0


@dataclass
class Story:
    """Generated story data"""
    title: str
    story_type: str
    scenes: List[StoryScene]
    moral: str
    hashtags: List[str]
    language: str
    total_duration: float = 0.0


class StoryGenerator:
    """Generate stories using Gemini with auto model fallback"""
    
    MODELS = [
        "gemini-2.0-flash",
        "gemini-2.5-flash",
        "gemini-2.0-flash-lite",
        "gemini-2.5-pro",
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
    
    def generate_story(self, story_type: str = "moral", language: str = "english") -> Story:
        """Generate a story based on type"""
        if story_type not in STORY_CATEGORIES:
            story_type = "moral"
        
        prompt = self._get_prompt(story_type, language)
        
        last_error = None
        attempts = 0
        
        while attempts < len(self.MODELS):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.9,
                        max_output_tokens=4096,
                    )
                )
                
                text = response.text.strip()
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0]
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0]
                
                data = json.loads(text.strip())
                
                scenes = [
                    StoryScene(
                        scene_number=i + 1,
                        narration=scene["narration"],
                        image_prompt=scene["image_prompt"],
                        duration=scene.get("duration", 5.0)
                    )
                    for i, scene in enumerate(data["scenes"])
                ]
                
                total_duration = sum(s.duration for s in scenes)
                
                logger.info(f"Story generated using {self.model}")
                
                return Story(
                    title=data["title"],
                    story_type=story_type,
                    scenes=scenes,
                    moral=data.get("moral", data.get("punchline", "")),
                    hashtags=data["hashtags"],
                    language=language.lower(),
                    total_duration=total_duration
                )
                
            except ClientError as e:
                last_error = e
                error_str = str(e)
                
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    logger.warning(f"Quota exhausted for {self.model}")
                    if self._switch_model():
                        attempts += 1
                        time.sleep(1)
                        continue
                    else:
                        break
                else:
                    logger.error(f"Error with {self.model}: {e}")
                    if self._switch_model():
                        attempts += 1
                        continue
                    break
                    
            except Exception as e:
                last_error = e
                logger.error(f"Story generation failed: {e}")
                if self._switch_model():
                    attempts += 1
                    continue
                break
        
        raise last_error or Exception("All models failed")
    
    def _get_prompt(self, story_type: str, language: str) -> str:
        """Get prompt based on story type"""
        prompts = {
            "moral": self._moral_story_prompt,
            "funny": self._funny_story_prompt,
            "anime": self._anime_story_prompt,
            "horror": self._horror_story_prompt,
            "inspirational": self._inspirational_story_prompt,
        }
        
        prompt_func = prompts.get(story_type, self._moral_story_prompt)
        return prompt_func(language)
    
    def _moral_story_prompt(self, language: str) -> str:
        lang_instruction = "in Hindi (Devanagari script)" if language == "hindi" else "in English"
        return f"""Create a short moral story {lang_instruction} for a 60-second video reel.

REQUIREMENTS:
1. Story should have a clear moral lesson
2. 4-5 scenes, each 10-15 seconds of narration
3. Simple but impactful narrative
4. Universal theme (kindness, honesty, perseverance, etc.)
5. Suitable for all ages

Respond in JSON only:
{{
    "title": "Story title",
    "scenes": [
        {{
            "narration": "Scene 1 narration text (2-3 sentences)",
            "image_prompt": "Detailed image description for this scene (style: warm, storybook illustration)",
            "duration": 12
        }},
        {{
            "narration": "Scene 2 narration text",
            "image_prompt": "Image description for scene 2",
            "duration": 12
        }},
        {{
            "narration": "Scene 3 narration text",
            "image_prompt": "Image description for scene 3",
            "duration": 12
        }},
        {{
            "narration": "Scene 4 - conclusion with moral",
            "image_prompt": "Image description for final scene",
            "duration": 12
        }}
    ],
    "moral": "The moral of the story in one sentence",
    "hashtags": ["moralstory", "lifelesson", "motivation", "storytime", "wisdom"]
}}"""

    def _funny_story_prompt(self, language: str) -> str:
        lang_instruction = "in Hindi (Devanagari script)" if language == "hindi" else "in English"
        return f"""Create a short funny story {lang_instruction} for a 45-second video reel.

REQUIREMENTS:
1. Light-hearted and genuinely funny
2. 3-4 scenes with a punchline ending
3. Relatable everyday humor
4. Clean humor suitable for all ages
5. Unexpected twist or punchline

Respond in JSON only:
{{
    "title": "Funny story title",
    "scenes": [
        {{
            "narration": "Scene 1 - Setup (2-3 sentences)",
            "image_prompt": "Cartoon-style illustration description for scene 1",
            "duration": 10
        }},
        {{
            "narration": "Scene 2 - Building up",
            "image_prompt": "Cartoon-style illustration for scene 2",
            "duration": 10
        }},
        {{
            "narration": "Scene 3 - The funny twist/punchline",
            "image_prompt": "Humorous illustration for the punchline",
            "duration": 12
        }}
    ],
    "punchline": "The funny punchline or tagline",
    "hashtags": ["funny", "comedy", "humor", "lol", "viral"]
}}"""

    def _anime_story_prompt(self, language: str) -> str:
        lang_instruction = "in Hindi (Devanagari script)" if language == "hindi" else "in English"
        return f"""Create a short anime-style story {lang_instruction} for a 60-second video reel.

REQUIREMENTS:
1. Anime/manga storytelling style
2. 4-5 dramatic scenes with emotional moments
3. Can include: action, friendship, determination themes
4. Epic or emotional climax
5. Memorable dialogue

Respond in JSON only:
{{
    "title": "Anime story title (epic sounding)",
    "scenes": [
        {{
            "narration": "Scene 1 - Introduction with dramatic flair",
            "image_prompt": "Anime-style illustration: detailed description with dramatic lighting, character pose, background",
            "duration": 12
        }},
        {{
            "narration": "Scene 2 - Rising tension or challenge",
            "image_prompt": "Anime-style: action or emotional scene description",
            "duration": 12
        }},
        {{
            "narration": "Scene 3 - Climax moment",
            "image_prompt": "Anime-style: epic climax scene with dramatic effects",
            "duration": 12
        }},
        {{
            "narration": "Scene 4 - Resolution with powerful message",
            "image_prompt": "Anime-style: triumphant or emotional conclusion",
            "duration": 12
        }}
    ],
    "moral": "Powerful message or quote from the story",
    "hashtags": ["anime", "animeedit", "animestory", "manga", "otaku"]
}}"""

    def _horror_story_prompt(self, language: str) -> str:
        lang_instruction = "in Hindi (Devanagari script)" if language == "hindi" else "in English"
        return f"""Create a short horror/thriller story {lang_instruction} for a 60-second video reel.

REQUIREMENTS:
1. Suspenseful and creepy (not gory)
2. 4 scenes building tension
3. Unexpected twist ending
4. Psychological horror preferred
5. Suitable for teens and adults

Respond in JSON only:
{{
    "title": "Creepy story title",
    "scenes": [
        {{
            "narration": "Scene 1 - Eerie setup",
            "image_prompt": "Dark, atmospheric illustration: scene description with shadows and mystery",
            "duration": 12
        }},
        {{
            "narration": "Scene 2 - Something is wrong",
            "image_prompt": "Unsettling scene with growing tension",
            "duration": 12
        }},
        {{
            "narration": "Scene 3 - The revelation begins",
            "image_prompt": "Creepy scene building to climax",
            "duration": 12
        }},
        {{
            "narration": "Scene 4 - Twist ending",
            "image_prompt": "Shocking or eerie final scene",
            "duration": 12
        }}
    ],
    "moral": "The chilling final thought",
    "hashtags": ["horror", "scary", "creepy", "thriller", "storytime"]
}}"""

    def _inspirational_story_prompt(self, language: str) -> str:
        lang_instruction = "in Hindi (Devanagari script)" if language == "hindi" else "in English"
        return f"""Create a short inspirational true-story style narrative {lang_instruction} for a 60-second video reel.

REQUIREMENTS:
1. Based on themes of real success stories
2. 4 scenes showing struggle to triumph
3. Emotionally powerful
4. Motivational message
5. Relatable protagonist

Respond in JSON only:
{{
    "title": "Inspirational title",
    "scenes": [
        {{
            "narration": "Scene 1 - The humble beginning or challenge",
            "image_prompt": "Cinematic scene: person facing difficulty, dramatic lighting",
            "duration": 12
        }},
        {{
            "narration": "Scene 2 - The struggle and perseverance",
            "image_prompt": "Person working hard, determined expression",
            "duration": 12
        }},
        {{
            "narration": "Scene 3 - The breakthrough moment",
            "image_prompt": "Triumphant moment, hopeful lighting",
            "duration": 12
        }},
        {{
            "narration": "Scene 4 - Success and message",
            "image_prompt": "Successful outcome, inspiring scene",
            "duration": 12
        }}
    ],
    "moral": "The inspirational takeaway message",
    "hashtags": ["inspiration", "motivation", "success", "nevergiveup", "dreambig"]
}}"""


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    print("Testing Story Generator...")
    generator = StoryGenerator()
    
    try:
        story = generator.generate_story("moral", "english")
        print(f"\n[SUCCESS] Title: {story.title}")
        print(f"Type: {story.story_type}")
        print(f"Scenes: {len(story.scenes)}")
        print(f"Moral: {story.moral}")
        for scene in story.scenes:
            print(f"\nScene {scene.scene_number}: {scene.narration[:50]}...")
    except Exception as e:
        print(f"\n[FAILED]: {e}")
