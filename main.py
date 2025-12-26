"""
AI Quote Reel Bot - Main Pipeline
Creates motivational reels with quotes, stories, and anime videos
Saves reel info for YouTube uploading
"""
import uuid
import json
import random
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
from loguru import logger

from config import get_settings, VISUAL_THEMES, QUOTE_CATEGORIES, STORY_CATEGORIES
from quote_engine import QuoteGenerator
from audio_engine import VoiceoverGenerator
from video_engine import ReelComposer
from seo_engine import SEOOptimizer
from story_engine import StoryGenerator, StoryReelComposer


class ReelPipeline:
    """Main pipeline for creating quote reels"""
    
    def __init__(self):
        self.settings = get_settings()
        self._ensure_dirs()
        
        self.quote_gen = QuoteGenerator()
        self.voice_gen = VoiceoverGenerator()
        self.video_composer = ReelComposer()
        self.seo = SEOOptimizer()
        
        # Story generators
        self.story_gen = StoryGenerator()
        self.story_composer = StoryReelComposer()
        
        # Reel info storage
        self.reels_data_file = self.settings.output_dir / "reels_data.json"
        
        logger.info("Pipeline initialized")
    
    def _ensure_dirs(self):
        """Create required directories"""
        dirs = [
            self.settings.output_dir,
            self.settings.temp_dir,
            # Quote directories
            self.settings.output_dir / "quotes",
            self.settings.output_dir / "quotes" / "reels",
            self.settings.output_dir / "quotes" / "thumbnails",
            # Story directories
            self.settings.output_dir / "stories",
            self.settings.output_dir / "stories" / "thumbnails",
            # Anime directories
            self.settings.output_dir / "anime",
            self.settings.output_dir / "anime" / "thumbnails",
            # Temp directories
            self.settings.temp_dir / "audio",
            self.settings.temp_dir / "images",
            self.settings.temp_dir / "story_images",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
    
    def create_reel(self, category: str = None, language: str = "english") -> dict:
        """Create a single reel"""
        reel_id = str(uuid.uuid4())[:8]
        logger.info(f"Creating reel {reel_id} ({language})...")
        
        try:
            # 1. Generate quote
            logger.info("Step 1/4: Generating quote...")
            quote = self.quote_gen.generate_quote(category, language)
            
            # 2. Set voice for language and generate voiceover (NO author)
            logger.info("Step 2/4: Generating voiceover with epic music...")
            self.voice_gen.set_voice(language)
            audio = self.voice_gen.generate_quote_audio(
                hook=quote.hook,
                quote=quote.text,
                filename=f"reel_{reel_id}",
                category=quote.category
            )
            
            # 3. Create video
            logger.info("Step 3/4: Creating video...")
            theme = random.choice(VISUAL_THEMES)
            video = self.video_composer.create_reel(
                quote=quote.text,
                author="",  # No author on video
                audio_path=audio.file_path,
                audio_duration=audio.duration,
                theme=theme,
                filename=f"reel_{reel_id}"
            )
            
            # 4. Generate YouTube metadata
            logger.info("Step 4/4: Generating YouTube metadata...")
            metadata = self.seo.generate_metadata(
                quote=quote.text,
                category=quote.category,
                hook=quote.hook,
                language=language
            )
            
            # Compile reel info
            reel_info = {
                'reel_id': reel_id,
                'created_at': datetime.now().isoformat(),
                'language': language,
                'quote': {
                    'text': quote.text,
                    'category': quote.category,
                    'hook': quote.hook,
                    'hashtags': quote.hashtags
                },
                'files': {
                    'video': str(video.file_path),
                    'thumbnail': str(video.thumbnail_path),
                    'audio': str(audio.file_path)
                },
                'video_info': {
                    'duration': video.duration,
                    'resolution': '1080x1920'
                },
                'youtube': {
                    'title': metadata.title,
                    'description': metadata.description,
                    'tags': metadata.tags,
                    'hashtags': metadata.hashtags,
                    'keywords': ', '.join(metadata.tags[:10])
                }
            }
            
            # Save reel info
            self._save_reel_info(reel_info)
            
            logger.info(f"Reel {reel_id} created and saved!")
            return reel_info
            
        except Exception as e:
            logger.error(f"Failed to create reel: {e}")
            raise
    
    def _save_reel_info(self, reel_info: dict):
        """Save reel info to JSON file"""
        # Load existing data
        if self.reels_data_file.exists():
            with open(self.reels_data_file, 'r', encoding='utf-8') as f:
                all_reels = json.load(f)
        else:
            all_reels = {'reels': []}
        
        # Add new reel
        all_reels['reels'].append(reel_info)
        all_reels['last_updated'] = datetime.now().isoformat()
        all_reels['total_reels'] = len(all_reels['reels'])
        
        # Save
        with open(self.reels_data_file, 'w', encoding='utf-8') as f:
            json.dump(all_reels, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved reel info to {self.reels_data_file}")
    
    def create_batch(self, count: int = 5, language: str = "english") -> list[dict]:
        """Create multiple reels"""
        results = []
        
        for i in range(count):
            category = QUOTE_CATEGORIES[i % len(QUOTE_CATEGORIES)]
            logger.info(f"\n{'='*40}")
            logger.info(f"Reel {i+1}/{count} - {category} ({language})")
            logger.info(f"{'='*40}")
            
            try:
                result = self.create_reel(category=category, language=language)
                results.append(result)
            except Exception as e:
                logger.error(f"Reel {i+1} failed: {e}")
        
        logger.info(f"\nBatch complete: {len(results)}/{count} reels")
        return results
    
    def create_story(self, story_type: str = "moral", language: str = "english") -> dict:
        """Create a cinematic story video"""
        story_id = str(uuid.uuid4())[:8]
        logger.info(f"Creating {story_type} story {story_id} ({language})...")
        
        try:
            # 1. Generate story
            logger.info("Step 1/4: Generating story...")
            story = self.story_gen.generate_story(story_type, language)
            logger.info(f"Story: {story.title} ({len(story.scenes)} scenes)")
            
            # 2. Generate continuous voiceover for entire story
            logger.info("Step 2/4: Generating continuous narration...")
            self.voice_gen.set_voice(language)
            
            # Combine all narrations with pauses
            full_narration = " ... ".join([scene.narration for scene in story.scenes])
            
            # Add story intro
            if story_type == "moral":
                intro = "Listen to this story..."
            elif story_type == "funny":
                intro = "Here's a funny story for you..."
            elif story_type == "anime":
                intro = ""
            elif story_type == "horror":
                intro = "Are you ready for a chilling tale?..."
            else:
                intro = ""
            
            if intro:
                full_narration = f"{intro} {full_narration}"
            
            # Add moral at the end
            if story.moral:
                full_narration = f"{full_narration} ... {story.moral}"
            
            audio = self.voice_gen.generate_voiceover(
                full_narration,
                f"story_{story_id}",
                category=story_type
            )
            
            # 3. Create cinematic video with all scenes
            logger.info("Step 3/4: Creating cinematic video...")
            video = self.story_composer.create_story_reel(
                story=story,
                audio_path=audio.file_path,
                audio_duration=audio.duration,
                filename=f"story_{story_id}"
            )
            
            # 4. Generate metadata
            logger.info("Step 4/4: Generating metadata...")
            metadata = self.seo.generate_metadata(
                quote=story.moral,
                category=story_type,
                hook=story.title,
                language=language
            )
            
            # Compile story info
            story_info = {
                'story_id': story_id,
                'created_at': datetime.now().isoformat(),
                'content_type': 'story',
                'language': language,
                'story': {
                    'title': story.title,
                    'type': story_type,
                    'moral': story.moral,
                    'scene_count': len(story.scenes),
                    'hashtags': story.hashtags
                },
                'files': {
                    'video': str(video.file_path),
                    'thumbnail': str(video.thumbnail_path),
                },
                'video_info': {
                    'duration': video.duration,
                    'scene_count': video.scene_count,
                    'resolution': '1080x1920'
                },
                'youtube': {
                    'title': f"{story.title} | {story_type.capitalize()} Story",
                    'description': metadata.description,
                    'tags': metadata.tags + ['story', story_type, 'storytime'],
                    'hashtags': story.hashtags,
                }
            }
            
            # Save story info
            self._save_reel_info(story_info)
            
            logger.info(f"Story {story_id} created and saved!")
            return story_info
            
        except Exception as e:
            logger.error(f"Failed to create story: {e}")
            raise
    
    def create_story_batch(self, count: int = 3, language: str = "english") -> list[dict]:
        """Create multiple stories of different types"""
        results = []
        
        for i in range(count):
            story_type = STORY_CATEGORIES[i % len(STORY_CATEGORIES)]
            logger.info(f"\n{'='*40}")
            logger.info(f"Story {i+1}/{count} - {story_type} ({language})")
            logger.info(f"{'='*40}")
            
            try:
                result = self.create_story(story_type=story_type, language=language)
                results.append(result)
            except Exception as e:
                logger.error(f"Story {i+1} failed: {e}")
        
        logger.info(f"\nBatch complete: {len(results)}/{count} stories")
        return results


def main():
    """CLI entry point"""
    import argparse
    from dotenv import load_dotenv
    load_dotenv()
    
    parser = argparse.ArgumentParser(description='AI Quote Reel Bot - Create quotes, stories, and anime videos')
    parser.add_argument('--type', choices=['quote', 'story'], default='quote',
                       help='Content type: quote or story')
    parser.add_argument('--mode', choices=['single', 'batch'], default='single',
                       help='Create single or batch content')
    parser.add_argument('--count', type=int, default=5,
                       help='Number of items to create in batch mode')
    parser.add_argument('--category', type=str, default=None,
                       help='Quote category (motivation, success, etc.)')
    parser.add_argument('--story-type', type=str, default='moral',
                       choices=['moral', 'funny', 'anime', 'horror', 'inspirational'],
                       help='Story type for story mode')
    parser.add_argument('--language', choices=['english', 'hindi'], default='english',
                       help='Language for content')
    
    args = parser.parse_args()
    
    # Setup logging
    logger.add("logs/reel_{time}.log", rotation="1 day", retention="7 days")
    
    pipeline = ReelPipeline()
    
    if args.type == 'quote':
        if args.mode == 'single':
            result = pipeline.create_reel(category=args.category, language=args.language)
            print(f"\nReel created: {result['files']['video']}")
            title = result['youtube']['title'].encode('ascii', 'ignore').decode()
            print(f"YouTube Title: {title}")
            print(f"Info saved to: output/reels_data.json")
            
        elif args.mode == 'batch':
            results = pipeline.create_batch(count=args.count, language=args.language)
            print(f"\nCreated {len(results)} reels")
            print(f"Info saved to: output/reels_data.json")
    
    elif args.type == 'story':
        if args.mode == 'single':
            result = pipeline.create_story(story_type=args.story_type, language=args.language)
            print(f"\nStory created: {result['files']['video']}")
            title = result['youtube']['title'].encode('ascii', 'ignore').decode()
            print(f"YouTube Title: {title}")
            print(f"Story type: {args.story_type}")
            print(f"Scenes: {result['video_info']['scene_count']}")
            print(f"Info saved to: output/reels_data.json")
            
        elif args.mode == 'batch':
            results = pipeline.create_story_batch(count=args.count, language=args.language)
            print(f"\nCreated {len(results)} stories")
            print(f"Info saved to: output/reels_data.json")


if __name__ == "__main__":
    main()
