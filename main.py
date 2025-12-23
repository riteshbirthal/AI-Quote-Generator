"""
AI Quote Reel Bot - Main Pipeline
Creates motivational reels with quotes and epic background music
Saves reel info for YouTube uploading
"""
import uuid
import json
import random
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
from loguru import logger

from config import get_settings, VISUAL_THEMES, QUOTE_CATEGORIES
from quote_engine import QuoteGenerator
from audio_engine import VoiceoverGenerator
from video_engine import ReelComposer
from seo_engine import SEOOptimizer


class ReelPipeline:
    """Main pipeline for creating quote reels"""
    
    def __init__(self):
        self.settings = get_settings()
        self._ensure_dirs()
        
        self.quote_gen = QuoteGenerator()
        self.voice_gen = VoiceoverGenerator()
        self.video_composer = ReelComposer()
        self.seo = SEOOptimizer()
        
        # Reel info storage
        self.reels_data_file = self.settings.output_dir / "reels_data.json"
        
        logger.info("Pipeline initialized")
    
    def _ensure_dirs(self):
        """Create required directories"""
        for d in [self.settings.output_dir, self.settings.temp_dir,
                  self.settings.output_dir / "reels",
                  self.settings.output_dir / "thumbnails",
                  self.settings.temp_dir / "audio",
                  self.settings.temp_dir / "images"]:
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


def main():
    """CLI entry point"""
    import argparse
    from dotenv import load_dotenv
    load_dotenv()
    
    parser = argparse.ArgumentParser(description='AI Quote Reel Bot')
    parser.add_argument('--mode', choices=['single', 'batch'], default='single')
    parser.add_argument('--count', type=int, default=5)
    parser.add_argument('--category', type=str, default=None)
    parser.add_argument('--language', choices=['english', 'hindi'], default='english')
    
    args = parser.parse_args()
    
    # Setup logging
    logger.add("logs/reel_{time}.log", rotation="1 day", retention="7 days")
    
    pipeline = ReelPipeline()
    
    if args.mode == 'single':
        result = pipeline.create_reel(category=args.category, language=args.language)
        print(f"\nReel created: {result['files']['video']}")
        # Handle Unicode for Windows console
        title = result['youtube']['title'].encode('ascii', 'ignore').decode()
        print(f"YouTube Title: {title}")
        print(f"Info saved to: output/reels_data.json")
        
    elif args.mode == 'batch':
        results = pipeline.create_batch(count=args.count, language=args.language)
        print(f"\nCreated {len(results)} reels")
        print(f"Info saved to: output/reels_data.json")


if __name__ == "__main__":
    main()
