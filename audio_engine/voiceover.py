"""
Audio Engine with Edge-TTS and Strong Background Music
"""
import asyncio
import math
import struct
import wave
import random
from pathlib import Path
from dataclasses import dataclass
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_settings


@dataclass
class AudioResult:
    """Result of audio generation"""
    file_path: Path
    duration: float


class BackgroundMusicGenerator:
    """Generate powerful background music based on mood"""
    
    # Strong, impactful music settings
    MOODS = {
        "motivation": {
            "base_freq": 55,  # Deep bass
            "chord": [1, 1.25, 1.5, 2],  # Power chord ratios
            "tempo": 0.8,
            "intensity": 0.9,
            "style": "epic"
        },
        "success": {
            "base_freq": 65,
            "chord": [1, 1.25, 1.5, 2],
            "tempo": 0.7,
            "intensity": 0.85,
            "style": "triumphant"
        },
        "mindset": {
            "base_freq": 50,
            "chord": [1, 1.33, 1.5, 2],
            "tempo": 0.6,
            "intensity": 0.8,
            "style": "deep"
        },
        "discipline": {
            "base_freq": 60,
            "chord": [1, 1.25, 1.5, 2],
            "tempo": 0.9,
            "intensity": 0.9,
            "style": "powerful"
        },
        "perseverance": {
            "base_freq": 55,
            "chord": [1, 1.2, 1.5, 2],
            "tempo": 0.7,
            "intensity": 0.85,
            "style": "building"
        },
        "wisdom": {
            "base_freq": 45,
            "chord": [1, 1.33, 1.5, 2],
            "tempo": 0.5,
            "intensity": 0.75,
            "style": "profound"
        },
        "leadership": {
            "base_freq": 60,
            "chord": [1, 1.25, 1.5, 2],
            "tempo": 0.8,
            "intensity": 0.9,
            "style": "commanding"
        },
        "growth": {
            "base_freq": 55,
            "chord": [1, 1.25, 1.5, 2],
            "tempo": 0.7,
            "intensity": 0.8,
            "style": "ascending"
        },
    }
    
    def __init__(self):
        self.settings = get_settings()
        self.output_dir = self.settings.temp_dir / "audio"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate = 44100
    
    def generate_music(self, category: str, duration: float, filename: str) -> Path:
        """Generate powerful background music"""
        output_path = self.output_dir / f"{filename}_music.wav"
        
        mood = self.MOODS.get(category.lower(), self.MOODS["motivation"])
        samples = self._generate_epic_music(mood, duration)
        self._save_wav(samples, output_path)
        
        logger.info(f"Generated epic background music: {output_path}")
        return output_path
    
    def _generate_epic_music(self, mood: dict, duration: float) -> list:
        """Generate epic, cinematic background music"""
        num_samples = int(self.sample_rate * duration)
        samples = []
        
        base_freq = mood["base_freq"]
        chord_ratios = mood["chord"]
        tempo = mood["tempo"]
        intensity = mood["intensity"]
        
        # Pre-calculate some values for drums
        kick_interval = int(self.sample_rate / tempo)
        
        for i in range(num_samples):
            t = i / self.sample_rate
            sample = 0
            
            # 1. Deep bass drone with movement
            bass_mod = 1 + 0.1 * math.sin(2 * math.pi * 0.1 * t)
            bass = 0.4 * math.sin(2 * math.pi * base_freq * bass_mod * t)
            bass += 0.2 * math.sin(2 * math.pi * base_freq * 2 * t)  # Octave
            sample += bass * intensity
            
            # 2. Power chord pad (evolving)
            pad_volume = 0.3 + 0.15 * math.sin(2 * math.pi * 0.05 * t)
            for ratio in chord_ratios:
                freq = base_freq * 2 * ratio
                # Add slight detuning for richness
                detune = 1 + random.uniform(-0.002, 0.002)
                sample += pad_volume * 0.15 * math.sin(2 * math.pi * freq * detune * t)
            
            # 3. Sub-bass pulse (kick-like)
            kick_phase = (i % kick_interval) / kick_interval
            if kick_phase < 0.1:
                kick_env = 1 - (kick_phase / 0.1)
                kick_freq = base_freq * 0.5 * (1 + kick_phase * 2)
                sample += 0.5 * kick_env * math.sin(2 * math.pi * kick_freq * t) * intensity
            
            # 4. Rising tension (cinematic swell)
            progress = t / duration
            swell = 0.2 * progress * math.sin(2 * math.pi * base_freq * 3 * t)
            sample += swell * intensity
            
            # 5. Subtle high shimmer
            shimmer = 0.05 * math.sin(2 * math.pi * base_freq * 8 * t)
            shimmer *= 0.5 + 0.5 * math.sin(2 * math.pi * 0.3 * t)
            sample += shimmer
            
            # Master envelope - fade in/out
            if t < 1.5:
                sample *= (t / 1.5) ** 0.5
            elif t > duration - 2:
                sample *= ((duration - t) / 2) ** 0.5
            
            # Soft limiting
            sample = max(-0.95, min(0.95, sample))
            samples.append(sample)
        
        return samples
    
    def _save_wav(self, samples: list, path: Path):
        """Save samples as WAV file"""
        with wave.open(str(path), 'w') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            
            for sample in samples:
                sample = max(-1, min(1, sample))
                wav.writeframes(struct.pack('<h', int(sample * 32767)))


class AudioMixer:
    """Mix voiceover with background music"""
    
    def mix(self, voice_path: Path, music_path: Path, output_path: Path, music_volume: float = 0.25) -> Path:
        """Mix voice and music with music more prominent"""
        from pydub import AudioSegment
        
        voice = AudioSegment.from_mp3(str(voice_path))
        music = AudioSegment.from_wav(str(music_path))
        
        # Music at good volume (not too quiet)
        music = music - (20 * (1 - music_volume))
        
        # Match lengths
        if len(music) < len(voice):
            music = music * ((len(voice) // len(music)) + 1)
        music = music[:len(voice) + 2000]
        
        # Mix with music bed
        mixed = voice.overlay(music)
        mixed.export(str(output_path), format="mp3")
        
        logger.info(f"Mixed audio with epic music: {output_path}")
        return output_path


class VoiceoverGenerator:
    """Generate voiceovers with powerful background music (no author)"""
    
    VOICES = {
        "male_us": "en-US-GuyNeural",
        "female_us": "en-US-JennyNeural",
        "male_uk": "en-GB-RyanNeural",
        "female_uk": "en-GB-SoniaNeural",
        "male_hindi": "hi-IN-MadhurNeural",
        "female_hindi": "hi-IN-SwaraNeural",
    }
    
    def __init__(self, voice: str = "male_us"):
        self.settings = get_settings()
        self.voice = self.VOICES.get(voice, self.VOICES["male_us"])
        self.output_dir = self.settings.temp_dir / "audio"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.music_gen = BackgroundMusicGenerator()
        self.mixer = AudioMixer()
    
    def set_voice(self, language: str = "english"):
        """Set voice based on language"""
        if language.lower() == "hindi":
            self.voice = self.VOICES["male_hindi"]
        else:
            self.voice = self.VOICES["male_us"]
    
    def generate_voiceover(self, text: str, filename: str, category: str = "motivation") -> AudioResult:
        """Generate voiceover with epic background music"""
        voice_path = self.output_dir / f"{filename}_voice.mp3"
        final_path = self.output_dir / f"{filename}.mp3"
        
        try:
            asyncio.run(self._generate_edge_tts(text, voice_path))
            duration = self._get_duration(voice_path)
            
            try:
                music_path = self.music_gen.generate_music(category, duration + 3, filename)
                self.mixer.mix(voice_path, music_path, final_path, music_volume=0.3)
                logger.info(f"Generated voiceover with epic music: {final_path} ({duration:.1f}s)")
            except Exception as mix_err:
                logger.warning(f"Music mixing failed: {mix_err}")
                import shutil
                shutil.copy(voice_path, final_path)
                logger.info(f"Generated voiceover (no music): {final_path} ({duration:.1f}s)")
            
            return AudioResult(file_path=final_path, duration=duration)
            
        except Exception as e:
            logger.error(f"Voiceover generation failed: {e}")
            raise
    
    async def _generate_edge_tts(self, text: str, output_path: Path):
        """Generate using Edge-TTS"""
        import edge_tts
        communicate = edge_tts.Communicate(text, self.voice)
        await communicate.save(str(output_path))
    
    def generate_quote_audio(self, hook: str, quote: str, filename: str, category: str = "motivation") -> AudioResult:
        """Generate voiceover for quote (NO author name)"""
        script = f"{hook}... {quote}"
        return self.generate_voiceover(script, filename, category)
    
    def _get_duration(self, file_path: Path) -> float:
        """Get audio duration"""
        try:
            from mutagen.mp3 import MP3
            return MP3(str(file_path)).info.length
        except:
            return file_path.stat().st_size / 16000


if __name__ == "__main__":
    print("Testing Epic Background Music Generator...")
    
    gen = BackgroundMusicGenerator()
    path = gen.generate_music("motivation", 10, "test_epic")
    print(f"Generated: {path}")
