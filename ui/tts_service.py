import streamlit as st
import requests
import hashlib
from pathlib import Path
from typing import Optional
import time

class KokoroTTSService:
    """Production-ready Kokoro TTS service with caching and error handling."""
    
    def __init__(self, api_key: str, api_url: str = "https://api.kokoro.ai/v1/synthesize"):
        self.api_key = api_key
        self.api_url = api_url
        self.cache_dir = Path("./tts_cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {api_key}"})
        
    def _get_cache_key(self, text: str, voice: str, speed: float) -> str:
        """Generate a unique cache key for the TTS request."""
        cache_input = f"{text}_{voice}_{speed}"
        return hashlib.md5(cache_input.encode()).hexdigest()
    
    def _get_cached_audio(self, text: str, voice: str, speed: float) -> Optional[bytes]:
        """Retrieve cached audio if available."""
        cache_key = self._get_cache_key(text, voice, speed)
        cache_file = self.cache_dir / f"{cache_key}.mp3"
        
        if cache_file.exists():
            return cache_file.read_bytes()
        return None
    
    def _cache_audio(self, text: str, voice: str, speed: float, audio_data: bytes) -> None:
        """Save audio to cache."""
        cache_key = self._get_cache_key(text, voice, speed)
        cache_file = self.cache_dir / f"{cache_key}.mp3"
        cache_file.write_bytes(audio_data)
    
    def synthesize(
        self,
        text: str,
        voice: str = "af_bella",
        speed: float = 1.0,
        use_cache: bool = True,
        max_retries: int = 3
    ) -> Optional[bytes]:
        """
        Synthesize text to speech using Kokoro TTS.
        
        Args:
            text: Text to synthesize
            voice: Voice name (e.g., "af_bella", "am_michael")
            speed: Speech speed (0.5-2.0)
            use_cache: Whether to use cached audio
            max_retries: Number of retry attempts
            
        Returns:
            Audio bytes or None if failed
        """
        
        # Validate input
        if not text or not text.strip():
            return None
        
        text = text.strip()[:1000]  # Limit to 1000 chars
        speed = max(0.5, min(2.0, speed))  # Clamp speed
        
        # Check cache first
        if use_cache:
            cached = self._get_cached_audio(text, voice, speed)
            if cached:
                return cached
        
        # Retry logic
        for attempt in range(max_retries):
            try:
                payload = {
                    "text": text,
                    "voice": voice,
                    "speed": speed,
                    "format": "mp3"
                }
                
                response = self.session.post(
                    self.api_url,
                    json=payload,
                    timeout=30
                )
                response.raise_for_status()
                
                audio_data = response.content
                
                # Cache successful result
                if use_cache and audio_data:
                    self._cache_audio(text, voice, speed, audio_data)
                
                return audio_data
                
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                return None
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:  # Rate limited
                    if attempt < max_retries - 1:
                        time.sleep(5 * (attempt + 1))
                        continue
                return None
            except Exception as e:
                st.warning(f"TTS Error: {str(e)}")
                return None
        
        return None
    
    def get_available_voices(self) -> list:
        """Get list of available voices."""
        return [
            ("af_bella", "Bella (Female)"),
            ("af_sarah", "Sarah (Female)"),
            ("am_michael", "Michael (Male)"),
            ("am_james", "James (Male)"),
            ("af_nicole", "Nicole (Female)"),
            ("am_adam", "Adam (Male)"),
        ]
    
    def clear_cache(self) -> None:
        """Clear all cached audio files."""
        for file in self.cache_dir.glob("*.mp3"):
            file.unlink()

@st.cache_resource
def get_tts_service() -> Optional[KokoroTTSService]:
    """Get or create TTS service instance."""
    try:
        api_key = st.secrets.get("KOKORO_API_KEY") or st.secrets.get("KOKORO_TTS_API_KEY")
        if not api_key:
            return None
        return KokoroTTSService(api_key)
    except Exception as e:
        st.warning(f"TTS Service unavailable: {e}")
        return None

