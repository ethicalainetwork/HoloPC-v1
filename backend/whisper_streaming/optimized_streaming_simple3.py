#!/usr/bin/env python3
"""
Speech Processor with Translation and .env Configuration Support
"""

import asyncio
import websockets
import json
import numpy as np
import threading
import queue
import time
import logging
import os
import sys
import io
from typing import Optional
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    
    # Look for .env file in current directory, then parent directories
    env_path = Path('.env')
    if not env_path.exists():
        env_path = Path('..') / '.env'
    
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ Loaded environment from {env_path}")
    else:
        print("⚠️ No .env file found, using system environment variables")
        
except ImportError:
    print("⚠️ python-dotenv not installed. Install with: pip install python-dotenv")
    print("📝 Using system environment variables only")

# Force CPU-only mode if specified
if os.getenv('CUDA_VISIBLE_DEVICES', '') == '':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

try:
    import requests
    import soundfile as sf
except ImportError as e:
    print(f"❌ Missing basic dependencies: {e}")
    print("Please install: pip install requests soundfile numpy python-dotenv")
    sys.exit(1)

# Check for whisper backends but don't import yet
WHISPER_BACKEND = None
try:
    import faster_whisper
    WHISPER_BACKEND = "faster-whisper"
    print("✅ faster-whisper available (will load lazily)")
except ImportError:
    try:
        import whisper
        WHISPER_BACKEND = "openai-whisper"
        print("✅ openai-whisper available (will load lazily)")
    except ImportError:
        print("❌ No Whisper backend found!")
        sys.exit(1)

# Configure logging
log_level = getattr(logging, os.getenv('LOG_LEVEL', 'INFO').upper())
logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LazyInitSpeechProcessor:
    def __init__(self):
        # Load configuration from environment variables
        self.model_size = os.getenv('WHISPER_MODEL', 'tiny.en')
        self.language = os.getenv('SPEECH_LANGUAGE', 'en')
        self.server_url = os.getenv('DISPLAY_SERVER_URL', 'http://localhost:3000')
        self.websocket_port = int(os.getenv('WEBSOCKET_PORT', '8765'))
        
        # Translation settings
        self.translate_to = os.getenv('TRANSLATE_TO')  # None if not set
        self.libretranslate_url = os.getenv('LIBRETRANSLATE_URL', 'http://localhost:5000')
        self.libretranslate_api_key = os.getenv('LIBRETRANSLATE_API_KEY')
        self.translation_enabled = self.translate_to is not None and self.translate_to.strip() != ''
        
        # Audio processing settings
        self.sample_rate = int(os.getenv('SAMPLE_RATE', '16000'))
        self.audio_buffer_seconds = float(os.getenv('AUDIO_BUFFER_SECONDS', '2.0'))
        self.processing_interval = float(os.getenv('PROCESSING_INTERVAL_SECONDS', '1.0'))
        
        # Audio processing
        self.audio_queue = queue.Queue()
        self.audio_buffer = np.array([], dtype=np.float32)
        
        # Whisper model (will be loaded lazily)
        self.model = None
        self.model_initialized = False
        self.model_loading = False
        self.skip_model_loading = os.getenv('SKIP_MODEL_LOADING', 'false').lower() == 'true'
        
        # Processing state
        self.is_processing = False
        self.current_transcript = ""
        self.current_translation = ""
        self.message_count = 0
        
        # WebSocket server
        self.websocket_server = None
        self.connected_clients = set()
        
        # Authentication
        self.auth_password = os.getenv('SPEECH_PASSWORD', 'default_password')
        if self.auth_password == 'default_password':
            logger.warning("⚠️ Using default password! Set SPEECH_PASSWORD in .env file")
        
        # Development settings
        self.dev_mode = os.getenv('DEV_MODE', 'false').lower() == 'true'
        self.debug_audio = os.getenv('DEBUG_AUDIO', 'false').lower() == 'true'
        
        # Test translation service on startup
        if self.translation_enabled and not self.skip_model_loading:
            asyncio.create_task(self.test_translation_service())
        
        self._log_configuration()
    
    def _log_configuration(self):
        """Log current configuration"""
        logger.info("🔧 Configuration loaded:")
        logger.info(f"   📝 Whisper Model: {self.model_size}")
        logger.info(f"   🗣️ Language: {self.language}")
        logger.info(f"   🌐 WebSocket Port: {self.websocket_port}")
        logger.info(f"   🎯 Display Server: {self.server_url}")
        
        if self.translation_enabled:
            logger.info(f"   🌐 Translation: {self.language} → {self.translate_to}")
            logger.info(f"   🔗 LibreTranslate: {self.libretranslate_url}")
            if self.libretranslate_api_key:
                logger.info("   🔑 API Key: ✅ Configured")
        else:
            logger.info("   🚫 Translation: Disabled")
        
        if self.dev_mode:
            logger.info("   🛠️ Development Mode: Enabled")
        
        if self.skip_model_loading:
            logger.info("   ⏩ Model Loading: Skipped (testing mode)")
    
    async def test_translation_service(self):
        """Test if LibreTranslate service is available"""
        try:
            test_payload = {
                'q': 'Hello',
                'source': 'en',
                'target': self.translate_to or 'es'
            }
            
            if self.libretranslate_api_key:
                test_payload['api_key'] = self.libretranslate_api_key
            
            response = requests.post(
                f"{self.libretranslate_url}/translate",
                data=test_payload,
                timeout=5.0
            )
            
            if response.status_code == 200:
                logger.info("✅ LibreTranslate service is available")
                return True
            else:
                logger.warning(f"⚠️ LibreTranslate test failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.warning(f"⚠️ LibreTranslate service not available: {e}")
            logger.info("📝 Make sure LibreTranslate is running: libretranslate")
            return False
    
    async def translate_text(self, text: str) -> str:
        """Translate text using LibreTranslate"""
        if not self.translation_enabled or not text.strip():
            return text
        
        try:
            payload = {
                'q': text,
                'source': self.language,
                'target': self.translate_to,
                'format': 'text'
            }
            
            if self.libretranslate_api_key:
                payload['api_key'] = self.libretranslate_api_key
            
            response = requests.post(
                f"{self.libretranslate_url}/translate",
                data=payload,
                timeout=10.0
            )
            
            if response.status_code == 200:
                result = response.json()
                translated_text = result.get('translatedText', text)
                
                if self.debug_audio or logger.isEnabledFor(logging.DEBUG):
                    logger.info(f"🌐 Translated: \"{text[:30]}...\" → \"{translated_text[:30]}...\"")
                
                return translated_text
            else:
                logger.error(f"❌ Translation failed: {response.status_code} - {response.text}")
                return text
                
        except Exception as e:
            logger.error(f"❌ Translation error: {e}")
            return text
    
    def initialize_whisper_lazy(self):
        """Initialize Whisper model only when needed"""
        if self.model_initialized or self.model_loading or self.skip_model_loading:
            return self.model_initialized
        
        logger.info("🔄 Starting lazy Whisper initialization...")
        self.model_loading = True
        
        try:
            if WHISPER_BACKEND == "faster-whisper":
                from faster_whisper import WhisperModel
                logger.info(f"📦 Loading faster-whisper model: {self.model_size}")
                
                self.model = WhisperModel(
                    self.model_size,
                    device="cpu",
                    compute_type="int8",
                    num_workers=1
                )
                logger.info("✅ faster-whisper model loaded successfully")
                
            elif WHISPER_BACKEND == "openai-whisper":
                import whisper
                logger.info(f"📦 Loading openai-whisper model: {self.model_size}")
                
                self.model = whisper.load_model(self.model_size, device="cpu")
                logger.info("✅ openai-whisper model loaded successfully")
            
            self.model_initialized = True
            self.model_loading = False
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Whisper: {e}")
            self.model_loading = False
            return False
    
    async def transcribe_audio_safe(self, audio_data: np.ndarray) -> tuple[str, str]:
        """Safely transcribe and optionally translate audio"""
        if self.skip_model_loading:
            # Return mock data for testing
            mock_transcript = f"Mock transcript {time.time():.1f}"
            mock_translation = f"Mock translation {time.time():.1f}" if self.translation_enabled else ""
            return mock_transcript, mock_translation
        
        try:
            # Initialize model if needed
            if not self.model_initialized:
                if not self.initialize_whisper_lazy():
                    return "Model initialization failed", ""
            
            if len(audio_data) == 0:
                return "", ""
            
            # Basic audio validation
            if np.all(audio_data == 0):
                return "", ""
            
            # Normalize audio
            max_val = np.abs(audio_data).max()
            if max_val > 1.0:
                audio_data = audio_data / max_val
            elif max_val < 0.001:
                return "", ""
            
            start_time = time.time()
            
            # Transcribe
            if WHISPER_BACKEND == "faster-whisper":
                segments, info = self.model.transcribe(
                    audio_data,
                    language=self.language if self.language != 'auto' else None,
                    beam_size=1,
                    best_of=1,
                    temperature=0.0,
                    word_timestamps=False
                )
                
                transcript_parts = []
                for segment in segments:
                    if hasattr(segment, 'text') and segment.text:
                        transcript_parts.append(segment.text.strip())
                
                transcript = " ".join(transcript_parts).strip()
                
            elif WHISPER_BACKEND == "openai-whisper":
                result = self.model.transcribe(
                    audio_data,
                    language=self.language if self.language != 'auto' else None,
                    fp16=False
                )
                transcript = result.get("text", "").strip()
            
            # Translate if enabled
            translation = ""
            if self.translation_enabled and transcript:
                translation = await self.translate_text(transcript)
            
            processing_time = time.time() - start_time
            audio_duration = len(audio_data) / self.sample_rate
            
            if self.debug_audio or logger.isEnabledFor(logging.DEBUG):
                logger.info(f"⚡ Processed {audio_duration:.1f}s in {processing_time:.2f}s")
                logger.info(f"📝 Original: \"{transcript[:50]}...\"")
                if translation and translation != transcript:
                    logger.info(f"🌐 Translated: \"{translation[:50]}...\"")
            
            return transcript, translation
            
        except Exception as e:
            logger.error(f"❌ Processing error: {e}")
            return f"Processing error: {str(e)[:50]}", ""
    
    def safe_audio_decode(self, audio_b64: str) -> Optional[np.ndarray]:
        """Safely decode base64 audio data"""
        try:
            import base64
            audio_bytes = base64.b64decode(audio_b64)
            
            if len(audio_bytes) == 0:
                return None
            
            # Try soundfile first
            try:
                audio_data, sr = sf.read(io.BytesIO(audio_bytes))
            except Exception:
                # Fallback to raw data
                try:
                    audio_data = np.frombuffer(audio_bytes, dtype=np.float32)
                    sr = self.sample_rate
                except Exception:
                    return None
            
            if len(audio_data) == 0:
                return None
            
            # Convert to mono
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Resample if needed
            if sr != self.sample_rate and sr > 0:
                target_length = int(len(audio_data) * self.sample_rate / sr)
                if target_length > 0:
                    audio_data = np.interp(
                        np.linspace(0, len(audio_data), target_length),
                        np.arange(len(audio_data)),
                        audio_data
                    )
            
            return audio_data.astype(np.float32)
            
        except Exception as e:
            if self.debug_audio:
                logger.error(f"❌ Audio decode error: {e}")
            return None
    
    async def handle_websocket_connection(self, websocket, path=None):
        """Handle WebSocket connections"""
        client_addr = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logger.info(f"🔌 New connection from {client_addr}")
        
        try:
            # Step 1: Authentication
            auth_message = await asyncio.wait_for(websocket.recv(), timeout=30.0)
            
            try:
                auth_data = json.loads(auth_message)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON: {e}")
                await websocket.send(json.dumps({'type': 'error', 'message': 'Invalid JSON'}))
                return
            
            if auth_data.get('type') != 'auth' or auth_data.get('password') != self.auth_password:
                await websocket.send(json.dumps({'type': 'auth_failed', 'message': 'Invalid credentials'}))
                return
            
            # Step 2: Authentication successful
            self.connected_clients.add(websocket)
            
            await websocket.send(json.dumps({
                'type': 'auth_success',
                'message': 'Connected to speech processor with translation',
                'model_loaded': self.model_initialized,
                'backend': WHISPER_BACKEND,
                'translation_enabled': self.translation_enabled,
                'source_language': self.language,
                'target_language': self.translate_to,
                'dev_mode': self.dev_mode
            }))
            
            logger.info(f"✅ Client {client_addr} authenticated")
            
            # Step 3: Start processing if not already running
            if not self.is_processing:
                self.start_processing()
            
            # Step 4: Handle messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    message_type = data.get('type', 'unknown')
                    
                    if message_type == 'start_recording':
                        logger.info(f"🎙️ Recording started for {client_addr}")
                        status = "Recording started"
                        if self.skip_model_loading:
                            status += " (testing mode)"
                        elif not self.model_initialized:
                            status += ", initializing model..."
                        
                        if self.translation_enabled:
                            status += f" (will translate {self.language}→{self.translate_to})"
                            
                        self.current_transcript = status
                        await self._update_and_broadcast()
                        
                        # Try to initialize Whisper in background
                        if not self.model_initialized and not self.skip_model_loading:
                            asyncio.create_task(self._initialize_model_async())
                        
                    elif message_type == 'stop_recording':
                        logger.info(f"🛑 Recording stopped for {client_addr}")
                        await self._process_final_audio()
                        
                    elif message_type == 'audio_chunk':
                        self.message_count += 1
                        
                        # Try to decode and queue audio
                        audio_b64 = data.get('audio', '')
                        if audio_b64:
                            audio_data = self.safe_audio_decode(audio_b64)
                            if audio_data is not None:
                                self.audio_queue.put(audio_data)
                                
                                # Update status periodically
                                if self.message_count % 10 == 0:
                                    if self.skip_model_loading:
                                        status = f"Testing mode: received {self.message_count} chunks"
                                    elif self.model_loading:
                                        status = "Model loading..."
                                    elif self.model_initialized:
                                        status = f"Processing audio... ({self.message_count} chunks)"
                                        if self.translation_enabled:
                                            status += f" [{self.language}→{self.translate_to}]"
                                    else:
                                        status = f"Received {self.message_count} audio chunks"
                                    
                                    self.current_transcript = status
                                    await self._update_and_broadcast()
                        
                        if self.debug_audio:
                            logger.debug(f"Audio chunk {self.message_count} processed")
                        
                    else:
                        logger.debug(f"Unknown message type: {message_type}")
                        
                except json.JSONDecodeError:
                    logger.error("JSON decode error in message handling")
                    continue
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    continue
        
        except asyncio.TimeoutError:
            logger.warning(f"Timeout for {client_addr}")
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Connection closed for {client_addr}")
        except Exception as e:
            logger.error(f"Connection error for {client_addr}: {e}")
        finally:
            self.connected_clients.discard(websocket)
            if len(self.connected_clients) == 0:
                self.stop_processing()
    
    async def _initialize_model_async(self):
        """Initialize model asynchronously"""
        def init_model():
            return self.initialize_whisper_lazy()
        
        # Run in thread to avoid blocking
        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(None, init_model)
        
        if success:
            status = "Model loaded! Ready for transcription"
            if self.translation_enabled:
                status += f" and translation ({self.language}→{self.translate_to})"
            self.current_transcript = status + "."
        else:
            self.current_transcript = "Model loading failed. Using fallback mode."
        
        await self._update_and_broadcast()
    
    def start_processing(self):
        """Start audio processing thread"""
        if self.is_processing:
            return
        
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        logger.info("🎤 Processing thread started")
    
    def stop_processing(self):
        """Stop processing"""
        self.is_processing = False
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join(timeout=2.0)
        logger.info("🛑 Processing stopped")
    
    def _processing_loop(self):
        """Main processing loop"""
        last_process_time = time.time()
        
        while self.is_processing:
            try:
                current_time = time.time()
                
                # Collect audio chunks
                chunks = []
                while not self.audio_queue.empty():
                    try:
                        chunk = self.audio_queue.get_nowait()
                        if chunk is not None:
                            chunks.append(chunk)
                    except queue.Empty:
                        break
                
                if chunks:
                    # Add to buffer
                    combined_audio = np.concatenate(chunks)
                    self.audio_buffer = np.concatenate([self.audio_buffer, combined_audio])
                
                # Process if we have enough audio and model is ready
                buffer_duration = len(self.audio_buffer) / self.sample_rate
                time_since_last = current_time - last_process_time
                
                should_process = (
                    buffer_duration >= self.audio_buffer_seconds and 
                    time_since_last >= self.processing_interval and 
                    (self.model_initialized or self.skip_model_loading) and 
                    len(self.audio_buffer) > 0
                )
                
                if should_process:
                    if self.debug_audio:
                        logger.info(f"🔄 Processing {buffer_duration:.1f}s of audio")
                    
                    # Transcribe and translate
                    asyncio.run_coroutine_threadsafe(
                        self._process_audio_async(),
                        asyncio.get_event_loop()
                    )
                    
                    # Trim buffer (keep 1 second overlap)
                    overlap_samples = int(1.0 * self.sample_rate)
                    if len(self.audio_buffer) > overlap_samples:
                        self.audio_buffer = self.audio_buffer[-overlap_samples:]
                    
                    last_process_time = current_time
                
                # Sleep
                sleep_time = 0.1 if chunks else 0.2
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"❌ Error in processing loop: {e}")
                time.sleep(1.0)
    
    async def _process_audio_async(self):
        """Process audio asynchronously (with translation)"""
        try:
            transcript, translation = await self.transcribe_audio_safe(self.audio_buffer.copy())
            
            if transcript and transcript != self.current_transcript:
                self.current_transcript = transcript
                self.current_translation = translation
                await self._update_and_broadcast()
                
        except Exception as e:
            logger.error(f"❌ Error in async audio processing: {e}")
    
    async def _process_final_audio(self):
        """Process final audio buffer"""
        if len(self.audio_buffer) > 0 and (self.model_initialized or self.skip_model_loading):
            transcript, translation = await self.transcribe_audio_safe(self.audio_buffer)
            if transcript:
                self.current_transcript = f"Final: {transcript}"
                self.current_translation = translation
                await self._update_and_broadcast()
        
        self.audio_buffer = np.array([], dtype=np.float32)
    
    async def _update_and_broadcast(self):
        """Update display and broadcast to clients"""
        try:
            # Determine what text to display
            display_text = self.current_translation if self.current_translation else self.current_transcript
            
            # Update holographic display
            await self._update_display(display_text)
            
            # Broadcast to clients
            await self._broadcast_transcript(self.current_transcript, self.current_translation)
            
        except Exception as e:
            logger.error(f"Error in update/broadcast: {e}")
    
    async def _update_display(self, text: str):
        """Update holographic display"""
        try:
            response = requests.post(
                f"{self.server_url}/api/speech/update-text",
                json={"text": text},
                timeout=3.0
            )
            
            if response.status_code == 200:
                if self.debug_audio:
                    logger.debug("✅ Display updated")
            else:
                logger.warning(f"Display update failed: {response.status_code}")
                
        except Exception as e:
            if self.debug_audio:
                logger.debug(f"Display update error: {e}")
    
    async def _broadcast_transcript(self, transcript: str, translation: str = ""):
        """Broadcast to WebSocket clients"""
        if not self.connected_clients:
            return
        
        message = json.dumps({
            'type': 'streaming_transcript',
            'original_text': transcript,
            'translated_text': translation,
            'display_text': translation if translation else transcript,
            'model_loaded': self.model_initialized,
            'model_loading': self.model_loading,
            'translation_enabled': self.translation_enabled,
            'source_language': self.language,
            'target_language': self.translate_to,
            'timestamp': time.time()
        })
        
        disconnected = set()
        for websocket in list(self.connected_clients):
            try:
                await websocket.send(message)
            except Exception:
                disconnected.add(websocket)
        
        self.connected_clients -= disconnected
    
    async def start_websocket_server(self):
        """Start WebSocket server"""
        logger.info(f"🌐 Starting WebSocket server on port {self.websocket_port}")
        
        self.websocket_server = await websockets.serve(
            self.handle_websocket_connection,
            "0.0.0.0",
            self.websocket_port,
            ping_interval=None,
            ping_timeout=None,
            close_timeout=10
        )
        
        logger.info(f"✅ WebSocket server running on ws://localhost:{self.websocket_port}")
    
    async def run(self):
        """Run the processor"""
        try:
            # Start WebSocket server FIRST
            await self.start_websocket_server()
            
            logger.info("🎤 Speech Processor with Translation ready!")
            logger.info(f"🔐 Password: {self.auth_password}")
            logger.info(f"🌐 WebSocket: ws://localhost:{self.websocket_port}")
            logger.info(f"🎯 Display server: {self.server_url}")
            
            if not self.skip_model_loading:
                logger.info(f"📝 Whisper model: {self.model_size} (language: {self.language})")
                if self.translation_enabled:
                    logger.info(f"🌐 Translation: {self.language} → {self.translate_to}")
                    logger.info(f"🔗 LibreTranslate: {self.libretranslate_url}")
                else:
                    logger.info("🚫 Translation disabled")
            else:
                logger.info("⏩ Running in testing mode (no model loading)")
            
            # Keep running
            await asyncio.Future()
            
        except KeyboardInterrupt:
            logger.info("🛑 Shutting down...")
        except Exception as e:
            logger.error(f"❌ Error: {e}")
        finally:
            self.stop_processing()

def main():
    processor = LazyInitSpeechProcessor()
    asyncio.run(processor.run())

if __name__ == "__main__":
    main()