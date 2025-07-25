#!/usr/bin/env python3
"""
Lazy Initialization Backend - Initialize Whisper AFTER WebSocket is running
This prevents the 1011 internal error by ensuring WebSocket works before heavy model loading
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

# Force CPU-only mode
os.environ['CUDA_VISIBLE_DEVICES'] = ''

try:
    import requests
    import soundfile as sf
except ImportError as e:
    print(f"❌ Missing basic dependencies: {e}")
    print("Please install: pip install requests soundfile numpy")
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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LazyInitSpeechProcessor:
    def __init__(self, 
                 model_size: str = "tiny.en",
                 language: str = "en",
                 server_url: str = "http://localhost:3000",
                 websocket_port: int = 8765):
        
        self.model_size = model_size
        self.language = language
        self.server_url = server_url
        self.websocket_port = websocket_port
        
        # Audio processing
        self.audio_queue = queue.Queue()
        self.sample_rate = 16000
        self.audio_buffer = np.array([], dtype=np.float32)
        
        # Whisper model (will be loaded lazily)
        self.model = None
        self.model_initialized = False
        self.model_loading = False
        
        # Processing state
        self.is_processing = False
        self.current_transcript = ""
        self.message_count = 0
        
        # WebSocket server
        self.websocket_server = None
        self.connected_clients = set()
        
        # Authentication
        self.auth_password = os.getenv('SPEECH_PASSWORD', 'holographic2024')
        
        logger.info("✅ LazyInit processor created (Whisper will load on first use)")
    
    def initialize_whisper_lazy(self):
        """Initialize Whisper model only when needed"""
        if self.model_initialized or self.model_loading:
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
    
    def transcribe_audio_safe(self, audio_data: np.ndarray) -> str:
        """Safely transcribe audio with lazy initialization"""
        try:
            # Initialize model if needed
            if not self.model_initialized:
                if not self.initialize_whisper_lazy():
                    return "Model initialization failed"
            
            if len(audio_data) == 0:
                return ""
            
            # Basic audio validation
            if np.all(audio_data == 0):
                return ""
            
            # Normalize audio
            max_val = np.abs(audio_data).max()
            if max_val > 1.0:
                audio_data = audio_data / max_val
            elif max_val < 0.001:
                return ""
            
            start_time = time.time()
            
            if WHISPER_BACKEND == "faster-whisper":
                segments, info = self.model.transcribe(
                    audio_data,
                    language=self.language,
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
                    language=self.language,
                    fp16=False
                )
                transcript = result.get("text", "").strip()
            
            processing_time = time.time() - start_time
            audio_duration = len(audio_data) / self.sample_rate
            
            logger.info(f"⚡ Transcribed {audio_duration:.1f}s in {processing_time:.2f}s: \"{transcript[:50]}...\"")
            
            return transcript
            
        except Exception as e:
            logger.error(f"❌ Transcription error: {e}")
            return f"Transcription error: {str(e)[:50]}"
    
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
            logger.error(f"❌ Audio decode error: {e}")
            return None
    
    async def handle_websocket_connection(self, websocket, path=None):
        """Handle WebSocket connections - this should work even before Whisper is loaded"""
        client_addr = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logger.info(f"🔌 New connection from {client_addr}")
        
        try:
            # Step 1: Authentication (should always work)
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
                'message': 'Connected to lazy-init speech processor',
                'model_loaded': self.model_initialized,
                'backend': WHISPER_BACKEND
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
                        self.current_transcript = "Recording started, initializing model..."
                        await self._update_and_broadcast()
                        
                        # Try to initialize Whisper in background
                        if not self.model_initialized:
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
                                    status = "Model loading..." if self.model_loading else f"Received {self.message_count} audio chunks"
                                    if self.model_initialized:
                                        status = f"Processing audio... ({self.message_count} chunks)"
                                    
                                    self.current_transcript = status
                                    await self._update_and_broadcast()
                        
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
            self.current_transcript = "Model loaded! Ready for transcription."
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
                
                if (buffer_duration >= 2.0 and time_since_last >= 1.0 and 
                    self.model_initialized and len(self.audio_buffer) > 0):
                    
                    logger.info(f"🔄 Processing {buffer_duration:.1f}s of audio")
                    
                    # Transcribe
                    transcript = self.transcribe_audio_safe(self.audio_buffer)
                    
                    if transcript and transcript != self.current_transcript:
                        self.current_transcript = transcript
                        
                        # Send updates asynchronously
                        asyncio.run_coroutine_threadsafe(
                            self._update_and_broadcast(),
                            asyncio.get_event_loop()
                        )
                    
                    # Trim buffer (keep 1 second overlap)
                    overlap_samples = int(1.0 * self.sample_rate)
                    if len(self.audio_buffer) > overlap_samples:
                        self.audio_buffer = self.audio_buffer[-overlap_samples:]
                    
                    last_process_time = current_time
                
                # Sleep
                time.sleep(0.1 if chunks else 0.2)
                
            except Exception as e:
                logger.error(f"❌ Error in processing loop: {e}")
                time.sleep(1.0)
    
    async def _process_final_audio(self):
        """Process final audio buffer"""
        if len(self.audio_buffer) > 0 and self.model_initialized:
            transcript = self.transcribe_audio_safe(self.audio_buffer)
            if transcript:
                self.current_transcript = f"Final: {transcript}"
                await self._update_and_broadcast()
        
        self.audio_buffer = np.array([], dtype=np.float32)
    
    async def _update_and_broadcast(self):
        """Update display and broadcast to clients"""
        try:
            # Update holographic display
            await self._update_display(self.current_transcript)
            
            # Broadcast to clients
            await self._broadcast_transcript(self.current_transcript)
            
        except Exception as e:
            logger.error(f"Error in update/broadcast: {e}")
    
    async def _update_display(self, transcript: str):
        """Update holographic display"""
        try:
            response = requests.post(
                f"{self.server_url}/api/speech/update-text",
                json={"text": transcript},
                timeout=3.0
            )
            
            if response.status_code == 200:
                logger.debug("✅ Display updated")
            else:
                logger.warning(f"Display update failed: {response.status_code}")
                
        except Exception as e:
            logger.debug(f"Display update error: {e}")
    
    async def _broadcast_transcript(self, transcript: str):
        """Broadcast to WebSocket clients"""
        if not self.connected_clients:
            return
        
        message = json.dumps({
            'type': 'streaming_transcript',
            'text': transcript,
            'model_loaded': self.model_initialized,
            'model_loading': self.model_loading,
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
            # Start WebSocket server FIRST (before heavy model loading)
            await self.start_websocket_server()
            
            logger.info("🎤 Lazy-Init Speech Processor ready!")
            logger.info(f"🔐 Password: {self.auth_password}")
            logger.info(f"🌐 WebSocket: ws://localhost:{self.websocket_port}")
            logger.info(f"🎯 Display server: {self.server_url}")
            logger.info("📝 Whisper model will load on first audio processing")
            
            # Keep running
            await asyncio.Future()
            
        except KeyboardInterrupt:
            logger.info("🛑 Shutting down...")
        except Exception as e:
            logger.error(f"❌ Error: {e}")
        finally:
            self.stop_processing()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Lazy-Init Speech Processor")
    parser.add_argument("--model", default="tiny.en", help="Whisper model size")
    parser.add_argument("--language", default="en", help="Language")
    parser.add_argument("--server-url", default="http://localhost:3000", help="Display server URL")
    parser.add_argument("--websocket-port", type=int, default=8765, help="WebSocket port")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    processor = LazyInitSpeechProcessor(
        model_size=args.model,
        language=args.language,
        server_url=args.server_url,
        websocket_port=args.websocket_port
    )
    
    asyncio.run(processor.run())

if __name__ == "__main__":
    main()