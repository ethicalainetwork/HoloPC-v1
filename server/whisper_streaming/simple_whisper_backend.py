#!/usr/bin/env python3
"""
Simplified Whisper Backend - Direct integration without whisper-online
Uses faster-whisper or openai-whisper directly for better stability
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
from typing import Optional, Dict, Any

try:
    import requests
    import soundfile as sf
except ImportError as e:
    print(f"❌ Missing basic dependencies: {e}")
    print("Please install: pip install requests soundfile numpy")
    sys.exit(1)

# Try to import whisper backends
WHISPER_BACKEND = None
whisper_model = None

# Try faster-whisper first (recommended)
try:
    from faster_whisper import WhisperModel
    WHISPER_BACKEND = "faster-whisper"
    print("✅ Using faster-whisper backend")
except ImportError:
    try:
        import whisper
        WHISPER_BACKEND = "openai-whisper"
        print("✅ Using openai-whisper backend")
    except ImportError:
        print("❌ No Whisper backend found!")
        print("Install one of:")
        print("  pip install faster-whisper")
        print("  pip install openai-whisper")
        sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleSpeechProcessor:
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
        self.buffer_duration = 5.0  # Process every 5 seconds
        self.last_process_time = time.time()
        
        # Speech recognition
        self.model = None
        self.is_processing = False
        self.current_transcript = ""
        
        # WebSocket server
        self.websocket_server = None
        self.connected_clients = set()
        
        # Authentication
        self.auth_password = os.getenv('SPEECH_PASSWORD', 'holographic2024')
        
        # Initialize Whisper
        self.initialize_whisper()
    
    def initialize_whisper(self):
        """Initialize Whisper model"""
        try:
            logger.info(f"🎤 Initializing Whisper model: {self.model_size}")
            
            # Force CPU-only mode to avoid CUDA issues
            import os
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            
            if WHISPER_BACKEND == "faster-whisper":
                # faster-whisper initialization - force CPU
                self.model = WhisperModel(
                    self.model_size,
                    device="cpu",  # Force CPU to avoid CUDA issues
                    compute_type="int8"  # Reduce memory usage
                )
                logger.info("✅ faster-whisper model loaded (CPU mode)")
                
            elif WHISPER_BACKEND == "openai-whisper":
                # openai-whisper initialization
                import whisper
                self.model = whisper.load_model(self.model_size, device="cpu")
                logger.info("✅ openai-whisper model loaded (CPU mode)")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Whisper: {e}")
            raise
    
    def transcribe_audio(self, audio_data: np.ndarray) -> str:
        """Transcribe audio using the loaded model"""
        try:
            if WHISPER_BACKEND == "faster-whisper":
                # faster-whisper transcription
                segments, info = self.model.transcribe(
                    audio_data,
                    language=self.language,
                    beam_size=1,  # Faster decoding
                    best_of=1,
                    temperature=0.0
                )
                
                # Combine all segments
                transcript = " ".join([segment.text.strip() for segment in segments])
                
            elif WHISPER_BACKEND == "openai-whisper":
                # openai-whisper transcription
                result = self.model.transcribe(
                    audio_data,
                    language=self.language,
                    fp16=False  # Use fp32 for CPU
                )
                transcript = result["text"].strip()
            
            return transcript
            
        except Exception as e:
            logger.error(f"❌ Transcription error: {e}")
            return ""
    
    async def handle_websocket_connection(self, websocket, path):
        """Handle WebSocket connections from web clients"""
        logger.info(f"🔌 New WebSocket connection")
        
        try:
            # Authentication
            auth_message = await websocket.recv()
            auth_data = json.loads(auth_message)
            
            if auth_data.get('type') != 'auth' or auth_data.get('password') != self.auth_password:
                await websocket.send(json.dumps({
                    'type': 'auth_failed',
                    'message': 'Invalid password'
                }))
                await websocket.close()
                return
            
            # Authentication successful
            self.connected_clients.add(websocket)
            await websocket.send(json.dumps({
                'type': 'auth_success',
                'message': 'Connected to speech recognition'
            }))
            
            logger.info("✅ Client authenticated")
            
            # Start speech processing if not already running
            if not self.is_processing:
                self.start_speech_processing()
            
            # Handle incoming audio data
            async for message in websocket:
                try:
                    data = json.loads(message)
                    
                    if data['type'] == 'audio_chunk':
                        # Receive audio data (base64 encoded)
                        import base64
                        audio_bytes = base64.b64decode(data['audio'])
                        
                        # Convert WebM/Opus to raw audio using soundfile
                        try:
                            audio_data, sr = sf.read(io.BytesIO(audio_bytes))
                            
                            # Convert to mono if stereo
                            if len(audio_data.shape) > 1:
                                audio_data = np.mean(audio_data, axis=1)
                            
                            # Resample to 16kHz if needed
                            if sr != self.sample_rate:
                                # Simple resampling (not perfect but works)
                                audio_data = np.interp(
                                    np.linspace(0, len(audio_data), int(len(audio_data) * self.sample_rate / sr)),
                                    np.arange(len(audio_data)),
                                    audio_data
                                )
                            
                            # Add to processing queue
                            self.audio_queue.put(audio_data.astype(np.float32))
                            
                        except Exception as audio_error:
                            logger.warning(f"⚠️ Audio processing error: {audio_error}")
                        
                    elif data['type'] == 'start_recording':
                        logger.info("🎙️ Recording started")
                        
                    elif data['type'] == 'stop_recording':
                        logger.info("🛑 Recording stopped")
                        await self.process_final_audio()
                        
                except json.JSONDecodeError:
                    logger.error("❌ Invalid JSON received")
                except Exception as e:
                    logger.error(f"❌ Error processing message: {e}")
        
        except websockets.exceptions.ConnectionClosed:
            logger.info("🔌 WebSocket connection closed")
        except Exception as e:
            logger.error(f"❌ WebSocket error: {e}")
        finally:
            self.connected_clients.discard(websocket)
            if len(self.connected_clients) == 0:
                self.stop_speech_processing()
    
    def start_speech_processing(self):
        """Start the speech processing thread"""
        if self.is_processing:
            return
        
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._speech_processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        logger.info("🎤 Speech processing started")
    
    def stop_speech_processing(self):
        """Stop speech processing"""
        self.is_processing = False
        self.audio_buffer = np.array([], dtype=np.float32)
        
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join(timeout=2.0)
        
        logger.info("🛑 Speech processing stopped")
    
    def _speech_processing_loop(self):
        """Main speech processing loop"""
        while self.is_processing:
            try:
                current_time = time.time()
                
                # Collect audio chunks from queue
                while not self.audio_queue.empty():
                    try:
                        chunk = self.audio_queue.get_nowait()
                        self.audio_buffer = np.concatenate([self.audio_buffer, chunk])
                    except queue.Empty:
                        break
                
                # Process if we have enough audio and enough time has passed
                buffer_duration = len(self.audio_buffer) / self.sample_rate
                time_since_last_process = current_time - self.last_process_time
                
                if buffer_duration >= self.buffer_duration and time_since_last_process >= 2.0:
                    # Transcribe the audio
                    transcript = self.transcribe_audio(self.audio_buffer)
                    
                    if transcript and transcript != self.current_transcript:
                        self.current_transcript = transcript
                        logger.info(f"💬 Transcribed: \"{transcript}\"")
                        
                        # Send to holographic display server
                        asyncio.run_coroutine_threadsafe(
                            self._update_holographic_display(transcript),
                            asyncio.get_event_loop()
                        )
                        
                        # Send to WebSocket clients
                        asyncio.run_coroutine_threadsafe(
                            self._broadcast_transcript(transcript),
                            asyncio.get_event_loop()
                        )
                    
                    # Keep some audio for context (overlap)
                    overlap_samples = int(1.0 * self.sample_rate)  # 1 second overlap
                    if len(self.audio_buffer) > overlap_samples:
                        self.audio_buffer = self.audio_buffer[-overlap_samples:]
                    
                    self.last_process_time = current_time
                
                # Sleep briefly to prevent busy waiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"❌ Error in speech processing loop: {e}")
                time.sleep(1.0)
    
    async def process_final_audio(self):
        """Process any remaining audio when recording stops"""
        if len(self.audio_buffer) > 0:
            transcript = self.transcribe_audio(self.audio_buffer)
            if transcript:
                self.current_transcript = transcript
                await self._update_holographic_display(transcript)
                await self._broadcast_transcript(transcript)
            
            self.audio_buffer = np.array([], dtype=np.float32)
    
    async def _update_holographic_display(self, transcript: str):
        """Send transcript to holographic display server"""
        try:
            response = requests.post(
                f"{self.server_url}/api/speech/update-text",
                json={"text": transcript},
                timeout=3.0
            )
            
            if response.status_code == 200:
                logger.info(f"✅ Updated holographic display")
            else:
                logger.warning(f"⚠️ Failed to update display: {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ Error updating holographic display: {e}")
    
    async def _broadcast_transcript(self, transcript: str):
        """Broadcast transcript to all connected WebSocket clients"""
        if not self.connected_clients:
            return
        
        message = json.dumps({
            'type': 'transcript',
            'text': transcript,
            'timestamp': time.time()
        })
        
        # Send to all clients
        disconnected_clients = set()
        for websocket in self.connected_clients:
            try:
                await websocket.send(message)
            except Exception:
                disconnected_clients.add(websocket)
        
        # Remove disconnected clients
        self.connected_clients -= disconnected_clients
    
    async def start_websocket_server(self):
        """Start the WebSocket server"""
        logger.info(f"🌐 Starting WebSocket server on port {self.websocket_port}")
        
        self.websocket_server = await websockets.serve(
            self.handle_websocket_connection,
            "0.0.0.0",
            self.websocket_port
        )
        
        logger.info(f"✅ WebSocket server running on ws://localhost:{self.websocket_port}")
    
    async def run(self):
        """Run the speech processor"""
        try:
            await self.start_websocket_server()
            
            logger.info("🎤 Simple Speech Processor ready!")
            logger.info(f"🔐 Password: {self.auth_password}")
            logger.info(f"🌐 WebSocket: ws://localhost:{self.websocket_port}")
            logger.info(f"🎯 Display server: {self.server_url}")
            
            # Keep running
            await asyncio.Future()  # Run forever
            
        except KeyboardInterrupt:
            logger.info("🛑 Shutting down...")
        except Exception as e:
            logger.error(f"❌ Error: {e}")
        finally:
            self.stop_speech_processing()

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple Whisper Speech Recognition")
    parser.add_argument("--model", default="tiny.en", help="Whisper model size")
    parser.add_argument("--language", default="en", help="Language")
    parser.add_argument("--server-url", default="http://localhost:3000", help="Display server URL")
    parser.add_argument("--websocket-port", type=int, default=8765, help="WebSocket port")
    
    args = parser.parse_args()
    
    processor = SimpleSpeechProcessor(
        model_size=args.model,
        language=args.language,
        server_url=args.server_url,
        websocket_port=args.websocket_port
    )
    
    asyncio.run(processor.run())

if __name__ == "__main__":
    main()