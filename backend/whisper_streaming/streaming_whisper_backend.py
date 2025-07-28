#!/usr/bin/env python3
"""
Streaming Whisper Backend - Uses whisper_online OnlineASRProcessor pattern
Provides real-time word-by-word streaming transcription
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

# Force CPU-only mode to avoid CUDA issues
os.environ['CUDA_VISIBLE_DEVICES'] = ''

try:
    import requests
    import soundfile as sf
except ImportError as e:
    print(f"❌ Missing basic dependencies: {e}")
    print("Please install: pip install requests soundfile numpy")
    sys.exit(1)

# Import whisper_online components
try:
    from whisper_online import OnlineASRProcessor, FasterWhisperASR, WhisperTimestampedASR, asr_factory
    print("✅ Successfully imported whisper_online components")
except ImportError as e:
    print(f"❌ Could not import whisper_online: {e}")
    print("Make sure whisper_online.py is in the current directory or PYTHONPATH")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StreamingWhisperProcessor:
    def __init__(self, 
                 model_size: str = "tiny.en",
                 language: str = "en",
                 backend: str = "faster-whisper",
                 server_url: str = "http://localhost:3000",
                 websocket_port: int = 8765,
                 min_chunk_size: float = 1.0,
                 use_vad: bool = True):
        
        self.model_size = model_size
        self.language = language
        self.backend = backend
        self.server_url = server_url
        self.websocket_port = websocket_port
        self.min_chunk_size = min_chunk_size
        self.use_vad = use_vad
        
        # Audio processing
        self.audio_queue = queue.Queue()
        self.sample_rate = 16000
        
        # Streaming transcription
        self.asr = None
        self.online_processor = None
        self.is_processing = False
        self.current_transcript = ""
        self.last_confirmed_transcript = ""
        
        # WebSocket server
        self.websocket_server = None
        self.connected_clients = set()
        
        # Authentication
        self.auth_password = os.getenv('SPEECH_PASSWORD', 'holographic2024')
        
        # Initialize Whisper streaming components
        self.initialize_streaming_whisper()
    
    def initialize_streaming_whisper(self):
        """Initialize Whisper with OnlineASRProcessor for streaming"""
        try:
            logger.info(f"🎤 Initializing streaming Whisper: {self.model_size} ({self.backend})")
            
            # Create args object for asr_factory
            class Args:
                def __init__(self, model, language, backend, use_vad, min_chunk_size):
                    self.model = model
                    self.lan = language
                    self.language = language
                    self.backend = backend
                    self.vad = use_vad
                    self.vac = False  # Voice Activity Controller (more complex)
                    self.min_chunk_size = min_chunk_size
                    self.vac_chunk_size = 0.04
                    self.model_cache_dir = None
                    self.model_dir = None
                    self.task = "transcribe"
                    self.buffer_trimming = "segment"
                    self.buffer_trimming_sec = 15
                    self.log_level = "INFO"
            
            args = Args(self.model_size, self.language, self.backend, self.use_vad, self.min_chunk_size)
            
            # Use asr_factory to create ASR and OnlineASRProcessor
            self.asr, self.online_processor = asr_factory(args, logfile=sys.stderr)
            
            logger.info("✅ Streaming Whisper initialization complete")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize streaming Whisper: {e}")
            raise
    
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
                'message': 'Connected to streaming speech recognition'
            }))
            
            logger.info("✅ Client authenticated for streaming")
            
            # Start speech processing if not already running
            if not self.is_processing:
                self.start_streaming_processing()
            
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
                                # Simple resampling
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
                        logger.info("🎙️ Recording started - initializing streaming")
                        self.online_processor.init()  # Reset the streaming processor
                        
                    elif data['type'] == 'stop_recording':
                        logger.info("🛑 Recording stopped - finalizing")
                        await self.finalize_streaming()
                        
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
                self.stop_streaming_processing()
    
    def start_streaming_processing(self):
        """Start the streaming processing thread"""
        if self.is_processing:
            return
        
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._streaming_processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        logger.info("🎤 Streaming processing started")
    
    def stop_streaming_processing(self):
        """Stop streaming processing"""
        self.is_processing = False
        
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join(timeout=2.0)
        
        logger.info("🛑 Streaming processing stopped")
    
    def _streaming_processing_loop(self):
        """Main streaming processing loop using OnlineASRProcessor"""
        last_process_time = time.time()
        
        while self.is_processing:
            try:
                current_time = time.time()
                
                # Collect audio chunks from queue
                audio_chunks = []
                while not self.audio_queue.empty():
                    try:
                        chunk = self.audio_queue.get_nowait()
                        audio_chunks.append(chunk)
                    except queue.Empty:
                        break
                
                # Process audio chunks if we have any
                if audio_chunks:
                    # Combine all chunks
                    combined_audio = np.concatenate(audio_chunks)
                    
                    # Insert into online processor
                    self.online_processor.insert_audio_chunk(combined_audio)
                    
                    # Process and get streaming result
                    beg_timestamp, end_timestamp, transcript = self.online_processor.process_iter()
                    
                    # Check if we have new confirmed text
                    if transcript and transcript.strip():
                        if transcript != self.current_transcript:
                            self.current_transcript = transcript
                            logger.info(f"💬 Streaming: \"{transcript}\"")
                            
                            # Send to holographic display server
                            asyncio.run_coroutine_threadsafe(
                                self._update_holographic_display(transcript),
                                asyncio.get_event_loop()
                            )
                            
                            # Send to WebSocket clients with streaming info
                            asyncio.run_coroutine_threadsafe(
                                self._broadcast_streaming_transcript(transcript, beg_timestamp, end_timestamp),
                                asyncio.get_event_loop()
                            )
                    
                    last_process_time = current_time
                
                # Sleep briefly to prevent busy waiting
                time.sleep(0.05)  # 50ms
                
            except Exception as e:
                logger.error(f"❌ Error in streaming processing loop: {e}")
                time.sleep(0.1)
    
    async def finalize_streaming(self):
        """Finalize streaming and get final transcript"""
        try:
            if self.online_processor:
                # Get final result
                beg_timestamp, end_timestamp, final_transcript = self.online_processor.finish()
                
                if final_transcript and final_transcript.strip():
                    self.current_transcript = final_transcript
                    logger.info(f"✅ Final transcript: \"{final_transcript}\"")
                    
                    # Send final transcript
                    await self._update_holographic_display(final_transcript)
                    await self._broadcast_streaming_transcript(final_transcript, beg_timestamp, end_timestamp, is_final=True)
                
        except Exception as e:
            logger.error(f"❌ Error finalizing streaming: {e}")
    
    async def _update_holographic_display(self, transcript: str):
        """Send transcript to holographic display server"""
        try:
            response = requests.post(
                f"{self.server_url}/api/speech/update-text",
                json={"text": transcript},
                timeout=3.0
            )
            
            if response.status_code == 200:
                logger.debug(f"✅ Updated holographic display")
            else:
                logger.warning(f"⚠️ Failed to update display: {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ Error updating holographic display: {e}")
    
    async def _broadcast_streaming_transcript(self, transcript: str, beg_timestamp=None, end_timestamp=None, is_final=False):
        """Broadcast streaming transcript to all connected WebSocket clients"""
        if not self.connected_clients:
            return
        
        message = json.dumps({
            'type': 'streaming_transcript',
            'text': transcript,
            'beg_timestamp': beg_timestamp,
            'end_timestamp': end_timestamp,
            'is_final': is_final,
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
        logger.info(f"🌐 Starting streaming WebSocket server on port {self.websocket_port}")
        
        self.websocket_server = await websockets.serve(
            self.handle_websocket_connection,
            "0.0.0.0",
            self.websocket_port
        )
        
        logger.info(f"✅ Streaming WebSocket server running on ws://localhost:{self.websocket_port}")
    
    async def run(self):
        """Run the streaming speech processor"""
        try:
            await self.start_websocket_server()
            
            logger.info("🎤 Streaming Whisper Processor ready!")
            logger.info(f"🧠 Model: {self.model_size} ({self.backend})")
            logger.info(f"🎯 Language: {self.language}")
            logger.info(f"⚙️ VAD: {'enabled' if self.use_vad else 'disabled'}")
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
            self.stop_streaming_processing()

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Streaming Whisper Speech Recognition")
    parser.add_argument("--model", default="tiny.en", help="Whisper model size")
    parser.add_argument("--language", default="en", help="Language")
    parser.add_argument("--backend", default="faster-whisper", 
                       choices=["faster-whisper", "whisper_timestamped"],
                       help="Whisper backend")
    parser.add_argument("--server-url", default="http://localhost:3000", help="Display server URL")
    parser.add_argument("--websocket-port", type=int, default=8765, help="WebSocket port")
    parser.add_argument("--min-chunk-size", type=float, default=1.0, help="Minimum chunk size in seconds")
    parser.add_argument("--no-vad", action="store_true", help="Disable VAD")
    parser.add_argument("--password", help="Authentication password")
    
    args = parser.parse_args()
    
    # Set password if provided
    if args.password:
        os.environ['SPEECH_PASSWORD'] = args.password
    
    processor = StreamingWhisperProcessor(
        model_size=args.model,
        language=args.language,
        backend=args.backend,
        server_url=args.server_url,
        websocket_port=args.websocket_port,
        min_chunk_size=args.min_chunk_size,
        use_vad=not args.no_vad
    )
    
    asyncio.run(processor.run())

if __name__ == "__main__":
    main()