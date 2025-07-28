#!/usr/bin/env python3
"""
Real-time Speech Recognition Backend using Whisper
Integrates with the holographic display server
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
from typing import Optional, Dict, Any

try:
    from whisper_online import *
    import soundfile as sf
    import requests
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("Please install required packages:")
    print("pip install librosa soundfile requests")
    print("pip install faster-whisper  # or whisper-timestamped")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HolographicSpeechProcessor:
    def __init__(self, 
                 model_size: str = "base.en",
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
        self.chunk_duration = 0.5  # 500ms chunks
        self.min_chunk_size = 1.0  # Minimum 1 second for processing
        
        # Speech recognition
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
        
        # Initialize Whisper
        self.initialize_whisper()
    
    def initialize_whisper(self):
        """Initialize Whisper ASR model"""
        try:
            logger.info(f"🎤 Initializing Whisper model: {self.model_size}")
            
            # Try faster-whisper first (recommended)
            try:
                self.asr = FasterWhisperASR(self.language, self.model_size)
                logger.info("✅ Using faster-whisper backend")
            except Exception as e:
                logger.warning(f"⚠️ faster-whisper failed: {e}")
                try:
                    # Fallback to whisper-timestamped
                    self.asr = WhisperTimestampedASR(self.language, self.model_size)
                    logger.info("✅ Using whisper-timestamped backend")
                except Exception as e:
                    logger.error(f"❌ All Whisper backends failed: {e}")
                    raise
            
            # Configure ASR options
            self.asr.use_vad()  # Enable Voice Activity Detection
            
            # Initialize online processor
            self.online_processor = OnlineASRProcessor(self.asr)
            
            logger.info("✅ Whisper initialization complete")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Whisper: {e}")
            raise
    
    async def handle_websocket_connection(self, websocket, path):
        """Handle WebSocket connections from web clients"""
        logger.info(f"🔌 New WebSocket connection from {websocket.remote_address}")
        
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
            
            logger.info("✅ Client authenticated successfully")
            
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
                        
                        # Convert to numpy array (assuming 16-bit PCM)
                        audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                        
                        # Add to processing queue
                        self.audio_queue.put(audio_data)
                        
                    elif data['type'] == 'start_recording':
                        logger.info("🎙️ Recording started")
                        
                    elif data['type'] == 'stop_recording':
                        logger.info("🛑 Recording stopped")
                        self.finish_speech_processing()
                        
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
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join(timeout=2.0)
        
        logger.info("🛑 Speech processing stopped")
    
    def _speech_processing_loop(self):
        """Main speech processing loop"""
        audio_buffer = np.array([], dtype=np.float32)
        last_process_time = time.time()
        
        while self.is_processing:
            try:
                # Collect audio chunks
                current_time = time.time()
                
                # Get audio from queue (non-blocking)
                while not self.audio_queue.empty():
                    try:
                        chunk = self.audio_queue.get_nowait()
                        audio_buffer = np.concatenate([audio_buffer, chunk])
                    except queue.Empty:
                        break
                
                # Process if we have enough audio and enough time has passed
                buffer_duration = len(audio_buffer) / self.sample_rate
                time_since_last_process = current_time - last_process_time
                
                if buffer_duration >= self.min_chunk_size and time_since_last_process >= self.chunk_duration:
                    # Process the audio
                    self.online_processor.insert_audio_chunk(audio_buffer)
                    transcript = self.online_processor.process_iter()
                    
                    if transcript and transcript != self.current_transcript:
                        self.current_transcript = transcript
                        
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
                    
                    # Clear processed audio (keep some overlap)
                    overlap_samples = int(0.2 * self.sample_rate)  # 200ms overlap
                    audio_buffer = audio_buffer[-overlap_samples:] if len(audio_buffer) > overlap_samples else np.array([], dtype=np.float32)
                    last_process_time = current_time
                
                # Sleep briefly to prevent busy waiting
                time.sleep(0.05)  # 50ms
                
            except Exception as e:
                logger.error(f"❌ Error in speech processing loop: {e}")
                time.sleep(0.1)
    
    def finish_speech_processing(self):
        """Finish processing and get final transcript"""
        try:
            if self.online_processor:
                final_transcript = self.online_processor.finish()
                if final_transcript and final_transcript != self.current_transcript:
                    self.current_transcript = final_transcript
                    
                    # Send final transcript
                    asyncio.run_coroutine_threadsafe(
                        self._update_holographic_display(final_transcript),
                        asyncio.get_event_loop()
                    )
                    
                    asyncio.run_coroutine_threadsafe(
                        self._broadcast_transcript(final_transcript),
                        asyncio.get_event_loop()
                    )
                
                # Reset for next session
                self.online_processor.init()
                
        except Exception as e:
            logger.error(f"❌ Error finishing speech processing: {e}")
    
    async def _update_holographic_display(self, transcript: str):
        """Send transcript to holographic display server"""
        try:
            response = requests.post(
                f"{self.server_url}/api/speech/update-text",
                json={"text": transcript},
                timeout=2.0
            )
            
            if response.status_code == 200:
                logger.info(f"✅ Updated holographic display: \"{transcript}\"")
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
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(websocket)
            except Exception as e:
                logger.error(f"❌ Error sending to client: {e}")
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
            # Start WebSocket server
            await self.start_websocket_server()
            
            logger.info("🎤 Holographic Speech Processor ready!")
            logger.info(f"🔐 Authentication password: {self.auth_password}")
            logger.info(f"🌐 WebSocket server: ws://localhost:{self.websocket_port}")
            logger.info(f"🎯 Holographic display server: {self.server_url}")
            
            # Keep running
            await asyncio.Future()  # Run forever
            
        except KeyboardInterrupt:
            logger.info("🛑 Shutting down...")
        except Exception as e:
            logger.error(f"❌ Error running speech processor: {e}")
        finally:
            self.stop_speech_processing()

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Holographic Speech Recognition Backend")
    parser.add_argument("--model", default="base.en", help="Whisper model size")
    parser.add_argument("--language", default="en", help="Source language")
    parser.add_argument("--server-url", default="http://localhost:3000", help="Holographic display server URL")
    parser.add_argument("--websocket-port", type=int, default=8765, help="WebSocket server port")
    parser.add_argument("--password", help="Authentication password (or set SPEECH_PASSWORD env var)")
    
    args = parser.parse_args()
    
    # Set password if provided
    if args.password:
        os.environ['SPEECH_PASSWORD'] = args.password
    
    # Create and run processor
    processor = HolographicSpeechProcessor(
        model_size=args.model,
        language=args.language,
        server_url=args.server_url,
        websocket_port=args.websocket_port
    )
    
    # Run the async event loop
    asyncio.run(processor.run())

if __name__ == "__main__":
    main()