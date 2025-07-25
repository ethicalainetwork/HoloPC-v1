#!/usr/bin/env python3

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

os.environ['CUDA_VISIBLE_DEVICES'] = ''

try:
    import requests
    import soundfile as sf
except ImportError as e:
    print(f"❌ Missing basic dependencies: {e}")
    print("Please install: pip install requests soundfile numpy websockets")
    sys.exit(1)

WHISPER_BACKEND = None
try:
    import faster_whisper
    WHISPER_BACKEND = "faster-whisper"
except ImportError:
    try:
        import whisper
        WHISPER_BACKEND = "openai-whisper"
    except ImportError:
        print("❌ No Whisper backend found!")
        sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LazyInitSpeechProcessor:
    def __init__(self, model_size="tiny.en", language="en", server_url="http://localhost:3000", websocket_port=8765):
        self.model_size = model_size
        self.language = language
        self.server_url = server_url
        self.websocket_port = websocket_port
        self.audio_queue = queue.Queue()
        self.sample_rate = 16000
        self.audio_buffer = np.array([], dtype=np.float32)
        self.model = None
        self.model_initialized = False
        self.model_loading = False
        self.is_processing = False
        self.current_transcript = ""
        self.message_count = 0
        self.websocket_server = None
        self.connected_clients = set()
        self.auth_password = os.getenv('SPEECH_PASSWORD', 'holographic2024')
        self.loop = None  # to be set on run()
        logger.info("✅ Processor initialized (lazy Whisper loading)")

    def initialize_whisper_lazy(self):
        if self.model_initialized or self.model_loading:
            return self.model_initialized
        logger.info("🔄 Initializing Whisper model...")
        self.model_loading = True
        try:
            if WHISPER_BACKEND == "faster-whisper":
                from faster_whisper import WhisperModel
                self.model = WhisperModel(self.model_size, device="cpu", compute_type="int8", num_workers=1)
            elif WHISPER_BACKEND == "openai-whisper":
                import whisper
                self.model = whisper.load_model(self.model_size, device="cpu")
            self.model_initialized = True
            logger.info("✅ Whisper model initialized")
        except Exception as e:
            logger.error(f"❌ Whisper init failed: {e}")
        finally:
            self.model_loading = False
        return self.model_initialized

    def transcribe_audio_safe(self, audio_data: np.ndarray) -> str:
        try:
            if not self.model_initialized and not self.initialize_whisper_lazy():
                return "Model init failed"

            if len(audio_data) == 0 or not np.isfinite(audio_data).all():
                return ""

            audio_data = audio_data.astype(np.float32)
            audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=0.0, neginf=0.0)
            max_val = np.abs(audio_data).max()
            if max_val < 0.001:
                return ""
            if max_val > 1.0:
                audio_data = audio_data / max_val

            start = time.time()
            if WHISPER_BACKEND == "faster-whisper":
                segments, _ = self.model.transcribe(audio_data, language=self.language, beam_size=1, best_of=1)
                transcript = " ".join(segment.text.strip() for segment in segments if hasattr(segment, 'text'))
            else:
                result = self.model.transcribe(audio_data, language=self.language, fp16=False)
                transcript = result.get("text", "").strip()

            logger.info(f"⚡ Transcribed in {time.time() - start:.2f}s: {transcript[:60]}...")
            return transcript

        except Exception as e:
            logger.error(f"❌ Transcription error: {e}")
            return ""

    def safe_audio_decode(self, audio_b64: str) -> Optional[np.ndarray]:
        try:
            import base64
            audio_bytes = base64.b64decode(audio_b64)
            audio_data, sr = sf.read(io.BytesIO(audio_bytes))
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            if sr != self.sample_rate:
                ratio = self.sample_rate / sr
                audio_data = np.interp(
                    np.linspace(0, len(audio_data), int(len(audio_data) * ratio)),
                    np.arange(len(audio_data)),
                    audio_data
                )
            return audio_data.astype(np.float32)
        except Exception as e:
            logger.warning(f"Decode error: {e}")
            return None

    async def handle_websocket_connection(self, websocket, path):
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logger.info(f"🔌 Connection from {client_id}")
        try:
            msg = await asyncio.wait_for(websocket.recv(), timeout=30.0)
            data = json.loads(msg)
            if data.get("type") != "auth" or data.get("password") != self.auth_password:
                await websocket.send(json.dumps({"type": "auth_failed"}))
                return

            await websocket.send(json.dumps({"type": "auth_success", "model_loaded": self.model_initialized}))
            self.connected_clients.add(websocket)

            if not self.is_processing:
                self.start_processing()

            async for msg in websocket:
                try:
                    data = json.loads(msg)
                    if data.get("type") == "start_recording":
                        self.current_transcript = "Listening..."
                        if not self.model_initialized:
                            asyncio.create_task(self._initialize_model_async())
                    elif data.get("type") == "stop_recording":
                        await self._process_final_audio()
                    elif data.get("type") == "audio_chunk":
                        audio = self.safe_audio_decode(data.get("audio", ""))
                        if audio is not None:
                            self.audio_queue.put(audio)
                            self.message_count += 1
                except Exception as e:
                    logger.warning(f"Message error: {e}")
        except Exception as e:
            logger.warning(f"Client error: {e}")
        finally:
            self.connected_clients.discard(websocket)
            if not self.connected_clients:
                self.stop_processing()
            logger.info(f"🔌 Disconnected: {client_id}")

    def start_processing(self):
        self.is_processing = True
        threading.Thread(target=self._processing_loop, daemon=True).start()
        logger.info("🎤 Processing started")

    def stop_processing(self):
        self.is_processing = False
        logger.info("🛑 Processing stopped")

    def _processing_loop(self):
        last_time = time.time()
        while self.is_processing:
            try:
                chunks = []
                while not self.audio_queue.empty():
                    chunks.append(self.audio_queue.get_nowait())
                if chunks:
                    self.audio_buffer = np.concatenate([self.audio_buffer] + chunks)
                dur = len(self.audio_buffer) / self.sample_rate
                if dur >= 2.0 and self.model_initialized and time.time() - last_time > 1:
                    transcript = self.transcribe_audio_safe(self.audio_buffer)
                    if transcript and transcript != self.current_transcript:
                        self.current_transcript = transcript
                        if self.loop:
                            asyncio.run_coroutine_threadsafe(self._update_and_broadcast(), self.loop)
                    self.audio_buffer = self.audio_buffer[-self.sample_rate:]  # Keep 1 sec
                    last_time = time.time()
                time.sleep(0.2)
            except Exception as e:
                logger.error(f"Loop error: {e}")
                time.sleep(1)

    async def _process_final_audio(self):
        if len(self.audio_buffer) > 0 and self.model_initialized:
            transcript = self.transcribe_audio_safe(self.audio_buffer)
            if transcript:
                self.current_transcript = f"Final: {transcript}"
                await self._update_and_broadcast()
        self.audio_buffer = np.array([], dtype=np.float32)

    async def _initialize_model_async(self):
        success = await asyncio.get_event_loop().run_in_executor(None, self.initialize_whisper_lazy)
        self.current_transcript = "Model ready!" if success else "Model failed"
        await self._update_and_broadcast()

    async def _update_and_broadcast(self):
        await self._update_display(self.current_transcript)
        await self._broadcast_transcript(self.current_transcript)

    async def _update_display(self, transcript: str):
        try:
            requests.post(f"{self.server_url}/api/speech/update-text", json={"text": transcript}, timeout=3)
        except Exception:
            pass

    async def _broadcast_transcript(self, transcript: str):
        if not self.connected_clients:
            return
        msg = json.dumps({"type": "streaming_transcript", "text": transcript})
        for client in list(self.connected_clients):
            try:
                await client.send(msg)
            except:
                self.connected_clients.discard(client)

    async def run(self):
        self.loop = asyncio.get_running_loop()
        server = await websockets.serve(
            self.handle_websocket_connection, "0.0.0.0", self.websocket_port,
            ping_interval=None, ping_timeout=None, close_timeout=10
        )
        logger.info(f"✅ WebSocket running on ws://0.0.0.0:{self.websocket_port}/ws/")
        await asyncio.Future()  # Keep running

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="tiny.en")
    parser.add_argument("--language", default="en")
    parser.add_argument("--server-url", default="http://localhost:3000")
    parser.add_argument("--websocket-port", type=int, default=8765)
    args = parser.parse_args()

    processor = LazyInitSpeechProcessor(
        model_size=args.model,
        language=args.language,
        server_url=args.server_url,
        websocket_port=args.websocket_port
    )
    asyncio.run(processor.run())

if __name__ == "__main__":
    main()
