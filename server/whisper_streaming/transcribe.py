#!/usr/bin/env python3
"""
Enhanced Lazy Initialization Backend - Full support for Raw PCM and improved audio processing
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
import concurrent.futures

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

class EnhancedSpeechProcessor:
    def __init__(self, 
                 model_size: str = "tiny.en",
                 language: str = "en",
                 server_url: str = "http://localhost:3000",
                 websocket_port: int = 8765,
                 use_vad: bool = False):
        
        self.model_size = model_size
        self.language = language
        self.server_url = server_url
        self.websocket_port = websocket_port
        self.use_vad = use_vad
        
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
        
        # Event loop for async operations from threads
        self.main_loop = None
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        
        # Audio format statistics
        self.format_stats = {
            'pcm_raw': 0,
            'wav_fallback': 0,
            'unknown_format': 0,
            'decode_failures': 0
        }
        
        logger.info("✅ Enhanced processor created with Raw PCM support")
    
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
                
                # Test the model with a simple sine wave
                logger.info("🧪 Testing model with synthetic audio...")
                test_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)).astype(np.float32) * 0.1
                
                try:
                    with np.errstate(invalid='ignore', over='ignore'):
                        segments, info = self.model.transcribe(
                            test_audio,
                            language=self.language,
                            beam_size=1,
                            best_of=1,
                        )
                        test_result = " ".join([seg.text.strip() for seg in segments if hasattr(seg, 'text')]).strip()
                    
                    logger.info(f"🧪 Model test result: '{test_result}' (empty is normal for sine wave)")
                except Exception as test_error:
                    logger.warning(f"🧪 Model test failed: {test_error}")
                
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
    
    def safe_audio_decode_enhanced(self, audio_b64: str, format_info: dict = None) -> Optional[np.ndarray]:
        """Enhanced audio decode with support for raw PCM data"""
        try:
            import base64
            audio_bytes = base64.b64decode(audio_b64)
            
            if len(audio_bytes) == 0:
                logger.debug("Received empty audio bytes")
                return None
            
            format_type = format_info.get('format', 'unknown') if format_info else 'unknown'
            logger.debug(f"📦 Received {len(audio_bytes)} bytes, format: {format_type}")

            # Handle raw PCM data first (PRIORITY)
            if format_info and format_info.get('format') == 'pcm_s16le':
                try:
                    # Raw PCM signed 16-bit little endian
                    audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
                    
                    # Convert to float32 and normalize to [-1, 1]
                    audio_data = audio_data.astype(np.float32) / 32768.0
                    
                    sr = format_info.get('sampleRate', self.sample_rate)
                    logger.debug(f"✅ Raw PCM decode: {len(audio_data)} samples at {sr}Hz")
                    
                    # Validate sample count if provided
                    expected_samples = format_info.get('samplesCount')
                    if expected_samples and len(audio_data) != expected_samples:
                        logger.warning(f"⚠️ Sample count mismatch: got {len(audio_data)}, expected {expected_samples}")
                    
                    self.format_stats['pcm_raw'] += 1
                    
                except Exception as e:
                    logger.error(f"❌ Raw PCM decode failed: {e}")
                    self.format_stats['decode_failures'] += 1
                    return None
            
            # Handle WAV fallback
            elif format_info and 'wav' in format_info.get('format', '').lower():
                try:
                    audio_data, sr = sf.read(io.BytesIO(audio_bytes))
                    logger.debug(f"✅ WAV decode: {len(audio_data)} samples at {sr}Hz")
                    self.format_stats['wav_fallback'] += 1
                except Exception as e:
                    logger.debug(f"❌ WAV decode failed: {e}")
                    self.format_stats['decode_failures'] += 1
                    return None
            
            # Try soundfile for unknown formats
            else:
                try:
                    audio_data, sr = sf.read(io.BytesIO(audio_bytes))
                    logger.debug(f"✅ Soundfile decode: {len(audio_data)} samples at {sr}Hz")
                    self.format_stats['unknown_format'] += 1
                except Exception as e:
                    logger.debug(f"❌ Soundfile failed: {e}, trying raw decode")
                    
                    # Fallback to raw data interpretation
                    try:
                        for dtype in [np.int16, np.float32, np.float64]:
                            try:
                                audio_data = np.frombuffer(audio_bytes, dtype=dtype)
                                if len(audio_data) > 0:
                                    # Normalize if needed
                                    if dtype == np.int16:
                                        audio_data = audio_data.astype(np.float32) / 32768.0
                                    elif dtype == np.float64:
                                        audio_data = audio_data.astype(np.float32)
                                    
                                    sr = self.sample_rate
                                    logger.debug(f"✅ Raw decode with {dtype}: {len(audio_data)} samples")
                                    self.format_stats['unknown_format'] += 1
                                    break
                            except Exception:
                                continue
                        else:
                            logger.warning("❌ All raw decode attempts failed")
                            self.format_stats['decode_failures'] += 1
                            return None
                    except Exception as e2:
                        logger.debug(f"❌ Raw decode also failed: {e2}")
                        self.format_stats['decode_failures'] += 1
                        return None
            
            if len(audio_data) == 0:
                logger.debug("Audio data is empty after decoding")
                return None
            
            # Log audio statistics before processing
            original_stats = {
                'length': len(audio_data),
                'sample_rate': sr,
                'duration': len(audio_data) / sr if sr > 0 else 0,
                'dtype': audio_data.dtype,
                'shape': audio_data.shape,
                'min': float(np.min(audio_data)) if len(audio_data) > 0 else 0,
                'max': float(np.max(audio_data)) if len(audio_data) > 0 else 0,
                'mean': float(np.mean(audio_data)) if len(audio_data) > 0 else 0,
                'rms': float(np.sqrt(np.mean(audio_data**2))) if len(audio_data) > 0 else 0
            }
            logger.debug(f"📊 Raw audio stats: {original_stats}")

            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                logger.debug(f"Converting from {audio_data.shape} to mono")
                audio_data = np.mean(audio_data, axis=1)

            # Resample if needed
            if sr != self.sample_rate and sr > 0:
                try:
                    from scipy import signal
                    target_length = int(len(audio_data) * self.sample_rate / sr)
                    audio_data = signal.resample(audio_data, target_length)
                    logger.debug(f"🔄 Resampled from {sr}Hz to {self.sample_rate}Hz: {len(audio_data)} samples")
                except ImportError:
                    target_length = int(len(audio_data) * self.sample_rate / sr)
                    if target_length > 0:
                        audio_data = np.interp(
                            np.linspace(0, len(audio_data), target_length),
                            np.arange(len(audio_data)),
                            audio_data
                        )
                        logger.debug(f"🔄 Linear resampled: {len(audio_data)} samples")

            # Clean the audio data
            audio_data = np.nan_to_num(audio_data.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            
            # Log final audio statistics
            final_stats = {
                'length': len(audio_data),
                'dtype': audio_data.dtype,
                'min': float(np.min(audio_data)),
                'max': float(np.max(audio_data)),
                'mean': float(np.mean(audio_data)),
                'rms': float(np.sqrt(np.mean(audio_data**2))),
                'non_zero_samples': int(np.count_nonzero(audio_data)),
                'zero_samples': int(len(audio_data) - np.count_nonzero(audio_data))
            }
            
            # Only log full stats every 20 chunks to reduce noise
            if self.message_count % 20 == 0:
                logger.info(f"🎵 Final audio stats: {final_stats}")
            else:
                logger.debug(f"🎵 Final audio stats: {final_stats}")
            
            # Check if we have meaningful audio
            if final_stats['rms'] < 1e-6:
                logger.debug(f"⚠️ Audio RMS very low ({final_stats['rms']:.8f}) - might be silent")
            
            if final_stats['non_zero_samples'] == 0:
                logger.debug("⚠️ All audio samples are zero - definitely silent")
                return None
                
            return audio_data
            
        except Exception as e:
            logger.error(f"❌ Audio decode error: {e}")
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            self.format_stats['decode_failures'] += 1
            return None
    
    def transcribe_audio_safe(self, audio_data: np.ndarray) -> str:
        """Safely transcribe audio with lazy initialization"""
        try:
            # Initialize model if needed
            if not self.model_initialized:
                if not self.initialize_whisper_lazy():
                    return "Model initialization failed"
            
            if len(audio_data) == 0:
                return ""
            
            # Enhanced audio validation and cleaning
            if np.all(audio_data == 0):
                logger.debug("All audio is silent")
                return ""
            
            # Check for NaN or inf values FIRST
            if np.any(np.isnan(audio_data)) or np.any(np.isinf(audio_data)):
                logger.warning("⚠️ Found NaN/inf values in audio, cleaning...")
                audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Clip audio to prevent overflow BEFORE calculating RMS
            audio_data = np.clip(audio_data, -1.0, 1.0)
            
            # Check audio levels for debugging
            try:
                audio_rms = np.sqrt(np.mean(audio_data**2))
                audio_max = np.abs(audio_data).max()
                logger.debug(f"Audio levels: RMS={audio_rms:.4f}, Max={audio_max:.4f}")
            except Exception as e:
                logger.warning(f"Could not calculate audio levels: {e}")
                audio_rms = 0.0
                audio_max = 0.0
            
            # Normalize audio more carefully
            if audio_max > 1.0:
                audio_data = audio_data / audio_max
                logger.debug(f"Audio normalized from max {audio_max:.3f}")
            elif audio_max < 0.001:
                logger.debug(f"Audio very quiet (max={audio_max:.6f}), but trying anyway")
            
            start_time = time.time()
            
            if WHISPER_BACKEND == "faster-whisper":
                # Suppress numpy warnings during transcription
                with np.errstate(invalid='ignore', over='ignore'):
                    if self.use_vad:
                        logger.debug("Using transcription with VAD filtering")
                        try:
                            segments, info = self.model.transcribe(
                                audio_data,
                                language=self.language,
                                beam_size=1,
                                best_of=1,
                                temperature=0.0,
                                word_timestamps=False,
                                vad_filter=True,
                                vad_parameters=dict(
                                    threshold=0.3,
                                    min_speech_duration_ms=100,
                                    min_silence_duration_ms=1000,
                                    window_size_samples=512,
                                    speech_pad_ms=200
                                )
                            )
                        except Exception as vad_error:
                            logger.warning(f"VAD filtering failed: {vad_error}, trying without VAD")
                            segments, info = self.model.transcribe(
                                audio_data,
                                language=self.language,
                                beam_size=1,
                                best_of=1,
                                temperature=0.0,
                                word_timestamps=False,
                                vad_filter=False
                            )
                    else:
                        logger.debug("Using transcription WITHOUT VAD filtering")
                        segments, info = self.model.transcribe(
                            audio_data,
                            language=self.language,
                            beam_size=1,
                            best_of=1,
                            temperature=0.0,
                            word_timestamps=False,
                            vad_filter=False
                        )
                
                transcript_parts = []
                segment_count = 0
                
                for segment in segments:
                    segment_count += 1
                    if hasattr(segment, 'text') and segment.text:
                        text = segment.text.strip()
                        if len(text) > 0 and not self._is_garbage_text(text):
                            transcript_parts.append(text)
                
                if segment_count == 0:
                    if self.use_vad:
                        logger.warning("⚠️ VAD removed all audio - no speech segments detected")
                    else:
                        logger.debug("No speech segments detected (VAD disabled)")
                else:
                    logger.debug(f"Processed {segment_count} segments, VAD={'enabled' if self.use_vad else 'disabled'}")
                
                transcript = " ".join(transcript_parts).strip()
                
            elif WHISPER_BACKEND == "openai-whisper":
                with np.errstate(invalid='ignore', over='ignore'):
                    result = self.model.transcribe(
                        audio_data,
                        language=self.language,
                        fp16=False
                    )
                transcript = result.get("text", "").strip()
                
                if self._is_garbage_text(transcript):
                    transcript = ""
            
            processing_time = time.time() - start_time
            audio_duration = len(audio_data) / self.sample_rate
            
            # Log results
            if transcript and len(transcript.strip()) > 0:
                logger.info(f"⚡ Transcribed {audio_duration:.1f}s in {processing_time:.2f}s: \"{transcript[:50]}...\"")
            else:
                logger.info(f"🔇 No speech detected in {audio_duration:.1f}s audio (processed in {processing_time:.2f}s)")
            
            return transcript
            
        except Exception as e:
            logger.error(f"❌ Transcription error: {e}")
            return ""
    
    def _is_garbage_text(self, text: str) -> bool:
        """Check if transcribed text is likely garbage"""
        if not text or len(text.strip()) == 0:
            return True
        
        # Check for repetitive characters
        if len(set(text.replace(' ', ''))) <= 2 and len(text) > 10:
            return True
        
        # Check for very repetitive patterns
        cleaned = text.replace(' ', '').replace('.', '').replace(',', '')
        if len(cleaned) > 5:
            most_common_char = max(set(cleaned), key=cleaned.count)
            if cleaned.count(most_common_char) / len(cleaned) > 0.8:
                return True
        
        return False
    
    async def handle_websocket_connection(self, websocket, path=None):
        """Handle WebSocket connections with enhanced error handling"""
        client_addr = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logger.info(f"🔌 New connection from {client_addr}")
        
        try:
            # Step 1: Authentication
            auth_message = await asyncio.wait_for(websocket.recv(), timeout=30.0)
            
            try:
                auth_data = json.loads(auth_message)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON from {client_addr}: {e}")
                await websocket.send(json.dumps({'type': 'error', 'message': 'Invalid JSON'}))
                return
            
            if auth_data.get('type') != 'auth' or auth_data.get('password') != self.auth_password:
                logger.warning(f"Authentication failed for {client_addr}")
                await websocket.send(json.dumps({'type': 'auth_failed', 'message': 'Invalid credentials'}))
                return
            
            # Step 2: Authentication successful
            self.connected_clients.add(websocket)
            
            await websocket.send(json.dumps({
                'type': 'auth_success',
                'message': 'Connected to enhanced speech processor',
                'model_loaded': self.model_initialized,
                'backend': WHISPER_BACKEND,
                'features': ['raw_pcm', 'wav_fallback', 'enhanced_decoding']
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
                        # Enhanced logging for audio configuration
                        primary_format = data.get('primaryFormat', 'unknown')
                        fallback_format = data.get('fallbackFormat', 'none')
                        sample_rate = data.get('sampleRate', 16000)
                        channels = data.get('channels', 1)
                        buffer_size = data.get('bufferSize', 'unknown')
                        
                        logger.info(f"🎙️ Recording started for {client_addr}")
                        logger.info(f"📊 Audio config: primary={primary_format}, fallback={fallback_format}, sr={sample_rate}Hz, ch={channels}, buffer={buffer_size}")
                        
                        self.current_transcript = f"Recording started with {primary_format} format, initializing model..."
                        await self._update_and_broadcast()
                        
                        # Initialize Whisper in background
                        if not self.model_initialized:
                            asyncio.create_task(self._initialize_model_async())
                        
                    elif message_type == 'stop_recording':
                        logger.info(f"🛑 Recording stopped for {client_addr}")
                        logger.info(f"📊 Format stats: {self.format_stats}")
                        await self._process_final_audio()
                        
                    elif message_type in ['audio_chunk', 'audio_chunk_raw', 'audio_chunk_wav']:
                        self.message_count += 1
                        
                        # Extract audio data and format info
                        audio_b64 = data.get('audio', '')
                        format_info = {
                            'format': data.get('format', 'unknown'),
                            'sampleRate': data.get('sampleRate', self.sample_rate),
                            'channels': data.get('channels', 1),
                            'samplesCount': data.get('samplesCount')
                        }
                        
                        if self.message_count % 50 == 0:  # Log format info periodically
                            logger.debug(f"Received {message_type} with format: {format_info}")
                        
                        if audio_b64:
                            # Use enhanced decoder
                            audio_data = self.safe_audio_decode_enhanced(audio_b64, format_info)
                            if audio_data is not None and len(audio_data) > 0:
                                self.audio_queue.put(audio_data)
                                
                                # Update status
                                if self.message_count % 10 == 0:
                                    status = "Model loading..." if self.model_loading else f"Listening... ({self.message_count} chunks)"
                                    if self.model_initialized:
                                        status = f"Processing speech in real-time... ({self.message_count} chunks)"
                                    
                                    if not self.current_transcript or "chunks" in self.current_transcript:
                                        self.current_transcript = status
                                        await self._update_and_broadcast()
                        
                        if self.message_count % 100 == 0:  # Log less frequently
                            logger.info(f"Audio chunk {self.message_count} processed, stats: {self.format_stats}")
                        
                    else:
                        logger.debug(f"Unknown message type from {client_addr}: {message_type}")
                        
                except json.JSONDecodeError:
                    logger.error(f"JSON decode error from {client_addr}")
                    continue
                except Exception as e:
                    logger.error(f"Error processing message from {client_addr}: {e}")
                    continue
        
        except asyncio.TimeoutError:
            logger.warning(f"Authentication timeout for {client_addr}")
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
        """Main processing loop with enhanced audio handling"""
        last_process_time = time.time()
        
        while self.is_processing:
            try:
                current_time = time.time()
                
                # Collect audio chunks
                chunks = []
                while not self.audio_queue.empty():
                    try:
                        chunk = self.audio_queue.get_nowait()
                        if chunk is not None and len(chunk) > 0:
                            chunks.append(chunk)
                    except queue.Empty:
                        break
                
                if chunks:
                    # Combine chunks
                    combined_audio = np.concatenate(chunks)
                    
                    # Clean the combined audio
                    combined_audio = np.nan_to_num(combined_audio, nan=0.0, posinf=0.0, neginf=0.0)
                    combined_audio = np.clip(combined_audio, -1.0, 1.0)
                    
                    self.audio_buffer = np.concatenate([self.audio_buffer, combined_audio])
                    
                    # Debug logging
                    try:
                        chunk_rms = np.sqrt(np.mean(combined_audio**2))
                        chunk_max = np.abs(combined_audio).max()
                        logger.debug(f"Added {len(chunks)} chunks: RMS={chunk_rms:.4f}, Max={chunk_max:.4f}")
                    except Exception as e:
                        logger.debug(f"Could not calculate chunk stats: {e}")
                
                # Process if conditions are met
                buffer_duration = len(self.audio_buffer) / self.sample_rate
                time_since_last = current_time - last_process_time
                
                if (buffer_duration >= 0.8 and time_since_last >= 0.5 and 
                    self.model_initialized and len(self.audio_buffer) > 0):
                    
                    logger.info(f"🔄 Processing {buffer_duration:.1f}s of audio")
                    
                    # Clean buffer before processing
                    self.audio_buffer = np.nan_to_num(self.audio_buffer, nan=0.0, posinf=0.0, neginf=0.0)
                    self.audio_buffer = np.clip(self.audio_buffer, -1.0, 1.0)
                    
                    try:
                        buffer_rms = np.sqrt(np.mean(self.audio_buffer**2))
                        buffer_max = np.abs(self.audio_buffer).max()
                        logger.debug(f"Buffer stats: RMS={buffer_rms:.4f}, Max={buffer_max:.4f}, Duration={buffer_duration:.1f}s")
                    except Exception as e:
                        logger.debug(f"Could not calculate buffer stats: {e}")
                    
                    # Transcribe
                    transcript = self.transcribe_audio_safe(self.audio_buffer.copy())
                    
                    logger.debug(f"Transcription result: '{transcript}' (length: {len(transcript) if transcript else 0})")
                    
                    if transcript and transcript != self.current_transcript and len(transcript.strip()) > 0:
                        # Filter out status messages
                        if not any(status_word in transcript.lower() for status_word in 
                                 ['loading', 'chunks', 'processing', 'listening', 'model', 'recording']):
                            
                            logger.info(f"📝 New transcript: '{transcript}'")
                            self.current_transcript = transcript
                            
                            # Schedule async update
                            if self.main_loop and not self.main_loop.is_closed():
                                future = asyncio.run_coroutine_threadsafe(
                                    self._update_and_broadcast(),
                                    self.main_loop
                                )
                        else:
                            logger.debug(f"Filtered out status message: '{transcript}'")
                    
                    # Keep smaller buffer for responsive streaming
                    overlap_samples = int(1.5 * self.sample_rate)
                    if len(self.audio_buffer) > overlap_samples:
                        self.audio_buffer = self.audio_buffer[-overlap_samples:]
                    
                    last_process_time = current_time
                
                # Sleep
                time.sleep(0.05 if chunks else 0.1)
                
            except Exception as e:
                logger.error(f"❌ Error in processing loop: {e}")
                time.sleep(1.0)
    
    async def _process_final_audio(self):
        """Process final audio buffer"""
        if len(self.audio_buffer) > 0 and self.model_initialized:
            transcript = self.transcribe_audio_safe(self.audio_buffer.copy())
            if transcript and len(transcript.strip()) > 0:
                self.current_transcript = f"Final: {transcript}"
                await self._update_and_broadcast()
        
        self.audio_buffer = np.array([], dtype=np.float32)
    
    async def _update_and_broadcast(self):
        """Update display and broadcast to clients"""
        try:
            # Update holographic display
            asyncio.create_task(self._update_display(self.current_transcript))
            
            # Broadcast to clients
            await self._broadcast_transcript(self.current_transcript)
            
        except Exception as e:
            logger.error(f"Error in update/broadcast: {e}")
    
    async def _update_display(self, transcript: str):
        """Update holographic display"""
        try:
            loop = asyncio.get_event_loop()
            
            def sync_request():
                try:
                    response = requests.post(
                        f"{self.server_url}/api/speech/update-text",
                        json={"text": transcript},
                        timeout=2.0
                    )
                    return response.status_code == 200
                except Exception:
                    return False
            
            success = await loop.run_in_executor(None, sync_request)
            
            if success:
                logger.debug("✅ Display updated")
            else:
                logger.debug("Display update failed")
                
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
            'timestamp': time.time(),
            'format_stats': self.format_stats
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
        logger.info(f"🌐 Starting Enhanced WebSocket server on port {self.websocket_port}")
        
        self.websocket_server = await websockets.serve(
            self.handle_websocket_connection,
            "0.0.0.0",
            self.websocket_port,
            ping_interval=30,
            ping_timeout=10,
            close_timeout=10,
            max_size=10 * 1024 * 1024  # 10MB max message size
        )
        
        logger.info(f"✅ Enhanced WebSocket server running on ws://localhost:{self.websocket_port}")
    
    async def run(self):
        """Run the enhanced processor"""
        try:
            # Store the main event loop
            self.main_loop = asyncio.get_event_loop()
            
            # Start WebSocket server
            await self.start_websocket_server()
            
            logger.info("🎤 Enhanced Speech Processor ready!")
            logger.info(f"🔐 Password: {self.auth_password}")
            logger.info(f"🌐 WebSocket: ws://localhost:{self.websocket_port}")
            logger.info(f"🎯 Display server: {self.server_url}")
            logger.info("📝 Features: Raw PCM support, Enhanced decoding, Better error handling")
            logger.info("📊 Supported formats: pcm_s16le (primary), WAV (fallback), Auto-detect (legacy)")
            
            # Keep running
            await asyncio.Future()
            
        except KeyboardInterrupt:
            logger.info("🛑 Shutting down...")
        except Exception as e:
            logger.error(f"❌ Error: {e}")
        finally:
            self.stop_processing()
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=True)

def test_harvard_streaming(processor, harvard_path):
    """Test streaming transcription with harvard.wav file"""
    import os
    import soundfile as sf
    import time
    
    if not os.path.exists(harvard_path):
        print(f"❌ Harvard test file not found: {harvard_path}")
        print("Please ensure harvard.wav is in the same directory as this script")
        return False
    
    try:
        print(f"🎯 Loading Harvard test audio: {harvard_path}")
        
        # Load the harvard.wav file
        audio_data, sample_rate = sf.read(harvard_path)
        
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            print(f"Resampling from {sample_rate}Hz to 16000Hz")
            target_length = int(len(audio_data) * 16000 / sample_rate)
            audio_data = np.interp(
                np.linspace(0, len(audio_data), target_length),
                np.arange(len(audio_data)),
                audio_data
            )
        
        audio_data = audio_data.astype(np.float32)
        
        print(f"📊 Harvard audio: {len(audio_data)/16000:.1f}s, {len(audio_data)} samples")
        print(f"📊 Audio levels: RMS={np.sqrt(np.mean(audio_data**2)):.4f}, Max={np.abs(audio_data).max():.4f}")
        
        # Initialize the model
        if not processor.initialize_whisper_lazy():
            print("❌ Failed to initialize Whisper model")
            return False
        
        print("🚀 Starting Harvard streaming test...")
        
        # Simulate streaming by sending chunks
        chunk_size = int(0.5 * 16000)  # 500ms chunks
        total_chunks = len(audio_data) // chunk_size
        
        print(f"📦 Sending {total_chunks} chunks of {chunk_size/16000:.1f}s each")
        
        processor.current_transcript = "Harvard streaming test starting..."
        
        for i in range(total_chunks):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, len(audio_data))
            chunk = audio_data[start_idx:end_idx]
            
            if len(chunk) > 0:
                processor.audio_queue.put(chunk)
                print(f"📤 Sent chunk {i+1}/{total_chunks} ({len(chunk)} samples)")
                time.sleep(0.1)
        
        # Wait for processing
        print("⏳ Waiting for processing to complete...")
        time.sleep(5.0)
        
        # Process final buffer
        if len(processor.audio_buffer) > 0:
            print("🔄 Processing final buffer...")
            final_transcript = processor.transcribe_audio_safe(processor.audio_buffer.copy())
            if final_transcript and len(final_transcript.strip()) > 0:
                processor.current_transcript = f"Final Harvard result: {final_transcript}"
                print(f"✅ Final transcript: {final_transcript}")
        
        print(f"🎯 Harvard test completed. Final transcript: '{processor.current_transcript}'")
        print(f"📊 Format statistics: {processor.format_stats}")
        return True
        
    except Exception as e:
        print(f"❌ Harvard test failed: {e}")
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Speech Processor with Raw PCM Support")
    parser.add_argument("--model", default="tiny.en", help="Whisper model size")
    parser.add_argument("--language", default="en", help="Language")
    parser.add_argument("--server-url", default="http://localhost:3000", help="Display server URL")
    parser.add_argument("--websocket-port", type=int, default=8765, help="WebSocket port")
    parser.add_argument("--use-vad", action="store_true", help="Enable VAD filtering")
    parser.add_argument("--test-harvard", action="store_true", help="Test with harvard.wav file")
    parser.add_argument("--harvard-path", default="harvard.wav", help="Path to harvard.wav file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("🐛 Debug logging enabled")
    
    logger.info("🎤 Starting Enhanced Speech Processor...")
    logger.info(f"📊 Configuration: model={args.model}, language={args.language}, vad={args.use_vad}")
    
    processor = EnhancedSpeechProcessor(
        model_size=args.model,
        language=args.language,
        server_url=args.server_url,
        websocket_port=args.websocket_port,
        use_vad=args.use_vad
    )
    
    # Harvard test mode
    if args.test_harvard:
        logger.info("🎯 Harvard streaming test mode enabled")
        processor.start_processing()
        
        success = test_harvard_streaming(processor, args.harvard_path)
        
        if success:
            print("✅ Harvard test completed successfully!")
        else:
            print("❌ Harvard test failed!")
        
        try:
            import time
            time.sleep(10)
        except KeyboardInterrupt:
            pass
        
        processor.stop_processing()
        return
    
    # Normal WebSocket server mode
    try:
        asyncio.run(processor.run())
    except KeyboardInterrupt:
        logger.info("🛑 Received interrupt signal")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")

if __name__ == "__main__":
    main()