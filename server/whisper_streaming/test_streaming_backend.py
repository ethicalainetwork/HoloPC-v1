#!/usr/bin/env python3
"""
Test script for streaming Whisper backend
Simulates real-time audio streaming similar to whisper_online.py
"""

import sys
import os
import numpy as np
import time
import soundfile as sf
import threading
import queue

# Force CPU mode
os.environ['CUDA_VISIBLE_DEVICES'] = ''

try:
    from whisper_online import OnlineASRProcessor, asr_factory, load_audio_chunk, load_audio
    print("✅ Successfully imported whisper_online components")
except ImportError as e:
    print(f"❌ Could not import whisper_online: {e}")
    print("Make sure whisper_online.py is in the current directory")
    sys.exit(1)

def simulate_streaming_transcription(audio_file="harvard.wav", 
                                   model_size="tiny.en", 
                                   backend="faster-whisper",
                                   min_chunk_size=1.0,
                                   use_vad=True):
    """Simulate streaming transcription like whisper_online.py"""
    
    print(f"🎤 Streaming Transcription Test")
    print("=" * 40)
    print(f"📂 Audio: {audio_file}")
    print(f"🧠 Model: {model_size} ({backend})")
    print(f"⚙️ Chunk size: {min_chunk_size}s")
    print(f"🔊 VAD: {'enabled' if use_vad else 'disabled'}")
    print("")
    
    if not os.path.exists(audio_file):
        print(f"❌ Audio file not found: {audio_file}")
        return None
    
    try:
        # Create args object for asr_factory
        class Args:
            def __init__(self, model, language, backend, use_vad, min_chunk_size):
                self.model = model
                self.lan = language
                self.language = language
                self.backend = backend
                self.vad = use_vad
                self.vac = False  # Voice Activity Controller
                self.min_chunk_size = min_chunk_size
                self.vac_chunk_size = 0.04
                self.model_cache_dir = None
                self.model_dir = None
                self.task = "transcribe"
                self.buffer_trimming = "segment"
                self.buffer_trimming_sec = 15
                self.log_level = "INFO"
        
        args = Args(model_size, "en", backend, use_vad, min_chunk_size)
        
        # Initialize streaming components
        print("🚀 Initializing streaming components...")
        start_init = time.time()
        
        asr, online_processor = asr_factory(args, logfile=sys.stderr)
        
        init_time = time.time() - start_init
        print(f"✅ Initialization complete ({init_time:.2f}s)")
        
        # Load audio file info
        duration = len(load_audio(audio_file)) / 16000
        print(f"📊 Audio duration: {duration:.2f} seconds")
        print("")
        
        # Simulate real-time streaming
        print("🔄 Starting streaming simulation...")
        print("📝 Transcript updates:")
        print("-" * 40)
        
        start_time = time.time()
        beg = 0
        end = 0
        transcript_parts = []
        
        # Streaming loop (similar to whisper_online.py online mode)
        while True:
            # Calculate current time and wait if needed
            now = time.time() - start_time
            if now < end + min_chunk_size:
                time.sleep(min_chunk_size + end - now)
            
            end = time.time() - start_time
            
            # Load audio chunk
            if end > duration:
                end = duration
            
            if beg < duration:
                audio_chunk = load_audio_chunk(audio_file, beg, end)
                beg = end
                
                # Insert audio into streaming processor
                online_processor.insert_audio_chunk(audio_chunk)
                
                # Process and get streaming result
                try:
                    beg_timestamp, end_timestamp, transcript = online_processor.process_iter()
                    
                    # Output streaming result (like whisper_online.py format)
                    current_time = time.time() - start_time
                    if beg_timestamp is not None and transcript.strip():
                        print(f"{current_time*1000:8.1f}ms {beg_timestamp*1000:6.0f}-{end_timestamp*1000:6.0f}ms: {transcript}")
                        transcript_parts.append(transcript)
                    
                except Exception as e:
                    print(f"❌ Processing error: {e}")
                    continue
            
            if end >= duration:
                break
        
        # Get final result
        print("\n🏁 Getting final result...")
        try:
            beg_timestamp, end_timestamp, final_transcript = online_processor.finish()
            if final_transcript and final_transcript.strip():
                current_time = time.time() - start_time
                print(f"{current_time*1000:8.1f}ms {beg_timestamp*1000 if beg_timestamp else 0:6.0f}-{end_timestamp*1000 if end_timestamp else 0:6.0f}ms: {final_transcript}")
                if final_transcript not in transcript_parts:
                    transcript_parts.append(final_transcript)
        except Exception as e:
            print(f"⚠️ Final processing error: {e}")
        
        # Results summary
        total_time = time.time() - start_time
        print("")
        print("=" * 40)
        print("📊 STREAMING RESULTS")
        print("=" * 40)
        
        if transcript_parts:
            # Get the longest/most complete transcript
            full_transcript = transcript_parts[-1] if transcript_parts else ""
            print(f"📝 Final transcript: \"{full_transcript}\"")
            print(f"📊 Transcript parts: {len(transcript_parts)}")
            print(f"⏱️  Total processing time: {total_time:.2f}s")
            print(f"⚡ Real-time factor: {total_time / duration:.2f}x")
            
            # Show streaming progress
            print(f"🔄 Streaming updates:")
            for i, part in enumerate(transcript_parts, 1):
                print(f"   {i:2d}: \"{part[:60]}{'...' if len(part) > 60 else ''}\"")
            
            return full_transcript
        else:
            print("❌ No transcript generated")
            return None
            
    except Exception as e:
        print(f"❌ Streaming test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_different_configurations():
    """Test different streaming configurations"""
    print("🧪 Testing Different Streaming Configurations")
    print("=" * 50)
    
    configs = [
        {"model": "tiny.en", "backend": "faster-whisper", "chunk": 1.0, "vad": True},
        {"model": "tiny.en", "backend": "faster-whisper", "chunk": 0.5, "vad": True},
        {"model": "base.en", "backend": "faster-whisper", "chunk": 1.0, "vad": True},
        {"model": "tiny.en", "backend": "whisper_timestamped", "chunk": 1.0, "vad": True},
    ]
    
    results = []
    
    for i, config in enumerate(configs, 1):
        print(f"\n🔧 Configuration {i}/{len(configs)}:")
        print(f"   Model: {config['model']}")
        print(f"   Backend: {config['backend']}")
        print(f"   Chunk size: {config['chunk']}s")
        print(f"   VAD: {config['vad']}")
        
        try:
            start_time = time.time()
            transcript = simulate_streaming_transcription(
                audio_file="harvard.wav",
                model_size=config['model'],
                backend=config['backend'],
                min_chunk_size=config['chunk'],
                use_vad=config['vad']
            )
            test_time = time.time() - start_time
            
            if transcript:
                results.append({
                    "config": config,
                    "transcript": transcript,
                    "time": test_time,
                    "success": True
                })
                print(f"✅ SUCCESS ({test_time:.1f}s)")
            else:
                results.append({
                    "config": config,
                    "transcript": "",
                    "time": test_time,
                    "success": False
                })
                print(f"❌ FAILED ({test_time:.1f}s)")
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
            results.append({
                "config": config,
                "transcript": "",
                "time": 0,
                "success": False,
                "error": str(e)
            })
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 CONFIGURATION TEST SUMMARY")
    print("=" * 50)
    
    successful_configs = [r for r in results if r["success"]]
    
    if successful_configs:
        print(f"✅ Successful configurations: {len(successful_configs)}/{len(results)}")
        
        # Find fastest configuration
        fastest = min(successful_configs, key=lambda x: x["time"])
        print(f"⚡ Fastest: {fastest['config']['model']} ({fastest['config']['backend']}) - {fastest['time']:.1f}s")
        
        # Find best transcript (longest)
        best = max(successful_configs, key=lambda x: len(x["transcript"]))
        print(f"📝 Best transcript: {best['config']['model']} ({best['config']['backend']}) - {len(best['transcript'])} chars")
        
        print(f"\n🎯 Recommended config:")
        print(f"   Model: {fastest['config']['model']}")
        print(f"   Backend: {fastest['config']['backend']}")
        print(f"   Chunk size: {fastest['config']['chunk']}s")
        
    else:
        print("❌ No configurations worked successfully")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Streaming Whisper Backend")
    parser.add_argument("--audio", default="harvard.wav", help="Audio file to test")
    parser.add_argument("--model", default="tiny.en", help="Whisper model")
    parser.add_argument("--backend", default="faster-whisper", help="Backend to use")
    parser.add_argument("--chunk-size", type=float, default=1.0, help="Chunk size in seconds")
    parser.add_argument("--no-vad", action="store_true", help="Disable VAD")
    parser.add_argument("--test-configs", action="store_true", help="Test multiple configurations")
    
    args = parser.parse_args()
    
    if args.test_configs:
        test_different_configurations()
    else:
        transcript = simulate_streaming_transcription(
            audio_file=args.audio,
            model_size=args.model,
            backend=args.backend,
            min_chunk_size=args.chunk_size,
            use_vad=not args.no_vad
        )
        
        if transcript:
            print(f"\n🎯 Final result: \"{transcript}\"")
        else:
            print("\n❌ Test failed - no transcript generated")