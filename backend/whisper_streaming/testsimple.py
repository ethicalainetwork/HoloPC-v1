#!/usr/bin/env python3
"""
Test script for the Simple Whisper Backend
Tests transcription with harvard.wav or any audio file
"""

import sys
import os
import numpy as np
import time
import json

# Add the current directory to path so we can import our backend
sys.path.append('.')

try:
    import soundfile as sf
    import requests
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("Please install: pip install soundfile requests numpy")
    sys.exit(1)

# Try to import our simplified backend
try:
    from simple_whisper_backend import SimpleSpeechProcessor
    print("✅ Successfully imported SimpleSpeechProcessor")
except ImportError as e:
    print(f"❌ Could not import SimpleSpeechProcessor: {e}")
    print("Make sure simple_whisper_backend.py is in the current directory")
    sys.exit(1)

def test_direct_transcription(audio_file="harvard.wav", model_size="tiny.en"):
    """Test direct transcription without WebSocket server"""
    print(f"🧪 Testing direct transcription with {audio_file}")
    print("=" * 50)
    
    try:
        # Check if audio file exists
        if not os.path.exists(audio_file):
            print(f"❌ Audio file not found: {audio_file}")
            
            # Create a test audio file
            print("🎵 Creating test audio file...")
            sample_rate = 16000
            duration = 5
            t = np.linspace(0, duration, sample_rate * duration, False)
            # Create a simple tone + noise for testing
            audio = 0.1 * np.sin(2 * np.pi * 440 * t) + 0.05 * np.random.randn(len(t))
            sf.write("test_audio.wav", audio, sample_rate)
            audio_file = "test_audio.wav"
            print(f"✅ Created test audio: {audio_file}")
        
        # Load and check the audio file
        print(f"📂 Loading audio file: {audio_file}")
        audio_data, sample_rate = sf.read(audio_file)
        
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        print(f"📊 Audio info:")
        print(f"   Duration: {len(audio_data) / sample_rate:.2f} seconds")
        print(f"   Sample rate: {sample_rate} Hz")
        print(f"   Samples: {len(audio_data)}")
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            print(f"🔄 Resampling from {sample_rate}Hz to 16kHz...")
            # Simple resampling
            target_length = int(len(audio_data) * 16000 / sample_rate)
            audio_data = np.interp(
                np.linspace(0, len(audio_data), target_length),
                np.arange(len(audio_data)),
                audio_data
            )
            sample_rate = 16000
        
        # Initialize the speech processor
        print(f"🎤 Initializing speech processor with model: {model_size}")
        processor = SimpleSpeechProcessor(
            model_size=model_size,
            language="en",
            server_url="http://localhost:3000",
            websocket_port=8765
        )
        
        print("✅ Speech processor initialized successfully")
        
        # Test transcription
        print("🔊 Starting transcription...")
        start_time = time.time()
        
        transcript = processor.transcribe_audio(audio_data.astype(np.float32))
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print("\n" + "=" * 50)
        print("🎯 TRANSCRIPTION RESULTS")
        print("=" * 50)
        print(f"📝 Transcript: \"{transcript}\"")
        print(f"⏱️  Processing time: {processing_time:.2f} seconds")
        print(f"⚡ Real-time factor: {processing_time / (len(audio_data) / sample_rate):.2f}x")
        
        if transcript.strip():
            print("✅ SUCCESS: Transcription completed!")
        else:
            print("⚠️  WARNING: Empty transcription result")
        
        # Clean up test file if we created it
        if audio_file == "test_audio.wav" and os.path.exists("test_audio.wav"):
            os.remove("test_audio.wav")
            print("🧹 Cleaned up test audio file")
        
        return transcript
        
    except Exception as e:
        print(f"❌ ERROR during transcription test: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_server_integration(transcript_text="This is a test message"):
    """Test integration with the holographic display server"""
    print(f"\n🌐 Testing server integration")
    print("=" * 50)
    
    try:
        # Test connection to holographic display server
        print("🔍 Checking if holographic display server is running...")
        
        response = requests.get("http://localhost:3000/api/health", timeout=3)
        
        if response.status_code == 200:
            print("✅ Holographic display server is running")
            
            # Test sending transcript
            print("📤 Sending test transcript to server...")
            
            update_response = requests.post(
                "http://localhost:3000/api/speech/update-text",
                json={"text": transcript_text},
                timeout=3
            )
            
            if update_response.status_code == 200:
                result = update_response.json()
                print(f"✅ Server update successful: {result.get('message', 'OK')}")
                print(f"📝 Current text on server: \"{result.get('currentText', 'N/A')}\"")
                return True
            else:
                print(f"❌ Server update failed: {update_response.status_code}")
                return False
                
        else:
            print(f"❌ Server health check failed: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to holographic display server")
        print("💡 Make sure the server is running: node server.js")
        return False
    except Exception as e:
        print(f"❌ Server integration error: {e}")
        return False

def run_full_test():
    """Run complete test suite"""
    print("🚀 Starting Simple Whisper Backend Test Suite")
    print("=" * 60)
    
    # Test 1: Direct transcription
    print("TEST 1: Direct Transcription")
    transcript = test_direct_transcription("harvard.wav", "tiny.en")
    
    if transcript:
        # Test 2: Server integration
        print("\nTEST 2: Server Integration")
        server_ok = test_server_integration(transcript)
        
        # Summary
        print("\n" + "=" * 60)
        print("📋 TEST SUMMARY")
        print("=" * 60)
        print(f"✅ Direct transcription: {'PASSED' if transcript else 'FAILED'}")
        print(f"{'✅' if server_ok else '❌'} Server integration: {'PASSED' if server_ok else 'FAILED'}")
        
        if transcript and server_ok:
            print("\n🎉 ALL TESTS PASSED!")
            print("🎯 Your simple backend is ready for use!")
            print("\n💡 Next steps:")
            print("1. Start the backend: python3 simple_whisper_backend.py")
            print("2. Open the speech interface: http://localhost:3000/speech")
            print("3. Enter password: holographic2024")
            print("4. Start recording and see live transcription!")
        else:
            print("\n⚠️  Some tests failed. Check the output above for details.")
    else:
        print("\n❌ Transcription test failed. Cannot proceed with server test.")

def test_with_custom_audio(audio_file):
    """Test with a custom audio file"""
    print(f"🎵 Testing with custom audio file: {audio_file}")
    
    if not os.path.exists(audio_file):
        print(f"❌ Audio file not found: {audio_file}")
        return None
    
    return test_direct_transcription(audio_file)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Simple Whisper Backend")
    parser.add_argument("--audio", help="Audio file to test with (default: harvard.wav)")
    parser.add_argument("--model", default="tiny.en", help="Whisper model to use")
    parser.add_argument("--test-server", action="store_true", help="Test server integration only")
    parser.add_argument("--full-test", action="store_true", help="Run full test suite")
    
    args = parser.parse_args()
    
    if args.test_server:
        # Test server integration only
        test_server_integration("Test message from backend")
    elif args.full_test:
        # Run full test suite
        run_full_test()
    elif args.audio:
        # Test with custom audio file
        test_with_custom_audio(args.audio)
    else:
        # Default: test transcription with harvard.wav
        transcript = test_direct_transcription(args.audio or "harvard.wav", args.model)
        if transcript:
            print(f"\n🎯 Final result: \"{transcript}\"")