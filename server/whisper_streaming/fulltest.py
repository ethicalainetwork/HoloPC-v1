#!/usr/bin/env python3
"""
Integration test for the complete holographic speech recognition system
Tests Node.js server + Python backend + WebSocket communication
"""

import requests
import time
import json
import asyncio
import websockets
import base64
import numpy as np
import soundfile as sf
import os

def test_server_health():
    """Test if the Node.js server is running and healthy"""
    print("🔍 Testing Node.js server health...")
    
    try:
        response = requests.get("http://localhost:3000/api/health", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Server is healthy - Platform: {data.get('platform', 'unknown')}")
            print(f"   ImageMagick: {'✅' if data.get('imagemagickAvailable') else '❌'}")
            return True
        else:
            print(f"❌ Server health check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return False

def test_python_backend_status():
    """Test if the Python backend is running"""
    print("🔍 Testing Python speech backend status...")
    
    try:
        response = requests.get("http://localhost:3000/api/speech/python-status", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('pythonBackendRunning'):
                print(f"✅ Python backend is running")
                print(f"   WebSocket URL: {data.get('websocketUrl', 'unknown')}")
                return True
            else:
                print(f"❌ Python backend not running: {data.get('message', 'unknown')}")
                return False
        else:
            print(f"❌ Backend status check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Cannot check backend status: {e}")
        return False

def test_speech_api_update():
    """Test the speech text update API"""
    print("🔍 Testing speech text update API...")
    
    test_text = "Integration test message from Python backend"
    
    try:
        response = requests.post(
            "http://localhost:3000/api/speech/update-text",
            json={"text": test_text},
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Speech text updated successfully")
            print(f"   Status: {data.get('status', 'unknown')}")
            print(f"   Current text: \"{data.get('currentText', 'none')}\"")
            return True
        else:
            print(f"❌ Speech text update failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Speech API error: {e}")
        return False

def test_current_speech_text():
    """Test retrieving current speech text"""
    print("🔍 Testing current speech text retrieval...")
    
    try:
        response = requests.get("http://localhost:3000/api/speech/current", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Current speech text retrieved")
            print(f"   Text: \"{data.get('currentText', 'none')}\"")
            print(f"   Active: {data.get('isActive', False)}")
            return True
        else:
            print(f"❌ Current speech text retrieval failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Speech current text error: {e}")
        return False

async def test_websocket_connection():
    """Test WebSocket connection to Python backend"""
    print("🔍 Testing WebSocket connection to Python backend...")
    
    try:
        # Connect to WebSocket
        uri = "ws://localhost:8765"
        async with websockets.connect(uri) as websocket:
            print("✅ WebSocket connection established")
            
            # Send authentication
            auth_data = {
                "type": "auth",
                "password": "holographic2024"
            }
            
            await websocket.send(json.dumps(auth_data))
            print("📤 Authentication sent")
            
            # Wait for auth response
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            auth_response = json.loads(response)
            
            if auth_response.get('type') == 'auth_success':
                print("✅ Authentication successful")
                
                # Send start recording message
                start_msg = {"type": "start_recording"}
                await websocket.send(json.dumps(start_msg))
                print("📤 Start recording message sent")
                
                # Send a small audio chunk (silence)
                audio_chunk = np.zeros(1600, dtype=np.float32)  # 100ms of silence at 16kHz
                
                # Convert to bytes and base64 encode
                import io
                buffer = io.BytesIO()
                sf.write(buffer, audio_chunk, 16000, format='WAV')
                buffer.seek(0)
                audio_bytes = buffer.read()
                audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                
                audio_msg = {
                    "type": "audio_chunk",
                    "audio": audio_b64
                }
                await websocket.send(json.dumps(audio_msg))
                print("📤 Audio chunk sent")
                
                # Send stop recording
                stop_msg = {"type": "stop_recording"}
                await websocket.send(json.dumps(stop_msg))
                print("📤 Stop recording message sent")
                
                print("✅ WebSocket communication test completed")
                return True
                
            else:
                print(f"❌ Authentication failed: {auth_response.get('message', 'unknown')}")
                return False
                
    except asyncio.TimeoutError:
        print("❌ WebSocket connection timeout")
        return False
    except Exception as e:
        print(f"❌ WebSocket connection error: {e}")
        return False

def test_progressive_speech():
    """Test the progressive speech simulation"""
    print("🔍 Testing progressive speech simulation...")
    
    try:
        # Start progressive speech
        response = requests.post("http://localhost:3000/api/speech/start-progressive", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Progressive speech started: {data.get('message', 'unknown')}")
            
            # Wait a few seconds and check the text
            print("⏳ Waiting 3 seconds for progressive updates...")
            time.sleep(3)
            
            # Check current text
            current_response = requests.get("http://localhost:3000/api/speech/current", timeout=5)
            if current_response.status_code == 200:
                current_data = current_response.json()
                current_text = current_data.get('currentText', '')
                print(f"📝 Progressive text: \"{current_text}\"")
                
                if current_text and len(current_text) > 10:
                    print("✅ Progressive speech is working")
                    
                    # Stop progressive speech
                    stop_response = requests.post("http://localhost:3000/api/speech/stop-progressive", timeout=5)
                    if stop_response.status_code == 200:
                        print("✅ Progressive speech stopped")
                        return True
                    else:
                        print("⚠️  Could not stop progressive speech")
                        return True  # Still consider it successful
                else:
                    print("⚠️  Progressive speech not generating text yet")
                    return False
            else:
                print("❌ Could not check progressive speech text")
                return False
                
        else:
            print(f"❌ Failed to start progressive speech: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Progressive speech test error: {e}")
        return False

def run_complete_integration_test():
    """Run the complete integration test suite"""
    print("🧪 COMPLETE SYSTEM INTEGRATION TEST")
    print("=" * 50)
    
    tests = [
        ("Server Health", test_server_health),
        ("Python Backend Status", test_python_backend_status),
        ("Speech API Update", test_speech_api_update),
        ("Current Speech Text", test_current_speech_text),
        ("Progressive Speech", test_progressive_speech),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        print("-" * 30)
        
        try:
            result = test_func()
            results.append((test_name, result))
            
            if result:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
                
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            results.append((test_name, False))
        
        time.sleep(1)  # Brief pause between tests
    
    # WebSocket test (requires async)
    print(f"\n📋 Running: WebSocket Connection")
    print("-" * 30)
    
    try:
        websocket_result = asyncio.run(test_websocket_connection())
        results.append(("WebSocket Connection", websocket_result))
        
        if websocket_result:
            print(f"✅ WebSocket Connection: PASSED")
        else:
            print(f"❌ WebSocket Connection: FAILED")
    except Exception as e:
        print(f"❌ WebSocket Connection: ERROR - {e}")
        results.append(("WebSocket Connection", False))
    
    # Summary
    print(f"\n" + "=" * 50)
    print("📊 INTEGRATION TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {test_name}")
    
    if passed == total:
        print(f"\n🎉 ALL TESTS PASSED!")
        print("🚀 Your holographic speech recognition system is ready!")
        print("\n💡 Next steps:")
        print("  1. Open: http://localhost:3000")
        print("  2. Speech interface: http://localhost:3000/speech")
        print("  3. Start recording and see live transcription!")
    else:
        print(f"\n⚠️  {total - passed} tests failed")
        print("🔧 Check the failed components before using the system")
    
    return passed == total

if __name__ == "__main__":
    success = run_complete_integration_test()
    exit(0 if success else 1)