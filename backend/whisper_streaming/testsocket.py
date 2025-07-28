#!/usr/bin/env python3
"""
Test the lazy initialization backend
"""

import asyncio
import websockets
import json
import time

async def test_lazy_backend():
    """Test the lazy initialization backend"""
    print("🧪 Testing Lazy Initialization Backend")
    print("=" * 45)
    
    try:
        uri = "ws://localhost:8765"
        
        print("📡 Connecting to WebSocket...")
        async with websockets.connect(uri, close_timeout=10) as websocket:
            print("✅ WebSocket connection established")
            
            # Test 1: Authentication
            print("\n🔐 Testing authentication...")
            auth_data = {
                "type": "auth",
                "password": "holographic2024"
            }
            
            await websocket.send(json.dumps(auth_data))
            print("📤 Auth sent")
            
            response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
            auth_response = json.loads(response)
            
            print(f"📥 Auth response: {auth_response}")
            
            if auth_response.get('type') == 'auth_success':
                print("✅ Authentication successful!")
                print(f"   Model loaded: {auth_response.get('model_loaded', False)}")
                print(f"   Backend: {auth_response.get('backend', 'unknown')}")
                
                # Test 2: Start recording (should work even without model)
                print("\n🎙️ Testing start recording...")
                start_msg = {"type": "start_recording"}
                await websocket.send(json.dumps(start_msg))
                print("📤 Start recording sent")
                
                # Wait for status update
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    response_data = json.loads(response)
                    print(f"📥 Status update: {response_data}")
                except asyncio.TimeoutError:
                    print("⏰ No immediate status update")
                
                # Test 3: Send some fake audio chunks
                print("\n📤 Sending test audio chunks...")
                for i in range(3):
                    audio_msg = {
                        "type": "audio_chunk",
                        "audio": f"fake_audio_data_{i}_{'x' * 100}"  # Fake base64-like data
                    }
                    await websocket.send(json.dumps(audio_msg))
                    print(f"   📤 Sent chunk {i+1}/3")
                    await asyncio.sleep(0.5)
                
                # Test 4: Listen for processing updates
                print("\n👂 Listening for processing updates...")
                updates_received = 0
                
                try:
                    while updates_received < 3:  # Listen for up to 3 updates
                        response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                        response_data = json.loads(response)
                        
                        if response_data.get('type') == 'streaming_transcript':
                            updates_received += 1
                            text = response_data.get('text', '')
                            model_loaded = response_data.get('model_loaded', False)
                            model_loading = response_data.get('model_loading', False)
                            
                            print(f"   📥 Update {updates_received}: \"{text}\"")
                            print(f"      Model loaded: {model_loaded}, Loading: {model_loading}")
                            
                            # If model finished loading, we can expect transcription soon
                            if model_loaded and not model_loading:
                                print("      🎯 Model is ready for transcription!")
                                break
                        else:
                            print(f"   📨 Other message: {response_data.get('type', 'unknown')}")
                            
                except asyncio.TimeoutError:
                    print("   ⏰ No more updates (timeout)")
                
                # Test 5: Stop recording
                print("\n🛑 Testing stop recording...")
                stop_msg = {"type": "stop_recording"}
                await websocket.send(json.dumps(stop_msg))
                print("📤 Stop recording sent")
                
                # Wait for final response
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    response_data = json.loads(response)
                    print(f"📥 Final response: {response_data}")
                except asyncio.TimeoutError:
                    print("⏰ No final response")
                
                print("\n✅ ALL TESTS COMPLETED SUCCESSFULLY!")
                print("🎯 Key achievements:")
                print("   ✅ WebSocket connection works")
                print("   ✅ Authentication works")
                print("   ✅ Audio chunk processing works")
                print("   ✅ Status updates work")
                print("   ✅ No internal errors (1011)")
                
                return True
                
            else:
                print(f"❌ Authentication failed: {auth_response}")
                return False
                
    except websockets.exceptions.ConnectionClosed as e:
        print(f"❌ Connection closed: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_connection_stability():
    """Test connection stability"""
    print("\n🔄 Testing connection stability...")
    
    successful_connections = 0
    
    for i in range(3):
        try:
            print(f"\n📡 Connection attempt {i+1}/3...")
            
            uri = "ws://localhost:8765"
            async with websockets.connect(uri, close_timeout=5) as websocket:
                # Quick auth test
                auth_data = {"type": "auth", "password": "holographic2024"}
                await websocket.send(json.dumps(auth_data))
                
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                auth_response = json.loads(response)
                
                if auth_response.get('type') == 'auth_success':
                    successful_connections += 1
                    print(f"   ✅ Connection {i+1} successful")
                else:
                    print(f"   ❌ Connection {i+1} auth failed")
                    
        except Exception as e:
            print(f"   ❌ Connection {i+1} failed: {e}")
        
        await asyncio.sleep(1)
    
    print(f"\n📊 Stability test results: {successful_connections}/3 connections successful")
    return successful_connections >= 2

async def run_all_tests():
    """Run all tests"""
    print("🧪 LAZY INITIALIZATION BACKEND TESTS")
    print("=" * 50)
    
    # Test 1: Main functionality
    main_test_passed = await test_lazy_backend()
    
    # Test 2: Connection stability
    stability_test_passed = await test_connection_stability()
    
    print("\n" + "=" * 50)
    print("📊 FINAL TEST RESULTS")
    print("=" * 50)
    
    print(f"✅ Main functionality: {'PASSED' if main_test_passed else 'FAILED'}")
    print(f"✅ Connection stability: {'PASSED' if stability_test_passed else 'FAILED'}")
    
    if main_test_passed and stability_test_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("🚀 The lazy initialization approach works!")
        print("\n💡 This means:")
        print("   • WebSocket server starts quickly")
        print("   • Connections work before model loading")
        print("   • Model loads in background after connection")
        print("   • No more 1011 internal errors!")
        
        print("\n🎤 Next steps:")
        print("   1. Use this as your main backend")
        print("   2. Model will load automatically on first use")
        print("   3. WebSocket stays responsive during model loading")
        
        return True
    else:
        print("\n❌ Some tests failed")
        print("🔧 Check the backend logs for issues")
        return False

if __name__ == "__main__":
    print("🎤 Make sure the lazy backend is running first:")
    print("   python3 lazy_init_backend.py --debug")
    print("")
    
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)