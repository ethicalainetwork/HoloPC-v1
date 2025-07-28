#!/usr/bin/env python3
"""
Compare the original simple backend vs optimized streaming backend
Shows the difference in update frequency and responsiveness
"""

import time
import numpy as np
import soundfile as sf
import os

# Force CPU mode
os.environ['CUDA_VISIBLE_DEVICES'] = ''

def simulate_audio_streaming(audio_file="harvard.wav", chunk_duration=0.1):
    """Simulate real-time audio streaming by yielding chunks"""
    if not os.path.exists(audio_file):
        print(f"❌ Audio file not found: {audio_file}")
        return
    
    # Load audio
    audio_data, sample_rate = sf.read(audio_file)
    
    # Convert to mono if needed
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    # Resample to 16kHz if needed
    if sample_rate != 16000:
        audio_data = np.interp(
            np.linspace(0, len(audio_data), int(len(audio_data) * 16000 / sample_rate)),
            np.arange(len(audio_data)),
            audio_data
        )
        sample_rate = 16000
    
    # Yield audio chunks in real-time
    chunk_size = int(chunk_duration * sample_rate)
    total_duration = len(audio_data) / sample_rate
    
    print(f"🎵 Simulating {total_duration:.1f}s audio stream in {chunk_duration*1000:.0f}ms chunks")
    print("")
    
    start_time = time.time()
    position = 0
    
    while position < len(audio_data):
        # Get next chunk
        end_position = min(position + chunk_size, len(audio_data))
        chunk = audio_data[position:end_position]
        
        # Calculate timing
        current_time = position / sample_rate
        elapsed_real_time = time.time() - start_time
        
        # Wait for real-time if we're ahead
        if elapsed_real_time < current_time:
            time.sleep(current_time - elapsed_real_time)
        
        yield chunk, current_time
        position = end_position

def test_original_approach():
    """Test the original 5-second chunk approach"""
    print("🔵 TESTING ORIGINAL APPROACH (5-second chunks)")
    print("=" * 50)
    
    buffer = np.array([], dtype=np.float32)
    updates = []
    last_update_time = time.time()
    
    start_time = time.time()
    
    for chunk, current_time in simulate_audio_streaming():
        buffer = np.concatenate([buffer, chunk])
        buffer_duration = len(buffer) / 16000
        
        # Original logic: process every 5 seconds AND wait at least 2 seconds
        time_since_last = time.time() - last_update_time
        
        if buffer_duration >= 5.0 and time_since_last >= 2.0:
            processing_time = time.time() - start_time
            
            # Simulate transcription (would be actual Whisper call)
            mock_transcript = f"Transcript at {processing_time:.1f}s (buffer: {buffer_duration:.1f}s)"
            
            updates.append({
                'time': processing_time,
                'transcript': mock_transcript,
                'buffer_duration': buffer_duration
            })
            
            print(f"{processing_time:6.1f}s: {mock_transcript}")
            
            # Keep 1 second overlap
            overlap_samples = int(1.0 * 16000)
            if len(buffer) > overlap_samples:
                buffer = buffer[-overlap_samples:]
            
            last_update_time = time.time()
    
    total_time = time.time() - start_time
    print(f"\n📊 Original Results: {len(updates)} updates in {total_time:.1f}s")
    print(f"   Average interval: {total_time/len(updates) if updates else 0:.1f}s between updates")
    return updates

def test_optimized_approach():
    """Test the optimized streaming approach"""
    print("\n🟢 TESTING OPTIMIZED APPROACH (0.5-second updates)")
    print("=" * 50)
    
    buffer = np.array([], dtype=np.float32)
    updates = []
    last_update_time = time.time()
    
    start_time = time.time()
    
    for chunk, current_time in simulate_audio_streaming():
        buffer = np.concatenate([buffer, chunk])
        buffer_duration = len(buffer) / 16000
        
        # Optimized logic: process every 0.5 seconds with 1 second minimum
        time_since_last = time.time() - last_update_time
        
        should_process = False
        
        # Process if we have 1+ seconds and 0.5s has passed
        if buffer_duration >= 1.0 and time_since_last >= 0.5:
            should_process = True
        # Or if buffer gets too long (10s max)
        elif buffer_duration >= 10.0:
            should_process = True
        
        if should_process:
            processing_time = time.time() - start_time
            
            # Simulate transcription
            mock_transcript = f"Streaming update at {processing_time:.1f}s (buffer: {buffer_duration:.1f}s)"
            
            updates.append({
                'time': processing_time,
                'transcript': mock_transcript,
                'buffer_duration': buffer_duration
            })
            
            print(f"{processing_time:6.1f}s: {mock_transcript}")
            
            # Keep 0.5 second overlap
            overlap_samples = int(0.5 * 16000)
            if len(buffer) > overlap_samples:
                buffer = buffer[-overlap_samples:]
            
            last_update_time = time.time()
    
    total_time = time.time() - start_time
    print(f"\n📊 Optimized Results: {len(updates)} updates in {total_time:.1f}s")
    print(f"   Average interval: {total_time/len(updates) if updates else 0:.1f}s between updates")
    return updates

def compare_approaches():
    """Compare both approaches side by side"""
    print("🎤 STREAMING APPROACH COMPARISON")
    print("=" * 60)
    
    original_updates = test_original_approach()
    optimized_updates = test_optimized_approach()
    
    print("\n" + "=" * 60)
    print("📊 COMPARISON SUMMARY")
    print("=" * 60)
    
    print(f"📈 Update Frequency:")
    print(f"   Original:  {len(original_updates)} updates")
    print(f"   Optimized: {len(optimized_updates)} updates")
    print(f"   Improvement: {len(optimized_updates) - len(original_updates):+d} more updates")
    
    if original_updates and optimized_updates:
        orig_interval = (original_updates[-1]['time'] - original_updates[0]['time']) / (len(original_updates) - 1) if len(original_updates) > 1 else 0
        opt_interval = (optimized_updates[-1]['time'] - optimized_updates[0]['time']) / (len(optimized_updates) - 1) if len(optimized_updates) > 1 else 0
        
        print(f"\n⏱️  Update Intervals:")
        print(f"   Original:  {orig_interval:.1f}s average between updates")
        print(f"   Optimized: {opt_interval:.1f}s average between updates")
        
        if opt_interval > 0:
            improvement = (orig_interval / opt_interval)
            print(f"   Improvement: {improvement:.1f}x more frequent updates")
    
    print(f"\n🎯 Streaming Quality:")
    print(f"   Original:  Large chunks, less responsive, fewer updates")
    print(f"   Optimized: Small chunks, more responsive, frequent updates")
    
    print(f"\n💡 User Experience:")
    print(f"   Original:  User waits 5+ seconds for each update")
    print(f"   Optimized: User sees updates every 0.5-1 seconds")

def show_streaming_benefits():
    """Show the benefits of optimized streaming"""
    print("\n🎯 STREAMING OPTIMIZATION BENEFITS")
    print("=" * 50)
    
    print("📱 Original Simple Backend:")
    print("   • Processes every 5 seconds")
    print("   • Waits 2+ seconds between updates")
    print("   • 3-4 updates for 18-second audio")
    print("   • User sees: [wait 5s] → text → [wait 5s] → more text")
    
    print("\n⚡ Optimized Streaming Backend:")
    print("   • Processes every 0.5 seconds")
    print("   • Minimum 1 second audio buffer")
    print("   • 15-20+ updates for 18-second audio")
    print("   • User sees: text → [0.5s] → more text → [0.5s] → even more")
    
    print("\n🎬 Real-world Impact:")
    print("   • Word-by-word appearance (like live captions)")
    print("   • Immediate feedback when speaking")
    print("   • Smooth, progressive text building")
    print("   • Professional streaming experience")
    
    print("\n⚙️ Technical Improvements:")
    print("   • Context prompts for better accuracy")
    print("   • VAD filtering for cleaner audio")
    print("   • Overlap buffering for continuity")
    print("   • Performance statistics tracking")
    print("   • Smart update detection (avoid spam)")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare streaming approaches")
    parser.add_argument("--audio", default="harvard.wav", help="Audio file to test")
    parser.add_argument("--show-benefits", action="store_true", help="Show streaming benefits")
    
    args = parser.parse_args()
    
    if args.show_benefits:
        show_streaming_benefits()
    else:
        compare_approaches()
    
    print(f"\n🚀 To use the optimized version:")
    print(f"   python3 optimized_streaming_simple.py --model tiny.en")
    print(f"   # Then open: http://localhost:3000/speech")