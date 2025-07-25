#! /bin/bash  
screen -dmS fe bash -c 'npm run build && node server/server.js; exec sh' 
screen -dmS whisper bash -c 'source "bin/activate" && python3 server/whisper_streaming/optimized_streaming_simple.py --model tiny.en; exec sh' 


