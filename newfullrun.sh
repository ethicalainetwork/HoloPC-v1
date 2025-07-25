screen -dmS whisper bash -c 'source "bin/activate" && python3 server/whisper_streaming/transcribe.py --model tiny.en; exec sh' 

screen -dmS fe bash -c 'npm run build && node server/server.js; exec sh' 
