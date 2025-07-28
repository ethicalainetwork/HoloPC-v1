screen -dmS whisper bash -c 'source "bin/activate" && python3 backend/whisper_streaming/transcribe.py --model tiny.en; exec sh' 

screen -dmS fe bash -c 'cd frontend && npm run build && node server/server.js; exec sh' 
