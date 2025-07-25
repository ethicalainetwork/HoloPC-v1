# Speech Processor with Translation Setup Guide


# Install LibreTranslate
pip install libretranslate

# Start LibreTranslate service (in one terminal)
libretranslate

# Start speech processor with Chinese model and English translation (in another terminal)
python3 optimized_streaming_simple2.py --model small.zh --language zh --translate-to en




## 1. Install LibreTranslate

First, install LibreTranslate for translation services:

```bash
pip install libretranslate
```

## 2. Start LibreTranslate Service

In a separate terminal window, start the LibreTranslate service:

```bash
libretranslate
```

This will start the service on `http://localhost:5000` by default.

### Test the Translation Service

You can test it with curl:

```bash
# Test Chinese to English translation
curl -X POST http://localhost:5000/translate \
  -d "q=你好世界" \
  -d "source=zh" \
  -d "target=en"

# Expected response: {"translatedText":"Hello world"}
```

## 3. Usage Examples

### Example 1: Chinese Speech to English Text

For Chinese speech recognition with English translation:

```bash
python speech_processor.py \
  --model small.zh \
  --language zh \
  --translate-to en
```

### Example 2: Spanish Speech to English Text

```bash
python speech_processor.py \
  --model small.es \
  --language es \
  --translate-to en
```

### Example 3: English Speech (No Translation)

```bash
python speech_processor.py \
  --model tiny.en \
  --language en
```

### Example 4: With Custom LibreTranslate Server

If running LibreTranslate on a different server:

```bash
python speech_processor.py \
  --model small.zh \
  --language zh \
  --translate-to en \
  --libretranslate-url http://your-server:5000
```

### Example 5: With API Key (For LibreTranslate API)

```bash
python speech_processor.py \
  --model small.zh \
  --language zh \
  --translate-to en \
  --libretranslate-api-key your-api-key-here
```

## 4. Whisper Model Recommendations

| Source Language | Recommended Model | Model Code |
|----------------|-------------------|------------|
| Chinese | `small.zh` or `medium.zh` | `zh` |
| Spanish | `small.es` or `medium.es` | `es` |
| French | `small.fr` or `medium.fr` | `fr` |
| German | `small.de` or `medium.de` | `de` |
| Japanese | `small.ja` or `medium.ja` | `ja` |
| Korean | `small.ko` or `medium.ko` | `ko` |
| Russian | `small.ru` or `medium.ru` | `ru` |
| Any Language | `small` or `medium` | `auto` |

## 5. Language Codes for Translation

Common language codes for the `--translate-to` parameter:

| Language | Code |
|----------|------|
| English | `en` |
| Spanish | `es` |
| French | `fr` |
| German | `de` |
| Chinese (Simplified) | `zh` |
| Japanese | `ja` |
| Korean | `ko` |
| Russian | `ru` |
| Portuguese | `pt` |
| Italian | `it` |
| Dutch | `nl` |
| Arabic | `ar` |

## 6. WebSocket Message Format

The enhanced processor sends messages with both original and translated text:

```json
{
  "type": "streaming_transcript",
  "original_text": "你好，我的名字是李明",
  "translated_text": "Hello, my name is Li Ming",
  "display_text": "Hello, my name is Li Ming",
  "translation_enabled": true,
  "source_language": "zh",
  "target_language": "en",
  "timestamp": 1234567890.123
}
```

## 7. Features

✅ **Lazy Model Loading**: Whisper model loads only when needed  
✅ **Real-time Translation**: Translates speech as it's processed  
✅ **Multiple Backends**: Supports both faster-whisper and openai-whisper  
✅ **Language Detection**: Works with language-specific models  
✅ **Fallback Support**: Continues working even if translation fails  
✅ **WebSocket API**: Real-time updates to connected clients  
✅ **Holographic Display**: Updates your display server with translated text  

## 8. Troubleshooting

### LibreTranslate Not Starting

```bash
# Check if port 5000 is available
lsof -i :5000

# If needed, start on different port
libretranslate --port 5001
```

Then use `--libretranslate-url http://localhost:5001`

### Translation Service Unavailable

The processor will continue working with transcription only if translation fails.

### Model Loading Issues

For Chinese:
```bash
# Download model manually if needed
python -c "import whisper; whisper.load_model('small.zh')"
```

## 9. Performance Tips

- Use `small.*` models for good accuracy with reasonable speed
- Use `tiny.*` models for fastest processing
- Use `medium.*` or `large.*` models for best accuracy (slower)
- LibreTranslate runs on CPU by default - consider GPU setup for heavy usage

## 10. Example Complete Setup

```bash
# Terminal 1: Start LibreTranslate
libretranslate

# Terminal 2: Start Speech Processor (Chinese → English)
python speech_processor.py \
  --model small.zh \
  --language zh \
  --translate-to en \
  --debug

# The processor will:
# 1. Start WebSocket server immediately
# 2. Load Whisper model when first audio arrives
# 3. Transcribe Chinese speech
# 4. Translate to English via LibreTranslate
# 5. Send both original and translated text to clients
# 6. Update holographic display with translated text
```