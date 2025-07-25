const express = require('express');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');
const cors = require('cors');

const app = express();
const PORT = process.env.PORT || 3000;

// Trust proxy headers (optional, but useful for HTTPS setups)
app.set('trust proxy', true);

// Secure CORS config
const allowedOrigins = [
  'http://localhost:4200',
  'http://localhost:3000',
  'https://holopc.nla.la'
];

app.use(cors({
  origin: function (origin, callback) {
    if (!origin || allowedOrigins.includes(origin)) {
      callback(null, true);
    } else {
      callback(new Error('CORS not allowed from this origin'));
    }
  },
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization'],
  credentials: true
}));

// Middleware
app.use(express.json());
app.use(express.static(path.join(__dirname, 'dist/holodisplay/')));

// Quality presets
const QUALITY_PRESETS = {
  low: { width: 480, height: 270 },
  medium: { width: 720, height: 405 },
  high: { width: 1080, height: 607 }
};

// Speech-to-text storage - REAL SPEECH ONLY (no demo)
let currentSpeechText = '';
let speechHistory = [];
let speechRecognitionProcess = null;
const MAX_SPEECH_HISTORY = 10;

// Real-time speech state
let isRealSpeechActive = false;
let lastSpeechUpdate = 0;

/**
 * Capture screen using system commands (cross-platform)
 */
async function captureScreenWithSystemCommand(quality = 'medium') {
  const preset = QUALITY_PRESETS[quality];
  const timestamp = Date.now();
  const filename = `screenshot_${timestamp}.png`;
  const outputPath = path.join(__dirname, 'temp', filename);
  
  // Ensure temp directory exists
  const tempDir = path.join(__dirname, 'temp');
  if (!fs.existsSync(tempDir)) {
    fs.mkdirSync(tempDir, { recursive: true });
  }

  return new Promise((resolve, reject) => {
    let command, args;
    
    // Detect operating system and use appropriate screenshot command
    if (process.platform === 'win32') {
      // Windows - using PowerShell
      command = 'powershell';
      args = [
        '-Command',
        `Add-Type -AssemblyName System.Windows.Forms; Add-Type -AssemblyName System.Drawing; $bounds = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds; $bmp = New-Object System.Drawing.Bitmap $bounds.Width, $bounds.Height; $graphics = [System.Drawing.Graphics]::FromImage($bmp); $graphics.CopyFromScreen($bounds.X, $bounds.Y, 0, 0, $bounds.Size); $bmp.Save('${outputPath}'); $graphics.Dispose(); $bmp.Dispose()`
      ];
    } else if (process.platform === 'darwin') {
      // macOS
      command = 'screencapture';
      args = ['-x', outputPath];
    } else {
      // Linux - try different tools in order of preference
      if (commandExists('gnome-screenshot')) {
        command = 'gnome-screenshot';
        args = ['-f', outputPath];
      } else if (commandExists('scrot')) {
        command = 'scrot';
        args = [outputPath];
      } else if (commandExists('import')) {
        // ImageMagick
        command = 'import';
        args = ['-window', 'root', outputPath];
      } else if (commandExists('xwd')) {
        // X Window Dump
        command = 'sh';
        args = ['-c', `xwd -root | convert xwd:- png:${outputPath}`];
      } else {
        reject(new Error('No screenshot utility found. Please install gnome-screenshot, scrot, or ImageMagick'));
        return;
      }
    }

    const process_screenshot = spawn(command, args);
    
    process_screenshot.on('close', (code) => {
      if (code === 0 && fs.existsSync(outputPath)) {
        // Optionally resize image if needed
        if (preset.width !== 1920 || preset.height !== 1080) {
          resizeImage(outputPath, preset.width, preset.height)
            .then(() => resolve(outputPath))
            .catch(() => resolve(outputPath)); // Use original if resize fails
        } else {
          resolve(outputPath);
        }
      } else {
        reject(new Error(`Screenshot command failed with code ${code}`));
      }
    });

    process_screenshot.on('error', (error) => {
      reject(new Error(`Screenshot command error: ${error.message}`));
    });
  });
}

/**
 * Create text image - ONLY REAL SPEECH (no demo text)
 */
async function createTextImage(text, quality = 'medium') {
  const preset = QUALITY_PRESETS[quality];
  const timestamp = Date.now();
  const filename = `text_${timestamp}.png`;
  const outputPath = path.join(__dirname, 'temp', filename);
  
  // Ensure temp directory exists
  const tempDir = path.join(__dirname, 'temp');
  if (!fs.existsSync(tempDir)) {
    fs.mkdirSync(tempDir, { recursive: true });
  }

  return new Promise((resolve, reject) => {
    if (!commandExists('convert')) {
      reject(new Error('ImageMagick not available for text image creation'));
      return;
    }

    // REAL SPEECH ONLY - no demo fallback
    let displayText = '';
    
    if (!text || text.trim() === '') {
      // Check if real speech is active
      if (isRealSpeechActive) {
        displayText = 'Listening for speech...';
      } else {
        displayText = 'Speech recognition not active';
      }
    } else {
      // Show the real speech text
      displayText = text.trim();
    }
    
    // Smart text wrapping for better display
    const words = displayText.split(' ');
    let line1 = '', line2 = '';
    
    if (words.length <= 4) {
      line1 = displayText;
      line2 = '';
    } else {
      const midPoint = Math.ceil(words.length * 0.6);
      line1 = words.slice(0, midPoint).join(' ');
      line2 = words.slice(midPoint).join(' ');
    }
    
    // Calculate font size based on image dimensions and text length
    const baseFontSize = Math.max(32, Math.floor(preset.width / 15));
    const fontSize = Math.min(baseFontSize, 72);
    
    // Create ImageMagick command for better text rendering
    const convert = spawn('convert', [
      '-size', `${preset.width}x${preset.height}`,
      'xc:black',
      '-fill', 'white',
      '-font', 'Arial-Bold',
      '-pointsize', fontSize.toString(),
      '-gravity', 'center',
      '-interline-spacing', '10',
      '-annotate', line2 ? '+0-40' : '+0+0', line1,
      ...(line2 ? ['-annotate', '+0+40', line2] : []),
      '-background', 'black',
      '-shadow', '80x3+2+2',
      '+repage',
      outputPath
    ]);

    convert.on('close', (code) => {
      if (code === 0 && fs.existsSync(outputPath)) {
        resolve(outputPath);
      } else {
        reject(new Error('Text image creation failed'));
      }
    });

    convert.on('error', reject);
  });
}

/**
 * Check if a command exists on the system
 */
function commandExists(command) {
  try {
    const { execSync } = require('child_process');
    execSync(`which ${command}`, { stdio: 'ignore' });
    return true;
  } catch (error) {
    return false;
  }
}

/**
 * Resize image using ImageMagick (if available)
 */
async function resizeImage(inputPath, width, height) {
  return new Promise((resolve, reject) => {
    if (!commandExists('convert')) {
      reject(new Error('ImageMagick not available for resizing'));
      return;
    }

    const outputPath = inputPath.replace('.png', '_resized.png');
    const convert = spawn('convert', [
      inputPath,
      '-resize',
      `${width}x${height}!`,
      outputPath
    ]);

    convert.on('close', (code) => {
      if (code === 0 && fs.existsSync(outputPath)) {
        fs.renameSync(outputPath, inputPath);
        resolve(inputPath);
      } else {
        reject(new Error('Image resize failed'));
      }
    });

    convert.on('error', reject);
  });
}

/**
 * Convert image to GIF using ImageMagick
 */
async function convertToGif(imagePaths, outputPath, fps = 10) {
  return new Promise((resolve, reject) => {
    if (!commandExists('convert')) {
      reject(new Error('ImageMagick not available for GIF conversion'));
      return;
    }

    const delay = Math.round(100 / fps);
    const args = [
      '-delay', delay.toString(),
      '-loop', '0',
      ...imagePaths,
      outputPath
    ];

    const convert = spawn('convert', args);

    convert.on('close', (code) => {
      if (code === 0 && fs.existsSync(outputPath)) {
        resolve(outputPath);
      } else {
        reject(new Error('GIF conversion failed'));
      }
    });

    convert.on('error', reject);
  });
}

// Store recent screenshots for GIF creation
let recentScreenshots = [];
let recentTextImages = [];
const MAX_SCREENSHOTS = 10;

/**
 * API endpoint for screen capture
 */
app.post('/api/screen-capture', async (req, res) => {
  try {
    const { quality = 'medium', fps = 10, format = 'gif' } = req.body;
    
    console.log(`Capturing screen: quality=${quality}, fps=${fps}, format=${format}`);
    
    const screenshotPath = await captureScreenWithSystemCommand(quality);
    
    if (format === 'gif') {
      recentScreenshots.push(screenshotPath);
      if (recentScreenshots.length > MAX_SCREENSHOTS) {
        const oldPath = recentScreenshots.shift();
        if (fs.existsSync(oldPath)) {
          fs.unlinkSync(oldPath);
        }
      }

      if (recentScreenshots.length >= 3) {
        const gifPath = path.join(__dirname, 'temp', `animation_${Date.now()}.gif`);
        
        try {
          await convertToGif(recentScreenshots.slice(-5), gifPath, fps);
          
          res.set({
            'Content-Type': 'image/gif',
            'Cache-Control': 'no-cache',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type'
          });
          
          const gifStream = fs.createReadStream(gifPath);
          gifStream.pipe(res);
          
          gifStream.on('end', () => {
            setTimeout(() => {
              if (fs.existsSync(gifPath)) {
                fs.unlinkSync(gifPath);
              }
            }, 1000);
          });
          
        } catch (gifError) {
          console.warn('GIF creation failed, sending PNG instead:', gifError.message);
          res.set({
            'Content-Type': 'image/png',
            'Cache-Control': 'no-cache',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type'
          });
          
          const pngStream = fs.createReadStream(screenshotPath);
          pngStream.pipe(res);
        }
      } else {
        res.set({
          'Content-Type': 'image/png',
          'Cache-Control': 'no-cache',
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
          'Access-Control-Allow-Headers': 'Content-Type'
        });
        
        const pngStream = fs.createReadStream(screenshotPath);
        pngStream.pipe(res);
      }
    } else {
      res.set({
        'Content-Type': 'image/png',
        'Cache-Control': 'no-cache',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type'
      });
      
      const imageStream = fs.createReadStream(screenshotPath);
      imageStream.pipe(res);
      
      imageStream.on('end', () => {
        setTimeout(() => {
          if (fs.existsSync(screenshotPath)) {
            fs.unlinkSync(screenshotPath);
          }
        }, 1000);
      });
    }
  } catch (error) {
    console.error('Screen capture API error:', error);
    res.status(500).json({ 
      error: 'Screen capture failed', 
      message: error.message,
      suggestions: [
        'On Linux: Install gnome-screenshot, scrot, or ImageMagick',
        'On Windows: PowerShell should be available',
        'On macOS: screencapture should be available'
      ]
    });
  }
});

/**
 * API endpoint for speech-to-text streaming - REAL SPEECH ONLY
 */
app.post('/api/speech-text-stream', async (req, res) => {
  try {
    const { quality = 'medium', fps = 2, format = 'gif' } = req.body;
    
    console.log(`Creating speech text image: quality=${quality}, format=${format}, text="${currentSpeechText}"`);
    
    // Use ONLY real speech text (no demo/fallback)
    const textToDisplay = currentSpeechText;
    
    // Create text image
    const textImagePath = await createTextImage(textToDisplay, quality);
    
    if (format === 'gif') {
      recentTextImages.push(textImagePath);
      if (recentTextImages.length > MAX_SCREENSHOTS) {
        const oldPath = recentTextImages.shift();
        if (fs.existsSync(oldPath)) {
          fs.unlinkSync(oldPath);
        }
      }

      if (recentTextImages.length >= 2) {
        const gifPath = path.join(__dirname, 'temp', `text_animation_${Date.now()}.gif`);
        
        try {
          await convertToGif(recentTextImages.slice(-3), gifPath, fps);
          
          res.set({
            'Content-Type': 'image/gif',
            'Cache-Control': 'no-cache',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type'
          });
          
          const gifStream = fs.createReadStream(gifPath);
          gifStream.pipe(res);
          
          gifStream.on('end', () => {
            setTimeout(() => {
              if (fs.existsSync(gifPath)) {
                fs.unlinkSync(gifPath);
              }
            }, 1000);
          });
          
        } catch (gifError) {
          console.warn('Text GIF creation failed, sending PNG instead:', gifError.message);
          res.set({
            'Content-Type': 'image/png',
            'Cache-Control': 'no-cache',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type'
          });
          
          const pngStream = fs.createReadStream(textImagePath);
          pngStream.pipe(res);
        }
      } else {
        res.set({
          'Content-Type': 'image/png',
          'Cache-Control': 'no-cache',
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
          'Access-Control-Allow-Headers': 'Content-Type'
        });
        
        const pngStream = fs.createReadStream(textImagePath);
        pngStream.pipe(res);
      }
    } else {
      res.set({
        'Content-Type': 'image/png',
        'Cache-Control': 'no-cache',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type'
      });
      
      const imageStream = fs.createReadStream(textImagePath);
      imageStream.pipe(res);
      
      imageStream.on('end', () => {
        setTimeout(() => {
          if (fs.existsSync(textImagePath)) {
            fs.unlinkSync(textImagePath);
          }
        }, 1000);
      });
    }
  } catch (error) {
    console.error('Speech text stream API error:', error);
    res.status(500).json({ 
      error: 'Speech text stream failed', 
      message: error.message,
      suggestions: [
        'Make sure ImageMagick is installed for text image creation',
        'Make sure Python speech backend is running on ws://localhost:8765'
      ]
    });
  }
});

/**
 * CRITICAL: Real speech update endpoint (called by Python backend)
 */
app.post('/api/speech/update-text', (req, res) => {
  try {
    const { text } = req.body;
    
    if (text === undefined || text === null) {
      res.status(400).json({ 
        status: 'error',
        message: 'Text parameter is required'
      });
      return;
    }
    
    // Update real speech state
    const oldText = currentSpeechText;
    currentSpeechText = text;
    lastSpeechUpdate = Date.now();
    isRealSpeechActive = true;
    
    console.log(`🎤 Real speech text updated: "${text}" (was: "${oldText}")`);
    
    // Add to history if it's actual speech (not status messages)
    if (text && text.trim() && !text.includes('chunks') && !text.includes('Loading') && !text.includes('Model')) {
      speechHistory.push({
        text: text,
        timestamp: Date.now(),
        source: 'python_backend'
      });
      
      // Keep only recent history
      if (speechHistory.length > MAX_SPEECH_HISTORY) {
        speechHistory.shift();
      }
    }
    
    res.json({ 
      status: 'success',
      message: 'Real speech text updated successfully',
      currentText: currentSpeechText,
      isRealSpeech: true
    });
    
  } catch (error) {
    console.error('Speech text update API error:', error);
    res.status(500).json({ 
      status: 'error',
      message: error.message
    });
  }
});

/**
 * API endpoint to get current speech text
 */
app.get('/api/speech/current', (req, res) => {
  res.set({
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type'
  });
  
  // Check if speech is still active (timeout after 10 seconds of no updates)
  const speechTimeout = Date.now() - lastSpeechUpdate > 10000;
  if (speechTimeout && isRealSpeechActive) {
    isRealSpeechActive = false;
    console.log('🔇 Speech recognition timed out');
  }
  
  res.json({
    currentText: currentSpeechText,
    isActive: isRealSpeechActive,
    isRealSpeech: isRealSpeechActive,
    lastUpdate: lastSpeechUpdate,
    history: speechHistory.slice(-5),
    platform: process.platform
  });
});

/**
 * API endpoint to check if Python speech backend is running
 */
app.get('/api/speech/python-status', (req, res) => {
  res.set({
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type'
  });
  
  // Try to connect to Python backend WebSocket
  const WebSocket = require('ws');
  const ws = new WebSocket('ws://localhost:8765');
  
  const timeout = setTimeout(() => {
    ws.terminate();
    res.json({
      pythonBackendRunning: false,
      message: 'Python speech backend not responding',
      websocketUrl: 'ws://localhost:8765'
    });
  }, 2000);
  
  ws.on('open', () => {
    clearTimeout(timeout);
    ws.close();
    res.json({
      pythonBackendRunning: true,
      message: 'Python speech backend is running',
      websocketUrl: 'ws://localhost:8765',
      isRealSpeechActive: isRealSpeechActive
    });
  });
  
  ws.on('error', () => {
    clearTimeout(timeout);
    res.json({
      pythonBackendRunning: false,
      message: 'Python speech backend not running',
      websocketUrl: 'ws://localhost:8765'
    });
  });
});

/**
 * REMOVED: All demo/progressive speech endpoints
 * - /api/speech/start-progressive
 * - /api/speech/stop-progressive  
 * - startProgressiveSpeech function
 * - test sentences
 * These were causing interference with real speech
 */

/**
 * Health check endpoint
 */
app.get('/api/health', (req, res) => {
  res.set({
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type'
  });
  
  const availableTools = [];
  
  if (process.platform === 'win32') {
    availableTools.push('PowerShell (Windows)');
  } else if (process.platform === 'darwin') {
    availableTools.push('screencapture (macOS)');
  } else {
    if (commandExists('gnome-screenshot')) availableTools.push('gnome-screenshot');
    if (commandExists('scrot')) availableTools.push('scrot');
    if (commandExists('import')) availableTools.push('ImageMagick import');
    if (commandExists('xwd')) availableTools.push('xwd');
  }
  
  res.json({
    status: 'OK',
    platform: process.platform,
    availableScreenshotTools: availableTools,
    imagemagickAvailable: commandExists('convert'),
    canCreateGifs: commandExists('convert'),
    canCreateTextImages: commandExists('convert'),
    realSpeechActive: isRealSpeechActive,
    currentSpeechText: currentSpeechText,
    lastSpeechUpdate: lastSpeechUpdate,
    pythonBackendExpected: 'ws://localhost:8765'
  });
});

app.get('/', (req, res) => {
  res.redirect('/en-US');
});

// Handle preflight OPTIONS requests
app.options('/api/*', (req, res) => {
  res.set({
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    'Access-Control-Max-Age': '86400'
  });
  res.status(200).end();
});

// Serve Angular app for all routes
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'dist/holodisplay/en-US/index.html'));
});

// Clean up temp directory on startup
const tempDir = path.join(__dirname, 'temp');
if (fs.existsSync(tempDir)) {
  fs.readdirSync(tempDir).forEach(file => {
    fs.unlinkSync(path.join(tempDir, file));
  });
}

// Clean up on process exit
process.on('exit', () => {
  console.log('🛑 Server shutting down');
});

process.on('SIGINT', () => {
  console.log('🛑 Received SIGINT, shutting down gracefully');
  process.exit();
});

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
  console.log('🖥️  Screen streaming available at /api/screen-capture');
  console.log('🎤 Real speech-to-text streaming at /api/speech-text-stream');
  console.log('🔄 Speech update endpoint at /api/speech/update-text');
  console.log('📊 Health check available at /api/health');
  console.log('🐍 Expects Python speech backend on ws://localhost:8765');
  console.log(`Platform: ${process.platform}`);
  console.log('⚡ Real-time speech mode - no demo text');
});

module.exports = app;
