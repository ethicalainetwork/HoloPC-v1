const http = require('http');

console.log('🎤 Word-by-Word Speech Test');
console.log('===========================');
console.log('');

// Test phrases for word-by-word simulation
const testPhrases = [
  "Hello world, this is a test",
  "Welcome to the holographic display",
  "Speech to text conversion is working",
  "This is a demonstration of real-time text",
  "The quick brown fox jumps over the lazy dog",
  "Testing speech recognition with multiple words",
  "Artificial intelligence and machine learning",
  "Voice commands are being processed",
  "Real-time speech transcription in progress",
  "This text will appear on the holographic display",
  "Multiple lines of text can be displayed",
  "The system is working correctly",
  "End of speech recognition test"
];

let currentPhraseIndex = 0;
let isRunning = false;
let wordByWordInterval = null;

// Function to make API calls
function makeRequest(path, method = 'POST', data = null) {
  return new Promise((resolve, reject) => {
    const postData = data ? JSON.stringify(data) : null;
    
    const options = {
      hostname: 'localhost',
      port: 3000,
      path: path,
      method: method,
      headers: {
        'Content-Type': 'application/json',
      }
    };

    if (postData) {
      options.headers['Content-Length'] = Buffer.byteLength(postData);
    }

    const req = http.request(options, (res) => {
      let responseData = '';
      res.on('data', (chunk) => {
        responseData += chunk;
      });
      res.on('end', () => {
        try {
          const result = JSON.parse(responseData);
          resolve({ status: res.statusCode, data: result });
        } catch (e) {
          resolve({ status: res.statusCode, data: responseData });
        }
      });
    });

    req.on('error', (e) => {
      reject(e);
    });

    if (postData) {
      req.write(postData);
    }
    req.end();
  });
}

// Function to update speech text on server
function updateSpeechText(text) {
  return makeRequest('/api/speech/update-text', 'POST', { text: text });
}

// Function to start built-in progressive simulation
async function startBuiltInProgressive() {
  try {
    console.log('🚀 Starting built-in progressive speech simulation...');
    const response = await makeRequest('/api/speech/start-progressive');
    
    if (response.status === 200) {
      console.log('✅ Built-in progressive speech started!');
      console.log(`📝 Status: ${response.data.status}`);
      console.log(`💬 Message: ${response.data.message}`);
      console.log('');
      console.log('🎯 The server will now:');
      console.log('   • Add words every 500ms (realistic speech pace)');
      console.log('   • Complete sentences then move to next');
      console.log('   • Cycle through 10 test sentences');
      console.log('   • Display text word-by-word on holographic display');
      console.log('');
      console.log('🛑 Press Ctrl+C to stop');
    } else {
      console.log('❌ Failed to start built-in progressive speech');
      console.log(`Response: ${JSON.stringify(response.data, null, 2)}`);
    }
  } catch (error) {
    console.error('❌ Error starting built-in progressive speech:', error.message);
  }
}

// Function to stop built-in progressive simulation
async function stopBuiltInProgressive() {
  try {
    console.log('🛑 Stopping built-in progressive speech simulation...');
    const response = await makeRequest('/api/speech/stop-progressive');
    
    if (response.status === 200) {
      console.log('✅ Built-in progressive speech stopped!');
      console.log(`📝 Status: ${response.data.status}`);
      console.log(`💬 Message: ${response.data.message}`);
    } else {
      console.log('❌ Failed to stop built-in progressive speech');
    }
  } catch (error) {
    console.error('❌ Error stopping built-in progressive speech:', error.message);
  }
}

// Function to start custom word-by-word simulation
function startWordByWordTest() {
  if (isRunning) {
    console.log('⚠️ Word-by-word test is already running!');
    return;
  }

  isRunning = true;
  console.log('🎤 Starting custom word-by-word speech simulation...');
  console.log('📝 Will build phrases word by word, then move to next phrase');
  console.log('🛑 Press Ctrl+C to stop\n');

  // Start with first phrase
  simulateWordByWord(testPhrases[currentPhraseIndex]);
}

// Function to simulate word-by-word speech for a single phrase
async function simulateWordByWord(phrase) {
  console.log(`🎯 Starting phrase: "${phrase}"`);
  
  const words = phrase.split(' ');
  let builtText = '';
  let wordIndex = 0;

  wordByWordInterval = setInterval(async () => {
    if (wordIndex < words.length) {
      // Add next word
      const nextWord = words[wordIndex];
      builtText += (wordIndex > 0 ? ' ' : '') + nextWord;
      
      console.log(`💬 Added word: "${nextWord}" → "${builtText}"`);
      
      try {
        await updateSpeechText(builtText);
      } catch (error) {
        console.error('❌ Error updating text:', error.message);
      }
      
      wordIndex++;
    } else {
      // Phrase complete, move to next phrase
      console.log(`✅ Phrase complete: "${builtText}"\n`);
      
      clearInterval(wordByWordInterval);
      
      // Move to next phrase after a pause
      setTimeout(() => {
        if (isRunning) {
          currentPhraseIndex = (currentPhraseIndex + 1) % testPhrases.length;
          
          if (currentPhraseIndex === 0) {
            console.log('🔄 Completed one full cycle, starting over...\n');
          }
          
          simulateWordByWord(testPhrases[currentPhraseIndex]);
        }
      }, 2000); // 2 second pause between phrases
    }
  }, 600); // Add a word every 600ms
}

// Function to send a single custom phrase word-by-word
async function sendCustomWordByWord(phrase) {
  console.log(`🎤 Sending custom phrase word-by-word: "${phrase}"`);
  
  const words = phrase.split(' ');
  let builtText = '';
  
  for (let i = 0; i < words.length; i++) {
    const nextWord = words[i];
    builtText += (i > 0 ? ' ' : '') + nextWord;
    
    console.log(`💬 Added word: "${nextWord}" → "${builtText}"`);
    
    try {
      await updateSpeechText(builtText);
      
      // Wait before adding next word (except for last word)
      if (i < words.length - 1) {
        await new Promise(resolve => setTimeout(resolve, 600));
      }
    } catch (error) {
      console.error('❌ Error updating text:', error.message);
      break;
    }
  }
  
  console.log(`✅ Custom phrase complete: "${builtText}"`);
}

// Function to check server status
async function checkServerStatus() {
  try {
    console.log('🔍 Checking server status...');
    const response = await makeRequest('/api/speech/current', 'GET');
    
    if (response.status === 200) {
      console.log('📊 Server Status:');
      console.log(`   Current Text: "${response.data.currentText}"`);
      console.log(`   Recognition Active: ${response.data.isActive}`);
      console.log(`   Platform: ${response.data.platform}`);
    } else {
      console.log('❌ Failed to get server status');
    }
  } catch (error) {
    console.error('❌ Error checking server status:', error.message);
    console.log('Make sure your server is running on localhost:3000');
  }
}

// Handle graceful shutdown
process.on('SIGINT', () => {
  console.log('\n🛑 Stopping word-by-word test simulation...');
  
  isRunning = false;
  
  if (wordByWordInterval) {
    clearInterval(wordByWordInterval);
  }
  
  // Try to stop built-in progressive simulation
  stopBuiltInProgressive().then(() => {
    // Clear the text on server
    updateSpeechText('Word-by-word test stopped').then(() => {
      console.log('✅ Word-by-word test stopped');
      process.exit(0);
    }).catch(() => {
      process.exit(0);
    });
  }).catch(() => {
    process.exit(0);
  });
});

// Main execution
const command = process.argv[2];
const customText = process.argv.slice(3).join(' ');

switch (command) {
  case 'progressive':
    startBuiltInProgressive();
    break;
  case 'custom':
    if (customText) {
      sendCustomWordByWord(customText);
    } else {
      startWordByWordTest();
    }
    break;
  case 'stop':
    stopBuiltInProgressive();
    break;
  case 'status':
    checkServerStatus();
    break;
  default:
    console.log('Usage:');
    console.log('  node word-by-word-test.js progressive           - Use built-in server progressive simulation');
    console.log('  node word-by-word-test.js custom               - Start custom word-by-word with test phrases');
    console.log('  node word-by-word-test.js custom "Your text"   - Send custom text word-by-word');
    console.log('  node word-by-word-test.js stop                 - Stop progressive simulation');
    console.log('  node word-by-word-test.js status               - Check current server status');
    console.log('');
    console.log('🎯 Recommended usage:');
    console.log('  1. Start your server: npm start');
    console.log('  2. Start speech streaming in the app: Click "🎤 Start Speech Stream"');
    console.log('  3. Run: node word-by-word-test.js progressive');
    console.log('  4. Watch text appear word-by-word on the holographic display!');
    console.log('');
    
    // Default to progressive if no command given
    if (!command) {
      console.log('🚀 Starting progressive simulation (default)...');
      startBuiltInProgressive();
    }
    break;
}