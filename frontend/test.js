const http = require('http');

// Test phrases to simulate speech recognition
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

// Function to update speech text on server
function updateSpeechText(text) {
  const postData = JSON.stringify({ text: text });
  
  const options = {
    hostname: 'localhost',
    port: 3000,
    path: '/api/speech/update-text',
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Content-Length': Buffer.byteLength(postData)
    }
  };

  const req = http.request(options, (res) => {
    console.log(`✅ Updated speech text: "${text}" (Status: ${res.statusCode})`);
  });

  req.on('error', (e) => {
    console.error(`❌ Error updating speech text: ${e.message}`);
  });

  req.write(postData);
  req.end();
}

// Function to start the test simulation
function startSpeechTest() {
  if (isRunning) {
    console.log('⚠️ Speech test is already running!');
    return;
  }

  isRunning = true;
  console.log('🎤 Starting speech-to-text test simulation...');
  console.log('📝 Will cycle through test phrases every 3 seconds');
  console.log('🛑 Press Ctrl+C to stop\n');

  // Start with first phrase
  updateSpeechText(testPhrases[currentPhraseIndex]);
  
  // Set up interval to change text every 3 seconds
  const interval = setInterval(() => {
    currentPhraseIndex = (currentPhraseIndex + 1) % testPhrases.length;
    updateSpeechText(testPhrases[currentPhraseIndex]);
    
    // Reset after going through all phrases
    if (currentPhraseIndex === 0) {
      console.log('\n🔄 Completed one full cycle, starting over...\n');
    }
  }, 3000);

  // Handle graceful shutdown
  process.on('SIGINT', () => {
    console.log('\n🛑 Stopping speech test simulation...');
    clearInterval(interval);
    
    // Clear the text on server
    updateSpeechText('Speech test stopped');
    
    setTimeout(() => {
      console.log('✅ Speech test stopped');
      process.exit(0);
    }, 1000);
  });
}

// Function to send a single test phrase
function sendTestPhrase(phrase) {
  if (!phrase) {
    console.log('❌ Please provide a phrase to send');
    console.log('Usage: node speech-test.js "Your custom text here"');
    return;
  }
  
  console.log(`🎤 Sending test phrase: "${phrase}"`);
  updateSpeechText(phrase);
}

// Main execution
const args = process.argv.slice(2);

if (args.length > 0) {
  // Send custom phrase
  const customPhrase = args.join(' ');
  sendTestPhrase(customPhrase);
} else {
  // Start automated test
  startSpeechTest();
}

// Display help information
console.log('📖 Speech-to-Text Test Script');
console.log('═══════════════════════════════');
console.log('Usage:');
console.log('  node speech-test.js                    - Start automated test with predefined phrases');
console.log('  node speech-test.js "Custom text"      - Send a single custom phrase');
console.log('');
console.log('Make sure your server is running on localhost:3000');
console.log('');