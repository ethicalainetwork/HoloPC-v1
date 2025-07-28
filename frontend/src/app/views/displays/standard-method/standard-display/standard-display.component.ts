import { CommonModule } from '@angular/common';
import {
  AfterViewInit,
  Component,
  ElementRef,
  Input, OnInit, OnDestroy,
  ViewChild
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Frame, ParsedFrame, ParsedGif, decompressFrames, parseGIF } from 'gifuct-js';
import { Observable, Subject, debounceTime, merge, Subscription } from 'rxjs';
import { StandardMethodCalculatorService } from 'src/app/services/calculators/standard-method/standard-method-calculator.service';
import { HelperService, Point } from 'src/app/services/helpers/helper.service';
import { SettingsBrokerService } from 'src/app/services/standard-display/settings-broker.service';
import { StandardDisplayFileSettings, MetaDataKeys, StandardDisplayGeneralSettings, StandardDisplaySettings } from 'src/app/services/standard-display/standard-display-settings.type';
import { TutorialService } from 'src/app/services/tutorial/tutorial.service';
import { ScreenStreamingService, StreamConfig, SavedGif } from 'src/app/services/screen-streaming.service';

@Component({
  selector: 'app-display-standard',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './standard-display.component.html',
  styleUrls: ['./standard-display.component.scss']
})
export class StandardDisplayComponent implements OnInit, AfterViewInit, OnDestroy {
  @Input() resizeEvent$!: Observable<Event>;
  @Input() calculate$!: Observable<void>;

  private readonly requestDraw$ = new Subject<void>();
  private readonly MY_SETTINGS_BROKER_ID = "StandardDisplayComponent";

  @ViewChild('displayCanvas') displayCanvas!: ElementRef<HTMLCanvasElement>;
  @ViewChild('container') container!: ElementRef<HTMLDivElement>;

  private innerEdgePoints: Point[] = [];
  private outerEdgePoints: Point[] = [];
  private canvasSize = 0;
  private angle = 0;
  private innerPolygonIncircleRadius = 0;
  private polygonInfo: { rotation: number, offset: {dx: number, dy: number}, sides: number } = {} as typeof this.polygonInfo;
  private transformationMatrices: DOMMatrix[][] = [];

  // Calculator properties
  calculatorDPI = 96;
  calculatorSlope = 45;
  calculatorImageWidthPx = 0;
  calculatorImageHeightPx = 0;
  calculatorJsPixelRatio = window.devicePixelRatio;

  // Screen streaming properties
  private isScreenStreaming = false;
  private streamingSubscription?: Subscription;
  private debugScreenshotInterval?: any;
  private animationInterval?: any;
  private currentGifIndex = 0;
  private isDarkFullscreen = false;
  public menuHidden = false;
  
  // Text streaming properties
  private currentSpeechText = '';
  private speechUpdateInterval?: any;
  
  // Speech Interface Properties (NEW)
  speechInterface = {
    isConnected: false,
    isRecording: false,
    status: 'Ready to connect...',
    transcript: 'Transcript will appear here...',
    audioLevel: 0,
    password: 'holographic2024',
    serverUrl: 'ws://localhost:8765'
  };
  // ===== PUBLIC METHODS FOR TEMPLATE =====

  /**
   * Get speech status text for UI display
   */
  getSpeechStatusText(): string {
    if (!this.speechInterface.isConnected) {
      return 'Disconnected';
    }
    if (this.speechInterface.isRecording) {
      return 'Recording';
    }
    return 'Connected';
  }

  /**
   * Get tooltip for speech stream button
   */
  getSpeechStreamButtonTooltip(): string {
    if (!this.speechInterface.isConnected) {
      return 'Connect to speech interface first';
    }
    if (!this.speechInterface.isRecording) {
      return 'Start recording in speech interface first';
    }
    if (this.isSpeechStreamingActive()) {
      return 'Stop displaying speech on screen';
    }
    return 'Display real-time speech transcripts on holographic screen';
  }

  private websocket: WebSocket | null = null;
  private mediaRecorder: MediaRecorder | null = null;
  private audioContext: AudioContext | null = null;
  private analyser: AnalyserNode | null = null;
  private microphone: MediaStreamAudioSourceNode | null = null;
  private animationFrameId?: number;
  
  streamConfig: StreamConfig = {
    quality: 'medium',
    fps: 10,
    format: 'gif',
    mode: 'screen'
  };

  constructor(
    private helperService: HelperService,
    private calculator: StandardMethodCalculatorService,
    private tutorial: TutorialService,
    public settingsBroker: SettingsBrokerService,
    private screenStreaming: ScreenStreamingService
  ) {
    settingsBroker.settings$.subscribe(({settings, changedBy}) => {
      if(changedBy == this.MY_SETTINGS_BROKER_ID) {
        return;
      }

      // update the calculated values (working with the general settings)
      this.recalculateValues(settings.generalSettings);

      // update the images (working with the file settings)
      this.updateImageSettings(settings);
    });

    this.requestDraw$.subscribe(() => this.draw());
  }

  ngOnInit(): void {
    this.resizeEvent$.pipe(debounceTime(20)).subscribe((event) => {
      this.resizeCanvas((event.target as Window).innerWidth, (event.target as Window).innerHeight);

      // scale the images again because the canvas size changed
      const settings = this.settingsBroker.getSettings();
      this.scaleImagesFromFileSetting(settings.fileSettings).then(() => {
        this.recalculateValues(settings.generalSettings);
        this.requestDraw$.next();
      });
    });

    // define the calculation function
    this.calculate$.subscribe(() => {
      this.toggleModal('calculatorExtraSettingsModal');
    });
  }

  ngAfterViewInit(): void {
    this.resizeCanvas(this.container.nativeElement.clientWidth, this.container.nativeElement.clientHeight);
    this.recalculateValues(this.settingsBroker.getSettings().generalSettings);

    if(!this.tutorial.isTutorialDeactivated('standardDisplay'))
      this.tutorial.startTutorial('standardDisplay');
  }

  ngOnDestroy(): void {
    this.stopScreenStreaming();
    this.stopDebugScreenshots();
    
    // Clean up speech interface (NEW)
    this.disconnectSpeechInterface();
    
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
    }
  }

  // ===== NEW SPEECH INTERFACE METHODS =====

  /**
   * Connect to the speech recognition WebSocket server
   */
  connectSpeechInterface(): void {
    if (!this.speechInterface.serverUrl || !this.speechInterface.password) {
      this.speechInterface.status = 'Please enter server URL and password';
      return;
    }

    try {
      this.speechInterface.status = 'Connecting...';
      
      this.websocket = new WebSocket(this.speechInterface.serverUrl);
      
      this.websocket.onopen = () => {
        if (this.websocket) {
          this.websocket.send(JSON.stringify({
            type: 'auth',
            password: this.speechInterface.password
          }));
        }
      };

      this.websocket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        switch(data.type) {
          case 'auth_success':
            this.speechInterface.isConnected = true;
            this.speechInterface.status = 'Connected successfully!';
            console.log('✅ Speech interface connected:', data);
            break;
            
          case 'auth_failed':
            this.speechInterface.status = 'Authentication failed: ' + data.message;
            if (this.websocket) {
              this.websocket.close();
            }
            break;
            
          case 'streaming_transcript':
            this.speechInterface.transcript = data.text;
            this.speechInterface.status = data.is_final ? 'Final transcript received' : 'Streaming transcript...';
            console.log('📝 Received transcript:', data.text);
            break;
            
          case 'transcript':
            this.speechInterface.transcript = data.text;
            break;
        }
      };

      this.websocket.onclose = () => {
        this.speechInterface.isConnected = false;
        this.speechInterface.status = 'Connection closed';
        console.log('🔌 Speech interface disconnected');
        if (this.speechInterface.isRecording) {
          this.stopSpeechRecording();
        }
      };

      this.websocket.onerror = (error) => {
        this.speechInterface.status = 'Connection error';
        console.error('❌ WebSocket error:', error);
      };

    } catch (error) {
      this.speechInterface.status = 'Failed to connect: ' + (error as Error).message;
      console.error('❌ Connection failed:', error);
    }
  }

  /**
   * Disconnect from the speech recognition server
   */
  disconnectSpeechInterface(): void {
    if (this.websocket) {
      this.websocket.close();
      this.websocket = null;
    }
    
    if (this.speechInterface.isRecording) {
      this.stopSpeechRecording();
    }
    
    this.speechInterface.isConnected = false;
    this.speechInterface.status = 'Disconnected';
  }

  /**
   * Toggle speech recording on/off
   */
  toggleSpeechRecording(): void {
    if (this.speechInterface.isRecording) {
      this.stopSpeechRecording();
    } else {
      this.startSpeechRecording();
    }
  }

  /**
 * Start recording audio for speech recognition (Raw PCM for best backend compatibility)
 */
async startSpeechRecording(): Promise<void> {
  if (!this.speechInterface.isConnected) {
    this.speechInterface.status = 'Not connected to server';
    return;
  }

  try {
    console.log('🎤 Requesting microphone access...');
    
    // Optimized constraints for speech recognition
    const constraints: MediaStreamConstraints = {
      audio: {
        sampleRate: { exact: 16000 },        // Fixed 16kHz for speech
        channelCount: { exact: 1 },          // Mono audio
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        // Remove latency and sampleSize constraints that might not be supported
      }
    };

    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia(constraints);
      console.log('✅ Microphone access granted with optimal settings');
    } catch (err) {
      console.warn('⚠️ Failed with exact constraints, trying fallback...');
      // Fallback with more flexible constraints
      stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true
        }
      });
    }

    // Create audio context with exact sample rate
    const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
    
    this.audioContext = new AudioContextClass({
      sampleRate: 16000,
      latencyHint: 'interactive'
    });

    // Handle mobile audio context state
    if (this.audioContext && this.audioContext.state === 'suspended') {
      console.log('🔊 Resuming audio context for mobile...');
      await this.audioContext.resume();
    }

    if (!this.audioContext) {
      throw new Error('Failed to create audio context');
    }

    console.log('🎛️ Audio context sample rate:', this.audioContext.sampleRate);

    this.microphone = this.audioContext.createMediaStreamSource(stream);
    this.analyser = this.audioContext.createAnalyser();
    this.analyser.fftSize = 2048;
    this.analyser.smoothingTimeConstant = 0.8;
    this.microphone.connect(this.analyser);

    // Audio level monitoring
    const bufferLength = this.analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    const updateLevels = () => {
      if (this.speechInterface.isRecording && this.analyser) {
        this.analyser.getByteFrequencyData(dataArray);
        const average = dataArray.reduce((a, b) => a + b) / bufferLength;
        this.speechInterface.audioLevel = Math.min(average / 128, 1);
        this.animationFrameId = requestAnimationFrame(updateLevels);
      }
    };
    updateLevels();

    // **NEW APPROACH**: Use ScriptProcessorNode for raw PCM data
    // This gives us direct access to audio samples without codec issues
    const bufferSize = 4096; // Process audio in chunks
    const processor = this.audioContext.createScriptProcessor(bufferSize, 1, 1);
    
    processor.onaudioprocess = (e) => {
      if (!this.speechInterface.isRecording) return;
      
      const inputBuffer = e.inputBuffer;
      const inputData = inputBuffer.getChannelData(0); // Get mono channel
      
      // Convert Float32Array to Int16Array for better backend compatibility
      const int16Buffer = new Int16Array(inputData.length);
      for (let i = 0; i < inputData.length; i++) {
        // Convert from -1.0 to 1.0 range to -32768 to 32767 range
        const sample = Math.max(-1, Math.min(1, inputData[i]));
        int16Buffer[i] = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
      }
      
      // Convert to base64 for transmission
      const uint8View = new Uint8Array(int16Buffer.buffer);
      const base64Audio = btoa(String.fromCharCode.apply(null, Array.from(uint8View)));
      
      if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
        this.websocket.send(JSON.stringify({
          type: 'audio_chunk_raw',
          audio: base64Audio,
          format: 'pcm_s16le',           // Raw PCM, signed 16-bit little endian
          sampleRate: this.audioContext?.sampleRate || 16000,
          channels: 1,
          samplesCount: int16Buffer.length
        }));
      }
    };

    // Connect the processor (with null checks)
    if (this.microphone && this.audioContext) {
      this.microphone.connect(processor);
      processor.connect(this.audioContext.destination);

      // Store processor reference for cleanup
      (this as any).audioProcessor = processor;
    }

    // **FALLBACK**: Also try MediaRecorder with WAV if available
    let mediaRecorderFallback = false;
    const wavMimeTypes = [
      'audio/wav',
      'audio/wave',
      'audio/x-wav'
    ];
    
    let selectedWavType = '';
    for (const mimeType of wavMimeTypes) {
      if (MediaRecorder.isTypeSupported(mimeType)) {
        selectedWavType = mimeType;
        mediaRecorderFallback = true;
        console.log('📱 WAV MediaRecorder available as fallback:', mimeType);
        break;
      }
    }

    if (mediaRecorderFallback) {
      this.mediaRecorder = new MediaRecorder(stream, {
        mimeType: selectedWavType
      });

      this.mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          const reader = new FileReader();
          reader.onload = () => {
            const audioData = new Uint8Array(reader.result as ArrayBuffer);
            const base64Audio = btoa(String.fromCharCode.apply(null, Array.from(audioData)));
            
            if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
              this.websocket.send(JSON.stringify({
                type: 'audio_chunk_wav',
                audio: base64Audio,
                format: selectedWavType,
                sampleRate: 16000,
                channels: 1
              }));
            }
          };
          reader.readAsArrayBuffer(event.data);
        }
      };

      // Start MediaRecorder as fallback
      this.mediaRecorder.start(250); // 250ms chunks
      console.log('🎙️ Started MediaRecorder fallback with WAV');
    }
    
    // Notify server about audio formats
    if (this.websocket) {
      this.websocket.send(JSON.stringify({
        type: 'start_recording',
        primaryFormat: 'pcm_s16le',
        fallbackFormat: selectedWavType || 'none',
        sampleRate: this.audioContext?.sampleRate || 16000,
        channels: 1,
        bufferSize: bufferSize
      }));
    }

    this.speechInterface.isRecording = true;
    this.speechInterface.status = 'Recording with raw PCM... Speak now!';
    this.speechInterface.transcript = 'Listening for speech...';

    console.log('🎙️ Recording started successfully with raw PCM + WAV fallback');

  } catch (error) {
    const errorMessage = (error as Error).message;
    this.speechInterface.status = 'Microphone access failed';
    
    console.error('❌ Recording error:', error);
    
    if (errorMessage.includes('Permission denied') || errorMessage.includes('NotAllowedError')) {
      alert('🎤 Microphone access is required for speech recognition.\n\nPlease:\n1. Allow microphone permission in your browser\n2. Check if another app is using the microphone\n3. Refresh the page and try again');
    } else if (errorMessage.includes('NotFoundError') || errorMessage.includes('DeviceNotFoundError')) {
      alert('🎤 No microphone found.\n\nPlease check that your device has a microphone and it\'s properly connected.');
    } else {
      alert('🎤 Failed to start recording.\n\nError: ' + errorMessage + '\n\nTry refreshing the page or using a different browser.');
    }
  }
}

/**
 * Stop recording audio (UPDATED with null checks)
 */
stopSpeechRecording(): void {
  console.log('🛑 Stopping speech recording...');
  
  this.speechInterface.isRecording = false;
  
  if (this.animationFrameId) {
    cancelAnimationFrame(this.animationFrameId);
    this.animationFrameId = undefined;
  }
  
  // Clean up ScriptProcessorNode
  if ((this as any).audioProcessor) {
    try {
      (this as any).audioProcessor.disconnect();
      (this as any).audioProcessor = null;
      console.log('🔇 Stopped audio processor');
    } catch (error) {
      console.warn('Warning: Error disconnecting audio processor:', error);
    }
  }
  
  if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
    try {
      this.mediaRecorder.stop();
      
      // Stop all tracks to free up the microphone
      this.mediaRecorder.stream.getTracks().forEach(track => {
        track.stop();
        console.log('🔇 Stopped audio track:', track.kind);
      });
      
      this.mediaRecorder = null;
    } catch (error) {
      console.warn('Warning: Error stopping MediaRecorder:', error);
    }
  }
  
  if (this.audioContext && this.audioContext.state !== 'closed') {
    try {
      this.audioContext.close();
      this.audioContext = null;
      console.log('🔇 Closed audio context');
    } catch (error) {
      console.warn('Warning: Error closing audio context:', error);
    }
  }
  
  // Clean up microphone reference
  this.microphone = null;
  this.analyser = null;
  
  if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
    this.websocket.send(JSON.stringify({
      type: 'stop_recording'
    }));
  }

  this.speechInterface.status = 'Recording stopped';
  this.speechInterface.audioLevel = 0;
  
  console.log('✅ Recording stopped successfully');
}



  /**
   * Detect if running on mobile device
   */
  private isMobileDevice(): boolean {
    return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) ||
           (navigator.maxTouchPoints !== undefined && navigator.maxTouchPoints > 1);
  }


  // ===== END NEW SPEECH INTERFACE METHODS =====

  private resizeCanvas(width: number, height: number): void {
    this.canvasSize = Math.min(width, height);

    const canvas = this.displayCanvas?.nativeElement;
    if(canvas) {
      canvas.width = this.canvasSize;
      canvas.height = this.canvasSize;
      canvas.style.width = `${this.canvasSize}px`;
      canvas.style.height = `${this.canvasSize}px`;
    }
  }

  private recalculateValues(generalSettings: StandardDisplayGeneralSettings): void {
    const sideCount = generalSettings.numberOfSides;

    this.angle = 2 * Math.PI / sideCount;
    let outerPolygon = this.helperService.getMaxRegPolygonPointsHeuristic(this.canvasSize, sideCount, false);
    this.polygonInfo = { rotation: outerPolygon.angle, offset: outerPolygon.offset, sides: sideCount };
    this.outerEdgePoints = this.helperService.centerPoints(outerPolygon.points, outerPolygon.offset).points;

    this.innerEdgePoints = [];
    for(let i = 0; i < sideCount; i++) {
      this.innerEdgePoints.push(this.helperService.getPointOnCircle(generalSettings.innerPolygonSize, i * this.angle - this.polygonInfo.rotation, {x: 0, y: 0}));
    }
    this.innerEdgePoints = this.helperService.centerPoints(this.innerEdgePoints, outerPolygon.offset).points;
    this.innerEdgePoints.reverse();
    const lastPoint = this.innerEdgePoints.pop() as Point;
    this.innerEdgePoints.unshift(lastPoint);

    this.innerPolygonIncircleRadius = this.helperService.getRadiusOfIncircleOfRegularPolygon((generalSettings.innerPolygonSize) / 2, sideCount);

    // reset transformation matrices because they are not valid anymore
    this.transformationMatrices = [];
  }

  // Screen Streaming Methods (EXISTING - UNCHANGED)
  startScreenStreaming(): void {
    if (this.isScreenStreaming) {
      this.stopScreenStreaming();
      return;
    }

    console.log(`🎬 Starting ${this.streamConfig.mode} streaming with saved GIF system`);

    // Clear existing display
    const settings = this.settingsBroker.getSettings();
    settings.fileSettings = [];
    this.settingsBroker.updateSettings(settings, this.MY_SETTINGS_BROKER_ID);

    this.currentGifIndex = 0;

    // Start debug screenshots only for screen mode
    if (this.streamConfig.mode === 'screen') {
      this.startDebugScreenshots();
    }

    console.log('🔗 Component subscribing to streaming service...');
    this.streamingSubscription = this.screenStreaming.startScreenStream(this.streamConfig)
      .subscribe({
        next: (savedGif: SavedGif) => {
          const mode = this.streamConfig.mode === 'speech' ? 'speech text' : 'screen';
          console.log(`🎯 Component received saved ${mode} GIF:`, savedGif.id, `(${savedGif.size} bytes)`);
          this.loadSavedGifToDisplay(savedGif);
        },
        error: (error) => {
          console.error('❌ Component streaming error:', error);
          this.stopScreenStreaming();
          
          if (error.message && error.message.includes('404')) {
            alert('Streaming failed. Please ensure the server is running on port 3000.\n\nStart the server with:\ncd holodisplay-server\nnpm start');
          } else {
            alert('Streaming failed with an unknown error. Check the console for details.');
          }
        },
        complete: () => {
          console.log('✅ Component streaming completed');
        }
      });
      
    console.log('📝 Component subscription created:', !!this.streamingSubscription);

    // Start animation cycle - switch GIFs every 3 seconds (faster for speech - every 1 second)
    const cycleInterval = this.streamConfig.mode === 'speech' ? 1000 : 3000;
    this.animationInterval = setInterval(() => {
      this.showNextSavedGif();
    }, cycleInterval);

    this.isScreenStreaming = true;
  }

  startSpeechStreaming(): void {
    if (this.isScreenStreaming) {
      this.stopScreenStreaming();
      return;
    }

    console.log('🎤 Starting direct speech text streaming');

    // Clear existing display
    const settings = this.settingsBroker.getSettings();
    settings.fileSettings = [];
    this.settingsBroker.updateSettings(settings, this.MY_SETTINGS_BROKER_ID);

    // Start speech recognition on server
    this.startSpeechRecognition();

    // Poll for speech text updates
    this.speechUpdateInterval = setInterval(() => {
      this.fetchCurrentSpeechText();
    }, 250); // Update every 250ms for responsive text

    this.isScreenStreaming = true;
    this.streamConfig.mode = 'speech';
  }

  private async startSpeechRecognition(): Promise<void> {
    try {
      const apiUrl = window.location.hostname === 'localhost' && window.location.port === '4200' 
        ? 'http://localhost:3000/api/speech/start-progressive'
        : '/api/speech/start-progressive';
      
      console.log('🎤 Starting progressive speech recognition...');
      
      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        }
      });

      if (response.ok) {
        const result = await response.json();
        console.log('✅ Progressive speech recognition started:', result.message);
      }
    } catch (error) {
      console.error('❌ Error starting progressive speech recognition:', error);
    }
  }

  private async stopSpeechRecognition(): Promise<void> {
    try {
      const apiUrl = window.location.hostname === 'localhost' && window.location.port === '4200' 
        ? 'http://localhost:3000/api/speech/stop-progressive'
        : '/api/speech/stop-progressive';
      
      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        }
      });

      if (response.ok) {
        console.log('✅ Speech recognition stopped');
      }
    } catch (error) {
      console.error('❌ Error stopping speech recognition:', error);
    }
  }

  private async fetchCurrentSpeechText(): Promise<void> {
    try {
      const apiUrl = window.location.hostname === 'localhost' && window.location.port === '4200' 
        ? 'http://localhost:3000/api/speech/current'
        : '/api/speech/current';
      
      const response = await fetch(apiUrl);
      
      if (response.ok) {
        const result = await response.json();
        const newText = result.currentText || '';
        
        if (newText !== this.currentSpeechText) {
          this.currentSpeechText = newText;
          console.log(`💬 Speech text updated: "${this.currentSpeechText}"`);
          this.requestDraw$.next(); // Trigger redraw with new text
        }
      }
    } catch (error) {
      console.error('❌ Error fetching speech text:', error);
    }
  }

  stopScreenStreaming(): void {
    if (this.streamingSubscription) {
      this.streamingSubscription.unsubscribe();
      this.streamingSubscription = undefined;
    }
    
    if (this.speechUpdateInterval) {
      clearInterval(this.speechUpdateInterval);
      this.speechUpdateInterval = undefined;
    }
    
    this.screenStreaming.stopScreenStream();
    this.isScreenStreaming = false;
    
    // Stop debug screenshots and animation
    this.stopDebugScreenshots();
    
    if (this.animationInterval) {
      clearInterval(this.animationInterval);
      this.animationInterval = null;
    }
    
    // Stop speech recognition if it was running
    if (this.streamConfig.mode === 'speech') {
      this.stopSpeechRecognition();
      this.currentSpeechText = '';
    }
    
    // Remove streamed content from display
    this.removeStreamedContent();
    
    // Clear saved GIFs
    this.screenStreaming.clearSavedGifs();
  }

  toggleScreenStreaming(): void {
    if (this.isScreenStreaming) {
      this.stopScreenStreaming();
    } else {
      this.streamConfig.mode = 'screen'; // Ensure screen mode
      this.streamConfig.fps = 10; // Reset to normal FPS
      this.startScreenStreaming();
    }
  }

  toggleSpeechStreaming(): void {
    if (this.isScreenStreaming && this.streamConfig.mode === 'speech') {
      this.stopScreenStreaming();
    } else {
      this.startSpeechStreaming();
    }
  }

  isScreenStreamingActive(): boolean {
    return this.isScreenStreaming && this.streamConfig.mode === 'screen';
  }

  isSpeechStreamingActive(): boolean {
    return this.isScreenStreaming && this.streamConfig.mode === 'speech';
  }

  getCurrentStreamingMode(): string {
    if (!this.isScreenStreaming) return 'none';
    return this.streamConfig.mode || 'screen';
  }

  // Dark fullscreen methods
  toggleDarkFullscreen(): void {
    this.isDarkFullscreen = !this.isDarkFullscreen;
    
    if (this.isDarkFullscreen) {
      // Enter dark fullscreen
      document.documentElement.requestFullscreen();
      document.body.style.backgroundColor = '#000000';
      document.body.style.overflow = 'hidden';
    } else {
      // Exit dark fullscreen
      if (document.fullscreenElement) {
        document.exitFullscreen();
      }
      document.body.style.backgroundColor = '';
      document.body.style.overflow = '';
    }
  }

  isDarkFullscreenActive(): boolean {
    return this.isDarkFullscreen;
  }

  // Menu visibility methods
  toggleMenuVisibility(): void {
    this.menuHidden = !this.menuHidden;
  }

  getGifStats() {
    return this.screenStreaming.getStorageStats();
  }

  applyStreamConfig(): void {
    console.log('Applying stream config:', this.streamConfig);
    
    if (this.isScreenStreaming) {
      // Restart streaming with new config
      this.stopScreenStreaming();
      setTimeout(() => {
        this.startScreenStreaming();
      }, 500);
    }
    
    this.toggleModal('streamConfigModal');
  }

  // Saved GIF handling methods (EXISTING - UNCHANGED)
  private loadSavedGifToDisplay(savedGif: SavedGif): void {
    console.log(`🎯 Loading saved GIF to display: ${savedGif.id} (index: ${savedGif.index})`);
    
    const settings = this.settingsBroker.getSettings();
    
    // Clear existing files
    settings.fileSettings = [];
    
    // Create new file entry using the SAME method as settings loading
    const streamedFile = this.settingsBroker.fillMissingFileValues({
      unique_id: `stream-${savedGif.id}`,
      src: savedGif.dataUrl, // Use data URL just like settings files
      mimeType: 'image/gif',
      displayIndex: 0,
      files: {
        original: [],
        scaled: [],
        currentFileIndex: 0
      },
      metaData: {
        [MetaDataKeys.LOADING_PROGRESS]: undefined,
        ...({} as any)
      },
      fileName: `Stream GIF ${savedGif.index}`,
      scalingFactor: 100,
      position: 0,
      rotation: 0,
      brightness: 100,
      flips: { v: false, h: false }
    } as any);
    
    // Add metadata
    (streamedFile.metaData as any)['isStreamedContent'] = true;
    (streamedFile.metaData as any)['streamSource'] = 'screen';
    (streamedFile.metaData as any)['savedGifId'] = savedGif.id;
    (streamedFile.metaData as any)['savedGifIndex'] = savedGif.index;
    
    settings.fileSettings.push(streamedFile);
    
    // Update settings with a DIFFERENT ID to trigger processing
    console.log(`📤 Updating settings with saved GIF: ${savedGif.id} using different broker ID`);
    this.settingsBroker.updateSettings(settings, 'ScreenStreamingLoader');
    
    console.log(`📊 Settings after update:`, {
      fileCount: settings.fileSettings.length,
      file: settings.fileSettings[0] ? {
        id: settings.fileSettings[0].unique_id,
        src: settings.fileSettings[0].src?.substring(0, 50) + '...',
        mimeType: settings.fileSettings[0].mimeType,
        hasOriginal: settings.fileSettings[0].files.original.length,
        hasScaled: settings.fileSettings[0].files.scaled.length
      } : 'none'
    });
  }

  private showNextSavedGif(): void {
    const savedGifs = this.screenStreaming.getSavedGifs();
    
    if (savedGifs.length === 0) {
      console.log('No saved GIFs to display');
      return;
    }
    
    // Cycle through saved GIFs
    this.currentGifIndex = (this.currentGifIndex + 1) % savedGifs.length;
    const currentGif = savedGifs[this.currentGifIndex];
    
    console.log(`🔄 Switching to saved GIF ${this.currentGifIndex + 1}/${savedGifs.length}: ${currentGif.id}`);
    this.loadSavedGifToDisplay(currentGif);
  }

  private removeStreamedContent(): void {
    const settings = this.settingsBroker.getSettings();
    const streamedFileIndex = settings.fileSettings.findIndex(f => 
      (f.metaData as any)?.['isStreamedContent'] === true
    );
    
    if (streamedFileIndex !== -1) {
      const streamedFile = settings.fileSettings[streamedFileIndex];
      if (streamedFile.fps) {
        window.clearInterval(streamedFile.fps.intervalId);
      }
      
      settings.fileSettings.splice(streamedFileIndex, 1);
      this.settingsBroker.updateSettings(settings, this.MY_SETTINGS_BROKER_ID);
      this.requestDraw$.next();
    }
  }

  // Debug methods for testing screenshot API (EXISTING - UNCHANGED)
  private startDebugScreenshots(): void {
    console.log('Starting debug screenshots every 20 seconds...');
    this.debugScreenshotInterval = setInterval(() => {
      this.saveDebugScreenshot();
    }, 20000);
    
    // Take one immediately
    setTimeout(() => this.saveDebugScreenshot(), 1000);
  }

  private stopDebugScreenshots(): void {
    if (this.debugScreenshotInterval) {
      clearInterval(this.debugScreenshotInterval);
      this.debugScreenshotInterval = null;
      console.log('Stopped debug screenshots');
    }
  }

  public async saveDebugScreenshot(): Promise<void> {
    try {
      console.log('Taking debug screenshot...');
      
      const apiUrl = window.location.hostname === 'localhost' && window.location.port === '4200' 
        ? 'http://localhost:3000/api/screen-capture'
        : '/api/screen-capture';
      
      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          quality: this.streamConfig.quality,
          fps: 1,
          format: 'gif'
        }),
        cache: 'no-cache'
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status} - ${response.statusText}`);
      }

      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      
      console.log('Debug screenshot captured successfully, size:', blob.size, 'bytes');
      
      // Download the debug screenshot
      const link = document.createElement('a');
      link.href = url;
      link.download = `debug-screenshot-${new Date().toISOString().replace(/[:.]/g, '-')}.gif`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
      // Clean up
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      
      console.log('Debug screenshot saved to downloads');
      
    } catch (error) {
      console.error('Debug screenshot failed:', error);
    }
  }

  // Test method to verify server connection (EXISTING - UNCHANGED)
  public async testServerConnection(): Promise<void> {
    try {
      console.log('🔍 Testing server connection...');
      
      const healthUrl = window.location.hostname === 'localhost' && window.location.port === '4200' 
        ? 'http://localhost:3000/api/health'
        : '/api/health';
      
      const response = await fetch(healthUrl);
      
      if (!response.ok) {
        throw new Error(`Health check failed: ${response.status}`);
      }
      
      const data = await response.json();
      console.log('✅ Server is running:', data);
      alert(`Server is running!\nPlatform: ${data.platform}\nScreenshot tools: ${data.availableScreenshotTools.join(', ')}\nCan create GIFs: ${data.canCreateGifs}`);
      
    } catch (error) {
      console.error('❌ Server connection test failed:', error);
      alert('Server connection failed. Make sure the server is running on port 3000.');
    }
  }

  // Image scaling and processing methods (EXISTING - UNCHANGED)
  private scaleImagesFromFileSetting(fileSettings: StandardDisplayFileSettings[]) {
    let settingsFinished = 0;

    fileSettings.forEach((fileSetting) => {
      const scalingFactor = fileSetting.scalingFactor;
      const originalFiles = fileSetting.files.original;

      if((fileSetting.mimeType === 'image/gif' || fileSetting.mimeType.startsWith('video')) && originalFiles.length > 0) {
        const newlyScaledFiles: HTMLImageElement[] = [];
        let loadedImages = 0;

        originalFiles.forEach((image) => {
          const scaled = document.createElement('img');
          scaled.src = image.src;
          scaled.width = image.width * scalingFactor/100;
          scaled.height = image.height * scalingFactor/100;

          scaled.onload = () => {
            newlyScaledFiles.push(scaled);
            loadedImages++;
          }

          scaled.onerror = () => {
            console.error('Failed to load gif frame');
            loadedImages++;
          }

        });

        // wait until all images are loaded
        const intervalId = window.setInterval(() => {
          if(loadedImages == originalFiles.length) {
            window.clearInterval(intervalId);
            fileSetting.files.scaled = newlyScaledFiles;
            settingsFinished++;
          }
        }, 100);
      }

      else if(fileSetting.mimeType.startsWith('image') && originalFiles.length > 0) {
        const scaled = document.createElement('img');
        scaled.src = originalFiles[0].src;
        scaled.width = originalFiles[0].width * scalingFactor/100;
        scaled.height = originalFiles[0].height * scalingFactor/100;

        scaled.onload = () => {
          fileSetting.files.scaled = [scaled];
          settingsFinished++;
        }

        scaled.onerror = () => {
          console.error('Failed to load image');
          settingsFinished++;
        }
      }
    });

    return new Promise<void>((resolve) => {
      const intervalId = window.setInterval(() => {
        if(settingsFinished == fileSettings.length) {
          clearInterval(intervalId);
          resolve();
        }
      }, 100);
    });
  }


  private updateImageSettings(settings: StandardDisplaySettings) {
    settings.fileSettings.forEach((latestFile) => {
      const unique_ids_displayed = settings.fileSettings.map((file) => file.unique_id);

      latestFile = this.settingsBroker.fillMissingFileValues(latestFile);

      // update the file if it is already displayed
      if(unique_ids_displayed.includes(latestFile.unique_id) && latestFile.files.original.length > 0) {
        const existingFile = settings.fileSettings.find((file) => file.unique_id == latestFile.unique_id)!;

        // update all changeable settings of the already existing file
        existingFile.brightness = latestFile.brightness;
        existingFile.flips = latestFile.flips;
        existingFile.metaData = latestFile.metaData;
        existingFile.position = latestFile.position;
        existingFile.rotation = latestFile.rotation;
        existingFile.scalingFactor = latestFile.scalingFactor;

        if(existingFile.mimeType === 'image/gif' || existingFile.mimeType.startsWith('video')) {
          const framerate = existingFile.fps?.framerate || (latestFile.mimeType === 'image/gif' ? 10 : 30);

          this.scaleImagesFromFileSetting([existingFile]).then(() => {
            const updatedSettings = this.settingsBroker.getSettings();
            const updatedFileIndex = updatedSettings.fileSettings.findIndex((f) => f.unique_id == existingFile.unique_id);

            if(updatedFileIndex == -1) return;

            if(existingFile.fps)
              window.clearInterval(existingFile.fps.intervalId);

            existingFile.fps = {
              framerate,
              intervalId: window.setInterval(() => {
                const updatedSettings = this.settingsBroker.getSettings();
  
                if(updatedSettings.fileSettings.findIndex((f) => f.unique_id == existingFile.unique_id) == -1) return;
  
                const upToDateFile = updatedSettings.fileSettings.find((f) => f.unique_id == existingFile.unique_id)!;
  
                upToDateFile.files.currentFileIndex = (upToDateFile.files.currentFileIndex + 1) % upToDateFile.files.original.length;
  
                this.requestDraw$.next();
                this.settingsBroker.updateSettings(updatedSettings, this.MY_SETTINGS_BROKER_ID);
              }, 1000/framerate)
            };

            updatedSettings.fileSettings[updatedFileIndex] = existingFile;
            this.settingsBroker.updateSettings(updatedSettings, this.MY_SETTINGS_BROKER_ID);
          });
        }

        else if(existingFile.mimeType.startsWith('image')) {
          this.scaleImagesFromFileSetting([existingFile]).then(() => {
            const updatedSettings = this.settingsBroker.getSettings();
            const updatedFileIndex = updatedSettings.fileSettings.findIndex((f) => f.unique_id == existingFile.unique_id);

            if(updatedFileIndex == -1) return;

            updatedSettings.fileSettings[updatedFileIndex] = existingFile;
            this.settingsBroker.updateSettings(updatedSettings, this.MY_SETTINGS_BROKER_ID);
            this.requestDraw$.next();
          });
        }
      }

      // load the file if it is not already displayed
      else {
        if(!latestFile.src) {
          console.error("Passed file has no src attribute and therefore cannot be loaded!", latestFile);
          return;
        }
        
        if(latestFile.fps)
          window.clearInterval(latestFile.fps.intervalId);

        if(latestFile.mimeType == 'image/gif') {
          // prepare a request to load the gif
          let xhr = new XMLHttpRequest();
          xhr.open('GET', latestFile.src!, true);
          xhr.responseType = 'arraybuffer';

          xhr.onload = () => {
            let arrayBuffer = xhr.response;
            
            if(arrayBuffer) {
              // parse the gif and load the frames
              let gif = parseGIF(arrayBuffer);
              let gifFrames = decompressFrames(gif, true);

              let gifImages: HTMLImageElement[] = [];
              let gifImagesLoaded = 0;

              // wait until all images are loaded by checking the number of loaded images every 100ms
              const interval = window.setInterval(() => {
                if(gifImagesLoaded == gifFrames.length) {
                  window.clearInterval(interval);
    
                  latestFile.files.original = gifImages;
                  latestFile.files.currentFileIndex = 0;
                  this.scaleImagesFromFileSetting([latestFile]).then(() => {
                    const updatedSettings = this.settingsBroker.getSettings();
                    const updatedFileIndex = updatedSettings.fileSettings.findIndex((f) => f.unique_id == latestFile.unique_id);
    
                    if(updatedFileIndex == -1) return;

                    if(latestFile.fps)
                      window.clearInterval(latestFile.fps.intervalId);
      
                    const framerate = latestFile.fps?.framerate || 10;

                    latestFile.fps = {
                      framerate,
                      intervalId: window.setInterval(() => {
                        const updatedSettings = this.settingsBroker.getSettings();
      
                        if(updatedSettings.fileSettings.findIndex((f) => f.unique_id == latestFile.unique_id) == -1) return;
      
                        const upToDateFile = updatedSettings.fileSettings.find((f) => f.unique_id == latestFile.unique_id)!;
      
                        upToDateFile.files.currentFileIndex = (upToDateFile.files.currentFileIndex + 1) % upToDateFile.files.original.length;
  
                        this.requestDraw$.next();
                      }, 1000/framerate)
                    }
    
                    updatedSettings.fileSettings[updatedFileIndex] = latestFile;
                    this.settingsBroker.updateSettings(updatedSettings, this.MY_SETTINGS_BROKER_ID);
                  });
                }
              })

              // load the images
              gifFrames.forEach((frame) => {
                let imageData = new ImageData(frame.patch, frame.dims.width, frame.dims.height);
                let canvas = document.createElement('canvas');
                canvas.width = frame.dims.width;
                canvas.height = frame.dims.height;
                let ctx = canvas.getContext('2d');
                ctx?.putImageData(imageData, 0, 0);

                let image = new Image();
                image.src = canvas.toDataURL();
                image.width = canvas.width;
                image.height = canvas.height;

                gifImages.push(image);

                image.onload = () => {
                  gifImagesLoaded++;
                }
              });
            }

            else {
              const settings = this.settingsBroker.getSettings();
              const latestFileIndex = settings.fileSettings.findIndex((f) => f.unique_id == latestFile.unique_id);

              settings.fileSettings.splice(latestFileIndex, 1);
              alert('Failed to load gif');
              this.settingsBroker.updateSettings(settings, this.MY_SETTINGS_BROKER_ID);
              return;
            }
          }

          xhr.onerror = () => {
            const settings = this.settingsBroker.getSettings();
            const latestFileIndex = settings.fileSettings.findIndex((f) => f.unique_id == latestFile.unique_id);

            settings.fileSettings.splice(latestFileIndex, 1);
            alert('Failed to load gif');
            this.settingsBroker.updateSettings(settings, this.MY_SETTINGS_BROKER_ID);
            return;
          }

          xhr.send();
        }
        
        else if(['image/jpeg', 'image/png', 'image/webp'].includes(latestFile.mimeType)) {          
          const originalImage = new Image();
          originalImage.src = latestFile.src || '';

          latestFile.files.original = [originalImage];
          latestFile.files.currentFileIndex = 0;

          originalImage.onload = () => this.scaleImagesFromFileSetting([latestFile]).then(() => {
              const updatedSettings = this.settingsBroker.getSettings();
              const updatedFileIndex = updatedSettings.fileSettings.findIndex((f) => f.unique_id == latestFile.unique_id);

              if(updatedFileIndex == -1) return;

              updatedSettings.fileSettings[updatedFileIndex] = latestFile;
              this.settingsBroker.updateSettings(updatedSettings, this.MY_SETTINGS_BROKER_ID);
              this.requestDraw$.next();
            });
        }

        else if(latestFile.mimeType.startsWith('video')) {
          // clear the interval id if it exists
          if(latestFile.fps)
            window.clearInterval(latestFile.fps.intervalId);

          // init a video element to load the video
          let video = document.createElement('video');
          video.src = latestFile.src!;
          
          video.onloadeddata = () => {
            // load the video and extract the frames to handle it as a gif
            const videoFrames = require('video-frames');

            videoFrames({
              url: video.src,
              count: video.duration * 30, // extract 30 frames per second
              width: 720,
              onProgress: (framesExtracted: number, totalFrames: number) => {
                const updatedSettings = this.settingsBroker.getSettings();

                if(updatedSettings.fileSettings.findIndex((f) => f.unique_id == latestFile.unique_id) == -1)
                  return;

                updatedSettings.fileSettings.find((f) => f.unique_id == latestFile.unique_id)!.metaData[MetaDataKeys.LOADING_PROGRESS] = $localize`${framesExtracted} of ${totalFrames} frames`;
              }
            }).then((frames: { offset: number, image: string }[]) => {
              const updatedSettings = this.settingsBroker.getSettings();
              const updatedFile = updatedSettings.fileSettings.find((f) => f.unique_id == latestFile.unique_id);

              if(!updatedFile) return;

              updatedFile.metaData[MetaDataKeys.LOADING_PROGRESS] = $localize`Finalizing...`;
              this.settingsBroker.updateSettings(updatedSettings, this.MY_SETTINGS_BROKER_ID);

              let videoImages: HTMLImageElement[] = [];
              let videoImagesLoaded = 0;

              // wait until all images are loaded by checking the number of loaded images every 100ms
              const interval = window.setInterval(() => {
                if(videoImagesLoaded == frames.length) {
                  window.clearInterval(interval);
              
                  if(updatedFile.fps)
                    window.clearInterval(updatedFile.fps.intervalId);

                  delete latestFile.metaData[MetaDataKeys.LOADING_PROGRESS];
                  latestFile.files.original = videoImages;
                  latestFile.files.currentFileIndex = 0;
    
                  if(latestFile.fps)
                    window.clearInterval(latestFile.fps.intervalId);

                  const framerate = latestFile.fps?.framerate || 30;
    
                  this.scaleImagesFromFileSetting([latestFile]).then(() => {
                    latestFile.fps = {
                      framerate,
                      intervalId: window.setInterval(() => {
                        const updatedSettings = this.settingsBroker.getSettings();
      
                        if(updatedSettings.fileSettings.findIndex((f) => f.unique_id == latestFile.unique_id) == -1) return;
      
                        const upToDateFile = updatedSettings.fileSettings.find((f) => f.unique_id == latestFile.unique_id)!;
                        upToDateFile.files.currentFileIndex = (upToDateFile.files.currentFileIndex + 1) % upToDateFile.files.original.length;
      
                        this.requestDraw$.next();
                      }, 1000/framerate)
                    };
    
                    const updatedSettings = this.settingsBroker.getSettings();
                    const i = updatedSettings.fileSettings.findIndex((f) => f.unique_id == latestFile.unique_id);
                    updatedSettings.fileSettings[i] = latestFile;
      
                    this.settingsBroker.updateSettings(updatedSettings!, this.MY_SETTINGS_BROKER_ID);
                  });
                }
              }, 100);

              // load the images
              frames.forEach((frame) => {
                let image = new Image();
                image.src = frame.image;

                videoImages.push(image);

                image.onload = () => {
                  videoImagesLoaded++;
                }
              });
            });
          }
        }
      }
    });

    // remove all files that are not in the latest settings anymore
    settings.fileSettings.forEach((file) => {
      if(settings.fileSettings.findIndex((f) => f.unique_id == file.unique_id) == -1) {
        if(file.fps)
          window.clearInterval(file.fps.intervalId);
      }
    });

    this.settingsBroker.updateSettings(settings, this.MY_SETTINGS_BROKER_ID);
  }


  // Drawing methods (EXISTING - UNCHANGED)
  private draw(): void {
    const canvas = this.displayCanvas?.nativeElement;
    if(!canvas) return;

    const ctx = canvas.getContext('2d') as CanvasRenderingContext2D;
    ctx.save();
    ctx.resetTransform();
    canvas.width = canvas.width;  // clears canvas as a side effect

    ctx.translate(canvas.width / 2, canvas.height / 2);
    this.helperService.connectPointsWithStraightLines(ctx, this.innerEdgePoints, 'blue');
    this.helperService.connectPointsWithStraightLines(ctx, this.outerEdgePoints, 'red');

    const settings = this.settingsBroker.getSettings();

    // Check if we're in speech mode and have text to display
    if (this.streamConfig.mode === 'speech' && this.currentSpeechText) {
      this.drawSpeechText(ctx);
      ctx.restore();
      return;
    }

    console.log('Drawing with', settings.fileSettings.length, 'file settings:', 
      settings.fileSettings.map(f => ({ id: f.unique_id, displayIndex: f.displayIndex, hasImages: f.files?.original?.length || 0 }))
    );

    if (settings.fileSettings.length === 0) {
      console.log('No files to display');
      return;
    }

    const maxDisplayIndex = Math.max(...settings.fileSettings.map((file) => file.displayIndex)) + 1;

    for(let iSide = 0; iSide < settings.generalSettings.numberOfSides; iSide++) {
      const iImage = iSide % maxDisplayIndex;
      const imageData = settings.fileSettings.find((entry) => entry.displayIndex === iImage);

      if(!imageData) {
        continue;
      };

      console.log(`Side ${iSide}: Using image ${iImage} (${imageData.unique_id}) with ${imageData.files?.scaled?.length || 0} scaled images`);

      // load the image
      const image = imageData.files.scaled[Math.min(imageData.files.currentFileIndex, imageData.files.scaled.length - 1)];
      if(!image) {
        console.error(`No image found for side ${iSide} in image data!`, imageData);
        continue;
      }

      // reset the clip mask
      ctx.restore();
      ctx.resetTransform();
      ctx.save();

      // store or apply transformation matrix instead of rotating and translating manually
      if(this.transformationMatrices[iSide] && this.transformationMatrices[iSide][0]) {
        ctx.setTransform(this.transformationMatrices[iSide][0]);
      } else {
        // apply the translation and rotation
        ctx.translate((this.canvasSize/2 - this.polygonInfo.offset.dx), (this.canvasSize/2 - this.polygonInfo.offset.dy));
        ctx.rotate(iSide * this.angle);

        // store the transformation matrix
        this.transformationMatrices[iSide] = [ctx.getTransform()];
      }

      // create the clip mask
      ctx.beginPath();
      ctx.moveTo(this.innerEdgePoints[0].x + this.polygonInfo.offset.dx, this.innerEdgePoints[0].y + this.polygonInfo.offset.dy);
      ctx.lineTo(this.outerEdgePoints[0].x + this.polygonInfo.offset.dx, this.outerEdgePoints[0].y + this.polygonInfo.offset.dy);
      ctx.lineTo(this.outerEdgePoints[1].x + this.polygonInfo.offset.dx, this.outerEdgePoints[1].y + this.polygonInfo.offset.dy);
      ctx.lineTo(this.innerEdgePoints[1].x + this.polygonInfo.offset.dx, this.innerEdgePoints[1].y + this.polygonInfo.offset.dy);
      ctx.closePath();
      ctx.clip();

      // undo the rotation
      ctx.rotate(-iSide * this.angle);
      
      // store or apply transformation matrix instead of rotating and translating manually
      if(this.transformationMatrices[iSide] && this.transformationMatrices[iSide][1]) {
        ctx.setTransform(this.transformationMatrices[iSide][1]);
      } else {
        ctx.rotate(Math.PI)
        ctx.rotate((iSide - (0.25 * (this.polygonInfo.sides - 2))) * this.angle + this.polygonInfo.rotation); // Why does this equation work?
        ctx.translate(0, -this.innerPolygonIncircleRadius - this.canvasSize/4 - imageData.position);
        ctx.rotate(imageData.rotation * Math.PI / 180);

        // store the transformation matrix
        this.transformationMatrices[iSide].push(ctx.getTransform());
      }

      // apply the flip
      ctx.scale(imageData.flips.h ? -1 : 1, imageData.flips.v ? -1 : 1);
      // apply the brightness change
      ctx.filter = `brightness(${imageData.brightness}%)`;

      // draw the image
      ctx.drawImage(
        image,
        -image.width/2,
        -image.height/2,
        image.width,
        image.height
      );
    }
  }

  private drawSpeechText(ctx: CanvasRenderingContext2D): void {
    console.log(`🎨 Drawing speech text: "${this.currentSpeechText}"`);

    if (!this.currentSpeechText.trim()) {
      // Show "Listening..." when no text on all sides
      for(let iSide = 0; iSide < this.polygonInfo.sides; iSide++) {
        ctx.save();
        
        // Position text around the holographic area
        const textRadius = this.canvasSize / 3.2;
        const sideAngle = iSide * this.angle;
        
        const textX = textRadius * Math.cos(sideAngle);
        const textY = textRadius * Math.sin(sideAngle);
        
        ctx.translate(textX, textY);
        ctx.rotate(sideAngle + Math.PI/2);

        ctx.font = `${Math.max(12, this.canvasSize / 50)}px Arial`;
        ctx.fillStyle = '#666666';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        
        ctx.fillText('Listening...', 0, 0);
        ctx.restore();
      }
      return;
    }

    // Draw the SAME text on all 4 sides for holographic effect
    for(let iSide = 0; iSide < this.polygonInfo.sides; iSide++) {
      ctx.save();
      
      // Position text around the holographic area (same radius for all sides)
      const textRadius = this.canvasSize / 3.2; // Adjust distance from center
      const sideAngle = iSide * this.angle;
      
      const textX = textRadius * Math.cos(sideAngle);
      const textY = textRadius * Math.sin(sideAngle);
      
      ctx.translate(textX, textY);
      
      // Rotate text to be readable from each viewing angle
      ctx.rotate(sideAngle + Math.PI/2);

      // Configure text rendering - same for all sides
      const fontSize = Math.max(14, this.canvasSize / 35);
      ctx.font = `bold ${fontSize}px Arial`;
      ctx.fillStyle = '#ffffff';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      
      // Add text shadow for better visibility
      ctx.shadowColor = '#000000';
      ctx.shadowBlur = 3;
      ctx.shadowOffsetX = 1;
      ctx.shadowOffsetY = 1;

      // Split text into lines if it's too long
      const maxCharsPerLine = 25;
      const words = this.currentSpeechText.split(' ');
      const lines = [];
      let currentLine = '';
      
      for (const word of words) {
        if ((currentLine + ' ' + word).length <= maxCharsPerLine) {
          currentLine += (currentLine ? ' ' : '') + word;
        } else {
          if (currentLine) lines.push(currentLine);
          currentLine = word;
        }
      }
      if (currentLine) lines.push(currentLine);

      // Draw each line with proper spacing
      const lineHeight = fontSize * 1.3;
      const totalHeight = lines.length * lineHeight;
      const startY = -totalHeight / 2 + lineHeight / 2;

      lines.forEach((line, index) => {
        const y = startY + (index * lineHeight);
        
        // Draw text with shadow
        ctx.fillText(line, 0, y);
        
        // Add outline for better visibility
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 1;
        ctx.strokeText(line, 0, y);
      });
      
      ctx.restore();
    }
  }

  // Calculator methods (EXISTING - UNCHANGED)
  onCalculateClick(): void {
    const settings = this.settingsBroker.getSettings();

    const canvas = this.calculator.calculateImage(
      settings.generalSettings.numberOfSides,
      this.calculatorSlope,
      settings.generalSettings.innerPolygonSize,
      this.canvasSize,
    );

    this.calculatorImageWidthPx = canvas?.width || -1;
    this.calculatorImageHeightPx = canvas?.height || -1;

    // download the image from the canvas
    if(canvas) {
      const link = document.createElement('a');
      link.download = 'mirror cutting template.png';
      link.href = canvas.toDataURL();
      link.click();
    }

    this.toggleModal('calculatorDownloadModal');
  }

  toggleModal(modalId: string): void {
    if(!document.getElementById(modalId)) return;

    document.getElementById(modalId)!.classList.toggle("hidden");
  }
}