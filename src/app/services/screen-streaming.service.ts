import { Injectable } from '@angular/core';
import { Observable, Subject, interval, switchMap, takeUntil, catchError, of } from 'rxjs';

export interface StreamConfig {
  quality: 'low' | 'medium' | 'high';
  fps: number;
  format: 'gif' | 'webm' | 'mp4';
  mode?: 'screen' | 'speech';
}

export interface SavedGif {
  id: string;
  index: number;
  timestamp: number;
  dataUrl: string;
  size: number;
}

export interface SpeechStatus {
  isConnected: boolean;
  isRecording: boolean;
  modelLoaded: boolean;
  modelLoading: boolean;
  currentText: string;
  error?: string;
}

@Injectable({
  providedIn: 'root'
})
export class ScreenStreamingService {
  private streamingSubject = new Subject<SavedGif>();
  private stopStreamingSubject = new Subject<void>();
  private isStreaming = false;
  private savedGifs: SavedGif[] = [];
  private gifIndex = 0;

  // NEW: Speech interface integration
  private speechStatusSubject = new Subject<SpeechStatus>();
  private currentSpeechStatus: SpeechStatus = {
    isConnected: false,
    isRecording: false,
    modelLoaded: false,
    modelLoading: false,
    currentText: 'Ready to connect'
  };

  constructor() {}

  /**
   * Start streaming screen from server - save GIFs locally and emit saved GIF info
   */
  startScreenStream(config: StreamConfig = { quality: 'medium', fps: 10, format: 'gif', mode: 'screen' }): Observable<SavedGif> {
    console.log('🚀 ScreenStreamingService.startScreenStream called, isStreaming:', this.isStreaming);
    
    if (this.isStreaming) {
      console.log('⚠️ Already streaming, returning existing observable');
      return this.streamingSubject.asObservable();
    }

    this.isStreaming = true;
    console.log('🚀 ScreenStreamingService: Starting NEW stream with config:', config);
    
    // Clear previous saved GIFs
    this.savedGifs = [];
    this.gifIndex = 0;
    
    // For speech mode, we DON'T start speech recognition here anymore
    // The speech interface modal handles connection and recording
    // This service only handles displaying the speech text as images
    
    // Poll for new captures from server
    const pollInterval = 1000 / config.fps;
    console.log(`📅 Polling interval: ${pollInterval}ms (${config.fps} FPS)`);
    
    interval(pollInterval).pipe(
      switchMap(() => {
        console.log(`🔄 Fetching new ${config.mode} capture...`);
        return config.mode === 'speech' 
          ? this.fetchSpeechTextStream(config)
          : this.fetchAndSaveScreenCapture(config);
      }),
      catchError((error) => {
        console.error('❌ Screen streaming fetch error:', error);
        return of(null);
      }),
      takeUntil(this.stopStreamingSubject)
    ).subscribe({
      next: (savedGif) => {
        if (savedGif) {
          console.log('✅ ScreenStreamingService: Saved GIF locally, emitting to component:', savedGif.id);
          this.streamingSubject.next(savedGif);
          console.log('📤 ScreenStreamingService: SavedGif emitted to component');
        } else {
          console.log('⚠️ Failed to save GIF locally');
        }
      },
      error: (error) => {
        console.error('❌ Screen streaming error:', error);
        this.stopScreenStream();
      },
      complete: () => {
        console.log('✅ Screen streaming completed');
        this.isStreaming = false;
      }
    });

    console.log('🔄 ScreenStreamingService: Returning observable to component');
    return this.streamingSubject.asObservable();
  }

  /**
   * Stop screen streaming
   */
  stopScreenStream(): void {
    this.isStreaming = false;
    this.stopStreamingSubject.next();
    console.log('🛑 Screen streaming stopped');
  }

  /**
   * Get speech status observable
   */
  getSpeechStatus(): Observable<SpeechStatus> {
    return this.speechStatusSubject.asObservable();
  }

  /**
   * Get current speech status
   */
  getCurrentSpeechStatus(): SpeechStatus {
    return { ...this.currentSpeechStatus };
  }

  /**
   * Update speech status (called by component)
   */
  updateSpeechStatus(status: Partial<SpeechStatus>): void {
    this.currentSpeechStatus = { ...this.currentSpeechStatus, ...status };
    this.speechStatusSubject.next(this.currentSpeechStatus);
    console.log('📊 Speech status updated:', this.currentSpeechStatus);
  }

  /**
   * Check if speech is ready for streaming (connected + recording)
   */
  isSpeechReadyForStreaming(): boolean {
    return this.currentSpeechStatus.isConnected && this.currentSpeechStatus.isRecording;
  }

  /**
   * Fetch screen capture and save locally, return saved GIF info
   */
  private async fetchAndSaveScreenCapture(config: StreamConfig): Promise<SavedGif | null> {
    try {
      const apiUrl = window.location.hostname === 'localhost' && window.location.port === '4200' 
        ? 'http://localhost:3000/api/screen-capture'
        : '/api/screen-capture';
      
      console.log(`📡 Fetching from: ${apiUrl}`);
        
      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(config),
        cache: 'no-cache'
      });

      console.log(`📈 Response status: ${response.status} ${response.statusText}`);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status} - ${response.statusText}`);
      }

      const blob = await response.blob();
      console.log(`📦 Received blob: ${blob.size} bytes, type: ${blob.type}`);
      
      if (blob.size === 0) {
        console.warn('⚠️ Received empty blob');
        return null;
      }
      
      // Convert blob to data URL for local storage
      const dataUrl = await this.blobToDataUrl(blob);
      
      const savedGif: SavedGif = {
        id: `gif_${Date.now()}_${this.gifIndex}`,
        index: this.gifIndex,
        timestamp: Date.now(),
        dataUrl: dataUrl,
        size: blob.size
      };
      
      // Store locally
      this.savedGifs.push(savedGif);
      this.gifIndex++;
      
      // Keep only last 10 GIFs to prevent memory issues
      if (this.savedGifs.length > 10) {
        const removed = this.savedGifs.shift();
        console.log(`🗑️ Removed old GIF: ${removed?.id}`);
      }
      
      console.log(`💾 Saved GIF locally: ${savedGif.id} (total: ${this.savedGifs.length})`);
      
      return savedGif;
      
    } catch (error) {
      console.error('❌ Failed to fetch and save screen capture:', error);
      return null;
    }
  }

  /**
   * Fetch speech text stream and save locally, return saved GIF info
   * This now gets the REAL speech text from the server that was updated by Python backend
   */
  private async fetchSpeechTextStream(config: StreamConfig): Promise<SavedGif | null> {
    try {
      const apiUrl = window.location.hostname === 'localhost' && window.location.port === '4200' 
        ? 'http://localhost:3000/api/speech-text-stream'
        : '/api/speech-text-stream';
      
      console.log(`🎤 Fetching REAL speech text from: ${apiUrl}`);
        
      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(config),
        cache: 'no-cache'
      });

      console.log(`📈 Speech response status: ${response.status} ${response.statusText}`);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status} - ${response.statusText}`);
      }

      const blob = await response.blob();
      console.log(`📦 Received speech blob: ${blob.size} bytes, type: ${blob.type}`);
      
      if (blob.size === 0) {
        console.warn('⚠️ Received empty speech blob');
        return null;
      }
      
      // Convert blob to data URL for local storage
      const dataUrl = await this.blobToDataUrl(blob);
      
      const savedGif: SavedGif = {
        id: `speech_gif_${Date.now()}_${this.gifIndex}`,
        index: this.gifIndex,
        timestamp: Date.now(),
        dataUrl: dataUrl,
        size: blob.size
      };
      
      // Store locally
      this.savedGifs.push(savedGif);
      this.gifIndex++;
      
      // Keep only last 5 speech images (they change less frequently)
      if (this.savedGifs.length > 5) {
        const removed = this.savedGifs.shift();
        console.log(`🗑️ Removed old speech GIF: ${removed?.id}`);
      }
      
      console.log(`🎤 Saved REAL speech GIF locally: ${savedGif.id} (total: ${this.savedGifs.length})`);
      
      return savedGif;
      
    } catch (error) {
      console.error('❌ Failed to fetch and save speech text stream:', error);
      return null;
    }
  }

  /**
   * Convert blob to data URL
   */
  private blobToDataUrl(blob: Blob): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result as string);
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  }

  /**
   * Get all saved GIFs
   */
  getSavedGifs(): SavedGif[] {
    return [...this.savedGifs];
  }

  /**
   * Get GIF by ID
   */
  getSavedGifById(id: string): SavedGif | null {
    return this.savedGifs.find(gif => gif.id === id) || null;
  }

  /**
   * Clear all saved GIFs
   */
  clearSavedGifs(): void {
    console.log(`🧹 Clearing ${this.savedGifs.length} saved GIFs`);
    this.savedGifs = [];
    this.gifIndex = 0;
  }

  /**
   * Check if streaming is currently active
   */
  isStreamingActive(): boolean {
    return this.isStreaming;
  }

  /**
   * Get storage stats
   */
  getStorageStats() {
    const totalSize = this.savedGifs.reduce((sum, gif) => sum + gif.size, 0);
    return {
      count: this.savedGifs.length,
      totalSizeBytes: totalSize,
      totalSizeMB: (totalSize / (1024 * 1024)).toFixed(2),
      oldest: this.savedGifs.length > 0 ? this.savedGifs[0].timestamp : null,
      newest: this.savedGifs.length > 0 ? this.savedGifs[this.savedGifs.length - 1].timestamp : null
    };
  }

  /**
   * Test server connection
   */
  async testServerConnection(): Promise<boolean> {
    try {
      const apiUrl = window.location.hostname === 'localhost' && window.location.port === '4200' 
        ? 'http://localhost:3000/api/screen-capture'
        : '/api/screen-capture';
        
      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ quality: 'low', fps: 1, format: 'gif' }),
        cache: 'no-cache'
      });
      return response.ok;
    } catch (error) {
      console.error('Server connection test failed:', error);
      return false;
    }
  }

  /**
   * Check Python backend status
   */
  async checkPythonBackendStatus(): Promise<any> {
    try {
      const apiUrl = window.location.hostname === 'localhost' && window.location.port === '4200' 
        ? 'http://localhost:3000/api/speech/python-status'
        : '/api/speech/python-status';
      
      const response = await fetch(apiUrl);
      return await response.json();
    } catch (error) {
      console.error('❌ Failed to check Python backend status:', error);
      return { pythonBackendRunning: false, message: 'Failed to check status' };
    }
  }
}
