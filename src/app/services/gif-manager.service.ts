import { Injectable } from '@angular/core';

export interface StoredGif {
  id: string;
  url: string;
  timestamp: number;
  blob: Blob;
  displayIndex: number;
}

@Injectable({
  providedIn: 'root'
})
export class GifManagerService {
  private storedGifs: Map<string, StoredGif> = new Map();
  private currentIndex = 0;
  private maxGifs = 10; // Store up to 10 recent GIFs
  private cleanupAge = 2 * 60 * 1000; // 2 minutes in milliseconds

  constructor() {
    // Clean up old GIFs every 30 seconds
    setInterval(() => this.cleanupOldGifs(), 30000);
  }

  /**
   * Store a new GIF from blob
   */
  storeGif(blob: Blob): StoredGif {
    const id = `gif_${Date.now()}_${this.currentIndex}`;
    const url = URL.createObjectURL(blob);
    const timestamp = Date.now();
    
    const storedGif: StoredGif = {
      id,
      url,
      timestamp,
      blob,
      displayIndex: this.currentIndex
    };
    
    this.storedGifs.set(id, storedGif);
    this.currentIndex++;
    
    console.log(`📁 Stored GIF ${id}, total stored: ${this.storedGifs.size}`);
    
    // Remove oldest if we exceed limit
    if (this.storedGifs.size > this.maxGifs) {
      this.removeOldestGif();
    }
    
    return storedGif;
  }

  /**
   * Get all stored GIFs sorted by timestamp (newest first)
   */
  getAllGifs(): StoredGif[] {
    return Array.from(this.storedGifs.values())
      .sort((a, b) => b.timestamp - a.timestamp);
  }

  /**
   * Get the most recent GIF
   */
  getLatestGif(): StoredGif | null {
    const gifs = this.getAllGifs();
    return gifs.length > 0 ? gifs[0] : null;
  }

  /**
   * Get GIF by ID
   */
  getGifById(id: string): StoredGif | null {
    return this.storedGifs.get(id) || null;
  }

  /**
   * Get GIFs for animation (last N GIFs)
   */
  getAnimationFrames(count: number = 5): StoredGif[] {
    return this.getAllGifs().slice(0, count);
  }

  /**
   * Remove a specific GIF
   */
  removeGif(id: string): boolean {
    const gif = this.storedGifs.get(id);
    if (gif) {
      URL.revokeObjectURL(gif.url);
      this.storedGifs.delete(id);
      console.log(`🗑️ Removed GIF ${id}`);
      return true;
    }
    return false;
  }

  /**
   * Remove oldest GIF
   */
  private removeOldestGif(): void {
    const gifs = this.getAllGifs();
    if (gifs.length > 0) {
      const oldest = gifs[gifs.length - 1];
      this.removeGif(oldest.id);
    }
  }

  /**
   * Clean up GIFs older than specified age
   */
  private cleanupOldGifs(): void {
    const now = Date.now();
    const toRemove: string[] = [];
    
    this.storedGifs.forEach((gif, id) => {
      if (now - gif.timestamp > this.cleanupAge) {
        toRemove.push(id);
      }
    });
    
    toRemove.forEach(id => this.removeGif(id));
    
    if (toRemove.length > 0) {
      console.log(`🧹 Cleaned up ${toRemove.length} old GIFs`);
    }
  }

  /**
   * Clear all stored GIFs
   */
  clearAll(): void {
    this.storedGifs.forEach((gif, id) => {
      URL.revokeObjectURL(gif.url);
    });
    this.storedGifs.clear();
    console.log('🗑️ Cleared all stored GIFs');
  }

  /**
   * Get storage stats
   */
  getStats() {
    const gifs = this.getAllGifs();
    const totalSize = gifs.reduce((sum, gif) => sum + gif.blob.size, 0);
    
    return {
      count: gifs.length,
      totalSizeBytes: totalSize,
      totalSizeMB: (totalSize / (1024 * 1024)).toFixed(2),
      oldest: gifs.length > 0 ? gifs[gifs.length - 1].timestamp : null,
      newest: gifs.length > 0 ? gifs[0].timestamp : null
    };
  }
}