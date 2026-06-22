class WebcamClient {
    constructor(options) {
        this.videoId = options.videoId;
        this.canvasId = options.canvasId;
        this.feedImgId = options.feedImgId;
        
        this.videoEl = document.getElementById(this.videoId);
        this.canvasEl = document.getElementById(this.canvasId);
        this.feedImgEl = document.getElementById(this.feedImgId);
        
        this.onFrameProcessed = options.onFrameProcessed || (() => {});
        this.onStatusChange = options.onStatusChange || (() => {});
        this.onError = options.onError || (() => {});
        
        this.stream = null;
        this.running = false;
        this.activeRequest = false;
        this.loopTimeoutId = null;
        
        // Adaptive Throttling State
        this.baseInterval = options.frameInterval || 300; // default 300ms
        this.frameInterval = this.baseInterval;
        this.quality = options.quality || 0.5; // jpeg compression quality
        this.targetWidth = options.targetWidth || 480;
        
        // Page visibility binding
        this._handleVisibilityChange = this._handleVisibilityChange.bind(this);
        document.addEventListener('visibilitychange', this._handleVisibilityChange);
    }

    async start() {
        if (this.running) return;
        
        // Refresh DOM elements in case they were dynamically recreated
        this.videoEl = document.getElementById(this.videoId);
        this.canvasEl = document.getElementById(this.canvasId);
        this.feedImgEl = document.getElementById(this.feedImgId);

        try {
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: "user"
                },
                audio: false
            });
            
            if (this.videoEl) {
                this.videoEl.srcObject = this.stream;
                // iOS Safari compatibility
                this.videoEl.setAttribute('playsinline', '');
                this.videoEl.setAttribute('muted', '');
                await this.videoEl.play();
            }
            
            this.running = true;
            this.activeRequest = false;
            this.onStatusChange('running');
            this._scheduleNextFrame();
        } catch (err) {
            this.onError(err);
            this.stop();
        }
    }

    stop() {
        this.running = false;
        if (this.loopTimeoutId) {
            clearTimeout(this.loopTimeoutId);
            this.loopTimeoutId = null;
        }
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        if (this.videoEl) {
            this.videoEl.srcObject = null;
        }
        this.activeRequest = false;
        this.onStatusChange('idle');
    }

    destroy() {
        this.stop();
        document.removeEventListener('visibilitychange', this._handleVisibilityChange);
    }

    _handleVisibilityChange() {
        if (document.hidden) {
            // Pause loop when page is hidden
            if (this.loopTimeoutId) {
                clearTimeout(this.loopTimeoutId);
                this.loopTimeoutId = null;
            }
            console.log("WebcamClient: Paused loop due to hidden tab");
        } else {
            // Resume loop if running
            if (this.running && !this.loopTimeoutId) {
                console.log("WebcamClient: Resumed loop due to visible tab");
                this._scheduleNextFrame();
            }
        }
    }

    _scheduleNextFrame() {
        if (!this.running || document.hidden) return;
        this.loopTimeoutId = setTimeout(async () => {
            if (!this.running || document.hidden) return;
            await this._captureAndSendFrame();
            this._scheduleNextFrame();
        }, this.frameInterval);
    }

    async _captureAndSendFrame() {
        // Explicit frame drop if previous request is in-flight
        if (this.activeRequest) {
            console.log("WebcamClient: Frame dropped - request in flight");
            return;
        }
        
        if (!this.videoEl || !this.canvasEl) return;
        
        // Ensure video is playing and has valid dimensions
        if (this.videoEl.paused || this.videoEl.ended || !this.videoEl.videoWidth) return;

        this.activeRequest = true;
        const requestStartTime = Date.now();
        
        try {
            const ctx = this.canvasEl.getContext('2d');
            const aspect = this.videoEl.videoWidth / this.videoEl.videoHeight;
            
            // Client-side resize
            this.canvasEl.width = this.targetWidth;
            this.canvasEl.height = this.targetWidth / aspect;
            
            ctx.drawImage(this.videoEl, 0, 0, this.canvasEl.width, this.canvasEl.height);
            
            // Compression quality control
            const base64Data = this.canvasEl.toDataURL('image/jpeg', this.quality);
            
            // Retrieve parameters and callback from target page logic
            const params = this.onFrameProcessed() || { query: '', threshold: 0.25, callback: () => {} };
            
            // Abort controller for network timeout
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout limit
            
            const response = await fetch('/process_webcam_frame', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image: base64Data,
                    query: params.query || '',
                    threshold: params.threshold || 0.25
                }),
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);
            const data = await response.json();
            const latency = Date.now() - requestStartTime;
            
            // Adjust frameInterval and quality dynamically based on latency (adaptive control)
            if (latency > 800) {
                // Network is slow: throttle frame rate, increase compression
                this.frameInterval = Math.min(this.frameInterval + 100, 1500);
                this.quality = Math.max(this.quality - 0.05, 0.2);
                console.log(`WebcamClient: Throttling (latency: ${latency}ms) -> interval: ${this.frameInterval}ms, quality: ${this.quality.toFixed(2)}`);
            } else if (latency < 350) {
                // Server is fast: recover to ideal frequency and quality
                this.frameInterval = Math.max(this.frameInterval - 50, this.baseInterval);
                this.quality = Math.min(this.quality + 0.05, 0.6);
            }
            
            if (data.success) {
                if (this.running && this.feedImgEl && !document.hidden) {
                    this.feedImgEl.src = data.data.image;
                }
                if (params.callback) {
                    params.callback(data.data.objects || [], data.meta || {});
                }
            } else {
                console.error("WebcamClient processing error:", data.error);
            }
        } catch (err) {
            console.error("WebcamClient communication error:", err);
        } finally {
            this.activeRequest = false;
        }
    }
}
window.WebcamClient = WebcamClient;
