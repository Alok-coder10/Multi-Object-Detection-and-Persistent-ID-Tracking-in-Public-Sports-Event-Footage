// demo.js — Main demo controller
(function () {
  const { ByteTracker, SimDetector } = window.TrackerSim;

  // ── Elements ────────────────────────────────────────────────────────────────
  const videoInput  = document.getElementById('videoInput');
  const srcVideo    = document.getElementById('srcVideo');
  const outCanvas   = document.getElementById('outputCanvas');
  const placeholder = document.getElementById('canvasPlaceholder');
  const runBtn      = document.getElementById('runBtn');
  const runBtnText  = document.getElementById('runBtnText');
  const uploadZone  = document.getElementById('uploadZone');
  const statusDot   = document.querySelector('.status-dot');
  const statusText  = document.getElementById('statusText');
  const playPauseBtn= document.getElementById('playPauseBtn');
  const downloadBtn = document.getElementById('downloadBtn');

  const statFrame   = document.getElementById('statFrame');
  const statTracks  = document.getElementById('statTracks');
  const statIds     = document.getElementById('statIds');
  const statDets    = document.getElementById('statDets');
  const statFps     = document.getElementById('statFps');
  const trackLog    = document.getElementById('trackLog');
  const trackLogCount= document.getElementById('trackLogCount');

  const ctx = outCanvas.getContext('2d');

  // ── State ────────────────────────────────────────────────────────────────────
  let videoLoaded = false;
  let running = false;
  let paused = false;
  let animId = null;
  let frameNum = 0;
  let uniqueIds = 0;
  let lastTime = 0;
  let heatmapData = null;

  // Tracker + detector instances
  let tracker = new ByteTracker();
  let detector = new SimDetector(35, 45);

  // Count over time for chart
  const countHistory = { labels: [], data: [] };
  let chart = null;

  // ── ID Color ────────────────────────────────────────────────────────────────
  function idColor(id, alpha=1) {
    const hue = (id * 137.508) % 360;
    return `hsla(${hue},75%,62%,${alpha})`;
  }
  function idColorHex(id) {
    const hue = (id * 137.508) % 360;
    const s=0.75, v=0.62;
    const h6=hue/60, i=Math.floor(h6), f=h6-i;
    const p=v*(1-s),q=v*(1-f*s),t=v*(1-(1-f)*s);
    let r,g,b;
    switch(i%6){case 0:r=v;g=t;b=p;break;case 1:r=q;g=v;b=p;break;case 2:r=p;g=v;b=t;break;case 3:r=p;g=q;b=v;break;case 4:r=t;g=p;b=v;break;default:r=v;g=p;b=q;}
    return '#'+[r,g,b].map(x=>Math.round(x*255).toString(16).padStart(2,'0')).join('');
  }

  // ── Controls ────────────────────────────────────────────────────────────────
  function getConf() { return parseInt(document.getElementById('confThresh').value); }
  function getIou()  { return parseInt(document.getElementById('iouThresh').value)/100; }
  function getMaxAge(){ return parseInt(document.getElementById('maxAge').value); }
  function getMinHits(){ return parseInt(document.getElementById('minHits').value); }
  function showTrails(){ return document.getElementById('showTrails').checked; }
  function showConf() { return document.getElementById('showConf').checked; }
  function showHeat() { return document.getElementById('showHeatmap').checked; }

  document.getElementById('confThresh').addEventListener('input', e => {
    document.getElementById('confVal').textContent = (e.target.value/100).toFixed(2);
  });
  document.getElementById('iouThresh').addEventListener('input', e => {
    document.getElementById('iouVal').textContent = (e.target.value/100).toFixed(2);
  });
  document.getElementById('maxAge').addEventListener('input', e => {
    document.getElementById('ageVal').textContent = e.target.value;
  });
  document.getElementById('minHits').addEventListener('input', e => {
    document.getElementById('hitsVal').textContent = e.target.value;
  });

  // ── Upload ──────────────────────────────────────────────────────────────────
  videoInput.addEventListener('change', e => {
    const file = e.target.files[0];
    if (file) loadVideo(URL.createObjectURL(file), file.name);
  });

  uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
  uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
  uploadZone.addEventListener('drop', e => {
    e.preventDefault(); uploadZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('video/')) loadVideo(URL.createObjectURL(file), file.name);
  });

  document.getElementById('loadSample').addEventListener('click', () => {
    // Generate a synthetic demo video using canvas recording
    generateSampleVideo();
  });

  function loadVideo(url, name) {
    srcVideo.src = url;
    srcVideo.load();
    srcVideo.onloadedmetadata = () => {
      videoLoaded = true;
      uploadZone.classList.add('has-video');
      uploadZone.querySelector('.upload-text').textContent = name || 'Video loaded';
      runBtn.disabled = false;
      setStatus('ready', `Video ready — ${Math.round(srcVideo.duration)}s`);
    };
  }

  function generateSampleVideo() {
    // Simulate a loaded video state for the demo
    videoLoaded = true;
    uploadZone.classList.add('has-video');
    uploadZone.querySelector('.upload-text').textContent = 'sample_cricket.mp4';
    runBtn.disabled = false;
    setStatus('ready', 'Sample loaded');
  }

  // ── Run ────────────────────────────────────────────────────────────────────
  runBtn.addEventListener('click', () => {
    if (!videoLoaded) return;
    if (running) {
      stopTracking();
    } else {
      startTracking();
    }
  });

  function startTracking() {
    running = true; paused = false;
    runBtnText.textContent = '■ Stop';
    runBtn.style.background = '#ff5533';

    placeholder.classList.add('hidden');
    playPauseBtn.disabled = false;
    downloadBtn.disabled = false;

    // Setup canvas size
    const W = srcVideo.videoWidth || 640;
    const H = srcVideo.videoHeight || 400;
    outCanvas.width = W; outCanvas.height = H;
    heatmapData = new Float32Array(W * H);

    // Reinit tracker + detector
    tracker = new ByteTracker({ maxAge: getMaxAge(), minHits: getMinHits() });
    detector = new SimDetector(getConf(), getIou());
    const numPeople = 6 + Math.floor(Math.random() * 6);
    detector.init(W, H, numPeople);
    frameNum = 0; uniqueIds = 0;
    countHistory.labels = []; countHistory.data = [];

    setStatus('active', 'Tracking…');

    if (srcVideo.src && !srcVideo.src.includes('blob:fake')) {
      srcVideo.play();
      renderVideoFrames();
    } else {
      renderSyntheticFrames();
    }
  }

  function stopTracking() {
    running = false;
    if (animId) { cancelAnimationFrame(animId); animId = null; }
    srcVideo.pause();
    runBtnText.textContent = '▶ Run Tracking';
    runBtn.style.background = '';
    setStatus('ready', 'Stopped');
    playPauseBtn.disabled = true;
  }

  // ── Play/Pause ──────────────────────────────────────────────────────────────
  playPauseBtn.addEventListener('click', () => {
    paused = !paused;
    playPauseBtn.textContent = paused ? '▶' : '⏸';
    if (!paused && srcVideo.src && !srcVideo.paused) srcVideo.play();
    else srcVideo.pause();
  });

  document.getElementById('resetBtn').addEventListener('click', () => {
    stopTracking();
    tracker = new ByteTracker();
    frameNum = 0; uniqueIds = 0;
    ctx.clearRect(0, 0, outCanvas.width, outCanvas.height);
    updateStats(0, 0, 0, 0, '—');
    updateTrackLog([]);
    placeholder.classList.remove('hidden');
    if (chart) { chart.data.labels = []; chart.data.datasets[0].data = []; chart.update('none'); }
  });

  downloadBtn.addEventListener('click', () => {
    const link = document.createElement('a');
    link.download = `trackfield_frame_${frameNum}.png`;
    link.href = outCanvas.toDataURL();
    link.click();
  });

  // ── Render: Real video ─────────────────────────────────────────────────────
  function renderVideoFrames() {
    if (!running) return;
    if (paused) { animId = requestAnimationFrame(renderVideoFrames); return; }

    const now = performance.now();
    const dt = now - lastTime;
    lastTime = now;

    ctx.drawImage(srcVideo, 0, 0, outCanvas.width, outCanvas.height);
    const imgData = ctx.getImageData(0, 0, outCanvas.width, outCanvas.height);

    processFrame(dt);

    if (srcVideo.ended) stopTracking();
    else animId = requestAnimationFrame(renderVideoFrames);
  }

  // ── Render: Synthetic (no real video) ──────────────────────────────────────
  function renderSyntheticFrames() {
    if (!running) return;
    if (paused) { animId = requestAnimationFrame(renderSyntheticFrames); return; }

    const now = performance.now();
    const dt = now - (lastTime||now);
    lastTime = now;

    const W = outCanvas.width, H = outCanvas.height;

    // Draw a realistic-looking field
    drawSyntheticField(W, H);
    processFrame(dt);

    animId = requestAnimationFrame(renderSyntheticFrames);
  }

  function drawSyntheticField(W, H) {
    // Green pitch
    ctx.fillStyle = '#1a3a12';
    ctx.fillRect(0, 0, W, H);
    // Stripes
    for (let i=0; i<8; i++) {
      ctx.fillStyle = i%2===0 ? 'rgba(0,0,0,0.06)' : 'rgba(255,255,255,0.03)';
      ctx.fillRect(0, i*(H/8), W, H/8);
    }
    // Pitch markings
    ctx.strokeStyle = 'rgba(255,255,255,0.25)';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(W*0.1, H*0.1, W*0.8, H*0.8);
    ctx.beginPath(); ctx.moveTo(W/2,H*0.1); ctx.lineTo(W/2,H*0.9); ctx.stroke();
    ctx.beginPath(); ctx.arc(W/2, H/2, H*0.15, 0, Math.PI*2); ctx.stroke();
    ctx.beginPath(); ctx.arc(W/2, H/2, 4, 0, Math.PI*2); ctx.fill();
  }

  // ── Core frame processing ──────────────────────────────────────────────────
  function processFrame(dt) {
    frameNum++;
    tracker.maxAge = getMaxAge();
    tracker.minHits = getMinHits();
    detector.conf = getConf();

    const detections = detector.detect(frameNum);
    const tracks = tracker.update(detections);

    // Heatmap update
    if (heatmapData && showHeat()) {
      tracks.forEach(t => {
        const cx = Math.round((t.bbox[0]+t.bbox[2])/2);
        const cy = Math.round((t.bbox[1]+t.bbox[3])/2);
        const W = outCanvas.width;
        for (let dy=-15;dy<=15;dy++) for (let dx=-15;dx<=15;dx++) {
          const dist = Math.hypot(dx,dy);
          if (dist<16) {
            const idx = (cy+dy)*W+(cx+dx);
            if (idx>=0 && idx<heatmapData.length) heatmapData[idx] += Math.exp(-dist*dist/50);
          }
        }
      });
      renderHeatmap();
    }

    // Draw tracks
    tracks.forEach(t => drawTrack(t));

    // Draw detection boxes (faint)
    detections.forEach(d => {
      if (showConf()) {
        ctx.strokeStyle = 'rgba(255,255,255,0.15)';
        ctx.lineWidth = 0.5;
        ctx.setLineDash([3,3]);
        ctx.strokeRect(d.bbox[0], d.bbox[1], d.bbox[2]-d.bbox[0], d.bbox[3]-d.bbox[1]);
        ctx.setLineDash([]);
      }
    });

    // HUD
    drawHUD(tracks.length, frameNum, detections.length);

    // Track count
    uniqueIds = TrackerSim.KalmanTracker.nextId - 1;

    // Stats
    const fps = dt > 0 ? Math.round(1000/dt) : 0;
    updateStats(frameNum, tracks.length, uniqueIds, detections.length, fps + ' fps');
    updateTrackLog(tracks);

    // Chart update
    if (frameNum % 5 === 0) updateChart(frameNum, tracks.length);
  }

  // ── Drawing helpers ────────────────────────────────────────────────────────
  function drawTrack(t) {
    const [x1,y1,x2,y2] = t.bbox.map(Math.round);
    const color = idColor(t.id);

    // Trail
    if (showTrails() && t.trail.length > 1) {
      for (let i=1; i<t.trail.length; i++) {
        const a = (i/t.trail.length)*0.8;
        ctx.strokeStyle = idColor(t.id, a);
        ctx.lineWidth = 1.5*(i/t.trail.length);
        ctx.beginPath();
        ctx.moveTo(t.trail[i-1].x, t.trail[i-1].y);
        ctx.lineTo(t.trail[i].x, t.trail[i].y);
        ctx.stroke();
      }
    }

    // Main box
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(x1, y1, x2-x1, y2-y1);

    // Corner accents
    const cs = 8;
    ctx.lineWidth = 2.5;
    [[x1,y1,1,1],[x2,y1,-1,1],[x1,y2,1,-1],[x2,y2,-1,-1]].forEach(([cx,cy,sx,sy]) => {
      ctx.beginPath();
      ctx.moveTo(cx+sx*cs, cy); ctx.lineTo(cx, cy); ctx.lineTo(cx, cy+sy*cs);
      ctx.stroke();
    });

    // Label
    const label = `#${t.id}${showConf() ? '' : ''}`;
    ctx.font = 'bold 10px "DM Mono", monospace';
    const tw = ctx.measureText(label).width;
    ctx.fillStyle = idColor(t.id, 0.9);
    ctx.fillRect(x1-1, y1-16, tw+10, 15);
    ctx.fillStyle = '#000';
    ctx.fillText(label, x1+4, y1-5);
  }

  function drawHUD(count, frame, dets) {
    const W = outCanvas.width;
    ctx.fillStyle = 'rgba(0,0,0,0.55)';
    ctx.fillRect(0, 0, W, 32);
    ctx.font = 'bold 11px "DM Mono", monospace';
    ctx.fillStyle = '#c8f135';
    ctx.fillText(`ACTIVE: ${count}`, 12, 21);
    ctx.fillStyle = 'rgba(255,255,255,0.5)';
    ctx.fillText(`FRAME: ${frame}`, W/2-40, 21);
    ctx.fillText(`DET: ${dets}`, W-80, 21);
    // Progress bar thin line
    ctx.fillStyle = 'rgba(200,241,53,0.4)';
    ctx.fillRect(0, 31, W*(frame/(frame+500)), 1);
  }

  function renderHeatmap() {
    const W = outCanvas.width, H = outCanvas.height;
    let max = 0;
    for (let i=0;i<heatmapData.length;i++) if(heatmapData[i]>max) max=heatmapData[i];
    if (max === 0) return;
    const imgData = ctx.getImageData(0, 0, W, H);
    for (let i=0;i<heatmapData.length;i++) {
      const v = heatmapData[i]/max;
      if (v < 0.01) continue;
      const px = i*4;
      // Jet colormap approximation
      const r = Math.min(255, Math.max(0, Math.round(255*Math.min(1, 1.5-Math.abs(v*4-3)))));
      const g = Math.min(255, Math.max(0, Math.round(255*Math.min(1, 1.5-Math.abs(v*4-2)))));
      const b = Math.min(255, Math.max(0, Math.round(255*Math.min(1, 1.5-Math.abs(v*4-1)))));
      const a = Math.round(v * 140);
      imgData.data[px]   = (imgData.data[px]*( 255-a) + r*a)/255;
      imgData.data[px+1] = (imgData.data[px+1]*(255-a) + g*a)/255;
      imgData.data[px+2] = (imgData.data[px+2]*(255-a) + b*a)/255;
    }
    ctx.putImageData(imgData, 0, 0);
  }

  // ── UI updates ─────────────────────────────────────────────────────────────
  function setStatus(state, text) {
    statusText.textContent = text;
    statusDot.className = 'status-dot' + (state==='active' ? ' active' : '');
  }

  function updateStats(frame, tracks, ids, dets, fps) {
    statFrame.textContent  = frame || '—';
    statTracks.textContent = tracks || '—';
    statIds.textContent    = ids || '—';
    statDets.textContent   = dets || '—';
    statFps.textContent    = fps || '—';
  }

  function updateTrackLog(tracks) {
    trackLogCount.textContent = `${tracks.length} tracks`;
    if (!tracks.length) {
      trackLog.innerHTML = '<div class="track-log-empty">No active tracks</div>';
      return;
    }
    trackLog.innerHTML = tracks.map(t => {
      const hex = idColorHex(t.id);
      return `<div class="track-pill" style="color:${hex};border-color:${hex}40;background:${hex}18">
        #${t.id} <span style="opacity:0.6;font-size:9px">h:${t.hits}</span>
      </div>`;
    }).join('');
  }

  function updateChart(frame, count) {
    countHistory.labels.push(frame);
    countHistory.data.push(count);
    if (countHistory.labels.length > 60) {
      countHistory.labels.shift();
      countHistory.data.shift();
    }
    if (!chart) {
      chart = new Chart(document.getElementById('countChart'), {
        type: 'line',
        data: {
          labels: countHistory.labels,
          datasets: [{
            label: 'Active tracks',
            data: countHistory.data,
            borderColor: '#c8f135',
            backgroundColor: 'rgba(200,241,53,0.08)',
            borderWidth: 1.5,
            pointRadius: 0,
            fill: true,
            tension: 0.3,
          }]
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          animation: false,
          plugins: { legend: { display: false } },
          scales: {
            x: { display: false },
            y: {
              min: 0, grid: { color: 'rgba(255,255,255,0.05)' },
              ticks: { color: '#5a5753', font: { size: 10, family: 'DM Mono' }, stepSize: 1 },
              border: { display: false }
            }
          }
        }
      });
    } else {
      chart.data.labels = [...countHistory.labels];
      chart.data.datasets[0].data = [...countHistory.data];
      chart.update('none');
    }
  }

  function idColorHex(id) {
    const hue = (id * 137.508) % 360;
    const s=0.75, v=0.62;
    const h6=hue/60, i=Math.floor(h6), f=h6-i;
    const p=v*(1-s),q=v*(1-f*s),t=v*(1-(1-f)*s);
    let r,g,b;
    switch(i%6){case 0:r=v;g=t;b=p;break;case 1:r=q;g=v;b=p;break;case 2:r=p;g=v;b=t;break;case 3:r=p;g=q;b=v;break;case 4:r=t;g=p;b=v;break;default:r=v;g=p;b=q;}
    return '#'+[r,g,b].map(x=>Math.round(x*255).toString(16).padStart(2,'0')).join('');
  }

})();
