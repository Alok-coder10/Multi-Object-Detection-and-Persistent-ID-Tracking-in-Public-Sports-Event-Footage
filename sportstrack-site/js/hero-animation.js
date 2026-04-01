// hero-animation.js — Simulates live tracking on the hero canvas
(function() {
  const canvas = document.getElementById('demoCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;

  // Generate distinct hue per ID (golden ratio)
  function idColor(id, alpha) {
    const hue = ((id * 137.508) % 360);
    return `hsla(${hue},80%,62%,${alpha})`;
  }

  // A simulated "player" track
  class Track {
    constructor(id) {
      this.id = id;
      this.x = 40 + Math.random() * (W - 80);
      this.y = 40 + Math.random() * (H - 80);
      this.vx = (Math.random() - 0.5) * 1.8;
      this.vy = (Math.random() - 0.5) * 1.8;
      this.w = 30 + Math.random() * 18;
      this.h = 50 + Math.random() * 24;
      this.trail = [];
      this.age = 0;
      this.maxAge = 300 + Math.random() * 400;
    }
    update() {
      this.x += this.vx + (Math.random() - 0.5) * 0.4;
      this.y += this.vy + (Math.random() - 0.5) * 0.4;
      // Bounce
      if (this.x < 10) { this.x = 10; this.vx = Math.abs(this.vx); }
      if (this.x > W - this.w - 10) { this.x = W - this.w - 10; this.vx = -Math.abs(this.vx); }
      if (this.y < 10) { this.y = 10; this.vy = Math.abs(this.vy); }
      if (this.y > H - this.h - 10) { this.y = H - this.h - 10; this.vy = -Math.abs(this.vy); }
      // Slight random turn
      this.vx += (Math.random() - 0.5) * 0.15;
      this.vy += (Math.random() - 0.5) * 0.15;
      const spd = Math.hypot(this.vx, this.vy);
      if (spd > 2.2) { this.vx *= 2.2/spd; this.vy *= 2.2/spd; }
      if (spd < 0.3) { this.vx *= 1.5; this.vy *= 1.5; }
      this.trail.push({x: this.x + this.w/2, y: this.y + this.h/2});
      if (this.trail.length > 40) this.trail.shift();
      this.age++;
    }
    draw() {
      // Trail
      for (let i = 1; i < this.trail.length; i++) {
        const a = (i / this.trail.length) * 0.7;
        ctx.strokeStyle = idColor(this.id, a);
        ctx.lineWidth = 1.5 * (i / this.trail.length);
        ctx.beginPath();
        ctx.moveTo(this.trail[i-1].x, this.trail[i-1].y);
        ctx.lineTo(this.trail[i].x, this.trail[i].y);
        ctx.stroke();
      }
      // Box
      ctx.strokeStyle = idColor(this.id, 0.9);
      ctx.lineWidth = 1.5;
      ctx.strokeRect(this.x, this.y, this.w, this.h);
      // Corner accents
      const cs = 6;
      ctx.strokeStyle = idColor(this.id, 1);
      ctx.lineWidth = 2;
      [[this.x,this.y],[this.x+this.w,this.y],[this.x,this.y+this.h],[this.x+this.w,this.y+this.h]].forEach(([cx,cy], idx) => {
        ctx.beginPath();
        ctx.moveTo(cx + (idx%2===0?0:cs*-1), cy);
        ctx.lineTo(cx + (idx%2===0?cs:0), cy);
        ctx.moveTo(cx, cy + (idx<2?0:cs*-1));
        ctx.lineTo(cx, cy + (idx<2?cs:0));
        ctx.stroke();
      });
      // Label
      const label = `#${this.id}`;
      ctx.fillStyle = idColor(this.id, 0.95);
      ctx.fillRect(this.x, this.y - 15, label.length * 7 + 6, 14);
      ctx.fillStyle = '#0a0a0a';
      ctx.font = '500 9px "DM Mono", monospace';
      ctx.fillText(label, this.x + 3, this.y - 4);
    }
  }

  let tracks = [];
  let nextId = 1;
  let frame = 0;
  const TARGET = 7;

  function spawnTrack() {
    tracks.push(new Track(nextId++));
  }
  for (let i = 0; i < TARGET; i++) spawnTrack();

  // Draw field lines (subtle)
  function drawField() {
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#0d1209';
    ctx.fillRect(0, 0, W, H);
    // Grid lines
    ctx.strokeStyle = 'rgba(200,241,53,0.04)';
    ctx.lineWidth = 0.5;
    for (let x = 0; x < W; x += 40) {
      ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,H); ctx.stroke();
    }
    for (let y = 0; y < H; y += 40) {
      ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(W,y); ctx.stroke();
    }
    // Center circle
    ctx.strokeStyle = 'rgba(200,241,53,0.07)';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.arc(W/2, H/2, 50, 0, Math.PI*2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(W/2, 0); ctx.lineTo(W/2, H); ctx.stroke();
  }

  let lastTime = performance.now();
  let fps = 0;
  let uniqueIds = nextId - 1;
  let frameCount = 0;

  function animate(now) {
    const dt = now - lastTime;
    lastTime = now;
    fps = Math.round(1000 / dt);
    frameCount++;

    // Spawn/remove
    if (frameCount % 120 === 0) {
      if (tracks.length < TARGET + 2) spawnTrack();
    }
    tracks = tracks.filter(t => t.age < t.maxAge);
    while (tracks.length < TARGET - 1) spawnTrack();

    uniqueIds = nextId - 1;

    drawField();
    tracks.forEach(t => { t.update(); t.draw(); });

    // Scan line effect
    const grad = ctx.createLinearGradient(0, 0, 0, H);
    grad.addColorStop(0, 'rgba(0,0,0,0)');
    grad.addColorStop(((frameCount * 0.5) % H) / H, 'rgba(200,241,53,0.03)');
    grad.addColorStop(Math.min(1, ((frameCount * 0.5) % H + 4) / H), 'rgba(0,0,0,0)');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, W, H);

    // Update stats
    const st = document.getElementById('s-tracks');
    const sf = document.getElementById('s-fps');
    const si = document.getElementById('s-ids');
    if (st) st.textContent = tracks.length;
    if (sf) sf.textContent = fps;
    if (si) si.textContent = uniqueIds;

    requestAnimationFrame(animate);
  }

  requestAnimationFrame(animate);
})();
