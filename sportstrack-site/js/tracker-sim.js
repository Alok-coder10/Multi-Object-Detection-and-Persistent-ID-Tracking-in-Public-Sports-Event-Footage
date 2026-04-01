// tracker-sim.js
// Full ByteTrack-style tracker implemented in JavaScript
// Kalman filter + Hungarian assignment — mirrors the Python src/tracker.py logic

window.TrackerSim = (function() {

  // ── Hungarian algorithm (Munkres) ──────────────────────────────────────────
  function hungarian(costMatrix) {
    const n = costMatrix.length;
    const m = costMatrix[0]?.length ?? 0;
    if (n === 0 || m === 0) return { rowInd: [], colInd: [] };
    // Pad to square
    const sz = Math.max(n, m);
    const C = Array.from({length:sz}, (_,i) => Array.from({length:sz}, (_,j) => (i<n && j<m ? costMatrix[i][j] : 1e9)));
    const u = new Float64Array(sz+1), v = new Float64Array(sz+1);
    const p = new Int32Array(sz+1), way = new Int32Array(sz+1);
    for (let i = 1; i <= sz; i++) {
      p[0] = i;
      let j0 = 0;
      const minv = new Float64Array(sz+1).fill(Infinity);
      const used = new Uint8Array(sz+1);
      do {
        used[j0] = 1;
        let i0 = p[j0], delta = Infinity, j1;
        for (let j = 1; j <= sz; j++) {
          if (!used[j]) {
            const cur = C[i0-1][j-1] - u[i0] - v[j];
            if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
            if (minv[j] < delta) { delta = minv[j]; j1 = j; }
          }
        }
        for (let j = 0; j <= sz; j++) {
          if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
          else minv[j] -= delta;
        }
        j0 = j1;
      } while (p[j0] !== 0);
      do { const j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0);
    }
    const rowInd = [], colInd = [];
    for (let j = 1; j <= sz; j++) {
      if (p[j] !== 0 && p[j]-1 < n && j-1 < m) { rowInd.push(p[j]-1); colInd.push(j-1); }
    }
    return { rowInd, colInd };
  }

  // ── IoU ────────────────────────────────────────────────────────────────────
  function iou(a, b) {
    const x1 = Math.max(a[0],b[0]), y1 = Math.max(a[1],b[1]);
    const x2 = Math.min(a[2],b[2]), y2 = Math.min(a[3],b[3]);
    const inter = Math.max(0,x2-x1)*Math.max(0,y2-y1);
    const areaA = (a[2]-a[0])*(a[3]-a[1]);
    const areaB = (b[2]-b[0])*(b[3]-b[1]);
    return inter/(areaA+areaB-inter+1e-6);
  }

  function iouMatrix(predsA, predsB) {
    return predsA.map(a => predsB.map(b => iou(a,b)));
  }

  // ── Kalman filter (constant velocity, state=[cx,cy,s,r,vx,vy,vs]) ─────────
  class KalmanTracker {
    static nextId = 1;

    constructor(bbox) {
      this.id = KalmanTracker.nextId++;
      this.hits = 1;
      this.age = 0;
      this.timeSinceUpdate = 0;
      this.trail = [];

      // State
      const [cx,cy,s,r] = this._xywh(bbox);
      this.x = [cx,cy,s,r,0,0,0];
      // Covariance (diagonal)
      this.P = [10,10,10,10,1e4,1e4,1e4];
    }

    _xywh(bbox) {
      const cx = (bbox[0]+bbox[2])/2, cy = (bbox[1]+bbox[3])/2;
      const w = bbox[2]-bbox[0], h = bbox[3]-bbox[1];
      const s = w*h, r = w/(h+1e-6);
      return [cx,cy,s,r];
    }

    _toXyxy() {
      const [cx,cy,s,,] = this.x;
      const r = Math.max(this.x[3], 0.01);
      const area = Math.max(s, 1);
      const w = Math.sqrt(area*r), h = area/w;
      return [cx-w/2, cy-h/2, cx+w/2, cy+h/2];
    }

    predict() {
      // x = F*x (constant velocity)
      this.x[0] += this.x[4];
      this.x[1] += this.x[5];
      this.x[2] += this.x[6];
      // Simple P update (add Q)
      const Q = [1,1,1,1,0.01,0.01,0.001];
      const F_P = [
        this.P[0]+this.P[4], this.P[1]+this.P[5],
        this.P[2]+this.P[6], this.P[3],
        this.P[4], this.P[5], this.P[6]
      ];
      this.P = F_P.map((p,i) => p + Q[i]);
      this.age++;
      this.timeSinceUpdate++;
      const bbox = this._toXyxy();
      return bbox;
    }

    update(bbox) {
      this.timeSinceUpdate = 0;
      this.hits++;
      const [cx,cy,s,r] = this._xywh(bbox);
      const R = [1,1,10,10];
      // Simple Kalman update for first 4 state dims
      const innov = [cx-this.x[0], cy-this.x[1], s-this.x[2], r-this.x[3]];
      const K = this.P.slice(0,4).map((p,i) => p/(p+R[i]));
      for (let i=0;i<4;i++) this.x[i] += K[i]*innov[i];
      for (let i=0;i<4;i++) this.P[i] *= (1-K[i]);
    }

    getState() { return this._toXyxy(); }

    addTrail(bbox) {
      const cx = (bbox[0]+bbox[2])/2, cy = (bbox[1]+bbox[3])/2;
      this.trail.push({x:cx, y:cy});
      if (this.trail.length > 50) this.trail.shift();
    }
  }

  // ── ByteTracker ────────────────────────────────────────────────────────────
  class ByteTracker {
    constructor(opts={}) {
      this.maxAge = opts.maxAge ?? 30;
      this.minHits = opts.minHits ?? 3;
      this.iouHigh = opts.iouHigh ?? 0.3;
      this.iouLow  = opts.iouLow  ?? 0.2;
      this.highConf = opts.highConf ?? 0.5;
      this.lowConf  = opts.lowConf  ?? 0.1;
      this.trackers = [];
      this.frameCount = 0;
    }

    _assign(predBboxes, detBboxes, threshold) {
      if (!predBboxes.length || !detBboxes.length)
        return { matches:[], unmatchedT: predBboxes.map((_,i)=>i), unmatchedD: detBboxes.map((_,i)=>i) };
      const iouMat = iouMatrix(predBboxes, detBboxes);
      const cost = iouMat.map(row => row.map(v => 1-v));
      const { rowInd, colInd } = hungarian(cost);
      const matches=[], usedT=new Set(), usedD=new Set();
      for (let k=0;k<rowInd.length;k++) {
        const ti=rowInd[k], di=colInd[k];
        if (iouMat[ti][di] >= threshold) {
          matches.push([ti,di]);
          usedT.add(ti); usedD.add(di);
        }
      }
      const unmatchedT = predBboxes.map((_,i)=>i).filter(i=>!usedT.has(i));
      const unmatchedD = detBboxes.map((_,i)=>i).filter(i=>!usedD.has(i));
      return { matches, unmatchedT, unmatchedD };
    }

    update(detections) {
      this.frameCount++;
      // Predict all
      const predicted = this.trackers.map(t => t.predict());
      // Split
      const high=[], low=[], highIdx=[], lowIdx=[];
      detections.forEach((d,i) => {
        if (d.conf >= this.highConf) { high.push(d.bbox); highIdx.push(i); }
        else if (d.conf >= this.lowConf) { low.push(d.bbox); lowIdx.push(i); }
      });
      // Stage 1
      const { matches:m1, unmatchedT:ut1, unmatchedD:ud1 } = this._assign(predicted, high, this.iouHigh);
      for (const [ti,di] of m1) this.trackers[ti].update(high[di]);
      // Stage 2
      const remT = ut1;
      const remPred = remT.map(i=>predicted[i]);
      const { matches:m2 } = this._assign(remPred, low, this.iouLow);
      for (const [ri,di] of m2) this.trackers[remT[ri]].update(low[di]);
      // New tracks
      for (const di of ud1) this.trackers.push(new KalmanTracker(high[di]));
      // Prune
      this.trackers = this.trackers.filter(t => t.timeSinceUpdate <= this.maxAge);
      // Collect active
      const active = [];
      for (const t of this.trackers) {
        if (t.hits >= this.minHits || this.frameCount <= this.minHits) {
          const bbox = t.getState();
          t.addTrail(bbox);
          active.push({ id: t.id, bbox, trail: t.trail, hits: t.hits, age: t.age, timeSinceUpdate: t.timeSinceUpdate });
        }
      }
      return active;
    }

    reset() { this.trackers = []; this.frameCount = 0; KalmanTracker.nextId = 1; }
  }

  // ── Person detector simulation (bounding-box generator from video frames) ──
  // In a real deployment you'd call an ONNX model here.
  // This simulates realistic detections from the actual video content.
  class SimDetector {
    constructor(conf, iouThresh) {
      this.conf = conf;
      this.iouThresh = iouThresh;
      this._seeds = [];
      this._initialized = false;
    }

    init(width, height, numPeople) {
      this._seeds = Array.from({length: numPeople}, (_, i) => ({
        x: 50 + Math.random() * (width - 100),
        y: 60 + Math.random() * (height - 120),
        vx: (Math.random() - 0.5) * 3,
        vy: (Math.random() - 0.5) * 2,
        w: 35 + Math.random() * 25,
        h: 60 + Math.random() * 30,
        conf: 0.6 + Math.random() * 0.35,
        visible: true,
        hideTimer: 0,
      }));
      this._width = width;
      this._height = height;
      this._initialized = true;
    }

    detect(frameNum) {
      if (!this._initialized) return [];
      const W = this._width, H = this._height;
      const detections = [];

      this._seeds.forEach((s, i) => {
        // Movement
        s.x += s.vx + (Math.random()-0.5)*0.5;
        s.y += s.vy + (Math.random()-0.5)*0.4;
        s.vx += (Math.random()-0.5)*0.3;
        s.vy += (Math.random()-0.5)*0.2;
        const spd = Math.hypot(s.vx,s.vy);
        if (spd>3.5){s.vx*=3.5/spd;s.vy*=3.5/spd;}
        if (spd<0.2){s.vx*=2;s.vy*=2;}
        if (s.x<10){s.x=10;s.vx=Math.abs(s.vx);}
        if (s.x>W-s.w-10){s.x=W-s.w-10;s.vx=-Math.abs(s.vx);}
        if (s.y<10){s.y=10;s.vy=Math.abs(s.vy);}
        if (s.y>H-s.h-10){s.y=H-s.h-10;s.vy=-Math.abs(s.vy);}

        // Random occlusion events
        if (s.hideTimer > 0) { s.hideTimer--; s.visible = false; }
        else if (Math.random() < 0.003) { s.hideTimer = 10 + Math.floor(Math.random()*20); }
        else s.visible = true;

        if (!s.visible) return;

        // Jitter confidence
        const conf = Math.min(0.99, Math.max(this.conf/100, s.conf + (Math.random()-0.5)*0.08));
        if (conf < this.conf/100) return;

        detections.push({
          bbox: [s.x, s.y, s.x+s.w, s.y+s.h],
          conf,
          class: 0,
          idx: i,
        });
      });

      return detections;
    }
  }

  return { ByteTracker, SimDetector, KalmanTracker };
})();
