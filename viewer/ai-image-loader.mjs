/**
 * AI Image Loading Animation
 * Inspired by Wensity AI Image Generator:
 * 1. Generating Phase: 2,880 dots rest on a dark floor, rise into an organic
 *    cloud of floating guesses, sample photo pixels, and settle into shape and color.
 * 2. Radial Sharpening: The developing image resolves through an offscreen blur
 *    and radial reveal mask as progress advances.
 * 3. Ring & Counter: Integrated circular progress ring, percentage counter (0 -> 100%),
 *    and elapsed timer badge.
 * 4. Mid-run Duration Bending: Paces smoothly during backend processing, then bends
 *    smoothly to 100% when the adapted design arrives without jump cuts.
 */

const REDUCED_MOTION_QUERY = '(prefers-reduced-motion: reduce)';
const RING_CIRCUMFERENCE = 2 * Math.PI * 5.25; // 32.9867...

function smoothstep(min, max, val) {
  const r = Math.min(1, Math.max(0, (val - min) / (max - min)));
  return r * r * (3 - 2 * r);
}

function pseudoHash(e) {
  const t = 43758.5453 * Math.sin(127.1 * e + 311.7);
  return t - Math.floor(t);
}

function parseTint(value) {
  if (!value) return { r: 255, g: 255, b: 255 };
  let str = String(value).trim();
  if (str.startsWith('#')) {
    let hex = str.slice(1);
    if (hex.length === 3) hex = hex.split('').map(c => c + c).join('');
    if (/^[0-9a-f]{6}$/i.test(hex)) {
      let r = parseInt(hex.slice(0, 2), 16);
      let g = parseInt(hex.slice(2, 4), 16);
      let b = parseInt(hex.slice(4, 6), 16);
      const lum = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
      if (lum < 0.55) {
        const boost = 0.55 / Math.max(lum, 0.05);
        r = Math.min(255, r * boost);
        g = Math.min(255, g * boost);
        b = Math.min(255, b * boost);
      }
      return { r, g, b };
    }
  }
  const match = str.match(/rgba?\(\s*([\d.]+)[,\s]+([\d.]+)[,\s]+([\d.]+)/);
  if (match) {
    let r = parseFloat(match[1]), g = parseFloat(match[2]), b = parseFloat(match[3]);
    const lum = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
    if (lum < 0.55) {
      const boost = 0.55 / Math.max(lum, 0.05);
      r = Math.min(255, r * boost);
      g = Math.min(255, g * boost);
      b = Math.min(255, b * boost);
    }
    return { r, g, b };
  }
  return { r: 255, g: 255, b: 255 };
}

function extractThumbnail(img) {
  if (!img || !img.complete || img.naturalWidth < 1) return null;
  try {
    const h = Math.max(1, Math.round(128 * img.naturalHeight / img.naturalWidth));
    const canvas = document.createElement('canvas');
    canvas.width = 128;
    canvas.height = h;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return null;
    ctx.drawImage(img, 0, 0, 128, h);
    return { width: 128, height: h, data: ctx.getImageData(0, 0, 128, h).data };
  } catch {
    return null;
  }
}

function createField(width, height, imgBounds, sampleData) {
  const count = 2880;
  const field = {
    width,
    height,
    count,
    seed: new Float32Array(count),
    depth: new Float32Array(count),
    floorX: new Float32Array(count),
    floorY: new Float32Array(count),
    guessX: new Float32Array(count),
    guessY: new Float32Array(count),
    realX: new Float32Array(count),
    realY: new Float32Array(count),
    shape: new Float32Array(count),
    red: new Float32Array(count),
    green: new Float32Array(count),
    blue: new Float32Array(count)
  };

  const floorStartY = 0.6 * height;
  const s = width / 72;
  const u = height / 40;

  for (let d = 0; d < 40; d++) {
    const rowFrac = d / 39;
    const depthCurve = rowFrac * rowFrac * 0.8 + 0.2 * rowFrac;
    const floorY = floorStartY + (height + u - floorStartY) * depthCurve;
    const floorSpread = 0.46 * width + (1.4 * width - 0.46 * width) * depthCurve;

    for (let i = 0; i < 72; i++) {
      const idx = 72 * d + i;
      const colFrac = (i + 0.5) / 72;
      const seed = pseudoHash(97 * d + 13 * i);
      const realX = colFrac * width + (pseudoHash(3.1 * idx) - 0.5) * s * 0.9;
      const realY = (d + 0.5) / 40 * height + (pseudoHash(7.7 * idx) - 0.5) * u * 0.9;
      const guessNoiseX = (pseudoHash(17 * d + 3 * i) + pseudoHash(41 * i + 7 * d)) / 2;
      const guessNoiseY = (pseudoHash(5 * d + 29 * i) + pseudoHash(11 * i + 23 * d)) / 2;

      field.seed[idx] = seed;
      field.depth[idx] = depthCurve;
      field.floorX[idx] = 0.5 * width + (colFrac - 0.5) * floorSpread;
      field.floorY[idx] = floorY;
      field.guessX[idx] = width * (0.5 + (guessNoiseX - 0.5) * 1.3);
      field.guessY[idx] = height * (0.46 + (guessNoiseY - 0.5) * 1.15);
      field.realX[idx] = realX;
      field.realY[idx] = realY;

      const uNorm = (realX - imgBounds.x) / imgBounds.w;
      const vNorm = (realY - imgBounds.y) / imgBounds.h;
      if (sampleData && uNorm >= 0 && uNorm <= 1 && vNorm >= 0 && vNorm <= 1) {
        const px = Math.min(sampleData.width - 1, Math.floor(uNorm * sampleData.width));
        const py = Math.min(sampleData.height - 1, Math.floor(vNorm * sampleData.height));
        const pidx = (py * sampleData.width + px) * 4;
        const pr = sampleData.data[pidx];
        const pg = sampleData.data[pidx + 1];
        const pb = sampleData.data[pidx + 2];
        const maxBright = Math.max(pr, pg, pb);
        field.shape[idx] = smoothstep(0.08, 0.36, maxBright / 255);
        const boost = Math.max(1, 170 / Math.max(maxBright, 1));
        field.red[idx] = Math.min(255, pr * boost);
        field.green[idx] = Math.min(255, pg * boost);
        field.blue[idx] = Math.min(255, pb * boost);
      } else {
        field.shape[idx] = 0.5;
        field.red[idx] = 230;
        field.green[idx] = 230;
        field.blue[idx] = 230;
      }
    }
  }

  return field;
}

function resampleField(field, imgBounds, sampleData) {
  if (!field || !sampleData) return;
  for (let idx = 0; idx < field.count; idx++) {
    const realX = field.realX[idx];
    const realY = field.realY[idx];
    const uNorm = (realX - imgBounds.x) / imgBounds.w;
    const vNorm = (realY - imgBounds.y) / imgBounds.h;
    if (uNorm >= 0 && uNorm <= 1 && vNorm >= 0 && vNorm <= 1) {
      const px = Math.min(sampleData.width - 1, Math.floor(uNorm * sampleData.width));
      const py = Math.min(sampleData.height - 1, Math.floor(vNorm * sampleData.height));
      const pidx = (py * sampleData.width + px) * 4;
      const pr = sampleData.data[pidx];
      const pg = sampleData.data[pidx + 1];
      const pb = sampleData.data[pidx + 2];
      const maxBright = Math.max(pr, pg, pb);
      field.shape[idx] = smoothstep(0.08, 0.36, maxBright / 255);
      const boost = Math.max(1, 170 / Math.max(maxBright, 1));
      field.red[idx] = Math.min(255, pr * boost);
      field.green[idx] = Math.min(255, pg * boost);
      field.blue[idx] = Math.min(255, pb * boost);
    }
  }
}

export class AiImageLoader {
  constructor({ container, canvas, badge, counter, timer, ringProgress, outputImg }) {
    this.container = container;
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d', { alpha: false });
    this.badge = badge;
    this.counter = counter;
    this.timer = timer;
    this.ringProgress = ringProgress;
    this.outputImg = outputImg;

    this.offscreenCanvas = document.createElement('canvas');
    this.offscreenCtx = this.offscreenCanvas.getContext('2d');

    this.state = 'idle'; // 'idle' | 'generating' | 'settling'
    this.field = null;
    this.activeImg = null;
    this.sampleData = null;
    this.animId = null;

    this.progress = 0; // 0..1
    this.startTime = 0;
    this.lastTime = 0;
    this.globalTime = 0;
    this.nominalDuration = 8;
    this.settleStartTime = 0;
    this.settleFromProgress = 0;
    this.settleDuration = 0.65;

    this.lastReportedPct = -1;
    this.lastReportedSec = -1;
    this.tint = { r: 255, g: 255, b: 255 };

    this.tick = this.tick.bind(this);
  }

  start({ targetWidth, targetHeight, sourceDesign, sourceImg }) {
    this.state = 'generating';
    this.progress = 0;
    this.settleStartTime = 0;
    this.lastReportedPct = -1;
    this.lastReportedSec = -1;
    this.nominalDuration = 8;

    // Pick tint color from sourceDesign if available
    let tintColor = '#ffffff';
    if (sourceDesign && Array.isArray(sourceDesign.elements)) {
      for (const el of sourceDesign.elements) {
        const c = el.color || el.fill_color;
        if (c && c !== '#ffffff' && c !== '#000000') {
          tintColor = c;
          break;
        }
      }
    }
    this.tint = parseTint(tintColor);

    // Compute bounding aspect fit inside artboard stage
    const artboard = this.container.parentElement;
    const artboardStyle = artboard ? getComputedStyle(artboard) : null;
    const padX = artboardStyle ? (parseFloat(artboardStyle.paddingLeft) + parseFloat(artboardStyle.paddingRight)) : 72;
    const padY = artboardStyle ? (parseFloat(artboardStyle.paddingTop) + parseFloat(artboardStyle.paddingBottom)) : 72;
    const availW = Math.max(20, (artboard ? artboard.clientWidth : 500) - padX);
    const availH = Math.max(20, (artboard ? artboard.clientHeight : 400) - padY);

    const fitScale = Math.min(availW / targetWidth, availH / targetHeight);
    const exactW = Math.max(30, Math.round(targetWidth * fitScale));
    const exactH = Math.max(20, Math.round(targetHeight * fitScale));

    this.width = exactW;
    this.height = exactH;

    this.container.style.width = `${exactW}px`;
    this.container.style.height = `${exactH}px`;
    this.container.classList.remove('visually-hidden');
    this.container.style.backgroundColor = `rgb(${Math.round(this.tint.r)}, ${Math.round(this.tint.g)}, ${Math.round(this.tint.b)})`;
    artboard?.classList.add('is-generating');
    artboard?.classList.remove('is-revealing');
    this.container.style.opacity = '1';

    if (this.outputImg) {
      this.outputImg.style.transition = 'none';
      this.outputImg.style.opacity = '0';
    }

    if (this.badge) {
      this.badge.classList.remove('visually-hidden');
      this.badge.style.opacity = '1';
    }

    this.resizeCanvas();

    // Keep the adapted artboard opaque while it is generating. The source
    // artwork is deliberately not sampled or shown in the target canvas.
    this.activeImg = null;
    this.sampleData = null;

    const imgBounds = this.getImgBounds(null);
    this.field = createField(this.canvas.width, this.canvas.height, imgBounds, this.sampleData);

    this.startTime = performance.now();
    this.lastTime = this.startTime;
    this.globalTime = 0;
    this.updateBadge(0, 0);

    if (window.matchMedia(REDUCED_MOTION_QUERY).matches) {
      this.progress = 1;
      this.updateBadge(100, 0);
      return;
    }

    if (!this.animId) {
      this.animId = requestAnimationFrame(this.tick);
    }
  }

  resizeCanvas() {
    this.dpr = Math.min(window.devicePixelRatio || 1, 2);
    const pixelW = Math.max(1, Math.round(this.width * this.dpr));
    const pixelH = Math.max(1, Math.round(this.height * this.dpr));

    if (this.canvas.width !== pixelW || this.canvas.height !== pixelH) {
      this.canvas.width = pixelW;
      this.canvas.height = pixelH;
      this.canvas.style.width = `${this.width}px`;
      this.canvas.style.height = `${this.height}px`;

      this.offscreenCanvas.width = pixelW;
      this.offscreenCanvas.height = pixelH;
    }
  }

  getImgBounds(img) {
    const W = this.canvas.width;
    const H = this.canvas.height;
    if (img && img.naturalWidth > 0) {
      const k = Math.max(W / img.naturalWidth, H / img.naturalHeight);
      const dw = img.naturalWidth * k;
      const dh = img.naturalHeight * k;
      return { x: (W - dw) / 2, y: (H - dh) / 2, w: dw, h: dh };
    }
    return { x: 0, y: 0, w: W, h: H };
  }

  async updateSourcePalette(sourceDesign, sourceImg) {
    // Source preview is shown in its own artboard only. Do not let its pixels
    // bleed into the adapted preview loader.
  }

  transitionToResult(outputPngUrl) {
    return new Promise(resolve => {
      const img = new Image();
      img.onload = () => {
        this.activeImg = img;
        this.sampleData = extractThumbnail(img);
        const bounds = this.getImgBounds(img);
        if (this.field) {
          resampleField(this.field, bounds, this.sampleData);
        }

        // Wait until the adapted image is fully decoded before settling to it.
        this.state = 'settling';
        this.settleStartTime = performance.now();
        this.settleFromProgress = this.progress;
        this.settleDuration = Math.max(0.45, Math.min(0.85, (1 - this.progress) * 0.9));
        this.onFinishCallback = resolve;
      };
      img.onerror = () => {
        this.activeImg = null;
        this.sampleData = null;
        this.state = 'settling';
        this.settleStartTime = performance.now();
        this.settleFromProgress = this.progress;
        this.settleDuration = 0.5;
        this.onFinishCallback = resolve;
      };
      img.src = outputPngUrl;
    });
  }

  updateBadge(pct, elapsedSec) {
    if (pct !== this.lastReportedPct) {
      this.lastReportedPct = pct;
      if (this.counter) {
        this.counter.textContent = `${pct}%`;
      }
      if (this.ringProgress) {
        const offset = RING_CIRCUMFERENCE * (1 - pct / 100);
        this.ringProgress.style.strokeDashoffset = String(offset);
      }
    }
    const sec = Math.floor(elapsedSec);
    if (sec !== this.lastReportedSec) {
      this.lastReportedSec = sec;
      if (this.timer) {
        this.timer.textContent = `${sec}s`;
      }
    }
  }

  tick(time) {
    if (this.state === 'idle') {
      this.animId = null;
      return;
    }

    const dt = this.lastTime ? Math.min(0.05, (time - this.lastTime) / 1000) : 0;
    this.lastTime = time;
    this.globalTime += dt;

    const elapsedTotal = (time - this.startTime) / 1000;

    if (this.state === 'settling') {
      const settleElapsed = (time - this.settleStartTime) / 1000;
      const t = Math.min(1, settleElapsed / this.settleDuration);
      const easedT = t * t * (3 - 2 * t);
      this.progress = this.settleFromProgress + (1 - this.settleFromProgress) * easedT;

      if (t >= 1) {
        this.progress = 1;
        this.updateBadge(100, elapsedTotal);
        this.renderFrame(1);
        setTimeout(() => {
          this.finish();
        }, 120);
        return;
      }
    } else {
      // Asymptotic progress towards 0.94 over nominalDuration
      const rawLinear = elapsedTotal / this.nominalDuration;
      // Soft asymptotic curve: never quite reaches 1 until backend resolves
      this.progress = Math.min(0.94, 1 - Math.exp(-rawLinear * 1.6));
    }

    const p = this.progress;
    // Authentic AI generation display formula from Wensity
    const displayPct = p >= 1 ? 100 : Math.min(99, Math.floor(100 * (0.55 * p + 0.45 * (1 - (1 - p) * (1 - p)))));
    this.updateBadge(displayPct, elapsedTotal);

    this.renderFrame(p);

    this.animId = requestAnimationFrame(this.tick);
  }

  renderFrame(p) {
    const W = this.canvas.width;
    const H = this.canvas.height;
    const ctx = this.ctx;
    const dpr = this.dpr;

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.globalCompositeOperation = 'source-over';
    ctx.globalAlpha = 1;
    ctx.fillStyle = `rgb(${Math.round(this.tint.r)}, ${Math.round(this.tint.g)}, ${Math.round(this.tint.b)})`;
    ctx.fillRect(0, 0, W, H);

    const bounds = this.getImgBounds(this.activeImg);

    // 1. Sharpening photograph unmasking
    if (this.activeImg && p > 0.45) {
      const photoProgress = smoothstep(0.45, 1.0, p);
      this.drawSharpeningPhoto(photoProgress, bounds);
    }

    // 2. Particle cloud
    if (this.field && p < 1.0) {
      this.drawParticles(p, bounds);
    }
  }

  drawSharpeningPhoto(photoProgress, bounds) {
    if (!this.activeImg || this.activeImg.naturalWidth < 1) return;
    const W = this.offscreenCanvas.width;
    const H = this.offscreenCanvas.height;
    const sCtx = this.offscreenCtx;
    const dpr = this.dpr;

    const invP = 1 - photoProgress;
    const scale = 1 + invP * invP * 0.06;
    const dw = bounds.w * scale;
    const dh = bounds.h * scale;

    sCtx.setTransform(1, 0, 0, 1, 0, 0);
    sCtx.globalCompositeOperation = 'source-over';
    sCtx.clearRect(0, 0, W, H);
    sCtx.imageSmoothingEnabled = true;
    sCtx.imageSmoothingQuality = 'high';

    const blurPx = invP * invP * 16 * dpr;
    sCtx.filter = blurPx > 0.3 ? `blur(${blurPx.toFixed(2)}px)` : 'none';
    sCtx.drawImage(this.activeImg, bounds.x - (dw - bounds.w) / 2, bounds.y - (dh - bounds.h) / 2, dw, dh);
    sCtx.filter = 'none';

    if (photoProgress < 0.999) {
      const cx = 0.5 * W;
      const cy = 0.48 * H;
      const radius = Math.hypot(W, H) * (0.08 + 0.9 * photoProgress);
      const grad = sCtx.createRadialGradient(cx, cy, 0, cx, cy, radius);
      grad.addColorStop(0, 'rgba(0,0,0,1)');
      grad.addColorStop(0.55, 'rgba(0,0,0,1)');
      grad.addColorStop(1, 'rgba(0,0,0,0)');
      sCtx.globalCompositeOperation = 'destination-in';
      sCtx.fillStyle = grad;
      sCtx.fillRect(0, 0, W, H);
      sCtx.globalCompositeOperation = 'source-over';
    }

    this.ctx.globalAlpha = smoothstep(0, 0.55, photoProgress);
    this.ctx.drawImage(this.offscreenCanvas, 0, 0);
    this.ctx.globalAlpha = 1;
  }

  drawParticles(p, bounds) {
    const F = this.field;
    const ctx = this.ctx;
    const dpr = this.dpr;
    const W = F.width;
    const H = F.height;
    const cx = 0.5 * W;
    const cy = 0.48 * H;
    const floorFlicker = 1 - smoothstep(0.02, 0.2, p);
    const tint = this.tint;

    for (let w = 0; w < F.count; w++) {
      const v = F.seed[w];
      const k = F.depth[w];
      const M = F.shape[w];
      const j = 0.12 * v;

      const C = smoothstep(0.04 + j, 0.26 + 0.5 * j, p);
      const A = smoothstep(0.24 + 0.6 * j, 0.56 + 0.3 * j, p);
      const R = smoothstep(0.76 + 0.08 * v, 0.90 + 0.06 * v, p);
      const shape = C * M;
      const invA = 1 - A;

      const noiseX = Math.sin(0.7 * this.globalTime + 6.283 * v) * W * 0.035 * invA;
      const noiseY = Math.cos(0.55 * this.globalTime + 9.1 * v) * H * 0.045 * invA;

      const gX = F.guessX[w] + noiseX;
      const gY = F.guessY[w] + noiseY;
      const targetX = gX + (F.realX[w] - gX) * A;
      const targetY = gY + (F.realY[w] - gY) * A;

      let posX = F.floorX[w] + (targetX - F.floorX[w]) * shape;
      let posY = F.floorY[w] + (targetY - F.floorY[w]) * shape;

      posX += (posX - cx) * R * 0.08;
      posY += (posY - cy) * R * 0.08 - 6 * R * dpr;

      if (posY < -4 || posY > H + 4) continue;

      const flicker = floorFlicker * (0.5 + 0.5 * Math.sin(2.4 * this.globalTime - 7 * k));
      const isVisible = M > 0.08 ? 1 : 1 - C;
      const riseAlpha = 0.7 + 0.25 * A;
      const baseAlpha = 0.1 + 0.5 * k + 0.25 * flicker;
      const alpha = (baseAlpha + (riseAlpha - baseAlpha) * shape) * isVisible * (1 - R);

      if (alpha < 0.025) continue;

      const colorBlend = shape * A;
      const r = tint.r + (F.red[w] - tint.r) * colorBlend;
      const g = tint.g + (F.green[w] - tint.g) * colorBlend;
      const b = tint.b + (F.blue[w] - tint.b) * colorBlend;
      const radius = (0.45 + 0.65 * k + 0.75 * shape) * (1 - 0.7 * R) * dpr;

      ctx.fillStyle = `rgba(${0 | r},${0 | g},${0 | b},${alpha.toFixed(3)})`;
      ctx.beginPath();
      ctx.arc(posX, posY, Math.max(0.35, radius), 0, 2 * Math.PI);
      ctx.fill();
    }
  }

  finish() {
    this.state = 'idle';
    if (this.animId) {
      cancelAnimationFrame(this.animId);
      this.animId = null;
    }
    this.container.classList.add('visually-hidden');
    const parent = this.container.parentElement;
    if (parent) {
      parent.classList.remove('is-revealing');
      parent.classList.remove('is-generating');
    }
    if (this.outputImg) {
      this.outputImg.style.opacity = '1';
    }
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    if (this.onFinishCallback) {
      const cb = this.onFinishCallback;
      this.onFinishCallback = null;
      cb();
    }
  }

  stop() {
    this.finish();
  }
}
