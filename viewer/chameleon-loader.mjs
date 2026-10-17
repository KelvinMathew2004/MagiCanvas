/**
 * Chameleon Organic Pebble Scale AI Generation Loader
 * Reverted to organic Voronoi pebble cobblestone shapes.
 * - Exact pixel sizing matching the target image footprint.
 * - Smooth, continuous Hermite color morph from original palette to generated image.
 * - Organic rounded cobblestone pebble blob shapes (no wobble).
 * - Radial dissolve from the center outward revealing the crisp generated design.
 */

function parseColor(str) {
  if (!str) return null;
  str = str.trim();
  if (str.startsWith('#')) {
    let hex = str.slice(1);
    if (hex.length === 3) hex = hex.split('').map(c => c + c).join('');
    if (hex.length === 6) {
      return [
        parseInt(hex.slice(0, 2), 16),
        parseInt(hex.slice(2, 4), 16),
        parseInt(hex.slice(4, 6), 16),
        1
      ];
    }
  }
  const match = str.match(/rgba?\(\s*([\d.]+)[,\s]+([\d.]+)[,\s]+([\d.]+)(?:[,\s/]+([\d.]+))?\s*\)/);
  if (match) {
    return [
      parseFloat(match[1]),
      parseFloat(match[2]),
      parseFloat(match[3]),
      match[4] !== undefined ? parseFloat(match[4]) : 1
    ];
  }
  return null;
}

export function extractPalette(sourceDesign, sourceImgElement) {
  const colors = [];

  // 1. Extract declared colors from design elements
  if (sourceDesign && Array.isArray(sourceDesign.elements)) {
    for (const el of sourceDesign.elements) {
      const candidates = [el.fill_color, el.color, el.stroke_color, el.background_color];
      for (const c of candidates) {
        const parsed = parseColor(c);
        if (parsed) colors.push(parsed);
      }
    }
  }

  // 2. Sample pixel colors from source preview image if available
  if (sourceImgElement && sourceImgElement.complete && sourceImgElement.naturalWidth > 0) {
    try {
      const canvas = document.createElement('canvas');
      canvas.width = 32;
      canvas.height = 32;
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      ctx.drawImage(sourceImgElement, 0, 0, 32, 32);
      const data = ctx.getImageData(0, 0, 32, 32).data;
      for (let i = 0; i < data.length; i += 32) {
        const r = data[i], g = data[i + 1], b = data[i + 2], a = data[i + 3] / 255;
        if (a > 0.4 && (r + g + b > 30) && (r + g + b < 730)) {
          colors.push([r, g, b, a]);
        }
      }
    } catch {
      // Fallback if canvas is tainted
    }
  }

  // Organic chameleon palette: turquoise, orange-amber, emerald green, warm rust
  if (colors.length < 2) {
    colors.push(
      [24, 180, 115, 1],
      [245, 166, 35, 1],
      [208, 60, 35, 1],
      [45, 156, 219, 1],
      [142, 68, 173, 1]
    );
  }

  return colors;
}

function clipPoly(poly, p0, p1) {
  const mx = (p0[0] + p1[0]) * 0.5;
  const my = (p0[1] + p1[1]) * 0.5;
  const nx = p1[0] - p0[0];
  const ny = p1[1] - p0[1];
  const out = [];

  for (let i = 0; i < poly.length; i++) {
    const a = poly[i];
    const b = poly[(i + 1) % poly.length];
    const da = (a[0] - mx) * nx + (a[1] - my) * ny;
    const db = (b[0] - mx) * nx + (b[1] - my) * ny;

    if (da <= 0) {
      out.push(a);
    }
    if ((da <= 0 && db > 0) || (da > 0 && db <= 0)) {
      const denom = da - db;
      const t = denom !== 0 ? da / denom : 0;
      out.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]);
    }
  }
  return out;
}

export class ChameleonLoader {
  constructor({ container, canvas, outputImg }) {
    this.container = container;
    this.canvas = canvas;
    this.outputImg = outputImg;
    this.ctx = canvas.getContext('2d');
    this.animId = null;
    this.state = 'idle'; // 'idle' | 'generating' | 'settling'
    this.blobs = [];
    this.palette = [];
    this.startTime = 0;
    this.settleStartTime = 0;
    this.targetImgData = null;
    this.width = 300;
    this.height = 300;
    this.dpr = window.devicePixelRatio || 1;
    this.maxDist = 1;
    this.centerX = 150;
    this.centerY = 150;
  }

  start({ targetWidth, targetHeight, sourceDesign, sourceImg }) {
    this.palette = extractPalette(sourceDesign, sourceImg);
    this.state = 'generating';
    this.startTime = performance.now();
    this.targetImgData = null;

    // Sizing to exact pixel box of the target image inside artboard
    const artboard = this.container.parentElement;
    const artboardStyle = getComputedStyle(artboard);
    const padX = parseFloat(artboardStyle.paddingLeft) + parseFloat(artboardStyle.paddingRight);
    const padY = parseFloat(artboardStyle.paddingTop) + parseFloat(artboardStyle.paddingBottom);
    const availW = Math.max(20, artboard.clientWidth - padX);
    const availH = Math.max(20, artboard.clientHeight - padY);

    const fitScale = Math.min(availW / targetWidth, availH / targetHeight);
    const exactW = Math.max(30, Math.round(targetWidth * fitScale));
    const exactH = Math.max(20, Math.round(targetHeight * fitScale));

    this.width = exactW;
    this.height = exactH;
    this.centerX = exactW / 2;
    this.centerY = exactH / 2;
    this.maxDist = Math.hypot(this.centerX, this.centerY) || 1;

    this.container.style.width = `${exactW}px`;
    this.container.style.height = `${exactH}px`;
    this.container.classList.remove('visually-hidden');
    this.container.parentElement.classList.add('is-generating');
    this.container.parentElement.classList.remove('is-revealing');
    this.container.style.opacity = '1';

    if (this.outputImg) {
      this.outputImg.style.transition = 'none';
      this.outputImg.style.opacity = '0';
    }

    this.resizeCanvas();
    this.initPebbleBlobs();

    if (!this.animId) {
      this.tick = this.tick.bind(this);
      this.animId = requestAnimationFrame(this.tick);
    }
  }

  resizeCanvas() {
    this.dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.round(this.width * this.dpr);
    this.canvas.height = Math.round(this.height * this.dpr);
    this.canvas.style.width = `${this.width}px`;
    this.canvas.style.height = `${this.height}px`;
    this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
  }

  initPebbleBlobs() {
    this.blobs = [];
    // Pebble scale spacing (organic chameleon skin cobblestones)
    const spacing = Math.max(7, Math.min(13, Math.round(Math.min(this.width, this.height) / 28)));
    const cols = Math.ceil(this.width / spacing) + 3;
    const rows = Math.ceil(this.height / spacing) + 3;

    // Pseudo-random seeded jitter grid
    const grid = [];
    for (let r = 0; r < rows; r++) {
      const row = [];
      for (let c = 0; c < cols; c++) {
        const jitterX = (Math.sin(r * 12.9898 + c * 78.233) * 0.5) * 0.45;
        const jitterY = (Math.cos(r * 4.1414 + c * 34.123) * 0.5) * 0.42;
        const x = (c - 1 + (r % 2 ? 0.5 : 0) + jitterX) * spacing;
        const y = (r - 1 + jitterY) * spacing;
        row.push([x, y]);
      }
      grid.push(row);
    }

    const boxRadius = spacing * 1.8;

    for (let r = 1; r < rows - 1; r++) {
      for (let c = 1; c < cols - 1; c++) {
        const p0 = grid[r][c];
        if (p0[0] < -spacing || p0[0] > this.width + spacing || p0[1] < -spacing || p0[1] > this.height + spacing) {
          continue;
        }

        // Bounding box around seed point
        let poly = [
          [p0[0] - boxRadius, p0[1] - boxRadius],
          [p0[0] + boxRadius, p0[1] - boxRadius],
          [p0[0] + boxRadius, p0[1] + boxRadius],
          [p0[0] - boxRadius, p0[1] + boxRadius]
        ];

        // Clip against 8 surrounding neighbor points to generate Voronoi pebble polygon
        for (let dr = -1; dr <= 1; dr++) {
          for (let dc = -1; dc <= 1; dc++) {
            if (dr === 0 && dc === 0) continue;
            const p1 = grid[r + dr][c + dc];
            poly = clipPoly(poly, p0, p1);
          }
        }

        if (poly.length < 3) continue;

        // Inset polygon vertices slightly toward seed center to create dark skin groove seams
        const insetFactor = 0.86;
        const insetPts = poly.map(pt => [
          p0[0] + (pt[0] - p0[0]) * insetFactor,
          p0[1] + (pt[1] - p0[1]) * insetFactor
        ]);

        const dist = Math.hypot(p0[0] - this.centerX, p0[1] - this.centerY);
        const colorIdx = Math.floor(Math.random() * this.palette.length);

        this.blobs.push({
          cx: p0[0],
          cy: p0[1],
          pts: insetPts,
          dist,
          relDist: Math.min(1, dist / this.maxDist),
          colorIdx,
          phase: Math.random() * Math.PI * 2,
          speed: 0.8 + Math.random() * 0.5,
          displayColor: [...this.palette[colorIdx]],
          startMorphColor: [...this.palette[colorIdx]],
          targetColor: [...this.palette[colorIdx]],
          opacity: 0.94 + Math.random() * 0.06
        });
      }
    }
  }

  async transitionToResult(outputPngUrl) {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.src = outputPngUrl;

    try {
      await new Promise((resolve, reject) => {
        img.onload = resolve;
        img.onerror = reject;
      });

      // Render into offscreen buffer to sample pixel colors from actual generated design
      const offscreen = document.createElement('canvas');
      offscreen.width = this.width;
      offscreen.height = this.height;
      const oCtx = offscreen.getContext('2d', { willReadFrequently: true });
      oCtx.drawImage(img, 0, 0, this.width, this.height);
      this.targetImgData = oCtx.getImageData(0, 0, this.width, this.height).data;

      // Assign target colors & snapshot current instantaneous display colors
      for (const b of this.blobs) {
        // Snapshot the EXACT color the blob has right now — ensures 0% abruptness!
        b.startMorphColor = [...b.displayColor];

        const sx = Math.max(0, Math.min(this.width - 1, Math.round(b.cx)));
        const sy = Math.max(0, Math.min(this.height - 1, Math.round(b.cy)));
        const idx = (sy * this.width + sx) * 4;
        const red = this.targetImgData[idx];
        const green = this.targetImgData[idx + 1];
        const blue = this.targetImgData[idx + 2];
        const alpha = this.targetImgData[idx + 3] / 255;
        b.targetColor = [red, green, blue, alpha];
      }
    } catch {
      // Keep generation colors if sampling fails
    }

    // Keep output image rendered behind canvas so dissolving from center reveals it
    if (this.outputImg) {
      this.outputImg.style.transition = 'none';
      this.outputImg.style.opacity = '1';
    }

    this.container.parentElement.classList.add('is-revealing');
    this.state = 'settling';
    this.settleStartTime = performance.now();
  }

  tick(time) {
    if (this.state === 'idle') {
      this.animId = null;
      return;
    }

    this.ctx.clearRect(0, 0, this.width, this.height);

    const elapsed = (time - this.startTime) / 1000;
    const isSettling = this.state === 'settling';
    const settleElapsed = isSettling ? (time - this.settleStartTime) / 1000 : 0;

    // Give the color match time to settle before the image reveal travels outward.
    let radialWave = 0;
    let backgroundFade = 1;

    if (isSettling) {
      const revealDelay = 0.58;
      const revealDuration = 1.42;
      if (settleElapsed > revealDelay) {
        const progress = Math.min(1, (settleElapsed - revealDelay) / revealDuration);
        const easedProgress = progress * progress * (3 - 2 * progress);
        radialWave = easedProgress * 1.34;
        backgroundFade = Math.max(0, 1 - easedProgress);
      }
      if (settleElapsed >= revealDelay + revealDuration + 0.16) {
        this.finish();
        return;
      }
    }

    // Draw dark chameleon skin interstitial background between pebble scales
    if (backgroundFade > 0.01) {
      this.ctx.fillStyle = `rgba(18, 20, 24, ${0.85 * backgroundFade})`;
      this.ctx.fillRect(0, 0, this.width, this.height);
    }

    for (const b of this.blobs) {
      let r, g, bCol, a;
      let scaleMult = 1;

      if (!isSettling) {
        // Continuous organic traveling color wave using source image palette
        const wave = Math.sin(elapsed * 1.7 + (b.cx * 0.016 + b.cy * 0.014) + b.phase);
        const norm = (wave + 1) / 2;
        const pLen = this.palette.length;
        const p1 = this.palette[b.colorIdx % pLen];
        const p2 = this.palette[(b.colorIdx + 1) % pLen];

        r = p1[0] * (1 - norm) + p2[0] * norm;
        g = p1[1] * (1 - norm) + p2[1] * norm;
        bCol = p1[2] * (1 - norm) + p2[2] * norm;
        a = b.opacity;

        // Save current color continuously for instantaneous morph snapshot
        b.displayColor[0] = r;
        b.displayColor[1] = g;
        b.displayColor[2] = bCol;
        b.displayColor[3] = a;
      } else {
        // Seamless Hermite S-curve interpolation from the exact instant of finish to final design
        const morphT = Math.min(1, settleElapsed / 0.88);
        const ease = morphT * morphT * (3 - 2 * morphT); // Smoothstep curve

        r = b.startMorphColor[0] * (1 - ease) + b.targetColor[0] * ease;
        g = b.startMorphColor[1] * (1 - ease) + b.targetColor[1] * ease;
        bCol = b.startMorphColor[2] * (1 - ease) + b.targetColor[2] * ease;
        a = (b.targetColor[3] || 1) * b.opacity;

        // Radial dissolve starting from the middle
        if (radialWave > 0) {
          if (b.relDist < radialWave) {
            const depth = Math.max(0, Math.min(1, (radialWave - b.relDist) / 0.34));
            const factor = 1 - depth * depth * (3 - 2 * depth);
            a *= factor;
            scaleMult = 0.5 + 0.5 * factor;
            if (a <= 0.01) continue; // Fully dissolved
          }
        }
      }

      // Draw the smooth organic pebble blob scale (no wobble in coordinates)
      this.drawPebbleBlob(b.cx, b.cy, b.pts, `rgba(${Math.round(r)}, ${Math.round(g)}, ${Math.round(bCol)}, ${a})`, scaleMult, a);
    }

    this.animId = requestAnimationFrame(this.tick);
  }

  drawPebbleBlob(cx, cy, pts, fillColor, scaleMult = 1, alpha = 1) {
    if (pts.length < 3) return;
    const ctx = this.ctx;
    ctx.save();

    if (scaleMult !== 1) {
      ctx.translate(cx, cy);
      ctx.scale(scaleMult, scaleMult);
      ctx.translate(-cx, -cy);
    }

    // Smooth rounded pebble curve through polygon midpoints
    ctx.beginPath();
    const midX = (pts[0][0] + pts[pts.length - 1][0]) / 2;
    const midY = (pts[0][1] + pts[pts.length - 1][1]) / 2;
    ctx.moveTo(midX, midY);

    for (let i = 0; i < pts.length; i++) {
      const next = pts[(i + 1) % pts.length];
      const curr = pts[i];
      const mx = (curr[0] + next[0]) / 2;
      const my = (curr[1] + next[1]) / 2;
      ctx.quadraticCurveTo(curr[0], curr[1], mx, my);
    }
    ctx.closePath();

    ctx.fillStyle = fillColor;
    ctx.fill();

    // Subtle 3D pebble bead highlight on top-left (authentic chameleon scale sheen)
    ctx.strokeStyle = `rgba(255, 255, 255, ${0.1 * alpha})`;
    ctx.lineWidth = 0.75;
    ctx.stroke();

    ctx.restore();
  }

  finish() {
    this.state = 'idle';
    if (this.animId) {
      cancelAnimationFrame(this.animId);
      this.animId = null;
    }
    this.container.classList.add('visually-hidden');
    this.container.parentElement.classList.remove('is-generating');
    this.container.parentElement.classList.remove('is-revealing');
    if (this.outputImg) {
      this.outputImg.style.opacity = '1';
    }
    this.ctx.clearRect(0, 0, this.width, this.height);
  }

  stop() {
    this.finish();
  }
}
