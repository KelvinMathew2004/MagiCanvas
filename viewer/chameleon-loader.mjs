/**
 * 3D Chrysalis Shed AI Generation Loader
 * Tactile Reptilian Chameleon Molt:
 * 1. Generating Phase: Organic cobblestone pebble scales breathe and cycle through the
 *    source design's palette across the dark interstitial skin with subtle 3D bevels.
 * 2. Color Harmonization: When the design is ready, scale colors smoothly morph into the
 *    adapted image's pixels via gamma-corrected blending (1.35s), assembling the new design.
 * 3. 3D Chrysalis Shed: The scales physically lift off the canvas, cast subtle 3D drop
 *    shadows, tilt in pseudo-3D space, and roll outward toward the edges like a shedding
 *    reptile skin, smoothly revealing the crisp photograph underneath.
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

  // Keep declared colors as palette accents
  if (sourceDesign && Array.isArray(sourceDesign.elements)) {
    for (const el of sourceDesign.elements) {
      for (const value of [el.fill_color, el.color, el.stroke_color, el.background_color]) {
        const color = parseColor(value);
        if (color) colors.push(color);
      }
    }
  }

  // Representative sample from actual rendered source
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
      // Use declared element colors if sampling fails
    }
  }

  if (colors.length < 2) {
    colors.push([24, 180, 115, 1], [245, 166, 35, 1], [208, 60, 35, 1], [45, 156, 219, 1], [142, 68, 173, 1]);
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
    this.scales = [];
    this.palette = [];
    this.paletteTransition = null;
    this.startTime = 0;
    this.settleStartTime = 0;
    this.targetImgData = null;
    this.width = 300;
    this.height = 300;
    this.dpr = window.devicePixelRatio || 1;
    this.centerX = 150;
    this.centerY = 150;
    this.maxDist = 1;
  }

  start({ targetWidth, targetHeight, sourceDesign, sourceImg }) {
    this.palette = extractPalette(sourceDesign, sourceImg);
    this.paletteTransition = null;
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
    this.initScales();

    if (!this.animId) {
      this.tick = this.tick.bind(this);
      this.animId = requestAnimationFrame(this.tick);
    }
  }

  async updateSourcePalette(sourceDesign, sourceImg) {
    if (sourceImg && !sourceImg.complete) {
      try { await sourceImg.decode(); } catch { /* Use declared design colors if decoding fails. */ }
    }
    const nextPalette = extractPalette(sourceDesign, sourceImg);
    if (this.state === 'idle' || !this.scales.length) {
      this.palette = nextPalette;
      return;
    }
    this.paletteTransition = {
      from: this.palette,
      to: nextPalette,
      start: performance.now(),
      duration: 420
    };
    this.palette = nextPalette;
  }

  resizeCanvas() {
    this.dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.round(this.width * this.dpr);
    this.canvas.height = Math.round(this.height * this.dpr);
    this.canvas.style.width = `${this.width}px`;
    this.canvas.style.height = `${this.height}px`;
    this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
  }

  initScales() {
    this.scales = [];
    // Spacing for tactile reptile cobblestone scales
    const spacing = Math.max(8, Math.min(15, Math.round(Math.min(this.width, this.height) / 25)));
    const cols = Math.ceil(this.width / spacing) + 3;
    const rows = Math.ceil(this.height / spacing) + 3;

    // Organic jitter grid for chameleon cobblestones
    const grid = [];
    for (let r = 0; r < rows; r++) {
      const row = [];
      for (let c = 0; c < cols; c++) {
        const jitterX = (Math.sin(r * 12.9898 + c * 78.233) * 0.5) * 0.40;
        const jitterY = (Math.cos(r * 4.1414 + c * 34.123) * 0.5) * 0.38;
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

        // Voronoi cell computation against 8 neighbors
        for (let dr = -1; dr <= 1; dr++) {
          for (let dc = -1; dc <= 1; dc++) {
            if (dr === 0 && dc === 0) continue;
            const p1 = grid[r + dr][c + dc];
            poly = clipPoly(poly, p0, p1);
          }
        }

        if (poly.length < 3) continue;

        // Inset polygon to leave narrow skin groove seams
        const insetFactor = 0.86;
        const insetPts = poly.map(pt => [
          p0[0] + (pt[0] - p0[0]) * insetFactor,
          p0[1] + (pt[1] - p0[1]) * insetFactor
        ]);

        const dist = Math.hypot(p0[0] - this.centerX, p0[1] - this.centerY);
        const relDist = Math.min(1, dist / this.maxDist);
        const angle = Math.atan2(p0[1] - this.centerY, p0[0] - this.centerX);
        const colorIdx = Math.floor(Math.random() * this.palette.length);

        this.scales.push({
          cx: p0[0],
          cy: p0[1],
          pts: insetPts,
          dist,
          relDist,
          angle,
          colorIdx,
          phase: Math.random() * Math.PI * 2,
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

      // Sample pixel colors from actual generated design
      const offscreen = document.createElement('canvas');
      offscreen.width = this.width;
      offscreen.height = this.height;
      const oCtx = offscreen.getContext('2d', { willReadFrequently: true });
      oCtx.drawImage(img, 0, 0, this.width, this.height);
      this.targetImgData = oCtx.getImageData(0, 0, this.width, this.height).data;

      // Assign target colors & snapshot current instantaneous display colors
      for (const s of this.scales) {
        s.startMorphColor = [...s.displayColor];

        const sx = Math.max(0, Math.min(this.width - 1, Math.round(s.cx)));
        const sy = Math.max(0, Math.min(this.height - 1, Math.round(s.cy)));
        const idx = (sy * this.width + sx) * 4;
        s.targetColor = [
          this.targetImgData[idx],
          this.targetImgData[idx + 1],
          this.targetImgData[idx + 2],
          this.targetImgData[idx + 3] / 255
        ];
      }
    } catch {
      // Keep generation colors if sampling fails
    }

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

    // Timeline:
    // 0.0s - 1.35s: Scale colors smoothly morph into the target design (gamma-correct S-curve)
    // 0.70s - 2.15s: 3D Chrysalis Shed peels and rolls outward toward the edges
    // 2.2s+: Finish
    let shedWave = 0;
    let backgroundFade = 1;

    if (isSettling) {
      const revealDelay = 0.70;
      const shedDuration = 1.45;
      if (settleElapsed > revealDelay) {
        const progress = Math.min(1, (settleElapsed - revealDelay) / shedDuration);
        const easedProgress = progress * progress * (3 - 2 * progress);
        shedWave = easedProgress * 1.35;
        backgroundFade = Math.max(0, 1 - easedProgress);
      }
      if (settleElapsed >= revealDelay + shedDuration + 0.16) {
        this.finish();
        return;
      }
    }

    // Dark under-skin interstitial background between scales
    if (backgroundFade > 0.01) {
      this.ctx.fillStyle = `rgba(16, 18, 22, ${0.88 * backgroundFade})`;
      this.ctx.fillRect(0, 0, this.width, this.height);
    }

    for (const s of this.scales) {
      let r, g, bCol, a;
      let scaleX = 1;
      let scaleY = 1;
      let transX = 0;
      let transY = 0;
      let lift = 0;

      if (!isSettling) {
        // Continuous traveling wave of source palette
        const wave = Math.sin(elapsed * 1.5 + (s.cx * 0.016 + s.cy * 0.014) + s.phase);
        const norm = (wave + 1) / 2;
        const sample = (palette) => {
          const c1 = palette[s.colorIdx % palette.length];
          const c2 = palette[(s.colorIdx + 1) % palette.length];
          return [0, 1, 2].map(i => Math.sqrt((1 - norm) * (c1[i] ** 2) + norm * (c2[i] ** 2)));
        };
        const nextColor = sample(this.palette);
        if (this.paletteTransition) {
          const blend = Math.min(1, (time - this.paletteTransition.start) / this.paletteTransition.duration);
          const oldColor = sample(this.paletteTransition.from);
          [r, g, bCol] = nextColor.map((val, i) => Math.sqrt((1 - blend) * (oldColor[i] ** 2) + blend * (val ** 2)));
          if (blend >= 1) this.paletteTransition = null;
        } else {
          [r, g, bCol] = nextColor;
        }
        a = s.opacity;

        s.displayColor[0] = r;
        s.displayColor[1] = g;
        s.displayColor[2] = bCol;
        s.displayColor[3] = a;
      } else {
        // Smooth gamma-corrected color morph into design
        const morphT = Math.min(1, settleElapsed / 1.35);
        const ease = morphT * morphT * (3 - 2 * morphT);

        const rSq = (1 - ease) * (s.startMorphColor[0] ** 2) + ease * (s.targetColor[0] ** 2);
        const gSq = (1 - ease) * (s.startMorphColor[1] ** 2) + ease * (s.targetColor[1] ** 2);
        const bSq = (1 - ease) * (s.startMorphColor[2] ** 2) + ease * (s.targetColor[2] ** 2);
        r = Math.sqrt(rSq);
        g = Math.sqrt(gSq);
        bCol = Math.sqrt(bSq);
        a = (s.targetColor[3] || 1) * s.opacity;

        // 3D Chrysalis Shed: Scales lift, tilt, and peel outward
        if (shedWave > 0) {
          if (s.relDist < shedWave) {
            const shedDepth = Math.min(1, (shedWave - s.relDist) / 0.35);
            const shedEase = shedDepth * shedDepth * (3 - 2 * shedDepth);

            lift = shedEase;

            // 3D tilt: scale rotates on edge and rolls outward
            const tiltAngle = shedEase * (Math.PI * 0.48);
            scaleX = Math.max(0.06, Math.cos(tiltAngle));
            scaleY = Math.max(0.35, 1 - shedEase * 0.55);

            // Outward radial slide / peel
            const slideDist = shedEase * 22;
            transX = Math.cos(s.angle) * slideDist;
            transY = Math.sin(s.angle) * slideDist - (shedEase * 4);

            a *= Math.max(0, 1 - shedEase * 1.15);
            if (a <= 0.01) continue;
          }
        }
      }

      this.drawShedScale(s, `rgba(${Math.round(r)}, ${Math.round(g)}, ${Math.round(bCol)}, ${a})`, transX, transY, scaleX, scaleY, lift, a);
    }

    this.animId = requestAnimationFrame(this.tick);
  }

  drawShedScale(s, fillColor, transX, transY, scaleX, scaleY, lift, alpha) {
    const pts = s.pts;
    if (pts.length < 3) return;
    const ctx = this.ctx;
    ctx.save();

    // Position and 3D orientation transform
    ctx.translate(s.cx + transX, s.cy + transY);
    ctx.rotate(s.angle);
    ctx.scale(scaleX, scaleY);
    ctx.rotate(-s.angle);
    ctx.translate(-s.cx, -s.cy);

    // Subtle 3D shadow as scale lifts during molt
    if (lift > 0.05 && alpha > 0.1) {
      ctx.shadowColor = `rgba(0, 0, 0, ${0.40 * lift * alpha})`;
      ctx.shadowBlur = 6 * lift;
      ctx.shadowOffsetX = Math.cos(s.angle) * (3 * lift);
      ctx.shadowOffsetY = Math.sin(s.angle) * (3 * lift) + (2 * lift);
    }

    // 1. Smooth rounded pebble curve through polygon midpoints
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

    // 2. Beveled 3D scale edge highlight
    const rimAlpha = Math.max(0.06, 0.18 * (1 - lift * 0.4)) * alpha;
    ctx.strokeStyle = `rgba(255, 255, 255, ${rimAlpha})`;
    ctx.lineWidth = 0.85;
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
    this.container.parentElement.classList.remove('is-revealing');
    this.container.parentElement.classList.remove('is-generating');
    if (this.outputImg) {
      this.outputImg.style.opacity = '1';
    }
    this.ctx.clearRect(0, 0, this.width, this.height);
  }

  stop() {
    this.finish();
  }
}
