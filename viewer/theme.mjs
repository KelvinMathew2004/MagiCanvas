// Content-derived theme. No sample IDs, asset names, or design-specific palettes.
const clamp = (value, min = 0, max = 1) => Math.min(max, Math.max(min, value));
const linear = value => (value /= 255) <= .04045 ? value / 12.92 : ((value + .055) / 1.055) ** 2.4;
const srgb = value => 255 * (value <= .0031308 ? 12.92 * value : 1.055 * value ** (1 / 2.4) - .055);
export const luminance = rgb => rgb.map(linear).reduce((sum, value, i) => sum + value * [.2126, .7152, .0722][i], 0);
export const contrast = (a, b) => (Math.max(luminance(a), luminance(b)) + .05) / (Math.min(luminance(a), luminance(b)) + .05);
const css = rgb => `rgb(${rgb.map(value => Math.round(clamp(value, 0, 255))).join(' ')})`;

function oklch(rgb) {
  const [r, g, b] = rgb.map(linear);
  const l = Math.cbrt(.4122214708*r + .5363325363*g + .0514459929*b);
  const m = Math.cbrt(.2119034982*r + .6806995451*g + .1073969566*b);
  const s = Math.cbrt(.0883024619*r + .2817188376*g + .6299787005*b);
  const a = 1.9779984951*l - 2.428592205*m + .4505937099*s;
  const bb = .0259040371*l + .7827717662*m - .808675766*s;
  return [.2104542553*l + .793617785*m - .0040720468*s, Math.hypot(a, bb), Math.atan2(bb, a)];
}

function tone(lightness, chroma, hue) {
  // Reduce chroma into sRGB rather than clipping bright colors to a new hue.
  for (let attempt = 0; attempt < 30; attempt++, chroma *= .85) {
    const a = chroma * Math.cos(hue), b = chroma * Math.sin(hue);
    const l = (lightness + .3963377774*a + .2158037573*b) ** 3;
    const m = (lightness - .1055613458*a - .0638541728*b) ** 3;
    const s = (lightness - .0894841775*a - 1.291485548*b) ** 3;
    const rgb = [srgb(4.0767416621*l - 3.3077115913*m + .2309699292*s),
      srgb(-1.2684380046*l + 2.6097574011*m - .3413193965*s),
      srgb(-.0041960863*l - .7034186147*m + 1.707614701*s)];
    if (rgb.every(v => v >= -.01 && v <= 255.01)) return rgb.map(v => Math.round(clamp(v, 0, 255)));
  }
  return [srgb(lightness ** 3), srgb(lightness ** 3), srgb(lightness ** 3)].map(Math.round);
}

function readableTone(lightness, chroma, hue, background, minimum = 4.5) {
  let rgb = tone(lightness, chroma, hue);
  const direction = luminance(background) > .4 ? -1 : 1;
  for (let i = 0; i < 80 && contrast(rgb, background) < minimum; i++) {
    lightness = clamp(lightness + direction * .01);
    rgb = tone(lightness, chroma, hue);
  }
  return rgb;
}

export function paletteFromSeed(seed, dark) {
  const [, sourceChroma, hue] = oklch(seed);
  // Surfaces stay almost neutral. Most of the source color lives in the action.
  const tint = Math.min(sourceChroma * .045, .0035);
  const chroma = sourceChroma < .018 ? sourceChroma * .5 : clamp(sourceChroma * 1.6, .075, .15);
  const page = tone(dark ? .19 : .978, tint, hue);
  const surface = tone(dark ? .235 : .995, tint, hue);
  const stage = tone(dark ? .165 : .947, tint, hue);
  const foreground = dark ? [239, 239, 242] : [36, 36, 40];
  const muted = readableTone(dark ? .70 : .48, tint, hue, surface);
  const accent = tone(dark ? .76 : .43, chroma, hue);
  const onAccent = contrast([255, 255, 255], accent) >= contrast([20, 20, 23], accent) ? [255,255,255] : [20,20,23];
  const focus = readableTone(dark ? .78 : .44, chroma, hue, surface, 3);
  const palette = {
    '--page': page, '--surface': surface, '--text': foreground, '--muted': muted,
    '--border': tone(dark ? .30 : .87, tint, hue),
    '--control-border': readableTone(dark ? .43 : .65, tint, hue, surface, 3),
    '--stage': stage, '--accent': accent, '--on-accent': onAccent,
    '--accent-hover': tone(dark ? .81 : .38, chroma, hue),
    '--hover': tone(dark ? .235 : .955, tint, hue), '--focus': focus,
    '--warning': readableTone(dark ? .78 : .45, .09, .85, surface),
  };
  return Object.fromEntries(Object.entries(palette).map(([key,value]) => [key, css(value)]));
}

export function parseColor(value) {
  if (typeof value !== 'string') return null;
  const hex = value.trim().match(/^#([\da-f]{3,8})$/i);
  if (hex && [3,4,6,8].includes(hex[1].length)) {
    let digits = hex[1];
    if (digits.length <= 4) digits = [...digits].map(v => v + v).join('');
    return [0,2,4].map(i => parseInt(digits.slice(i,i+2),16)).concat(digits.length === 8 ? parseInt(digits.slice(6,8),16)/255 : 1);
  }
  const rgb = value.trim().match(/^rgba?\((.*)\)$/i);
  if (rgb) {
    const parts = rgb[1].trim().split(/[\s,/]+/);
    if (parts.length < 3 || parts.length > 4) return null;
    const values = parts.map((v,i) => Number.parseFloat(v) * (v.endsWith('%') ? (i === 3 ? .01 : 2.55) : 1));
    if (!values.every(Number.isFinite)) return null;
    return values.slice(0,3).map(v => clamp(v,0,255)).concat(clamp(values[3] ?? 1));
  }
  return null;
}

export function seedFromDesign(design) {
  const {width, height} = design.canvas;
  const backgrounds = design.elements.filter(e => e.type !== 'text' && e.x <= 0 && e.y <= 0 && e.x + e.width >= width && e.y + e.height >= height && !e.rotation)
    .sort((a,b) => (b.z_index ?? 0) - (a.z_index ?? 0));
  const background = backgrounds[0];
  if (background?.type === 'image') return null; // Sample its real rendered pixels.
  if (background?.type === 'shape' && background.shape_type === 'rectangle') {
    const color = parseColor(background.fill_color);
    if (color && color[3] * (background.opacity ?? 1) >= .98) return color.slice(0,3);
  }
  return null;
}

export function dominantPixels(pixels, width, height) {
  const buckets = new Map();
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const offset = (y * width + x) * 4;
    if (pixels[offset+3] < 200) continue;
    const rgb = [pixels[offset], pixels[offset+1], pixels[offset+2]];
    const weight = (x < width*.14 || x > width*.86 || y < height*.14 || y > height*.86) ? 4 : 1;
    const key = rgb.map(v => Math.floor(v/24)).join(',');
    const bucket = buckets.get(key) ?? {weight:0, rgb:[0,0,0]};
    bucket.weight += weight;
    bucket.rgb = bucket.rgb.map((v,i) => v + rgb[i] * weight);
    buckets.set(key,bucket);
  }
  const winner = [...buckets.values()].sort((a,b) => b.weight-a.weight)[0];
  return winner ? winner.rgb.map(v => Math.round(v/winner.weight)) : null;
}

export async function themeSeed(design, preview) {
  const direct = seedFromDesign(design);
  if (direct) return direct;
  if (!preview) return null;
  const image = new Image();
  image.src = preview;
  await image.decode();
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = 64;
  const context = canvas.getContext('2d', {willReadFrequently:true});
  if (!context) return null;
  context.drawImage(image,0,0,64,64);
  return dominantPixels(context.getImageData(0,0,64,64).data,64,64);
}
