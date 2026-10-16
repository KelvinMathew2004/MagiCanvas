import assert from 'node:assert/strict';
import {contrast, paletteFromSeed, parseColor, seedFromDesign, dominantPixels} from '../viewer/theme.mjs';
const rgb = value => value.match(/[\d.]+/g).map(Number);
const seeds = [[0,0,0],[255,255,255],[128,128,128],[255,0,0],[0,255,0],[0,0,255],[255,255,0],[0,255,255],[255,0,255],[55,25,52],[252,245,220]];
for (const seed of seeds) for (const dark of [false,true]) {
  const p = Object.fromEntries(Object.entries(paletteFromSeed(seed,dark)).map(([k,v])=>[k,rgb(v)]));
  for (const bg of ['--page','--surface','--stage']) {
    assert.ok(contrast(p['--text'],p[bg])>=4.5,`${seed}: body contrast on ${bg}`);
    assert.ok(contrast(p['--muted'],p[bg])>=4.5,`${seed}: label contrast on ${bg}`);
  }
  assert.ok(contrast(p['--on-accent'],p['--accent'])>=4.5,`${seed}: primary button contrast`);
  assert.ok(contrast(p['--on-accent'],p['--accent-hover'])>=4.5,`${seed}: primary hover contrast`);
  assert.ok(contrast(p['--control-border'],p['--surface'])>=3,`${seed}: control boundary contrast`);
  for (const values of Object.values(p)) assert.ok(values.every(v=>Number.isFinite(v)&&v>=0&&v<=255));
}
assert.notEqual(paletteFromSeed([255,0,0],true)['--accent'],paletteFromSeed([0,0,255],true)['--accent']);
assert.deepEqual(parseColor('#abc'),[170,187,204,1]);
assert.deepEqual(parseColor('#11223380').slice(0,3),[17,34,51]);
assert.deepEqual(parseColor('rgba(120, 35, 200, 0.4)'),[120,35,200,.4]);
assert.equal(parseColor('not a color'),null);
const design={canvas:{width:100,height:100},elements:[{type:'shape',shape_type:'rectangle',x:0,y:0,width:100,height:100,fill_color:'#eebbaa'}]};
assert.deepEqual(seedFromDesign(design),[238,187,170]);
design.elements.push({type:'image',x:0,y:0,width:100,height:100,z_index:1});
assert.equal(seedFromDesign(design),null,'An image above a background fill must be sampled.');
// A bright central illustration must not overpower a dominant dark perimeter.
const pixels=new Uint8ClampedArray(16*16*4);
for(let y=0;y<16;y++)for(let x=0;x<16;x++) {
 const center=x>3&&x<12&&y>3&&y<12;
 pixels.set(center?[240,110,40,255]:[40,20,55,255],(y*16+x)*4);
}
assert.deepEqual(dominantPixels(pixels,16,16),[40,20,55]);
assert.equal(dominantPixels(new Uint8ClampedArray(64),4,4),null);
console.log('Theme checks passed: 22 light/dark palettes, contrast, source fills, image precedence, edge-weighted sampling, transparency.');
