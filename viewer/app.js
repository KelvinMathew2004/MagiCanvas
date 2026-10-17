import { paletteFromSeed, seedFromDesign, themeSeed } from '/theme.mjs';
import { ChameleonLoader } from '/chameleon-loader.mjs';

const $ = (id) => document.getElementById(id);
let task, result, samples = [], requestId = 0;
let backgroundSeed = null;
const appearance = matchMedia('(prefers-color-scheme: dark)');

let chameleonLoader = null;
function getLoader() {
  if (!chameleonLoader && $('loader-stage') && $('chameleon-canvas')) {
    chameleonLoader = new ChameleonLoader({
      container: $('loader-stage'),
      canvas: $('chameleon-canvas'),
      outputImg: $('output')
    });
  }
  return chameleonLoader;
}

function applyTheme(seed) {
  backgroundSeed = seed;
  for (const [name, value] of Object.entries(paletteFromSeed(seed, appearance.matches))) {
    document.documentElement.style.setProperty(name, value);
  }
}
appearance.addEventListener('change', () => { if (backgroundSeed) applyTheme(backgroundSeed); });
async function updateTheme(source, preview, id) {
  try {
    const seed = await themeSeed(source, preview);
    if (seed && id === requestId) applyTheme(seed);
  } catch { /* An unreadable preview keeps the neutral or previous theme. */ }
}

function setStatus(message, state = '') {
  $('status').textContent = message;
  $('status').dataset.state = state;
}
function setTask(value) {
  const next = value.source ? value : {source: value, target_canvas: value.canvas};
  if (!next.source?.canvas || !Array.isArray(next.source.elements)) throw new Error('Load a v0.9 challenge task or design with canvas and elements.');
  task = structuredClone(next);
  const immediateSeed = seedFromDesign(task.source);
  if (immediateSeed) applyTheme(immediateSeed);
  $('width').value = task.target_canvas?.width ?? task.source.canvas.width;
  $('height').value = task.target_canvas?.height ?? task.source.canvas.height;
  $('format').value = 'custom';
  result = null;
  ['json', 'png', 'bundle', 'source-preview', 'output-preview'].forEach(id => $(id).disabled = true);
}
function applyAdaptedResult(data, id, width, height, loader) {
  result = data;
  void loader?.updateSourcePalette(data.source, $('source'));
  void updateTheme(data.source, data.source_png, id);
  $('source').src = data.source_png;
  $('output').src = data.output_png;
  $('source-size').textContent = `${data.source.canvas.width} × ${data.source.canvas.height}`;
  $('output-size').textContent = `${width} × ${height}`;
  $('document').textContent = JSON.stringify(data.design, null, 2);
  $('warnings').replaceChildren(...data.warnings.map(message => {const li=document.createElement('li');li.textContent=message;return li;}));
  $('warning-count').textContent = `(${data.warnings.length})`;
  setStatus(`${data.design.elements.length} elements · ${data.seconds}s${data.warnings.length ? ` · ${data.warnings.length} rendering ${data.warnings.length === 1 ? 'note' : 'notes'}` : ''}`, data.warnings.length ? 'warning' : '');
  ['json','png','bundle','source-preview','output-preview'].forEach(key => $(key).disabled=false);
  return loader ? loader.transitionToResult(data.output_png) : Promise.resolve();
}
async function readAdaptStream(response, id, width, height, loader) {
  const reader=response.body?.getReader();
  if (!reader) throw new Error('The browser could not read the adaptation stream.');
  const decoder=new TextDecoder();
  let pending='', completed=false;
  const handle=async block => {
    if (!block.trim()) return;
    let type='message';const dataLines=[];
    for (const line of block.split('\n')) {
      if (line.startsWith('event:')) type=line.slice(6).trim();
      else if (line.startsWith('data:')) dataLines.push(line.slice(5).trimStart());
    }
    const payload=JSON.parse(dataLines.join('\n'));
    if (type==='error') throw new Error(payload.error || 'Adaptation failed.');
    if (id!==requestId) return;
    if (type==='source') {
      result={source:task.source,source_png:payload.source_png,design:null};
      $('source').src=payload.source_png;
      $('source-size').textContent=`${payload.source_size.width} × ${payload.source_size.height}`;
      $('source-preview').disabled=false;
      await loader?.updateSourcePalette(task.source,$('source'));
      if (id!==requestId) return;
      void updateTheme(task.source,payload.source_png,id);
      setStatus('Original ready · adapting layout…');
    } else if (type==='adapted') {
      completed=true;
      await applyAdaptedResult(payload,id,width,height,loader);
    }
  };
  while (true) {
    const {value,done}=await reader.read();
    pending+=decoder.decode(value || new Uint8Array(),{stream:!done}).replaceAll('\r\n','\n');
    let split;
    while ((split=pending.indexOf('\n\n'))>=0) {
      const block=pending.slice(0,split);pending=pending.slice(split+2);
      await handle(block);
    }
    if (done) break;
  }
  if (pending.trim()) await handle(pending);
  if (!completed) throw new Error('The adaptation stream ended before the layout was ready.');
}
async function adapt() {
  if (!task) return;
  const width = Number($('width').value), height = Number($('height').value);
  if (![width, height].every(v => Number.isInteger(v) && v > 0)) { setStatus('Enter positive whole-pixel dimensions.', 'error'); return; }
  const id = ++requestId;
  document.body.classList.add('busy'); $('adapt').disabled = true;
  setStatus('Rendering original…');

  const loader = getLoader();
  if (loader) {
    loader.start({
      targetWidth: width,
      targetHeight: height,
      sourceDesign: task.source,
      sourceImg: $('source')
    });
  }

  try {
    const response = await fetch('/api/adapt-stream', {method:'POST', headers:{'Content-Type':'application/json','Accept':'text/event-stream'}, body:JSON.stringify({source:task.source, target_canvas:{width,height}})});
    if (!response.ok) {
      const error=await response.json().catch(()=>({}));
      throw new Error(error.error || 'Adaptation failed.');
    }
    if (id !== requestId) {
      loader?.stop();
      return;
    }
    await readAdaptStream(response,id,width,height,loader);
  } catch (error) {
    if (id === requestId) {
      loader?.stop();
      setStatus(error.message, 'error');
    }
  }
  finally { if (id === requestId) {document.body.classList.remove('busy');$('adapt').disabled=false;} }
}
function download(data, name, type) {
  const url=URL.createObjectURL(new Blob([data], {type})); const link=document.createElement('a');link.href=url;link.download=name;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
const imageViewer = $('image-viewer');
const enlargedImage = $('image-viewer-image');
let actualSize = false;
let lightboxRequestId = 0;
let lightboxObjectUrl = null;
let lightboxFallback = false;
function fitImage() {
  actualSize = false;
  imageViewer.classList.remove('actual-size');
  enlargedImage.style.removeProperty('width');
  enlargedImage.style.removeProperty('height');
  $('image-viewer-zoom').textContent = '100%';
  $('image-viewer-zoom').setAttribute('aria-label', 'Show image at actual pixel size');
  $('image-viewer-zoom').setAttribute('aria-pressed', 'false');
}
function openImageViewer(kind) {
  if (!result) return;
  fitImage();
  const title = kind === 'source' ? 'Original' : 'Adapted';
  const design = kind === 'source' ? result.source : result.design;
  const preview = kind === 'source' ? result.source_png : result.output_png;
  const requestId = ++lightboxRequestId;
  if (lightboxObjectUrl) {
    URL.revokeObjectURL(lightboxObjectUrl);
    lightboxObjectUrl = null;
  }
  $('image-viewer-title').textContent = title;
  $('image-viewer-size').textContent = '';
  $('image-viewer-loading').hidden = false;
  enlargedImage.hidden = true;
  lightboxFallback = false;
  enlargedImage.alt = `${title} design`;
  enlargedImage.removeAttribute('src');
  document.body.classList.add('image-viewer-open');
  if (!imageViewer.open) imageViewer.showModal();

  fetch('/api/lightbox', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({design})
  }).then(async response => {
    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.error || 'Could not prepare the HD preview.');
    }
    return response.blob();
  }).then(blob => {
    if (requestId !== lightboxRequestId || !imageViewer.open) return;
    lightboxObjectUrl = URL.createObjectURL(blob);
    enlargedImage.src = lightboxObjectUrl;
  }).catch(() => {
    if (requestId === lightboxRequestId && imageViewer.open) {
      lightboxFallback = true;
      $('image-viewer-size').textContent = 'HD preview unavailable · showing standard preview';
      enlargedImage.src = preview;
    }
  });
}
enlargedImage.addEventListener('load', () => {
  $('image-viewer-size').textContent = `${lightboxFallback ? 'Standard preview · ' : ''}${enlargedImage.naturalWidth} × ${enlargedImage.naturalHeight} px`;
  if (actualSize) {
    enlargedImage.style.width = `${enlargedImage.naturalWidth}px`;
    enlargedImage.style.height = `${enlargedImage.naturalHeight}px`;
  }
  enlargedImage.hidden = false;
  $('image-viewer-loading').hidden = true;
});
$('source-preview').addEventListener('click', () => openImageViewer('source'));
$('output-preview').addEventListener('click', () => openImageViewer('output'));
$('image-viewer-close').addEventListener('click', () => imageViewer.close());
imageViewer.addEventListener('close', () => {
  lightboxRequestId++;
  document.body.classList.remove('image-viewer-open');
  enlargedImage.removeAttribute('src');
  enlargedImage.hidden = true;
  $('image-viewer-loading').hidden = false;
  $('image-viewer-size').textContent = '';
  if (lightboxObjectUrl) {
    URL.revokeObjectURL(lightboxObjectUrl);
    lightboxObjectUrl = null;
  }
  fitImage();
});
imageViewer.addEventListener('click', event => {
  if (event.target !== imageViewer) return;
  const bounds = imageViewer.getBoundingClientRect();
  if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) imageViewer.close();
});
$('image-viewer-zoom').addEventListener('click', () => {
  if (actualSize) { fitImage(); return; }
  actualSize = true;
  imageViewer.classList.add('actual-size');
  enlargedImage.style.width = `${enlargedImage.naturalWidth}px`;
  enlargedImage.style.height = `${enlargedImage.naturalHeight}px`;
  $('image-viewer-zoom').textContent = 'Fit';
  $('image-viewer-zoom').setAttribute('aria-label', 'Fit image to viewer');
  $('image-viewer-zoom').setAttribute('aria-pressed', 'true');
});
$('adapt').addEventListener('click', adapt);
$('sample').addEventListener('change', () => {const sample=samples[Number($('sample').value)];if(!sample)return;setTask(sample.task);adapt();});
$('upload').addEventListener('change', async (event) => {try {const file=event.target.files[0];if(!file)return;setTask(JSON.parse(await file.text()));$('sample').querySelector('[data-import]')?.remove();const option=document.createElement('option');option.dataset.import='true';option.value='imported';option.textContent=file.name;$('sample').append(option);$('sample').value='imported';await adapt();}catch(error){setStatus(error.message, 'error');}});
$('format').addEventListener('change', () => {if($('format').value==='custom')return;const [w,h]=$('format').value.split(',');$('width').value=w;$('height').value=h;adapt();});
['width','height'].forEach(id => {$(id).addEventListener('keydown', e=>{if(e.key==='Enter'){e.preventDefault();$(id).blur();}});$(id).addEventListener('change',()=>{$('format').value='custom';adapt();});});
$('json').onclick=()=>{if(result)download(JSON.stringify(result.design,null,2),'adapted-design.json','application/json');};
$('png').onclick=()=>{if(!result)return;const link=document.createElement('a');link.href=result.output_png;link.download='adapted-design.png';link.click();};
$('bundle').onclick=()=>{if(result)download(JSON.stringify({viewerVersion:1,...result}),'magicanvas-viewer-bundle.json','application/json');};
(async()=>{try{const response=await fetch('/api/samples');if(!response.ok)throw new Error('Could not load samples.');samples=await response.json();for(const [i,sample]of samples.entries()){const option=document.createElement('option');option.value=i;option.textContent=sample.name.replace('chameleon_sample_','').replaceAll('_',' ');$('sample').append(option);}if(samples.length){setTask(samples[0].task);await adapt();}else{setStatus('Import a design to begin.');}}catch(error){setStatus(error.message, 'error');}})();
