import { paletteFromSeed, seedFromDesign, themeSeed } from '/theme.mjs';

const $ = (id) => document.getElementById(id);
let task, result, samples = [], requestId = 0;
let backgroundSeed = null;
const appearance = matchMedia('(prefers-color-scheme: dark)');
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
  ['json', 'png', 'bundle'].forEach(id => $(id).disabled = true);
}
async function adapt() {
  if (!task) return;
  const width = Number($('width').value), height = Number($('height').value);
  if (![width, height].every(v => Number.isInteger(v) && v > 0)) { setStatus('Enter positive whole-pixel dimensions.', 'error'); return; }
  const id = ++requestId;
  document.body.classList.add('busy'); $('adapt').disabled = true;
  setStatus('Adapting…');
  try {
    const response = await fetch('/api/adapt', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({source:task.source, target_canvas:{width,height}})});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Adaptation failed.');
    if (id !== requestId) return;
    result = data;
    void updateTheme(data.source, data.source_png, id);
    $('source').src = data.source_png; $('output').src = data.output_png;
    $('source-size').textContent = `${data.source.canvas.width} × ${data.source.canvas.height}`;
    $('output-size').textContent = `${width} × ${height}`;
    $('document').textContent = JSON.stringify(data.design, null, 2);
    $('warnings').replaceChildren(...data.warnings.map(message => {const li=document.createElement('li');li.textContent=message;return li;}));
    $('warning-count').textContent = `(${data.warnings.length})`;
    setStatus(`${data.design.elements.length} elements · ${data.seconds}s${data.warnings.length ? ` · ${data.warnings.length} rendering ${data.warnings.length === 1 ? 'note' : 'notes'}` : ''}`, data.warnings.length ? 'warning' : '');
    ['json','png','bundle'].forEach(key => $(key).disabled=false);
  } catch (error) { if (id === requestId) setStatus(error.message, 'error'); }
  finally { if (id === requestId) {document.body.classList.remove('busy');$('adapt').disabled=false;} }
}
function download(data, name, type) {
  const url=URL.createObjectURL(new Blob([data], {type})); const link=document.createElement('a');link.href=url;link.download=name;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
$('adapt').addEventListener('click', adapt);
$('sample').addEventListener('change', () => {const sample=samples[Number($('sample').value)];if(!sample)return;setTask(sample.task);adapt();});
$('upload').addEventListener('change', async (event) => {try {const file=event.target.files[0];if(!file)return;setTask(JSON.parse(await file.text()));$('sample').querySelector('[data-import]')?.remove();const option=document.createElement('option');option.dataset.import='true';option.value='imported';option.textContent=file.name;$('sample').append(option);$('sample').value='imported';await adapt();}catch(error){setStatus(error.message, 'error');}});
$('format').addEventListener('change', () => {if($('format').value==='custom')return;const [w,h]=$('format').value.split(',');$('width').value=w;$('height').value=h;adapt();});
['width','height'].forEach(id => {$(id).addEventListener('keydown', e=>{if(e.key==='Enter'){e.preventDefault();$(id).blur();}});$(id).addEventListener('change',()=>{$('format').value='custom';adapt();});});
$('json').onclick=()=>{if(result)download(JSON.stringify(result.design,null,2),'adapted-design.json','application/json');};
$('png').onclick=()=>{if(!result)return;const link=document.createElement('a');link.href=result.output_png;link.download='adapted-design.png';link.click();};
$('bundle').onclick=()=>{if(result)download(JSON.stringify({viewerVersion:1,...result}),'magicanvas-viewer-bundle.json','application/json');};
(async()=>{try{const response=await fetch('/api/samples');if(!response.ok)throw new Error('Could not load samples.');samples=await response.json();for(const [i,sample]of samples.entries()){const option=document.createElement('option');option.value=i;option.textContent=sample.name.replace('chameleon_sample_','').replaceAll('_',' ');$('sample').append(option);}if(samples.length){setTask(samples[0].task);await adapt();}else{setStatus('Import a design to begin.');}}catch(error){setStatus(error.message, 'error');}})();
