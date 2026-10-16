import addOnUISdk from 'https://express.adobe.com/static/add-on-sdk/sdk.js';

const $ = id => document.getElementById(id);
let sdkReady=false, design=null, preview=null;
const status = message => {$('status').textContent=message;};
function update() {
  $('insert').disabled = !sdkReady || !preview;
  $('download').disabled = !design;
}
function setDesign(value) {
  if (!value?.canvas || !Array.isArray(value.elements)) throw new Error('Expected a challenge v0.9 design with canvas and elements.');
  design=value;
  $('json').textContent=JSON.stringify(design,null,2);
  $('output-size').textContent=`· ${design.canvas.width} × ${design.canvas.height}`;
  update();
}
function safePng(value) {
  if (typeof value !== 'string' || !value.startsWith('data:image/png;base64,')) throw new Error('Preview must be a bundled PNG.');
  return value;
}
$('bundle').addEventListener('change',async event=>{
  try {
    const file=event.target.files[0];if(!file)return;
    if(file.size>32*1024*1024)throw new Error('Bundle must be smaller than 32 MB.');
    const data=JSON.parse(await file.text());
    if(data.viewerVersion!==1)throw new Error('Load a bundle exported by the Layout lab, or use output files below.');
    const output=safePng(data.output_png), source=safePng(data.source_png);
    setDesign(data.design);
    preview=output;$('output').src=output;$('source').src=source;
    $('source-size').textContent=`· ${data.source.canvas.width} × ${data.source.canvas.height}`;
    $('warnings').replaceChildren(...(data.warnings||[]).map(message=>{const li=document.createElement('li');li.textContent=message;return li;}));
    update();status('Bundle loaded. Compare both layouts below.');
  }catch(error){status(error.message);}
});
$('files').addEventListener('change',async event=>{
  try {
    const files=[...event.target.files],json=files.find(f=>f.name.endsWith('.json')),png=files.find(f=>f.name.endsWith('.png'));
    if(!json||!png)throw new Error('Select the matching output JSON and PNG together.');
    if(json.size>4*1024*1024||png.size>24*1024*1024)throw new Error('Output files are too large.');
    const next=JSON.parse(await json.text());
    const dataUrl=await new Promise((resolve,reject)=>{const r=new FileReader();r.onload=()=>resolve(r.result);r.onerror=()=>reject(new Error('Could not read PNG.'));r.readAsDataURL(png);});
    setDesign(next);preview=safePng(dataUrl);$('output').src=preview;
    $('source').removeAttribute('src');$('source-size').textContent='';$('warnings').replaceChildren();
    update();status('Output loaded. Source preview is available through a viewer bundle.');
  }catch(error){status(error.message);}
});
$('insert').addEventListener('click',async()=>{
  if(!sdkReady||!preview)return;
  $('insert').disabled=true;
  try{const blob=await(await fetch(preview)).blob();await addOnUISdk.app.document.addImage(blob,{title:'MagiCanvas adapted layout'});status('Adapted preview added to Express.');}
  catch(error){status(`Could not insert preview: ${error.message}`);}
  finally{update();}
});
$('download').addEventListener('click',()=>{
  if(!design)return;
  const url=URL.createObjectURL(new Blob([JSON.stringify(design,null,2)],{type:'application/json'}));
  const a=document.createElement('a');a.href=url;a.download='adapted-design.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
addOnUISdk.ready.then(()=>{sdkReady=true;update();}).catch(error=>status(`Express SDK unavailable: ${error.message}`));
