"""Request-local exact raster cache with optional isolated render processes.

Renderers cache mutable instances. Separate processes, never shared SkCanvas/
SkSurface threads, own concurrent renders. Adaptive candidate decisions remain
serial and every candidate retains its family identity and measurements.
"""
import json,os,shutil,time
from hashlib import sha256
from pathlib import Path


def _render_job(doc,assets,path,mode,max_size):
    if mode=='dataset':
        from scripts.dataset_renderer import render_design_to_png
        fonts=Path(__file__).resolve().parents[1]/'data/crello_fonts.pickle'
        return render_design_to_png(doc,assets,path,fonts,max_size=max_size,adjust_tight_text_widths=False)
    from renderer.converter.convert import render_design_to_png
    return render_design_to_png(doc,assets,path,max_size=max_size)


class DiagnosticRenderCache:
    def __init__(self,renderer,assets,mode='brand',max_size=640):
        self.renderer=renderer;self.assets=assets;self.mode=mode;self.max_size=max_size
        self.cache={};self.executor=None
        self.stats=dict(requests=0,renders=0,cache_hits=0,render_seconds=0.,workers=1,parallel_fallbacks=0)
    def __enter__(self):
        count=int(os.environ.get('MAGICANVAS_RENDER_WORKERS','1'))
        if count not in (1,2):raise ValueError('MAGICANVAS_RENDER_WORKERS must be 1 or 2')
        if count==2:
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor
            self.executor=ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn'));self.stats['workers']=2
        return self
    def __exit__(self,*exc):
        if self.executor:self.executor.shutdown(wait=True,cancel_futures=True)
    def signature(self,doc):
        return sha256(json.dumps(dict(document=doc,assets=str(self.assets),mode=self.mode,max_size=self.max_size),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    def render_many(self,requests):
        pending={};answers=[None]*len(requests)
        for i,(doc,path) in enumerate(requests):
            self.stats['requests']+=1;key=self.signature(doc)
            if key in self.cache:
                old,warnings=self.cache[key]
                if str(old)!=str(path):shutil.copyfile(old,path)
                answers[i]=list(warnings);self.stats['cache_hits']+=1;continue
            # Coalesce exact duplicates inside the same concurrent batch too.
            if key in pending:
                pending[key]['aliases'].append((i,path));continue
            job=dict(doc=doc,path=path,index=i,aliases=[],started=time.perf_counter())
            if self.executor:job['future']=self.executor.submit(_render_job,doc,self.assets,path,self.mode,self.max_size)
            pending[key]=job
        for key,job in pending.items():
            try:
                warnings=(job['future'].result() if 'future' in job else self.renderer(job['doc'],self.assets,job['path'],max_size=self.max_size))
            except Exception:
                if 'future' not in job:raise
                self.stats['parallel_fallbacks']+=1
                warnings=self.renderer(job['doc'],self.assets,job['path'],max_size=self.max_size)
            self.stats['renders']+=1;self.stats['render_seconds']+=time.perf_counter()-job['started']
            self.cache[key]=(job['path'],list(warnings));answers[job['index']]=list(warnings)
            for i,path in job['aliases']:
                shutil.copyfile(job['path'],path);answers[i]=list(warnings);self.stats['cache_hits']+=1
        return answers
    def render(self,doc,path):return self.render_many([(doc,path)])[0]
