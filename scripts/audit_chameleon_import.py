#!/usr/bin/env python3
"""Read-only inventory and focused probes of an imported Chameleon checkout.

No checkout, fetch, installation, model inference, training, or production edits.
Writes evidence only to the explicitly supplied MagiCanvas output directory.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True)


def inventory(repo: Path, output: Path, magicanvas: Path) -> dict:
    before = git(repo, "status", "--porcelain")
    refs = [r for r in git(repo, "for-each-ref", "--format=%(refname:short)",
                          "refs/remotes/origin").splitlines() if r != "origin/HEAD"]
    # Git's short-name printer can abbreviate origin/HEAD to just origin.
    refs = [r for r in refs if r != "origin"]
    branches, blobs = [], {}
    for ref in refs:
        commit = git(repo, "rev-parse", ref).strip()
        parent = git(repo, "merge-base", "origin/main", ref).strip()
        files = git(repo, "ls-tree", "-r", "--name-only", ref).splitlines()
        code = [p for p in files if p.endswith(".py") and p.split('/')[0] in
                ("engine", "eval", "critic", "generator")]
        branch_blobs = {}
        for path in code:
            blob = git(repo, "rev-parse", f"{ref}:{path}").strip()
            branch_blobs[path] = blob
            if blob in blobs:
                continue
            text = git(repo, "show", f"{ref}:{path}")
            tree = ast.parse(text, filename=f"{ref}:{path}")
            blobs[blob] = dict(example=f"{ref}:{path}", lines=len(text.splitlines()),
                               sha256=hashlib.sha256(text.encode()).hexdigest(),
                               definitions=[n.name for n in tree.body if isinstance(n,
                                            (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))],
                               imports=[ast.unparse(n) for n in tree.body
                                        if isinstance(n, (ast.Import, ast.ImportFrom))])
        branches.append(dict(ref=ref, commit=commit, main_merge_base=parent,
                             ancestor_of_main=subprocess.run(["git", "-C", str(repo),
                                 "merge-base", "--is-ancestor", ref, "origin/main"],
                                 capture_output=True).returncode == 0,
                             divergent_changes=git(repo, "diff", "--name-status", parent, ref).splitlines(),
                             differences_from_main=git(repo, "diff", "--name-status", "origin/main", ref).splitlines(),
                             code_blobs=branch_blobs,
                             license_files=[p for p in files if Path(p).name.lower().startswith(
                                 ("license", "copying"))]))
    library = json.loads((repo/'engine/data/layout_library.json').read_text())
    result = dict(repo=str(repo), head=git(repo, "rev-parse", "HEAD").strip(),
                  magicanvas_head=git(magicanvas, "rev-parse", "HEAD").strip(),
                  remote_branches=len(branches), unique_python_blobs=len(blobs),
                  branches=branches, code_inventory=blobs,
                  tags=git(repo, "show-ref", "--tags").splitlines(),
                  library=dict(entries=len(library), bytes=(repo/'engine/data/layout_library.json').stat().st_size,
                               entry_keys=sorted(library[0]),
                               unit_count=sum(len(r['units']) for r in library)),
                  model_files={name:(repo/'engine/data'/name).exists()
                               for name in ('generator.onnx','critic.onnx','interface.json')},
                  installed_dependencies={name:bool(importlib.util.find_spec(name))
                                          for name in ('ortools','onnxruntime','numpy','torch','skia')},
                  imported_worktree_unchanged=git(repo, "status", "--porcelain")==before)
    (output/'branch_inventory.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def probes(repo: Path, output: Path) -> dict:
    # A fresh process prevents MagiCanvas's identically named engine package
    # from satisfying imports intended for the imported repository.
    program = r'''
import copy, itertools, json, math, random, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from engine.fit import Grid, Sizer
from engine.retrieve import assign
from engine.analyze import Unit, analyze
from generator.group_features import group_features
from generator.campaigns import cluster
from critic.damage import DAMAGES
from eval.score import score

result={}
rng=np.random.default_rng(2026)
for n in range(1,6):
 for m in range(n,n+3):
  for _ in range(10):
   cost=rng.normal(size=(n,m))
   answer=assign(cost)
   actual=sum(cost[i,j] for i,j in answer)
   expected=min(sum(cost[i,j] for i,j in enumerate(p)) for p in itertools.permutations(range(m),n))
   assert abs(actual-expected)<1e-9
result['hungarian_bruteforce_matrices']=150
checked=0
for tw,th in [(100,100),(728,90),(160,600),(1600,400)]:
 for seed in range(12):
  g=Grid(tw,th,3);r=random.Random(seed)
  for _ in range(7):
   x,y=r.random()*tw,r.random()*th
   g.mark((x,y,min(tw,x+r.random()*tw/4),min(th,y+r.random()*th/4)))
  free=g.free_rects()
  # Test cell-center coverage: every vacant interior cell must be represented,
  # and no returned rectangle may contain an occupied cell center.
  for rr in range(g.occ.shape[0]):
   for cc in range(g.occ.shape[1]):
    x,y=(cc+.5)*g.cell,(rr+.5)*g.cell
    if not (3<x<tw-3 and 3<y<th-3):continue
    covered=any(a<=x<b and c<=y<d for a,c,b,d in free)
    assert covered != bool(g.occ[rr,cc])
  checked+=1
result['free_rectangle_grids_checked']=checked

els={f'e{i}':dict(id=f'e{i}',type='image',x=i*110,y=0,width=100,height=60) for i in range(4)}
parts=[([f'e{i}'],(i*110,0,i*110+100,60)) for i in range(4)]
u=Unit('siblings',list(els),'rigid',(0,0,430,60),parts=parts)
out=[]
for tw,th in [(160,600),(600,160)]:
 s=Sizer(SimpleNamespace(els=els),tw,th)
 arrangements=s.arrangements(u)
 assert {r[0] for r in arrangements}=={1,2,3,4}
 for cols,w,h,origins in arrangements:
  assert set(origins)==set(els)
  for i,j in itertools.combinations(els,2):
   x,y=origins[i];xx,yy=origins[j]
   assert x+100<=xx or xx+100<=x or y+60<=yy or yy+60<=y
 extent=s.unit_extent(u,1.,tw,th)
 out.append(dict(target=[tw,th],columns=extent.cols,scale=extent.scale))
result['sibling_reflow']=out
# Labels make the readability constraint meaningful; pure ornament rows may
# legitimately remain in one row at a smaller scale.
for i in range(4):
 els[f't{i}']=dict(id=f't{i}',type='text',x=i*110+10,y=10,width=80,height=30,
                  font_size=20,font_family='Solstice Sans',content='Badge',color='rgba(0,0,0,1)')
 parts[i][0].append(f't{i}')
u=Unit('labelled_siblings',list(els),'rigid',(0,0,430,60),parts=parts)
s=Sizer(SimpleNamespace(els=els),160,600)
extent=s.unit_extent(u,1.,160,600)
assert extent.cols<4 and extent.scale>=s.read_scale(u)
result['labelled_sibling_reflow']=dict(columns=extent.cols,scale=extent.scale,
                                      minimum_readable_scale=s.read_scale(u))

sample=json.loads(Path('input/chameleon_sample_level3_advanced.json').read_text())['source']
analysis=analyze(sample)
groups=group_features(analysis)
assert sorted(i for g in groups for i in g['members'])==list(range(len(sample['elements'])))
result['group_partition']=dict(elements=len(sample['elements']),groups=len(groups))
damage=[]
for name,fn in DAMAGES.items():
 changed=fn(analysis,copy.deepcopy(sample),random.Random(2026))
 if changed is None:
  damage.append(dict(kind=name,applicable=False));continue
 assert [e['id'] for e in changed['elements']]==[e['id'] for e in sample['elements']]
 assert all(math.isfinite(e[k]) for e in changed['elements'] for k in ('x','y','width','height'))
 damage.append(dict(kind=name,applicable=True,score=score(sample,sample['canvas'],changed)['total']))
result['damage_probes']=damage
fp=lambda tokens:dict(types=['text','image'],texts=tokens)
records=[dict(design_hash='a',fingerprints=[fp(['1','2'])]),
         dict(design_hash='b',fingerprints=[fp(['1','2','3'])]),
         dict(design_hash='c',fingerprints=[fp(['2','3'])])]
components,edges=cluster(records)
assert len(set(components.values()))==1
result['campaign_transitive_bridge']=True
print(json.dumps(result))
'''
    run=subprocess.run([sys.executable,'-B','-c',program],cwd=repo,text=True,capture_output=True)
    (output/'geometry_probes.log').write_text(run.stdout+run.stderr)
    if run.returncode:
        return dict(returncode=run.returncode, error=run.stderr[-3000:])
    result=json.loads(run.stdout)
    (output/'geometry_probes.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--magicanvas',type=Path,default=Path.cwd(),
                   help='MagiCanvas checkout used for baseline provenance (default: cwd)')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    inv=inventory(a.repo.resolve(),a.output.resolve(),a.magicanvas.resolve())
    result=probes(a.repo.resolve(),a.output.resolve())
    print(json.dumps(dict(remote_branches=inv['remote_branches'],
                          unique_python_blobs=inv['unique_python_blobs'],
                          probes=result),indent=2))


if __name__=='__main__':
    main()
