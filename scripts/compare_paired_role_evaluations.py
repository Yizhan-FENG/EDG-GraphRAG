#!/usr/bin/env python3
"""Compare two role-evaluation folders by identical held-out sample IDs."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from statistics import fmean

def read(path: Path):
 return {r['sample_id']:r for r in (json.loads(x) for x in (path/'per_sample.jsonl').read_text(encoding='utf8').splitlines() if x.strip())}
def f1(row,key): return row.get(key,{}).get('f1',0.0) if row.get('json_valid') else 0.0
def main():
 p=argparse.ArgumentParser(); p.add_argument('v1'); p.add_argument('v2'); p.add_argument('--output',required=True); a=p.parse_args()
 one,two=read(Path(a.v1)),read(Path(a.v2)); ids=sorted(set(one)&set(two)); rows=[]
 for i in ids:
  x,y=one[i],two[i]
  rows.append({'sample_id':i,'v1_json_valid':x.get('json_valid',False),'v2_json_valid':y.get('json_valid',False),'entity_f1_delta':f1(y,'entity')-f1(x,'entity'),'claim_f1_delta':f1(y,'claim')-f1(x,'claim'),'evidence_f1_delta':f1(y,'evidence')-f1(x,'evidence')})
 summary={'paired_samples':len(rows),'v1_json_valid_rate':sum(r['v1_json_valid'] for r in rows)/len(rows) if rows else None,'v2_json_valid_rate':sum(r['v2_json_valid'] for r in rows)/len(rows) if rows else None,'mean_entity_f1_delta':fmean(r['entity_f1_delta'] for r in rows) if rows else None,'mean_claim_f1_delta':fmean(r['claim_f1_delta'] for r in rows) if rows else None,'mean_evidence_f1_delta':fmean(r['evidence_f1_delta'] for r in rows) if rows else None,'note':'Descriptive paired pilot only; report CI before publication claims.'}
 out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps({'summary':summary,'per_sample':rows},ensure_ascii=False,indent=2),encoding='utf8'); print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__': main()
