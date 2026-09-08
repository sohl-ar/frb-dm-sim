"""Verify saved audit artifacts; no simulations, training, or posterior draws."""
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from repository_audit import rank_coverage_bounds

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    path=ROOT/'results/repository-audit.json'
    a=json.loads(path.read_text())
    m=json.loads((ROOT/'results/training-data-manifest.json').read_text())
    p=json.loads((ROOT/'results/posterior-gates.json').read_text())['gates']['G-P4']
    s=json.loads((ROOT/'results/sbc-spotcheck-50.json').read_text())
    checks=[{'path':entry['path'],'match':sha(ROOT/'work/training-data'/entry['path'])==entry['sha256']}
            for entry in m['shards']]
    assert all(x['match'] for x in checks)
    a['dataset_shards_verified']=len(checks); a['dataset_hash_checks']=checks
    ranks=np.asarray(p['raw_ranks']); rows=[]
    for j,item in enumerate(p['parameters']):
        row={'parameter':item['parameter']}
        for level in (.68,.95):
            lo,hi=rank_coverage_bounds(ranks[:,j],level,p['posterior_samples_per_trial'])
            assert lo.mean()<=item['coverage'+str(round(level*100))]<=hi.mean()
            row[str(level)]=[float(lo.mean()),float(hi.mean())]
        rows.append(row)
    a['global_coverage_rank_bounds']=rows
    with np.load(ROOT/s['posterior_artifact']) as replay:
        for level in (.68,.95):
            low,high=np.quantile(replay['posterior'],[(1-level)/2,(1+level)/2],axis=1)
            actual=(replay['truth']>=low)&(replay['truth']<=high)
            for j in range(4):
                lo,hi=rank_coverage_bounds(ranks[:s['trials'],j],level,p['posterior_samples_per_trial'])
                assert np.all(lo<=actual[:,j]) and np.all(actual[:,j]<=hi)
    a['rank_bounds_verified_against_50_replay_intervals']=True
    assert sha(ROOT/'results/training.json')==a['training_json_sha256']
    assert sha(ROOT/'results/posterior-gates.json')==a['original_sbc_sha256']
    a['original_training_and_SBC_hashes_unchanged']=True
    assert sha(ROOT/'docs/resume_train_reviewed.py.txt')==a['resume']['sha256']
    a['reviewed_resume_copy_hash_matches']=True
    newline=[]
    for name in ('results/training-data-manifest.json','results/training.json','results/posterior-gates.json','results/phase2a_gates.json'):
        local=(ROOT/name).read_bytes()
        committed=subprocess.check_output(['git','-C',str(ROOT),'show',a['audit_git_sha']+':'+name])
        newline.append({'file':name,'local_sha256':hashlib.sha256(local).hexdigest(),
            'committed_blob_sha256':hashlib.sha256(committed).hexdigest(),
            'same_bytes':local==committed,'same_json':json.loads(local)==json.loads(committed),
            'difference_is_only_CRLF_to_LF':local.replace(b'\r\n',b'\n')==committed})
    a['newline_provenance']={'status':'FLAG: raw manifest hash not portable to LF checkout','files':newline,
        'impact':'Fresh checkout can fail the checkpoint manifest guard despite equal parsed contents. Local SBC remains valid; provenance repair requires human decision.'}
    path.write_text(json.dumps(a,indent=2,allow_nan=False),encoding='utf-8',newline='\n')
    print(json.dumps({'shards_checked':len(checks),'rank_bounds_check':'pass','original_local_evidence_check':'pass'}))


if __name__=='__main__':
    main()
