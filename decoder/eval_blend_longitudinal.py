import os, sys, json, numpy as np, torch
sys.path.insert(0,'decoder')
import training as T, diagnostics as D, eval_ckpt as E
from longitudinal_dataset import prior_vec_mask
CV='decoder/results/cv_long'; AUTO='decoder/results/auto'
# deterministic blend: visit-1 -> long_global fundus pred ; with-prior -> persistence (prior obs; fundus for masked-prior pts)
allP=[];allT=[];allHP=[]
for f in range(5):
    recs=json.load(open(f'{CV}/fold{f}_val.json'))
    m=E.load_model(f'{AUTO}/long_global_f{f}_best.pth')
    fp,ft=E.per_eye_preds(m, f'{CV}/fold{f}_val.json')   # fundus preds per record (query order)
    for r,fpred,ftrue in zip(recs,fp,ft):
        lat=r['Laterality']
        if r['has_prior']:
            pv,pm=prior_vec_mask(r['prior_hvf'],lat)   # 52 query order
            pred=np.where(pm>0, pv, fpred)             # persistence where prior observed, else fundus
        else:
            pred=fpred
        allP.append(pred.astype(np.float64)); allT.append(ftrue); allHP.append(r['has_prior'])
raw=D.pooled_metrics(allP,allT)
# calibrate (fit b on... use a simple variance match on the pooled, illustrative)
print('DETERMINISTIC BLEND (persistence follow-ups + fundus first-visits):')
print(' ',D.fmt(raw))
D.stratified_report(allP,allT)
wp=[i for i in range(len(allHP)) if allHP[i]]; v1=[i for i in range(len(allHP)) if not allHP[i]]
def sub(idx):
    pp=np.concatenate([allP[i][~np.isnan(allT[i])] for i in idx]); tt=np.concatenate([allT[i][~np.isnan(allT[i])] for i in idx])
    return np.abs(pp-tt).mean()
print(f'  with-prior {sub(wp):.3f} (n{len(wp)}) | visit-1 {sub(v1):.3f} (n{len(v1)})')
print(f'refs: trained model 3.751 ; fundus-only 4.29 ; persistence-blend target')
