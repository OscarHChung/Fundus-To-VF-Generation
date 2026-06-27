import os, sys, json, numpy as np, torch
sys.path.insert(0,'decoder')
import training as T, diagnostics as D
from longitudinal_model import LongitudinalVFModel
from longitudinal_dataset import LongitudinalDataset, collate_val
from torch.utils.data import DataLoader
CV='decoder/results/cv_long'; AUTO='decoder/results/auto'
def evalf(model, vj):
    ds=LongitudinalDataset(vj,mode='val',use_tta=True); dl=DataLoader(ds,batch_size=1,collate_fn=collate_val)
    P=[];Tt=[];HP=[];PV=[];PM=[]
    model.eval()
    with torch.no_grad():
        for xs,hvf,lat,pv,pm,dt,hp in dl:
            xs=xs.to(T.DEVICE);V=xs.shape[0];lt=model._encode(xs)
            pred=model.decode_latent_long(lt,pv[None].expand(V,-1).to(T.DEVICE),pm[None].expand(V,-1).to(T.DEVICE),
                dt[None].expand(V).to(T.DEVICE),hp[None].expand(V).to(T.DEVICE),laterality=[lat]*V,average_multi=True)
            vi=T.valid_indices_od if lat.startswith('OD') else T.valid_indices_os
            t=hvf.numpy()[vi].astype(np.float64);t[t>=99]=np.nan
            P.append(pred.cpu().numpy()[0]);Tt.append(t);HP.append(float(hp));PV.append(pv.numpy());PM.append(pm.numpy())
    return P,Tt,HP,PV,PM
allP=[];allT=[];allHP=[];allPV=[];allPM=[]
for f in range(5):
    m=LongitudinalVFModel(T.base_model,'decoder/pretrained_vf_ae.pth',global_head=True).to(T.DEVICE)
    sd=torch.load(f'{AUTO}/long_prior_f{f}_best.pth',map_location='cpu',weights_only=False)['model']
    m.load_state_dict(sd,strict=False)
    P,Tt,HP,PV,PM=evalf(m,f'{CV}/fold{f}_val.json'); allP+=P;allT+=Tt;allHP+=HP;allPV+=PV;allPM+=PM
def mae_sub(idx): 
    pp=np.concatenate([allP[i][~np.isnan(allT[i])] for i in idx]); tt=np.concatenate([allT[i][~np.isnan(allT[i])] for i in idx])
    return np.abs(pp-tt).mean(), len(idx)
wp=[i for i in range(len(allHP)) if allHP[i]>0]; v1=[i for i in range(len(allHP)) if allHP[i]==0]
print(f'MODEL  with-prior MAE {mae_sub(wp)[0]:.3f} (n={len(wp)})  |  visit-1 MAE {mae_sub(v1)[0]:.3f} (n={len(v1)})')
# persistence on with-prior (predict = prior_vec, only observed pts)
pe=[]
for i in wp:
    pv=allPV[i];pm=allPM[i];t=allT[i];msk=(~np.isnan(t))&(pm>0)
    pe.append(np.abs(pv[msk]-t[msk]))
print(f'PERSIST with-prior MAE {np.concatenate(pe).mean():.3f}   (model must beat this)')
print(f'refs: fundus-only visit-1 ~4.29 ; pooled model RAW 3.751')
