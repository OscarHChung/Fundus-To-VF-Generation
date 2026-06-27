import os, sys, json, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0,'decoder')
import training as T, diagnostics as D, eval_ckpt as E
from longitudinal_dataset import prior_vec_mask
CV='decoder/results/cv_long'; AUTO='decoder/results/auto'
P=[];Tt=[]
for f in range(5):
    recs=json.load(open(f'{CV}/fold{f}_val.json')); m=E.load_model(f'{AUTO}/long_global_f{f}_best.pth')
    fp,ft=E.per_eye_preds(m, f'{CV}/fold{f}_val.json')
    for r,fpred,ftrue in zip(recs,fp,ft):
        if r['has_prior']:
            pv,pm=prior_vec_mask(r['prior_hvf'],r['Laterality']); pred=np.where(pm>0,pv,fpred)
        else: pred=fpred
        P.append(pred);Tt.append(ftrue)
mets=D.pooled_metrics(P,Tt); strat=D.stratified_report(P,Tt,verbose=False)
tt=np.concatenate([Tt[i][~np.isnan(Tt[i])] for i in range(len(P))]); pp=np.concatenate([P[i][~np.isnan(Tt[i])] for i in range(len(P))])
rmse=np.sqrt(((pp-tt)**2).mean())
fig,ax=plt.subplots(figsize=(7.6,7.6))
ax.scatter(tt,pp,s=6,alpha=0.10,color='#1f4e79',edgecolors='none')
lo,hi=0,36
ax.plot([lo,hi],[lo,hi],'--',color='gray',lw=1.2,label='y = x (perfect)')
sl,ic=np.polyfit(tt,pp,1); xs=np.array([lo,hi])
ax.plot(xs,sl*xs+ic,'-',color='#c00000',lw=2.2,label=f'best fit: y = {sl:.2f}x + {ic:.1f}')
ax.axvspan(0,10,color='orange',alpha=0.07)
ax.text(5,1.0,'severe\n(0–10 dB)',ha='center',va='bottom',fontsize=8,color='#a0522d')
ax.set_xlim(lo,hi); ax.set_ylim(lo,hi); ax.set_aspect('equal','box')
ax.set_xlabel('True 24-2 sensitivity (dB)',fontsize=11); ax.set_ylabel('Predicted sensitivity (dB)',fontsize=11)
ax.set_title('Longitudinal 24-2 VF prediction (fundus + prior VF)\nhonest per-patient 5-fold CV, 631 records',fontsize=12,pad=12)
box=(f"This model (raw)\n"
     f"  pointwise MAE   {mets['mae']:.2f} dB\n"
     f"  RMSE            {rmse:.2f} dB\n"
     f"  mild/mod/severe {strat['mild']['mae']:.2f} / {strat['moderate']['mae']:.2f} / {strat['severe']['mae']:.2f}\n"
     f"  pointwise r     {mets['corr']:.2f}   slope {sl:.2f}\n"
     f"────────────────────────\n"
     f"TDV-Net, Graefe's 2026 (fundus-only,\n31k imgs, total-deviation target)\n"
     f"  pointwise MAE   3.91 dB\n"
     f"  mild/mod/severe 3.09 / 5.66 / 9.15")
ax.text(0.03,0.97,box,transform=ax.transAxes,va='top',ha='left',fontsize=8.5,family='monospace',
        bbox=dict(boxstyle='round',facecolor='white',edgecolor='#888',alpha=0.92))
ax.legend(loc='lower right',fontsize=9); ax.grid(alpha=0.15)
out='decoder/results/auto/longitudinal_scatter.png'
fig.savefig(out,dpi=140,bbox_inches='tight')
print(f"saved {out}")
print(f"OUR: MAE {mets['mae']:.3f} RMSE {rmse:.3f} slope {sl:.3f} corr {mets['corr']:.3f} | "
      f"mild {strat['mild']['mae']:.3f} mod {strat['moderate']['mae']:.3f} severe {strat['severe']['mae']:.3f}")
