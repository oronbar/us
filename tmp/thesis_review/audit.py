from pathlib import Path
import pandas as pd, json, hashlib, zipfile, xml.etree.ElementTree as ET
import fitz
from PIL import Image, ImageOps, ImageDraw
r=Path('D:/us'); out=r/'tmp/thesis_review'
t=pd.read_parquet(r/'cardiotoxicity_next_visit_gpu_results/next_visit_transitions.parquet')
v=pd.read_parquet(r/'amber_full_105_preprocessed/Ichilov_july_visits.parquet')
print('VISITS',len(v),'columns',list(v.columns)[:15])
print('TRANSITIONS',len(t),'patients',t.patient_id.nunique(),'primary',t['mask__mid_first_rel15'].sum(),'events',t.loc[t['mask__mid_first_rel15'],'label__mid_first_rel15'].sum())
f=pd.read_csv(r/'cardiotoxicity_next_visit_gpu_results/patient_fold_assignments.csv')
print('FOLDS',len(f),'unique patient repeat counts',f.groupby(['repeat','patient_id']).size().value_counts().to_dict())
print('HASH_MATCH',hashlib.sha256((r/'amber_full_105_preprocessed/Ichilov_july_visits.parquet').read_bytes()).hexdigest())
from sklearn.metrics import roc_auc_score,average_precision_score
p=pd.read_parquet(r/'cardiotoxicity_timeseries_round3_results/round3_oof_predictions.parquet')
for model in ['clinical_ridge','current_cnn','ensemble_equal_cnn_mantis_timemil']:
 x=p[p.model.eq(model)]; print('RECOMPUTED',model,len(x),roc_auc_score(x.label,x.score),average_precision_score(x.label,x.score))
for d in ['cardiotoxicity_timeseries_round3_results','cardiotoxicity_timeseries_round4_results']:
 m=pd.read_csv(r/d/(('round3' if 'round3' in d else 'round4')+'_metrics.csv'))
 print(d,m[m.model.str.contains('ensemble_equal')][['model','roc_auc','average_precision','roc_auc_ci_low','roc_auc_ci_high','average_precision_ci_low','average_precision_ci_high']].to_string(index=False))
texts=[]
for path in list(r.glob('*.pptx'))+list(r.glob('*.docx')):
 if path.name.startswith('~$'): continue
 with zipfile.ZipFile(path) as z:
  names=[n for n in z.namelist() if (n.startswith('ppt/slides/slide') and n.endswith('.xml')) or n=='word/document.xml']
  tx=[]
  for n in names:
   tx.append(' '.join(e.text for e in ET.fromstring(z.read(n)).iter() if e.tag.endswith('}t') and e.text))
 texts.append({'path':path.name,'text':'\n'.join(tx)})
(out/'presentation_text.json').write_text(json.dumps(texts,ensure_ascii=False,indent=2),encoding='utf-8')
print('PRESENTATIONS',[(x['path'],len(x['text'])) for x in texts])
paths=[Path('C:/Users/Oron/OneDrive - Technion/Attachments/Research Proposal (1).pdf'),Path('C:/Users/Oron/Documents/Echocardiography-based early prediction of cardio-toxicity - Technion, Shaare Zedek, Ichilov .pptx.pdf')]
for stem,path in zip(['proposal','grant'],paths):
 doc=fitz.open(path)
 for start in range(0,len(doc),12):
  sheet=Image.new('RGB',(1500,1600),'#ddd'); dr=ImageDraw.Draw(sheet)
  for k,page in enumerate(list(doc)[start:start+12]):
   pix=page.get_pixmap(matrix=fitz.Matrix(1,1)); im=Image.frombytes('RGB',[pix.width,pix.height],pix.samples)
   im.thumbnail((485,365)); x=(k%3)*500;y=(k//3)*400
   sheet.paste(im,(x,y+25));dr.text((x+5,y+4),f'{stem} page {start+k+1}',fill='black')
  sheet.save(out/f'{stem}_{start+1}.png')
