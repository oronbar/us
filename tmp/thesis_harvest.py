from pathlib import Path
import json, hashlib
from pypdf import PdfReader

root=Path('D:/us')
out=root/'tmp/thesis_review'
out.mkdir(parents=True,exist_ok=True)
sources=[Path('C:/Users/Oron/OneDrive - Technion/Attachments/Research Proposal (1).pdf'),Path('C:/Users/Oron/Documents/Echocardiography-based early prediction of cardio-toxicity - Technion, Shaare Zedek, Ichilov .pptx.pdf')]
for name,p in zip(['proposal','grant_background'],sources):
    reader=PdfReader(p)
    text='\n\n'.join(f'--- PAGE {i+1} ---\n'+(page.extract_text() or '') for i,page in enumerate(reader.pages))
    (out/f'{name}.txt').write_text(text,encoding='utf-8')
    print(name,len(reader.pages),'pages',len(text),'characters')
reports=[]
for p in sorted(root.rglob('*.md')):
    rel=p.relative_to(root)
    if any(x.startswith('.') or x in ['tmp','node_modules','build','dist','Echo-Vison-FM','USF-MAE'] for x in rel.parts): continue
    reports.append({'path':str(rel),'text':p.read_text(encoding='utf-8',errors='replace')})
(out/'reports.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2),encoding='utf-8')
print('reports',len(reports))
for x in reports: print(x['path'],len(x['text']))
