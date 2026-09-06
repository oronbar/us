from pathlib import Path
import re,json,hashlib,csv,html
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY,TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate,BaseDocTemplate,PageTemplate,Frame,Paragraph,Spacer,PageBreak,Table,TableStyle,Image,KeepTogether
from reportlab.platypus.tableofcontents import TableOfContents
root=Path('D:/us'); out=root/'output/thesis';out.mkdir(parents=True,exist_ok=True)
pdfout=root/'output/pdf';pdfout.mkdir(parents=True,exist_ok=True)
figdir=out/'figures';figdir.mkdir(exist_ok=True)
refs=[
('Suter TM and Ewer MS','Cancer drugs and the heart: importance and management','2013','European Heart Journal 34:1102-1111','10.1093/eurheartj/ehs181'),
('Lyon AR and colleagues','2022 ESC Guidelines on cardio-oncology','2022','European Heart Journal 43:4229-4361','10.1093/eurheartj/ehac244'),
('Lang RM and colleagues','Recommendations for cardiac chamber quantification by echocardiography in adults: an update from the ASE and EACVI','2015','Journal of the American Society of Echocardiography 28:1-39.e14','10.1016/j.echo.2014.10.003'),
('Herrmann J and colleagues','Defining cardiovascular toxicities of cancer therapies: an International Cardio-Oncology Society consensus statement','2022','European Heart Journal 43:280-299','10.1093/eurheartj/ehab674'),
('Negishi T and colleagues','Cardioprotection Using Strain-Guided Management of Potentially Cardiotoxic Cancer Therapy: 3-Year Results of the SUCCOUR Trial','2023','JACC Cardiovascular Imaging','10.1016/j.jcmg.2022.10.010'),
('Chang WT and colleagues','Layer-specific distribution of myocardial deformation from anthracycline-induced cardiotoxicity in patients with breast cancer - From bedside to bench','2020','International Journal of Cardiology 311:64-70','10.1016/j.ijcard.2020.01.036'),
('Kim MN and colleagues','Serial changes of layer-specific myocardial function according to chemotherapy regimen in patients with breast cancer','2022','European Heart Journal Open 2:oeac008','10.1093/ehjopen/oeac008'),
('Demissei BG and colleagues','Left ventricular segmental strain and the prediction of cancer therapy-related cardiac dysfunction','2021','European Heart Journal Cardiovascular Imaging 22:418-426','10.1093/ehjci/jeaa288'),
('Yahav A and Adam D','Early Detection of Left Ventricular Dysfunction With Machine Learning-Based Strain Imaging in Aortic Stenosis Patients','2024','Echocardiography 41:e70007','10.1111/echo.70007'),
('Farsalinos KE and colleagues','Head-to-Head Comparison of Global Longitudinal Strain Measurements among Nine Different Vendors: The EACVI/ASE Inter-Vendor Comparison Study','2015','Journal of the American Society of Echocardiography 28:1171-1181.e2','10.1016/j.echo.2015.06.011'),
('Khamis H and colleagues','Feasibility of reproducible vendor independent estimation of cardiac function based on first generation speckle tracking echocardiography','2016','Journal of Biomedical Engineering and Informatics 2:57; online December 2015','10.5430/jbei.v2n2p57'),
('Pineiro-Lamas B and colleagues','A cardiotoxicity dataset for breast cancer patients','2023','Scientific Data 10:527','10.1038/s41597-023-02419-1'),
('Ouyang D and colleagues','Video-based AI for beat-to-beat assessment of cardiac function','2020','Nature 580:252-256','10.1038/s41586-020-2145-8'),
('Kalliatakis G and colleagues','EchoRisk: A Multicentre Echocardiography Dataset and Benchmark for Cardio-Oncology','2026','arXiv preprint, submitted 1 July 2026','10.48550/arXiv.2607.01039'),
('Goswami M and colleagues','MOMENT: A Family of Open Time-series Foundation Models','2024','ICML 2024; arXiv version 3','10.48550/arXiv.2402.03885'),
('Feofanov V and colleagues','Mantis: Lightweight Foundation Model for Time Series Classification','2026','ICML 2026; arXiv version 2, first submitted 2025','10.48550/arXiv.2502.15637'),
('Chen X and colleagues','TimeMIL: Advancing Multivariate Time Series Classification via a Time-aware Multiple Instance Learning','2024','Original method paper, arXiv','10.48550/arXiv.2405.03140'),
('Lubba CH and colleagues','catch22: CAnonical Time-series CHaracteristics','2019','Data Mining and Knowledge Discovery 33:1821-1852','10.1007/s10618-019-00647-x'),
('Guillaume A and Vrain C and Elloumi W','Random Dilated Shapelet Transform: A New Approach for Time Series Shapelets','2021','Original arXiv manuscript; subsequent conference publication 2022','10.48550/arXiv.2109.13514'),
('Saito T and Rehmsmeier M','The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets','2015','PLOS ONE 10:e0118432','10.1371/journal.pone.0118432'),
('Varoquaux G','Cross-validation failure: Small sample sizes lead to large error bars','2018','NeuroImage 180:68-77','10.1016/j.neuroimage.2017.06.061'),
('Riley RD and colleagues','Calculating the sample size required for developing a clinical prediction model','2020','BMJ 368:m441','10.1136/bmj.m441'),
('Collins GS and colleagues','TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods','2024','BMJ 385:e078378','10.1136/bmj-2023-078378'),
('Moons KGM and colleagues','PROBAST+AI: an updated quality, risk of bias, and applicability assessment tool for prediction models using regression or artificial intelligence methods','2025','BMJ 388:e082505','10.1136/bmj-2024-082505'),
]
reftext=[];bib=[]
for i,(a,t,y,j,d) in enumerate(refs,1):
 reftext.append(f'[{i}] {a.replace(" and colleagues", " et al")}. {t}. {j}. {y}. [doi:{d}](https://doi.org/{d}).')
 bib.append('@article{ref'+str(i)+',\n  author = {'+a.replace(' and colleagues',' and others')+'},\n  title = {'+t+'},\n  year = {'+y+'},\n  journal = {'+j+'},\n  doi = {'+d+'},\n  url = {https://doi.org/'+d+'}\n}\n')
(out/'references.bib').write_text('\n'.join(bib),encoding='utf-8')
m3=pd.read_csv(root/'cardiotoxicity_timeseries_round3_results/round3_metrics.csv').set_index('model')
m4=pd.read_csv(root/'cardiotoxicity_timeseries_round4_results/round4_metrics.csv').set_index('model')
names=[('clinical_ridge','Clinical ridge'),('current_cnn','Retained CNN'),('mantis_random_frozen_curves_scalars','Random Mantis plus scalars'),('moment_small_frozen_curves_scalars','MOMENT plus scalars'),('timemil_curves_scalars','TimeMIL derived plus scalars'),('ensemble_equal_cnn_mantis_timemil','Equal CNN plus random Mantis plus TimeMIL')]
alt=[('ensemble_cnn_moment_catch22_xgb_curves_scalars','CNN plus MOMENT plus catch22'),('ensemble_cnn_moment_rdst_shapelet_curves_scalars','CNN plus MOMENT plus RDST'),('ensemble_cnn_moment_drcif_curves_scalar_blend','CNN plus MOMENT plus DrCIF')]
def metric_table(frame,sel):
 lines=['| Model | AUC with 95% CI | AP with 95% CI |','| --- | --- | --- |']
 for k,name in sel:
  x=frame.loc[k];lines.append(f'| {name} | {x.roc_auc:.3f} ({x.roc_auc_ci_low:.3f}-{x.roc_auc_ci_high:.3f}) | {x.average_precision:.3f} ({x.average_precision_ci_low:.3f}-{x.average_precision_ci_high:.3f}) |')
 return '\n'.join(lines)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9})
fig,axes=plt.subplots(1,2,figsize=(8.2,3.5),sharey=True)
labels=['Clinical ridge','Retained CNN','Random Mantis + scalars','MOMENT + scalars','TimeMIL derived + scalars','Fixed three-model ensemble']
for ax,metric,title,ref in zip(axes,['roc_auc','average_precision'],['Discrimination','Average precision'],[.5,49/238]):
 for y,(k,_) in enumerate(names):
  x=m3.loc[k];a=x[metric];lo=x[metric+'_ci_low'];hi=x[metric+'_ci_high'];ax.errorbar(a,y,xerr=[[a-lo],[hi-a]],fmt='o',color='#153c52',capsize=3)
 ax.axvline(ref,ls='--',color='#999',lw=1);ax.set_title(title);ax.grid(axis='x',alpha=.15);ax.set_xlabel('AUC' if metric=='roc_auc' else 'AP');ax.set_ylim(5.6,-.6);ax.spines[['top','right']].set_visible(False)
axes[0].set_yticks(range(6),labels);axes[0].set_xlim(.48,.84);axes[1].set_xlim(.15,.55)
fig.tight_layout();fig.savefig(figdir/'model_comparison.png',dpi=240,bbox_inches='tight');plt.close(fig)
md=(root/'tmp/thesis_review/manuscript.md').read_text(encoding='utf-8').replace('{{MAIN_TABLE}}',metric_table(m3,names)).replace('{{ROUND4_TABLE}}',metric_table(m4,alt)).replace('{{REFERENCES}}','\n\n'.join(reftext)).replace('{{PERFORMANCE_FIGURE}}','![Figure 1 Selected primary models with patient bootstrap 95% intervals](figures/model_comparison.png)\n\nFigure 1. Saved round-3 point estimates and patient-bootstrap intervals. Dashed lines mark AUC 0.5 and the cohort event fraction for AP. These intervals do not incorporate the full uncertainty from repeated model selection. Source: round3_metrics.csv.')
assert '{{' not in md
(out/'thesis_draft.md').write_text(md,encoding='utf-8')
for n,f in [('TimesAcademic','times.ttf'),('TimesAcademicBold','timesbd.ttf'),('TimesAcademicItalic','timesi.ttf')]:pdfmetrics.registerFont(TTFont(n,'C:/Windows/Fonts/'+f))
pdfmetrics.registerFontFamily('TimesAcademic',normal='TimesAcademic',bold='TimesAcademicBold',italic='TimesAcademicItalic',boldItalic='TimesAcademicBold')
styles={
 'body':ParagraphStyle('body',fontName='TimesAcademic',fontSize=11.3,leading=14.7,spaceAfter=6.4,alignment=TA_JUSTIFY,allowWidows=0,allowOrphans=0),
 'h1':ParagraphStyle('h1',fontName='TimesAcademicBold',fontSize=17,leading=21,spaceAfter=15,keepWithNext=True),
 'h2':ParagraphStyle('h2',fontName='TimesAcademicBold',fontSize=12.5,leading=16,spaceBefore=11,spaceAfter=7,keepWithNext=True),
 'title':ParagraphStyle('title',fontName='TimesAcademicBold',fontSize=25,leading=30,spaceAfter=20),
 'subtitle':ParagraphStyle('subtitle',fontName='TimesAcademic',fontSize=17,leading=22,spaceAfter=26),
 'cell':ParagraphStyle('cell',fontName='TimesAcademic',fontSize=9,leading=11.5,spaceAfter=0,wordWrap='CJK'),
 'caption':ParagraphStyle('caption',fontName='TimesAcademic',fontSize=9.5,leading=12,spaceAfter=10),
 'ref':ParagraphStyle('ref',fontName='TimesAcademic',fontSize=10,leading=13,spaceAfter=9),
}
def inline(s):
 s=html.escape(s)
 s=re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)',lambda m:f'<link href="{m[2]}" color="#153c52">{m[1]}</link>',s)
 return s
class ThesisDoc(BaseDocTemplate):
 def __init__(self,path):
  super().__init__(path,pagesize=letter,leftMargin=72,rightMargin=72,topMargin=62,bottomMargin=60,title='Early prediction of echocardiographic deterioration during cancer therapy',author='Oron Barazani')
  self.addPageTemplates(PageTemplate(id='normal',frames=[Frame(72,60,468,670,id='body',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)],onPage=self.page))
 def page(self,c,d):
  if d.page>1:
   c.saveState();c.setFont('TimesAcademic',9);c.setFillColor(colors.HexColor('#555555'));c.drawString(72,757,'Oron Barazani | MSc thesis working draft');c.drawRightString(540,36,str(d.page));c.restoreState()
 def afterFlowable(self,f):
  if isinstance(f,Paragraph) and f.style.name=='h1' and f.getPlainText()!='Contents':
   title=f.getPlainText();key='h'+str(self.seq.nextf('heading'));self.canv.bookmarkPage(key);self.canv.addOutlineEntry(title,key,0);self.notify('TOCEntry',(0,title,self.page,key))
story=[];lines=md.splitlines();i=0;cover=True;refmode=False
while i<len(lines):
 line=lines[i].strip()
 if not line:i+=1;continue
 if line.startswith('# '):story+=[Spacer(1,55),Paragraph(inline(line[2:]),styles['title'])];i+=1;continue
 if line.startswith('## Layer specific') and cover:story.append(Paragraph(inline(line[3:]),styles['subtitle']));i+=1;continue
 if line=='## Abstract':
  story.append(PageBreak());story.append(Paragraph('Contents',styles['h1']));toc=TableOfContents();toc.levelStyles=[ParagraphStyle('toc',fontName='TimesAcademic',fontSize=11,leading=17,spaceBefore=5)];story.append(toc);story.append(PageBreak());cover=False
 elif line.startswith('## '):story.append(PageBreak())
 if line.startswith('## '):
  refmode=line=='## References';story.append(Paragraph(inline(line[3:]),styles['h1']));i+=1;continue
 if line.startswith('### '):story.append(Paragraph(inline(line[4:]),styles['h2']));i+=1;continue
 if line.startswith('|'):
  rows=[]
  while i<len(lines) and lines[i].strip().startswith('|'):
   cells=[x.strip() for x in lines[i].strip().strip('|').split('|')]
   if not all(re.fullmatch('[-: ]+',x) for x in cells):rows.append(cells)
   i+=1
  n=len(rows[0]);widths=([215,126,127] if n==3 else [90,378])
  dat=[[Paragraph(('<b>'+inline(x)+'</b>') if y==0 else inline(x),styles['cell']) for x in row] for y,row in enumerate(rows)]
  tbl=Table(dat,colWidths=widths,repeatRows=1,hAlign='LEFT');tbl.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8edf0')),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#617786')),('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#cbd3d8')),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]));story.extend([tbl,Spacer(1,10)]);continue
 if line.startswith('!['):
  path=out/re.search(r'\]\((.+)\)',line)[1];im=Image(str(path));im.drawHeight*=468/im.drawWidth;im.drawWidth=468;story.extend([Spacer(1,7),im,Spacer(1,7)]);i+=1;continue
 para=[line];i+=1
 while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','![')):para.append(lines[i].strip());i+=1
 text=' '.join(para);style=styles['ref'] if refmode else styles['caption'] if text.startswith(('Figure 1.','Table 1')) else styles['body']
 story.append(Paragraph(inline(text),style))
doc=ThesisDoc(str(pdfout/'thesis_draft.pdf'));doc.multiBuild(story)
manifest=[]
sources=[root/x['path'] for x in json.loads((root/'tmp/thesis_review/reports.json').read_text(encoding='utf-8'))]
sources+=list(root.glob('cardiotoxicity*.py'))+list(root.glob('vvi*.py'))+list(root.glob('*.yaml'))+list(root.glob('*.pptx'))+list(root.glob('*.docx'))
for p in root.glob('*results/*'):
 if p.suffix in ['.csv','.json'] or p.name in ['round3_oof_predictions.parquet','next_visit_transitions.parquet','oof_predictions.parquet']:sources.append(p)
sources+=[root/'amber_full_105_preprocessed/Ichilov_july_visits.parquet',root/'amber_full_105_preprocessed/Ichilov_july_run_metadata.json',Path('C:/Users/Oron/OneDrive - Technion/Attachments/Research Proposal (1).pdf'),Path('C:/Users/Oron/Documents/Echocardiography-based early prediction of cardio-toxicity - Technion, Shaare Zedek, Ichilov .pptx.pdf')]
for p in sorted(set(sources)):
 if not p.is_file() or p.name.startswith('~$'):continue
 manifest.append({'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
with (out/'source_manifest.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=['path','bytes','sha256']);w.writeheader();w.writerows(manifest)
print(json.dumps({'words':len(md.split()),'references':len(refs),'manifest_files':len(manifest),'pdf':str(pdfout/'thesis_draft.pdf')},indent=2))
