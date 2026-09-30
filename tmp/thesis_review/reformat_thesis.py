from pathlib import Path
import re, html, json
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.platypus import BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, PageBreak, Table, TableStyle, Image, KeepTogether
from reportlab.platypus.tableofcontents import TableOfContents

root=Path('D:/us'); out=root/'output/thesis'; pdfout=root/'output/pdf'
source=(out/'thesis_draft.md').read_text(encoding='utf-8')
parts=re.split(r'^## (.+)\n',source,flags=re.M)
sec={parts[i]:parts[i+1].strip() for i in range(1,len(parts),2)}
replacements={
 'The Israel Innovation Authority presentation specifically emphasizes differences between myocardial layers and between successive examinations. It also describes earlier work on view classification, segmentation, physiological curve assessment and aortic stenosis. These provide technical background. Their reported accuracies belong to different populations and tasks and are not evidence that future cardiotoxicity has already been predicted at comparable accuracy.': 'This research is funded by the Israel Innovation Authority and focuses on differences between myocardial layers and between successive examinations. Earlier laboratory work on view classification, segmentation, physiological curve assessment and aortic stenosis provides the technical foundation. Performance on those earlier tasks does not establish accuracy for predicting future deterioration during cancer therapy.',
 'The supplied proposal and grant presentation were used to identify seed references.': 'The review builds on the clinical and methodological literature underlying the research programme.',
 ' The reference audit accompanying this draft records incomplete citations from the presentation rather than silently replacing them with guessed papers.': '',
 'The paper was published online in December 2015 and appears in the 2016 journal issue, resolving the date ambiguity in the grant slides. ': '',
 "The grant's physiological curve examples illustrate a useful distinction between normal, pathological and artifactual behavior.": 'Physiological quality assessment distinguishes normal, pathological and artifactual curve behavior.',
 'The principal investigation is a retrospective longitudinal prediction study. This draft reconstructs its methods and findings from saved research outputs. The two supplied documents establish the scientific plan. The us workspace provides preprocessing code, configurations, reports, aggregate tables, saved out-of-fold predictions and presentations. Project emails provide context about transfers, labeling and prospective validation readiness. They are not treated as patient-level clinical outcome evidence.': 'The principal investigation is a retrospective longitudinal prediction study. Study materials comprise structured echocardiographic exports, preprocessing code, model configurations, aggregate reports and saved out-of-fold predictions. Project records document data transfers, labeling and readiness for independent validation; clinical outcomes require linkage to the corresponding patient-level records.',
 'The grant target of greater than 85% accuracy and specificity': 'The research programme target of greater than 85% accuracy and specificity',
 'The prior aortic-stenosis and lung-ultrasound examples in the grant presentation are contextual achievements. Their sample sizes and accuracies are deliberately excluded from the oncology results tables.': 'Prior laboratory work in aortic stenosis and lung ultrasound provides methodological background. Its populations and performance estimates are separate from the oncology cohort analyzed here.',
 'This draft treats the next-visit experiment': 'This thesis treats the next-visit experiment',
 'The original proposal envisaged': 'The broader research programme envisages',
 'This draft does not assert a final approval number, waiver or approval date that was not verified. Those details must be inserted from the approved protocol and institutional records before the thesis is submitted.': 'The final approval number, waiver status and approval date require verification against the approved protocol and institutional records before submission.',
 'Finally, this manuscript was prepared from saved analyses. The primary metrics and cohort counts were checked, but every historical training run was not reproduced.': 'The analysis reported here uses saved research outputs. The primary metrics and cohort counts were checked, but not every historical training run was reproduced.',
}
for k,v in list(sec.items()):
 for a,b in replacements.items(): v=v.replace(a,b)
 sec[k]=v

objectives=sec['3 Research objectives'].split('\n\n')
hypothesis=objectives.pop()
goals='\n\n'.join(objectives)
goals+='\n\nThe funded research programme also calls for physiological quality assessment, identification of the curve parameters contributing to a prediction, repeatable conclusions and quantification of predictive uncertainty. Its planned validation and test milestones specify accuracy and specificity greater than 85%. These are development objectives for the broader system, not achieved results or substitutes for the AUC and AP measures used in the present study. The original programme considers three myocardial layers; the completed primary analysis uses matched Endo and Mid measurements.'
method=sec['4 Materials and methods']
method=re.sub(r'### 4\.(\d+)',lambda m:'### 5.'+str(int(m[1])+1),method)
method='### 5.1 Integrated System Pipeline Overview\n\nThe analysis pipeline links vendor exports to distinct examinations, organizes examinations into patient trajectories, constructs a first-crossing next-visit target, and transforms the available curves into scalar features or model inputs. Models are fitted and evaluated using patient-grouped partitions, after which saved out-of-fold scores support performance comparisons, uncertainty estimates and exploratory interpretation. Clinical outcome linkage and independent hospital evaluation form the subsequent validation stage.\n\n'+method
discussion=re.sub(r'### 6\.', '### 7.',sec['6 Discussion'])
discussion=discussion.replace('### 7.6 External validation and the multicentre plan','### 7.6 Future Work and External Validation')
discussion+='\n\n### 7.8 Comparison with Prior Work\n\nThe layer-specific studies of Chang and colleagues [6] and Kim and colleagues [7] provide biological motivation for retaining layer-resolved measurements, while Demissei and colleagues [8] support examination of regional strain beyond global summaries. The present investigation extends this motivation to repeated current-to-next-visit forecasting using full curve representations. However, its strain-derived endpoint, limited clinical baseline and internal patient-grouped evaluation differ from those studies. The observed performance therefore supports an engineering feasibility result and cannot be ranked directly against published clinical cardiotoxicity models.\n\n### 7.9 Future Work Priorities\n\nThe immediate priorities are to complete treatment and outcome linkage, verify pre-treatment reference examinations, harmonize the external hospital data and freeze the endpoint and modeling pipeline. All feature screening and learned ensemble construction should be confined to development partitions. Subsequent evaluation should assess measurement repeatability, a comprehensive clinical comparator, calibration and prespecified operating thresholds on untouched patients. Candidate pre-event layer patterns should be tested as prespecified hypotheses rather than selected again on the validation cohort.'
chapters=[('1. Introduction',sec['1 Introduction']),('2. Background',sec['2 Literature review']),('3. Hypothesis',hypothesis),('4. Research Goals',goals),('5. Methodology',method),('6. Results',re.sub(r'### 5\.','### 6.',sec['5 Results'])),('7. Discussion',discussion),('8. Conclusion',sec['7 Conclusions'])]

# Preserve source bibliography details while matching the reference's author-year convention.
refrows=sec['References'].split('\n\n')
authors=['Suter and Ewer','Lyon et al.','Lang et al.','Herrmann et al.','Negishi et al.','Chang et al.','Kim et al.','Demissei et al.','Yahav and Adam','Farsalinos et al.','Khamis et al.','Piñeiro-Lamas et al.','Ouyang et al.','Kalliatakis et al.','Goswami et al.','Feofanov et al.','Chen et al.','Lubba et al.','Guillaume et al.','Saito and Rehmsmeier','Varoquaux','Riley et al.','Collins et al.','Moons et al.']
years=[2013,2022,2015,2022,2023,2020,2022,2021,2024,2015,2016,2023,2020,2026,2024,2026,2024,2019,2021,2015,2018,2020,2024,2025]
def cite(s):
 return re.sub(r'\[(\d+(?:,\s*\d+)*)\]',lambda m:'('+ '; '.join(authors[int(x)-1]+', '+str(years[int(x)-1]) for x in m[1].split(','))+')',s)
bib=[]
for i,row in enumerate(refrows):
 row=re.sub(r'^\[\d+\] ','',row)
 # Move publication year to the author position; retain online/version qualifiers.
 row=row.replace('. '+str(years[i])+'. ','. ',1)
 a,t=row.split('. ',1) if 'et al.' not in row[:60] else (row[:row.index('et al.')+6],row[row.index('et al.')+7:])
 bib.append((authors[i],a+' ('+str(years[i])+'). '+t))
bibliography='\n\n'.join(x[1] for x in sorted(bib))
title='Early Prediction of Echocardiographic Deterioration During Cancer Therapy Using Layer Specific Strain Curves and Machine Learning'
ack='This research was conducted in the Faculty of Biomedical Engineering at the Technion - Israel Institute of Technology under the supervision of Professor Dan Adam.\n\nThe financial support of the Israel Innovation Authority is gratefully acknowledged.'
front=[('Acknowledgements',ack),('Contents','{{CONTENTS}}'),('Abstract',sec['Abstract']),('Abbreviations and Notations',sec['Abbreviations'])]
allsec=front+chapters+[('References',bibliography),('Appendix A Relationship to the Original Proposal',sec['Appendix A Relationship to the original proposal']),('Appendix B Reproducibility Map',sec['Appendix B Reproducibility map'])]
md='# '+title+'\n\nResearch Thesis\n\nSubmitted in Partial Fulfilment of the Requirements for the Degree of Master of Science in Biomedical Engineering\n\nOron Barazani\n\nSubmitted to the Senate of the Technion - Israel Institute of Technology\n\nSeptember 2026, Haifa\n\n'
for name,content in allsec:
 if name=='Contents':
  content='\n'.join('- '+n for n,_ in chapters)+'\n- References\n- Appendix A Relationship to the Original Proposal\n- Appendix B Reproducibility Map'
 md+='## '+name+'\n\n'+(content if name=='References' else cite(content))+'\n\n'
assert not re.search(r'\b(presentation|slides)\b',md,re.I)
assert len(refrows)==24
(out/'thesis_reformatted.md').write_text(md,encoding='utf-8')
notes='# Revision notes\n\nReformatted on 9 September 2026 using the colleague thesis PDF as the structural and visual reference. Original thesis_draft.md and thesis_draft.pdf are preserved. Results, uncertainty estimates and scientific limitations are retained.\n\nChanges: separate Hypothesis and Research Goals; Literature review renamed Background; Materials and methods renamed Methodology with pipeline overview; numbered chapters through Conclusion; acknowledgement of Israel Innovation Authority funding; removal of direct references to the funding presentation; author-year citations; Calibri and blue heading styles; reference-style cover and front matter. Appendices A and B are retained after References. The integrity declaration in the colleague thesis was not copied as a personal attestation. The author should review and approve the required institutional declaration before final submission.\n\nThe supplied reference is an unfinished thesis and does not establish official institutional format requirements. The existing working title and the September 2026 cover date require author confirmation for final submission. No new training, clinical outcome adjudication or literature update was performed in this formatting revision.\n\n'+sec['Appendix C Items required before submission']
(out/'thesis_revision_notes.md').write_text(notes,encoding='utf-8')

blue=colors.HexColor('#4F81BD')
for name,file in [('Calibri','calibri.ttf'),('CalibriB','calibrib.ttf'),('CalibriI','calibrii.ttf')]: pdfmetrics.registerFont(TTFont(name,'C:/Windows/Fonts/'+file))
pdfmetrics.registerFontFamily('Calibri',normal='Calibri',bold='CalibriB',italic='CalibriI',boldItalic='CalibriB')
styles={
 'body':ParagraphStyle('body',fontName='Calibri',fontSize=12,leading=15,spaceAfter=12,alignment=TA_JUSTIFY,allowWidows=0,allowOrphans=0),
 'h1':ParagraphStyle('h1',fontName='CalibriB',fontSize=18,leading=22,textColor=blue,spaceBefore=16,spaceAfter=14,keepWithNext=True),
 'h2':ParagraphStyle('h2',fontName='CalibriB',fontSize=16,leading=20,textColor=blue,spaceBefore=12,spaceAfter=14,keepWithNext=True),
 'title':ParagraphStyle('title',fontName='CalibriB',fontSize=22,leading=27,textColor=blue,alignment=TA_CENTER,spaceAfter=30),
 'cover':ParagraphStyle('cover',fontName='Calibri',fontSize=14,leading=18,alignment=TA_CENTER,spaceAfter=25),
 'author':ParagraphStyle('author',fontName='CalibriB',fontSize=20,leading=24,textColor=blue,alignment=TA_CENTER,spaceBefore=14,spaceAfter=30),
 'cell':ParagraphStyle('cell',fontName='Calibri',fontSize=10,leading=12,wordWrap='CJK'),
 'caption':ParagraphStyle('caption',fontName='Calibri',fontSize=10,leading=13,spaceAfter=12),
 'ref':ParagraphStyle('ref',fontName='Calibri',fontSize=11,leading=14,spaceAfter=10),
}
def inline(s):
 s=html.escape(s)
 return re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)',lambda m:f'<link href="{m[2]}" color="#4F81BD">{m[1]}</link>',s)
class Thesis(BaseDocTemplate):
 def __init__(self,path):
  super().__init__(path,pagesize=(612,792),leftMargin=72,rightMargin=72,topMargin=54,bottomMargin=54,title=title,author='Oron Barazani')
  self.addPageTemplates(PageTemplate(id='main',frames=[Frame(72,54,468,684,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)],onPage=self.page_draw))
 def page_draw(self,c,d):
  if d.page>1:
   c.saveState();c.setFont('Calibri',9);c.setFillColor(colors.HexColor('#555555'));c.drawCentredString(306,30,str(d.page));c.restoreState()
 def afterFlowable(self,f):
  if isinstance(f,Paragraph) and f.style.name in ['h1','h2']:
   s=f.getPlainText();level=0 if f.style.name=='h1' else 1
   if s in ['Acknowledgements','Contents','Abstract','Abbreviations and Notations']: return
   key='heading'+str(self.seq.nextf('h')); self.canv.bookmarkPage(key);self.canv.addOutlineEntry(s,key,level)
   self.notify('TOCEntry',(level,s,self.page,key))
story=[Spacer(1,44),Paragraph(title,styles['title']),Paragraph('Research Thesis',styles['cover']),Paragraph('Submitted in Partial Fulfilment of the Requirements for the Degree<br/>of Master of Science in Biomedical Engineering',styles['cover']),Paragraph('Oron Barazani',styles['author']),Paragraph('Submitted to the Senate of the Technion - Israel Institute of Technology<br/>September 2026, Haifa',styles['cover'])]
tablecount=0
for name,content in allsec:
 if name in ['Acknowledgements','Contents','Abstract','Abbreviations and Notations','1. Introduction','References'] or name.startswith('Appendix'): story.append(PageBreak())
 story.append(Paragraph(inline(name),styles['h1']))
 if name=='Contents':
  toc=TableOfContents();toc.levelStyles=[ParagraphStyle('toc0',fontName='CalibriB',fontSize=11,leading=14,spaceBefore=5,textColor=blue),ParagraphStyle('toc1',fontName='Calibri',fontSize=10.5,leading=13,leftIndent=14,firstLineIndent=0,spaceBefore=2)]
  toc.tableStyle=TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),0),('RIGHTPADDING',(0,0),(-1,-1),0),('TOPPADDING',(0,0),(-1,-1),0),('BOTTOMPADDING',(0,0),(-1,-1),2)])
  story.append(toc);continue
 if name!='References':content=cite(content)
 lines=content.splitlines();i=0
 while i<len(lines):
  line=lines[i].strip()
  if not line:i+=1;continue
  if line.startswith('### '): story.append(Paragraph(inline(line[4:]),styles['h2']));i+=1;continue
  if line.startswith('|'):
   rows=[]
   while i<len(lines) and lines[i].strip().startswith('|'):
    cells=[x.strip() for x in lines[i].strip().strip('|').split('|')]
    if not all(re.fullmatch('[-: ]+',x) for x in cells):rows.append(cells)
    i+=1
   n=len(rows[0]); widths=[200,134,134] if n==3 else ([92,376] if name.startswith('Abbreviations') else [145,323])
   if name.startswith('Appendix A'): widths=[132,178,158]
   data=[[Paragraph('<b>'+inline(x)+'</b>' if r==0 else inline(x),styles['cell']) for x in row] for r,row in enumerate(rows)]
   table=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT');table.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E9EFF7')),('LINEBELOW',(0,0),(-1,0),.6,blue),('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#CCCCCC')),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
   story.extend([table,Spacer(1,12)]);tablecount+=1;continue
  if line.startswith('!['):
   im=Image(str(out/re.search(r'\]\((.+)\)',line)[1]));im.drawHeight*=468/im.drawWidth;im.drawWidth=468
   i+=1
   while i<len(lines) and not lines[i].strip():i+=1
   group=[im,Spacer(1,7)]
   if i<len(lines) and lines[i].startswith('Figure '):group.append(Paragraph(inline(lines[i]),styles['caption']));i+=1
   story.append(KeepTogether(group));continue
  para=[line];i+=1
  while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','![')):para.append(lines[i].strip());i+=1
  text=' '.join(para);style=styles['ref'] if name=='References' else styles['caption'] if text.startswith(('Figure 1.','Table 1','Table 2.')) else styles['body']
  story.append(Paragraph(inline(text),style))
dest=pdfout/'thesis_reformatted.pdf';Thesis(str(dest)).multiBuild(story)
print(json.dumps({'pdf':str(dest),'words':len(md.split()),'tables':tablecount,'references':len(bib)}))
