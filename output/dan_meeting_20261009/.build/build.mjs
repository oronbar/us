import fs from 'node:fs/promises';
import path from 'node:path';
import {execFileSync} from 'node:child_process';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile} from '@oai/artifact-tool';
const skill='C:/Users/Oron/.codex/plugins/cache/openai-primary-runtime/presentations/26.915.20218/skills/presentations';
const root='D:/us/output/dan_meeting_20261009';
const tmp=root+'/.build';
const {applyPresentationChartFont,finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs'));
const data=JSON.parse(await fs.readFile(tmp+'/data.json','utf8'),(k,v)=>typeof v==='number'?Number(v.toFixed(6)):v);
const p=Presentation.create({slideSize:{width:1280,height:720}});
const C={ink:'#16324B',muted:'#536574',teal:'#168A91',pale:'#A3BACB',orange:'#C5802E',grid:'#E4E9ED',white:'#FFFFFF'};
const sources={email:'D:/us/output/ichilov3_email_summary_20261004/email_draft.md',figures:'D:/us/output/ichilov3_email_summary_20261004/make_figures.py',vendor:'D:/DS/ichilov3_temporal_trial_20260927/vendor_failure_analysis/vendor_metrics.csv',points:'D:/DS/ichilov3_temporal_trial_20260927/vendor_failure_analysis/prediction_points.csv',importance:'D:/us/cardiotoxicity_feature_importance_results/noncnn_feature_importance_top.csv',report:'D:/us/cardiotoxicity_feature_importance_results/noncnn_feature_importance_report.md',ensemble:'D:/us/cardiotoxicity_top_ensemble_feature_importance_results/ensemble_feature_importance.csv'};
function text(s,t,x,y,w,h,size=28,color=C.ink,bold=false,align='right'){
 const sh=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 sh.text=/[\u0590-\u05ff]/.test(t)?t.replace(/[A-Za-z][A-Za-z0-9]*(?:[ +–-]+[A-Za-z0-9]+)*/g,v=>'\u202A'+v+'\u202C'):t; sh.text.style={typeface:'Arial',fontSize:size,color,bold,alignment:align,verticalAlignment:'top',autoFit:'none',wrap:'square',insets:{top:0,bottom:0,left:0,right:0}}; return sh;
}
function slide(title,subtitle){const s=p.slides.add();s.background.fill=C.white;text(s,title,58,35,1164,65,44,C.ink,true);if(subtitle)text(s,subtitle,58,110,1164,65,24,C.muted);return s;}
function foot(s,t,n){text(s,t,72,654,1120,46,18,C.muted);text(s,String(n),24,672,30,26,16,C.muted,false,'left');}
function notes(s,t,src=[]){s.speakerNotes.textFrame.setText(t+'\n\nמקורות:\n'+src.join('\n'));}
const tx={typeface:'Arial',fontSize:20,fill:C.muted};
function bar(s,cat,series,pos,{max=1,unit=.2,title='',horizontal=false,legend=false,format='0.000',labelSize=20}={}){
 const ch=s.charts.add('bar',{position:pos,categories:cat,series:series.map(z=>({...z,valuesFormatCode:format})),hasLegend:legend,legend:{position:'bottom',textStyle:{...tx,fontSize:20}},barOptions:{direction:horizontal?'bar':'column',grouping:'clustered',gapWidth:95},title,titleTextStyle:{...tx,fontSize:24,bold:true,fill:C.ink},dataLabels:{showValue:true,position:'outEnd',textStyle:{...tx,fontSize:labelSize,fill:C.ink}},xAxis:horizontal?{min:0,max,majorUnit:unit,numberFormatCode:format,textStyle:tx,majorGridlines:{fill:C.grid,width:1}}:{textStyle:{...tx,fontSize:16,alignment:'center'},majorGridlines:null},yAxis:horizontal?{textStyle:{...tx,fontSize:16,alignment:'center'},majorGridlines:null}:{min:0,max,majorUnit:unit,numberFormatCode:format,textStyle:tx,majorGridlines:{fill:C.grid,width:1}},chartFill:C.white,plotAreaFill:C.white,chartLine:{fill:'none',width:0}});
 applyPresentationChartFont(ch,{fontFamily:'Arial'});return ch;
}

// 1. Context and definitions without a separate cover slide.
{
 const s=slide('סרטוני אקו וחיזוי הידרדרות','עדכון מחקר לפגישה עם דן  |  אורון  |  אוקטובר 2026');
 text(s,'האם הווידאו מוסיף מידע מעבר לעקומות strain ולנתונים הקליניים?',90,196,1100,88,36,C.ink,true);
 text(s,'103 מטופלים, 400 ביקורי strain לאחר איחוד אנליזות חוזרות.\nל־398 ביקורים נמצאה התאמה לווידאו.',90,300,1100,88,28);
 text(s,'שלושת מבטי האקו האפיקליים ששימשו לניתוח',90,417,1100,42,28,C.ink,true);
 text(s,'A2C',860,476,240,45,36,C.teal,true,'center');text(s,'שני מדורים',860,527,240,42,26,C.muted,false,'center');
 text(s,'A3C',525,476,240,45,36,C.teal,true,'center');text(s,'שלושה מדורים',525,527,240,42,26,C.muted,false,'center');
 text(s,'A4C',190,476,240,45,36,C.teal,true,'center');text(s,'ארבעה מדורים',190,527,240,42,26,C.muted,false,'center');
 foot(s,'הידרדרות: הופעה ראשונה של ירידה יחסית של לפחות 15% ב־Mid-GLS לעומת הבסיס.',1);
 notes(s,'בחרתי cine אחד לכל מבט בכל ביקור, באמצעות TOMTEC bookmark כשהיה אפשר ובאמצעות סיווג ובדיקה ידנית בשאר המקרים. ביצעתי crop וקידוד. בסך הכול נבחרו 1,233 סרטונים ל־411 ביקורי DICOM, אך לחיזוי GLS משתמשים ב־398 ביקורים מותאמים עם תווית. יש להבחין בין ביקורי DICOM לביקורי strain. בכל הערכת החיזוי שמרתי על הפרדה לפי מטופלים. ההגדרה כאן היא הידרדרות הדמייתית, ולא אבחנה קלינית מלאה של קרדיוטוקסיות.',[sources.email]);
}
// 2. Sampling and actual same-visit results.
{
 const s=slide('דגימה לאורך פעימה שיפרה את חיזוי ה־GLS','חיזוי Mid-GLS באותו ביקור מתוך הווידאו  •  398 ביקורים מותאמים');
 bar(s,['EchoPrime','PanEcho'],[{name:'Original sampling',values:[1.607,2.025],fill:C.pale},{name:'Estimated heartbeat',values:[1.459,1.603],fill:C.teal}],{left:55,top:205,width:700,height:400},{max:2.5,unit:.5,title:'MAE (GLS percentage points)',legend:true});
 text(s,'הדגימה המקורית',815,220,395,45,30,C.ink,true);
 text(s,'עד ארבעה קטעים לאורך הסרטון.\nבכל קטע 16 פריימים במרווח קבוע,\nללא התאמה לדופק.',790,273,420,123,26);
 text(s,'הדגימה החדשה',815,416,395,45,30,C.teal,true);
 text(s,'16 פריימים לאורך פעימה משוערת,\nלפי הדופק וזמני הפריימים.\nממוצע הקידודים של עד ארבעה קטעים.',770,468,440,125,26);
 foot(s,'MAE נמוך יותר = שגיאה קטנה יותר. הפעימה בניסוי הזה הוערכה לפי דופק, ללא סנכרון ל־ECG.',2);
 notes(s,'חלון הוא קטע זמן בתוך אותו סרטון, ולא מבט נוסף. בדגימה המקורית לכל מקודד היה מרווח פריימים קבוע, ולכן משך החלון לא הותאם לאורך הפעימה. בדגימה החדשה קבעתי את משך החלון לפי זמן מחזור משוער והשתמשתי בזמני הפריימים המקוריים. נשמרו אותם crops, צבעים וחלוקות לפי מטופלים. אלה תוצאות מקודדים ללא התאמה נוספת. אין לערבב כאן עם ניסוי PanEcho R-wave המאוחר יותר, שהושמט מהסיכום לפי בקשת אורון. שיפור בניבוי GLS נוכחי אינו מוכיח שיפור בהידרדרות עתידית.',[sources.email,sources.figures]);
}
// 3. Actual editable scatter plus vendor comparison.
{
 const s=slide('חיזוי ה־GLS היה מדויק יותר בסרטוני GE','EchoPrime עם דגימת פעימה משוערת  •  תחזיות על מטופלים שלא השתתפו באימון');
 const ch=s.charts.add('scatter',{position:{left:55,top:187,width:625,height:435},title:'GE Healthcare: predicted vs reported',titleTextStyle:{...tx,fontSize:24,bold:true,fill:C.ink},series:[{name:'Visits',xValues:data.ge.gt,values:data.ge.pred,fill:C.teal,line:{fill:'none',width:0},marker:{symbol:'circle',size:5}},{name:'Perfect agreement',xValues:[4,28],values:[4,28],line:{fill:C.pale,width:2},marker:{symbol:'none'}}],scatterOptions:{style:'lineWithMarkers'},hasLegend:false,xAxis:{title:{text:'Reported |Mid-GLS| (%)',textStyle:tx},min:4,max:28,majorUnit:4,textStyle:tx,majorGridlines:{fill:C.grid,width:1}},yAxis:{title:{text:'Predicted |Mid-GLS| (%)',textStyle:tx},min:4,max:28,majorUnit:4,textStyle:tx,majorGridlines:{fill:C.grid,width:1}},chartFill:C.white,plotAreaFill:C.white});applyPresentationChartFont(ch,{fontFamily:'Arial'});
 text(s,'GE Healthcare',715,203,480,42,30,C.teal,true,'left');
 text(s,'r = 0.817     MAE = 1.30',715,252,480,48,32,C.ink,true,'left');
 text(s,'230 ביקורים של 79 מטופלים',715,303,480,36,24,C.muted);
 bar(s,['GE Healthcare','GE Vingmed','Philips'],[{name:'Pearson r',values:[.81656382,.77721314,.58950082],fill:C.teal}],{left:705,top:365,width:510,height:244},{max:1,unit:.25,title:'Pearson r by vendor',labelSize:19});
 foot(s,'המודל אומן על כלל היצרנים. הפילוח לפי יצרן הוא תיאורי, ואינו מוכיח שהיצרן גורם לפער.',3);
 notes(s,'גרף הפיזור משתמש ב־230 זוגות תחזיות אמיתיים מתוך prediction_points.csv, ללא הסרת ערכי קיצון. זוהי תת־קבוצת GE Healthcare, ולא GE Vingmed. המודל המקורי אומן על כל היצרנים, בהפרדה לפי מטופלים. GE Vingmed: 68 ביקורים, r=0.777, MAE=1.649. Philips: 100 ביקורים, r=0.590, MAE=1.696. מספרי המטופלים בכל יצרן אינם ניתנים לחיבור, מפני שלמטופל יכולים להיות ביקורים ביותר מיצרן אחד. שונות באוכלוסייה, איכות צילום ודגם מכשיר עשויות להסביר חלק מהפער. זהות היצרן כקלט נוסף לא שיפרה משמעותית את הדיוק.',[sources.vendor,sources.points]);
}
// 4. Frozen encoders, identical cohort and score fusion.
{
 const s=slide('מקודדי וידאו מוכנים לא הוסיפו שיפור ברור','הידרדרות בביקור הבא  •  ללא Philips: ‏182 מעברים, 86 מטופלים, 38 אירועים');
 const cats=['Baseline','Echo\nPrime','Pan\nEcho','Echo\nFM','Echo\nJEPA'];
 const colors=[C.ink,C.teal,C.teal,C.teal,C.teal];
 bar(s,cats,[{name:'AUROC',values:[.726,.701,.722,.713,.720],fill:C.teal,points:colors.map((fill,idx)=>({idx,fill}))}],{left:55,top:209,width:573,height:363},{max:1,unit:.2,title:'AUROC'});
 bar(s,cats,[{name:'AP',values:[.379,.347,.378,.387,.368],fill:C.teal,points:colors.map((fill,idx)=>({idx,fill}))}],{left:650,top:209,width:573,height:363},{max:.5,unit:.1,title:'Average precision (AP)'});
 text(s,'כל עמודת וידאו מציגה שילוב של 25% ציון וידאו עם 75% ממודל הייחוס.',75,592,1130,43,25,C.ink);
 foot(s,'מודל הייחוס: strain ונתונים קליניים (CNN + MOMENT). ‏AP מסכם את איכות דירוג מקרי ההידרדרות.',4);
 notes(s,'ההשוואה על אותה תת־קבוצה ואותן חלוקות: 3 חזרות של 5 קפלים לפי מטופלים. כל המקודדים בשקף הזה ללא אימון נוסף. המספרים מייצגים ציונים משולבים ולא וידאו בלבד. בקוהורט המלא הנפרד היו 238 מעברים ו־49 אירועים: מודל הייחוס AUROC=0.698 ו־AP=0.353, ושילוב EchoPrime המקורי 0.694 ו־0.330. אין להשוות ישירות בין הקוהורטים. הערכים המוצגים הם אומדני נקודה של ניסויים חוזרים על אותה אוכלוסייה, ולא הוכחת עליונות. בדיקה חיצונית עדיין דרושה.',[sources.email,sources.figures]);
}
// 5. Adaptation and selection discipline.
{
 const s=slide('התאמת EchoPrime ובחירת מבטים','אותה תת־קבוצה ללא Philips ואותה הפרדה לפי מטופלים');
 const cats=['Baseline','Shared encoder','View-specific','A3C+A4C post hoc','Nested selection'];
 bar(s,cats,[{name:'AUROC',values:[.726,.710,.727,.736,.707],fill:C.teal},{name:'AP',values:[.379,.360,.355,.355,.345],fill:C.pale}],{left:60,top:198,width:1160,height:365},{max:1,unit:.2,title:'Next-visit deterioration',legend:true});
 text(s,'אימנתי מקודד משותף ומקודדים נפרדים למבטים, באימון self-supervised.',80,571,1120,39,25);
 text(s,'היתרון שנראה בבחירת שני מבטים בדיעבד נעלם בבחירה בתוך נתוני האימון.',80,614,1120,38,25,C.orange,true);
 foot(s,'בשתי גרסאות האימון נכללה גם היסטוריית וידאו כשזמינה. הצירוף שנבחר בדיעבד הוא ממצא חקרני.',5);
 notes(s,'סרטוני GE מהדאטה הנוכחי ומהדאטה הישן שימשו לאימון המקודד. הדאטה הישן לא שימש להערכת החיזוי. Shared encoder הוא מקודד משותף ל־A2C/A3C/A4C. View-specific הוא מקודד נפרד לכל מבט. שתי התוצאות כוללות היסטוריית וידאו בהתאם לניסוי המקורי. בצירוף A3C+A4C נבחרה התוצאה לאחר השוואת כל הצירופים, ולכן היא חשופה להטיית בחירה. בניסוי nested בחרתי את המבטים רק בתוך נתוני האימון ובדקתי על מטופלים מוחזקים בחוץ. גם שינוי משקל הווידאו בתוך האימון לא נתן שיפור ברור. אין לפרש ירידה זו כהשוואה סיבתית נקייה בין צירופי המבטים, משום שמדובר גם בפרוצדורת בחירה שונה.',[sources.email]);
}
// 6. Feature importance, not a conflation with the 101-patient landmark study.
{
 const s=slide('ההפרש בין שכבות ה־strain בלט בבדיקת הפיצ׳רים','בדיקה נפרדת של מודלים עם פיצ׳רים מחושבים  •  103 מטופלים, 238 מעברים, 49 אירועים');
 const labels=['Change in Endo–Mid gap pattern','Normalized Endo–Mid shape gap','Mid peak-strain dispersion'];
 const series=[];
 data.importance.forEach((r,i)=>{const y=3-i;series.push({name:labels[i]+' 95% CI',xValues:[r.auc_drop_boot_low,r.auc_drop_boot_high],values:[y,y],line:{fill:i===2?C.pale:C.teal,width:4},marker:{symbol:'none'}});series.push({name:labels[i],xValues:[r.auc_drop_mean],values:[y],fill:C.ink,line:{fill:'none',width:0},marker:{symbol:'circle',size:9}});});
 series.push({name:'Zero',xValues:[0,0],values:[.4,3.6],line:{fill:C.pale,width:1},marker:{symbol:'none'}});
 const ch=s.charts.add('scatter',{position:{left:55,top:220,width:730,height:350},series,scatterOptions:{style:'lineWithMarkers'},hasLegend:false,title:'AUROC decrease after feature shuffling',titleTextStyle:{...tx,fontSize:24,bold:true,fill:C.ink},xAxis:{min:-.02,max:.08,majorUnit:.02,numberFormatCode:'0.00',textStyle:tx,majorGridlines:{fill:C.grid,width:1}},yAxis:{min:.4,max:3.6,visible:false,tickLabelPosition:'none',line:{fill:'none',width:0},majorGridlines:null},chartFill:C.white});applyPresentationChartFont(ch,{fontFamily:'Arial'});
 text(s,'שינוי בדפוס ההפרש בין השכבות',795,275,420,38,26,C.ink,true);text(s,'מודל קליני + הפרש שכבות\nΔAUROC = 0.037',800,312,410,67,22,C.muted);
 text(s,'הבדל בצורת העקומות המנורמלות',795,360,420,38,26,C.ink,true);text(s,'Extra Trees\nΔAUROC = 0.017',800,397,410,64,22,C.muted);
 text(s,'פיזור peak strain בין סגמנטים',795,445,420,38,26,C.ink,true);text(s,'מודל קליני + שונות\nΔAUROC = 0.010',800,482,410,64,22,C.muted);
 text(s,'הקווים מציגים רווח סמך 95%. תרומת פיזור הסגמנטים פחות יציבה וכוללת אפס.',75,603,1140,40,24,C.ink);
 foot(s,'זהו מדד להסתמכות של כל מודל על פיצ׳ר, ולא הוכחה לסמן סיבתי או לשיפור של המודל הכולל.',6);
 notes(s,'זהו ניסוי next-visit נפרד של non-CNN, ולא ניסוי 101 המטופלים שבו ניבאנו את הביקור האחרון מתוך שני הראשונים. בדיקת חשיבות נעשתה על מטופלים מוחזקים בחוץ באמצעות ערבוב פיצ׳ר, עם patient-cluster bootstrap. הפיצ׳ר הראשון: בכל סגמנט מחשבים Endo פחות Mid לאורך המחזור, מחשבים שונות בין סגמנטים בכל זמן, מסכמים את דפוס הזמן באמצעות DCT04, ומחסרים את ערך הביקור הקודם. זה אינו פשוט גידול בשונות. חשיבות AUROC: 0.036713, CI 0.001082–0.065587. השני הוא הפרש צורת עקומות מנורמלות, חשיבות 0.016521, CI 0.008079–0.025740. השלישי פיזור רובסטי של שיאי Mid, חשיבות 0.010366, CI -0.014271–0.036233. בגרף כל שורה נמדדת במודל אחר, ולכן אין לראות בו תחרות ישירה בין פיצ׳רים באותו מודל. בניסוי נפרד של CNN+MOMENT+Catch22 בלט segment-max של MOMENT: ירידת AUROC=0.042, CI 0.002–0.082, וירידת AP=0.057. גם כאן מדובר בהסתמכות המודל על קבוצת embedding ולא בסמן פיזיולוגי מזוהה. אין להסיק חשיבות ביולוגית של סגמנט יחיד.',[sources.importance,sources.report,sources.ensemble]);
}
// 7. Next meeting decision.
{
 const s=slide('איך ממשיכים בניתוח הדאטה','הווידאו משקף את ה־GLS הנוכחי. התוספת לחיזוי ההידרדרות עדיין מוגבלת.');
 text(s,'ניתוח העקומות',80,218,1135,48,34,C.teal,true);
 text(s,'סיווג עקומות תקינות ולא תקינות עדיין לא ביצעתי.\nאפשר לבדוק את השפעת איכות העקומות ואת השונות בין סגמנטים בכל מבט.',80,277,1135,89,28);
 text(s,'מידע ישיר על התנועה',80,394,1135,48,34,C.teal,true);
 text(s,'לזהות סוף דיאסטולה וסוף סיסטולה ולמדוד את דפוס הכיווץ לאורך הפעימה.\nתחילה פיילוט קטן לבדיקת איכות המדידות, ואחריו בדיקת התוספת לחיזוי.',80,453,1135,92,28);
 text(s,'התאמת מקודדים נוספים היא אפשרות נוספת, לפי סדר העדיפויות שנקבע בפגישה.',80,578,1135,68,26,C.muted);
 foot(s,'אבחר את הניסויים בתוך נתוני האימון. תוצאה חיובית תצטרך אישור על מטופלים חדשים.',7);
 notes(s,'מטרת הדיון היא לבחור ניסוי המשך אחד ממוקד. במקרה של סיווג תקינות עקומות צריך להגדיר קודם מהי תקינות ומי נותן תווית, ולהימנע מהוצאת דוגמאות על בסיס שגיאת חיזוי התוצאה. תזמון R-wave תוחם את הפעימה אך אינו מזהה לבדו סוף סיסטולה. התאמת מקודדי וידאו אחרים, למשל PanEcho או EchoFM, יכולה להיות ניסוי נפרד ולא הנחה ששינוי המודל לבדו יפתור את הבעיה. הניסויים המאוחרים שהוצאו מהמייל אינם מוצגים כאן. חשוב להבחין בין בדיקת חשיבות פיצ׳רים שבוצעה לבין סיווג תקין/לא תקין שטרם בוצע.',[sources.email,sources.report]);
}

await fs.mkdir(tmp+'/previews',{recursive:true});
const draft=tmp+'/candidate.pptx';
await (await PresentationFile.exportPptx(p)).save(draft);
for(let i=0;i<p.slides.items.length;i++){
 const s=p.slides.items[i];
 const png=await p.export({slide:s,format:'png',scale:1});
 await fs.writeFile(tmp+`/previews/slide-${i+1}.png`,new Uint8Array(await png.arrayBuffer()));
 console.log('Rendered',i+1);
}
execFileSync('C:/Users/Oron/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',[tmp+'/fix_rtl.py',draft,tmp+'/candidate_rtl.pptx']);
const result=await finalizePresentation({workspaceDir:root,candidatePath:tmp+'/candidate_rtl.pptx',finalPath:root+'/deliverables/meeting_with_dan.pptx',pythonExecutable:'C:/Users/Oron/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],explicitTotalSlideCount:7,requiredNativeChartOwnerSlides:[2,3,4,5,6],materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:['Arial']},verifyArtifactToolImport:true,receiptPath:tmp+'/validation_meeting.json'});
console.log(JSON.stringify(result));
