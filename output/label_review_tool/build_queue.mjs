import fs from 'node:fs/promises';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const records=JSON.parse(await fs.readFile('D:/us/output/label_review_tool/queue.json','utf8'));
if(records.length!==60) throw new Error('Expected 60 review visits');
const source=['visit_id','patient_id','visit_date','study_uid','a2c_dicom','a3c_dicom','a4c_dicom',
  'reread_mid_gls','reread_endo_gls','reader_id','same_three_cines_confirmed','notes'];
const headers=['Visit ID','Patient ID','Visit date','Study UID','A2C DICOM','A3C DICOM','A4C DICOM',
  'Reread Mid GLS signed (%)','Reread Endo GLS signed (%)','Reader ID','Same 3 cines confirmed?','Notes'];
const matrix=[headers,...records.map(r=>source.map(k=>r[k]??''))];
const wb=Workbook.create();
const sheet=wb.worksheets.add('Blinded reread');
sheet.getRange('A1:L61').values=matrix;
sheet.getRange('A1:L61').format.font={name:'Arial',size:10,color:'#182230'};
sheet.getRange('A1:L1').format={fill:'#17324D',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},rowHeight:30};
sheet.getRange('H2:L61').format.fill='#FFF3C4';
sheet.getRange('H2:I61').format.numberFormat='0.00';
sheet.getRange('A:A').format.columnWidth=22;
sheet.getRange('B:B').format.columnWidth=18;
sheet.getRange('C:C').format.columnWidth=16;
sheet.getRange('D:D').format.columnWidth=36;
sheet.getRange('E:G').format.columnWidth=46;
sheet.getRange('H:I').format.columnWidth=24;
sheet.getRange('J:K').format.columnWidth=26;
sheet.getRange('L:L').format.columnWidth=42;
sheet.getRange('A1:L61').format.rowHeight=24;
sheet.getRange('A1:L1').format.rowHeight=32;
sheet.freezePanes.freezeRows(1);
sheet.freezePanes.freezeColumns(2);
sheet.showGridLines=false;
wb.recalculate();
const check=await wb.inspect({kind:'table',range:'Blinded reread!A1:L5',include:'values,formulas',tableMaxRows:5,tableMaxCols:12,maxChars:2800});
console.log(check.ndjson);
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:100},maxChars:500});
console.log(errors.ndjson);
const preview=await wb.render({sheetName:'Blinded reread',range:'A1:D9',scale:1.5,format:'png'});
await fs.writeFile('D:/us/output/label_review_tool/preview.png',new Uint8Array(await preview.arrayBuffer()));
const result=await SpreadsheetFile.exportXlsx(wb);
await result.save('D:/DS/ichilov3_temporal_trial_20260927/label_review/blinded_gls_reread_queue.xlsx');
