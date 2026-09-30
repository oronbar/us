const $ = id => document.getElementById(id);
let all=[], queue=[], cursor=0, current=null, selectedStatus=null, activeView='A2C', mediaGeneration=0;
let selectedCandidate=null, alternativeGeneration=0;
const reasons={wrong_view:'Wrong view',crop_or_anatomy:'Crop / missing anatomy',timing_or_motion:'Timing / motion',image_quality:'Image quality',report_or_label:'Report / GLS label',wrong_study:'Wrong study',other:'Other'};
const params=new URLSearchParams(location.search);

function score(v){return $('target').value==='endo'?v.endo_absolute_error:v.absolute_error}
function rebuild(keepId){
  const vendor=$('vendor').value,filter=$('reviewFilter').value,term=$('search').value.trim().toLowerCase();
  queue=all.filter(v=>(vendor==='all'||v.vendor===vendor)&&(!term||v.visit_id.toLowerCase().includes(term)||v.patient_id.toLowerCase().includes(term))&&
    (filter==='all'||(filter==='unreviewed'?!v.review:v.review?.status===filter)));
  queue.sort((a,b)=>score(b)-score(a)||a.visit_id.localeCompare(b.visit_id));
  const found=queue.findIndex(v=>v.visit_id===keepId);
  cursor=found>=0?found:Math.min(cursor,Math.max(queue.length-1,0));
  render();
}
function setStatus(status){
  selectedStatus=status;
  document.querySelectorAll('.statuses button').forEach(b=>b.classList.toggle('selected',b.dataset.status===status));
  $('reasons').hidden=status==='no_issue';
  if(status==='no_issue')document.querySelectorAll('#reasons input').forEach(i=>i.checked=false);
  $('saved').textContent='Unsaved changes';
  updateReplacementPanel();
}
function updateReplacementPanel(){
  const replacements=current?.replacements||{};
  $('replacementPanel').hidden=selectedStatus!=='suspected_issue'&&!Object.keys(replacements).length;
  const view=$('replacementView').value;
  $('revertReplacement').hidden=!replacements[view];
  if(replacements[view]&&!selectedCandidate)$('replacementStatus').textContent=`Approved replacement for ${view}: ${replacements[view].source_name}. Prediction above still uses the original cine.`;
}
function makeClip(clip,generation){
  const replacement=current.replacements?.[clip.view];
  const section=document.createElement('section');section.className='clip'+(clip.view===activeView?' active':'');section.dataset.view=clip.view;
  const head=document.createElement('div');head.className='cliphead';
  const title=document.createElement('h3');title.textContent=clip.view+(replacement?' · replacement':'');
  const mode=document.createElement('button');mode.className='mode';mode.textContent=replacement?'Original selected crop':'Original frame';mode.dataset.mode=replacement?'alternative':'crop';
  head.append(title,mode);
  const video=document.createElement('video');video.controls=true;video.loop=true;video.muted=true;video.playsInline=true;video.preload='metadata';
  const status=document.createElement('span');status.className='mediaStatus';status.textContent='Preparing full cine…';
  const meta=document.createElement('div');meta.className='clipmeta';
  meta.textContent=replacement?`Approved: ${replacement.source_name} · original: ${clip.source_name}`:
    `${clip.frames} frames · ${clip.source_name} · ${clip.selection_source}${clip.crop_flagged?' · prior crop flag':''}`;
  section.append(head,video,status,meta);
  mode.onclick=()=>{
    const old=mode.dataset.mode;
    const next=old==='alternative'?'crop':old==='crop'?'source':replacement?'alternative':'crop';
    mode.dataset.mode=next;
    mode.textContent=next==='crop'?'Original frame':next==='source'?(replacement?'Approved replacement':'Selected crop'):'Original selected crop';
    video.removeAttribute('src');video.load();loadMedia(clip,next,video,status,generation);
  };
  return section;
}
async function loadMedia(clip,mode,video,status,generation){
  status.textContent=mode==='alternative'?'Preparing approved replacement…':mode==='crop'?'Preparing selected crop cine…':'Decoding original DICOM…';
  for(let attempt=0;attempt<100;attempt++){
    if(generation!==mediaGeneration||!document.body.contains(video))return;
    try{
      const replacement=current.replacements?.[clip.view];
      const url=mode==='alternative'&&replacement?
        `/api/alternative_media/${encodeURIComponent(current.visit_id)}/${clip.view}/${replacement.candidate_id}`:
        `/api/media/${encodeURIComponent(clip.file_id)}/${mode}`;
      const response=await fetch(url);
      const result=await response.json();
      if(result.status==='ready'){video.src=result.url;status.textContent=mode==='alternative'?'Approved replacement · full cine':mode==='crop'?'Selected crop · full cine':'Original DICOM frame · full cine';return}
      if(result.status==='error'){status.textContent='Video error: '+result.error;return}
    }catch(err){status.textContent='Connection error; retrying…'}
    await new Promise(resolve=>setTimeout(resolve,650));
  }
  status.textContent='Video preparation timed out. Switch view to retry.';
}
function render(){
  const phil=all.filter(v=>v.vendor==='Philips'),done=phil.filter(v=>v.review).length;
  $('progress').textContent=`${done} / ${phil.length} Philips visits reviewed · ${all.filter(v=>v.review).length} / ${all.length} overall`;
  $('position').textContent=queue.length?`${cursor+1} of ${queue.length} in this queue`:'0 visits';
  $('empty').hidden=!!queue.length;$('case').hidden=!queue.length;
  if(!queue.length)return;
  mediaGeneration++;const generation=mediaGeneration;
  current=queue[cursor];const v=current;
  $('priority').textContent=`#${cursor+1} by ${$('target').value==='mid'?'Mid':'Endo'}-GLS error`;
  $('caseTitle').textContent=`${v.patient_id} · ${v.visit_date}`;
  $('caseSub').textContent=`${v.vendor} · ${v.all_bookmark?'All views from TOMTEC bookmark':'At least one view selected manually'} · ${v.visit_id}`;
  const useEndo=$('target').value==='endo';
  $('gt').textContent=(useEndo?v.endo_gt:v.gt).toFixed(2);
  $('pred').textContent=(useEndo?v.endo_prediction:v.prediction).toFixed(2);
  $('err').textContent=score(v).toFixed(2);
  $('secondary').textContent=`${useEndo?'Mid':'Endo'}-GLS: report ${(useEndo?v.gt:v.endo_gt).toFixed(2)}, prediction ${(useEndo?v.prediction:v.endo_prediction).toFixed(2)}, absolute error ${(useEndo?v.absolute_error:v.endo_absolute_error).toFixed(2)} points. Signed ${useEndo?'Endo':'Mid'} error: ${(useEndo?v.endo_prediction-v.endo_gt:v.signed_error).toFixed(2)}.`;
  $('clips').replaceChildren(...v.clips.map(c=>makeClip(c,generation)));
  v.clips.forEach((clip,index)=>{
    const section=$('clips').children[index];
    loadMedia(clip,v.replacements?.[clip.view]?'alternative':'crop',section.querySelector('video'),section.querySelector('.mediaStatus'),generation);
  });
  alternativeGeneration++;selectedCandidate=null;$('candidateList').replaceChildren();$('candidateVideo').hidden=true;$('candidateVideo').removeAttribute('src');$('candidateVideo').load();
  $('approveReplacement').disabled=true;$('replacementStatus').textContent='';$('replacementView').value=activeView;
  selectedStatus=null;document.querySelectorAll('.statuses button').forEach(b=>b.classList.remove('selected'));
  document.querySelectorAll('#reasons input').forEach(i=>i.checked=v.review?.reasons.includes(i.value)||false);
  $('note').value=v.review?.note||'';
  if(v.review)setStatus(v.review.status);else {$('reasons').hidden=false;updateReplacementPanel()}
  $('saved').textContent=v.review?'Saved '+new Date(v.review.updated).toLocaleString():'Not reviewed';
  $('previous').disabled=cursor===0;$('next').disabled=cursor===queue.length-1;
  document.querySelectorAll('.viewTabs button').forEach(b=>b.classList.toggle('active',b.dataset.view===activeView));
}
function navigate(delta){cursor=Math.max(0,Math.min(queue.length-1,cursor+delta));render();window.scrollTo({top:0,behavior:'smooth'});history.replaceState(null,'',`?visit=${encodeURIComponent(current.visit_id)}`)}
async function save({stay=false}={}){
  if(!selectedStatus){$('saved').textContent='Choose a review decision first';return false}
  const reasons=[...document.querySelectorAll('#reasons input:checked')].map(i=>i.value);
  $('saved').textContent='Saving…';$('save').disabled=true;
  try{
    const response=await fetch('/api/review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({visit_id:current.visit_id,status:selectedStatus,reasons,note:$('note').value})});
    const result=await response.json();if(!response.ok)throw Error(result.error||'Save failed');
    current.review=result.review;$('saved').textContent='Saved '+new Date(result.review.updated).toLocaleString();
    const id=current.visit_id;if($('reviewFilter').value==='unreviewed'&&!stay)rebuild(id);
    else $('progress').textContent=`${all.filter(v=>v.vendor==='Philips'&&v.review).length} / ${all.filter(v=>v.vendor==='Philips').length} Philips visits reviewed · ${all.filter(v=>v.review).length} / ${all.length} overall`;
    return true;
  }catch(err){$('saved').textContent='Could not save: '+err.message;return false}
  finally{$('save').disabled=false}
}
async function findAlternatives(){
  if(selectedStatus!=='suspected_issue'){$('replacementStatus').textContent='Select Suspected issue first.';return}
  const visit=current.visit_id,view=$('replacementView').value,generation=++alternativeGeneration;
  selectedCandidate=null;$('candidateList').replaceChildren();$('candidateVideo').hidden=true;$('candidateVideo').removeAttribute('src');$('candidateVideo').load();$('approveReplacement').disabled=true;
  $('replacementStatus').textContent=`Ranking other ${view} cines from this study…`;
  for(let attempt=0;attempt<200;attempt++){
    if(generation!==alternativeGeneration||current.visit_id!==visit)return;
    try{
      const response=await fetch(`/api/alternatives/${encodeURIComponent(visit)}/${view}`);
      const result=await response.json();
      if(!response.ok||result.status==='error')throw Error(result.error||'Ranking failed');
      if(result.status==='ready'){
        if(!result.candidates.length){$('replacementStatus').textContent='No confident unselected cine in this study was classified as '+view+'.';return}
        $('replacementStatus').textContent=`${result.candidates.length} ranked alternative${result.candidates.length===1?'':'s'} for ${view}. Preview before use.`;
        for(const candidate of result.candidates){
          const button=document.createElement('button');button.type='button';
          button.textContent=`#${candidate.rank} · ${candidate.name} · view probability ${(candidate.view_probability*100).toFixed(0)}% · score ${candidate.score.toFixed(2)}`;
          const small=document.createElement('small');small.textContent=`${candidate.frames} frames · ${candidate.duration_seconds.toFixed(2)} s${candidate.warnings.length?' · '+candidate.warnings.join('; '):''}`;
          button.append(small);button.onclick=()=>previewCandidate(candidate,visit,view,generation);
          $('candidateList').append(button);
        }
        $('candidateList').firstChild.click();return;
      }
      $('replacementStatus').textContent=result.step||'Ranking cines…';
    }catch(err){$('replacementStatus').textContent='Could not rank alternatives: '+err.message;return}
    await new Promise(resolve=>setTimeout(resolve,850));
  }
  $('replacementStatus').textContent='Ranking timed out; try again.';
}
async function previewCandidate(candidate,visit,view,generation){
  selectedCandidate=candidate;
  [...$('candidateList').children].forEach((node,index)=>node.classList.toggle('chosen',index+1===candidate.rank));
  $('approveReplacement').disabled=true;$('candidateVideo').hidden=true;$('candidateVideo').removeAttribute('src');$('candidateVideo').load();
  $('replacementStatus').textContent=`Preparing #${candidate.rank}: ${candidate.name}…`;
  for(let attempt=0;attempt<100;attempt++){
    if(generation!==alternativeGeneration||current.visit_id!==visit||selectedCandidate!==candidate)return;
    try{
      const response=await fetch(`/api/alternative_media/${encodeURIComponent(visit)}/${view}/${candidate.candidate_id}`);
      const result=await response.json();if(!response.ok||result.status==='error')throw Error(result.error||'Preview failed');
      if(result.status==='ready'){$('candidateVideo').src=result.url;$('candidateVideo').hidden=false;$('approveReplacement').disabled=false;$('replacementStatus').textContent=`Previewing #${candidate.rank}: ${candidate.name}. Check apex and walls throughout the cine.`;return}
    }catch(err){$('replacementStatus').textContent='Could not preview candidate: '+err.message;return}
    await new Promise(resolve=>setTimeout(resolve,650));
  }
  $('replacementStatus').textContent='Preview preparation timed out.';
}
async function approveReplacement(){
  if(!selectedCandidate||selectedStatus!=='suspected_issue')return;
  const visit=current.visit_id,view=$('replacementView').value,candidate=selectedCandidate;
  $('approveReplacement').disabled=true;
  if(!await save({stay:true})){ $('approveReplacement').disabled=false;return }
  try{
    const response=await fetch('/api/replacement',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({visit_id:visit,view,candidate_id:candidate.candidate_id})});
    const result=await response.json();if(!response.ok)throw Error(result.error||'Replacement failed');
    current.replacements[view]=result.replacement;render();$('replacementView').value=view;updateReplacementPanel();
    $('replacementStatus').textContent=`Using ${candidate.name} for ${view}. Original selection is preserved; current GLS prediction has not been recalculated.`;
  }catch(err){$('replacementStatus').textContent='Could not save replacement: '+err.message;$('approveReplacement').disabled=false}
}
async function revertReplacement(){
  const view=$('replacementView').value;
  if(!current.replacements?.[view])return;
  try{
    const response=await fetch('/api/replacement',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({visit_id:current.visit_id,view,candidate_id:null})});
    const result=await response.json();if(!response.ok)throw Error(result.error||'Restore failed');
    delete current.replacements[view];render();$('replacementView').value=view;updateReplacementPanel();
    $('replacementStatus').textContent=`Restored original ${view} selection.`;
  }catch(err){$('replacementStatus').textContent='Could not restore original: '+err.message}
}
async function init(){
  for(const [key,id] of [['vendor','vendor'],['target','target'],['reviewFilter','reviewFilter'],['search','search']])$(id).addEventListener('input',()=>rebuild(current?.visit_id));
  document.querySelectorAll('.statuses button').forEach(b=>b.addEventListener('click',()=>setStatus(b.dataset.status)));
  document.querySelectorAll('.viewTabs button').forEach(b=>b.addEventListener('click',()=>{activeView=b.dataset.view;$('replacementView').value=activeView;updateReplacementPanel();document.querySelectorAll('.viewTabs button').forEach(x=>x.classList.toggle('active',x===b));document.querySelectorAll('.clip').forEach(x=>x.classList.toggle('active',x.dataset.view===activeView))}));
  const reasonArea=$('reasons');for(const [value,label] of Object.entries(reasons)){const item=document.createElement('label');const input=document.createElement('input');input.type='checkbox';input.value=value;item.append(input,document.createTextNode(label));reasonArea.append(item)}
  $('save').addEventListener('click',()=>save());
  $('previous').addEventListener('click',()=>navigate(-1));
  $('next').addEventListener('click',()=>navigate(1));
  $('replacementView').addEventListener('change',()=>{alternativeGeneration++;selectedCandidate=null;$('candidateList').replaceChildren();$('candidateVideo').hidden=true;$('approveReplacement').disabled=true;$('replacementStatus').textContent='';updateReplacementPanel()});
  $('findReplacement').addEventListener('click',()=>findAlternatives());
  $('approveReplacement').addEventListener('click',()=>approveReplacement());
  $('revertReplacement').addEventListener('click',()=>revertReplacement());
  try{const response=await fetch('/api/state');if(!response.ok)throw Error('Cannot load visit queue');const data=await response.json();all=data.visits;rebuild(params.get('visit'));}
  catch(err){$('loadError').hidden=false;$('loadError').textContent=err.message;$('progress').textContent='Load failed'}
}
init();
