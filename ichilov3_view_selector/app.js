const $ = id => document.getElementById(id);
const VIEWS = ['A2C', 'A3C', 'A4C'];
const state = { visits: [], filter: 'no_bookmark', key: null, detail: null, views: {A2C:'',A3C:'',A4C:''}, sources: {A2C:'',A3C:'',A4C:''}, suggestions: null, suggestionTimer: null, shown: 12, viewerFile: null, clipUrl: null, clipRequest: 0, viewMode: 'cine', pendingSave: '', autoKeepDraft: false, saveVersion: 0, visitVersion: 0 };
let saveQueue = Promise.resolve();
const esc = value => String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
const fileById = id => state.detail?.files.find(file => file.id === id);
const frameUrl = (file, index, width) => `/api/frame?id=${encodeURIComponent(file.id)}&index=${index}&width=${width}`;

async function getJson(url, options) {
  const response = await fetch(url, options);
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || `Request failed (${response.status})`);
  return body;
}

function queueItems() {
  const search = $('search').value.trim().toLowerCase();
  return state.visits.filter(visit => {
    const match = state.filter === 'no_bookmark' ? visit.tomtec_bookmark !== 'Yes' :
      state.filter === 'unresolved' ? visit.resolved_views < 3 : true;
    return match && (!search || `${visit.patient} ${visit.date}`.toLowerCase().includes(search));
  });
}

function renderQueue() {
  const items = queueItems();
  $('queueCount').textContent = items.length;
  $('countNoBookmark').textContent = state.visits.filter(v => v.tomtec_bookmark !== 'Yes').length;
  $('countUnresolved').textContent = state.visits.filter(v => v.resolved_views < 3).length;
  $('countAll').textContent = state.visits.length;
  $('visitList').innerHTML = items.map(v => {
    const mark = v.manual_status === 'complete' ? '✓' : v.manual_status === 'partial' ? '◒' : ['unable', 'invalid'].includes(v.manual_status) ? '!' : '·';
    return `<button class="visit-item ${state.key === v.key ? 'selected' : ''}" data-key="${esc(v.key)}"><span><strong>${esc(v.patient)}</strong><small>${esc(v.date)} · ${v.video_count} cines</small></span><span class="queue-status ${esc(v.manual_status)}">${mark}</span></button>`;
  }).join('') || '<div class="queue-empty">No visits match this filter.</div>';
  $('visitList').querySelectorAll('.visit-item').forEach(el => el.addEventListener('click', () => selectVisit(el.dataset.key)));
  const queue = state.visits.filter(v => v.tomtec_bookmark !== 'Yes');
  const done = queue.filter(v => v.manual_status === 'complete').length;
  $('progressNumber').textContent = `${done} / ${queue.length}`;
  $('progressDetail').textContent = `${queue.filter(v => v.manual_status === 'partial').length} partial · ${queue.filter(v => ['unable', 'invalid'].includes(v.manual_status)).length} invalid`;
  $('progressBar').style.width = `${queue.length ? 100 * done / queue.length : 0}%`;
  $('progressCaption').textContent = 'no-bookmark visits complete';
}

async function selectVisit(key) {
  const visitVersion = ++state.visitVersion;
  stopPlayback();
  clearTimeout(state.suggestionTimer);
  state.key = key;
  state.pendingSave = '';
  state.suggestions = null;
  state.shown = 12;
  $('emptyState').hidden = true;
  $('reviewPanel').hidden = false;
  $('clipGrid').innerHTML = '<div class="loading">Loading cine metadata…</div>';
  renderQueue();
  try {
    await saveQueue;
    if (state.visitVersion !== visitVersion) return;
    const detail = await getJson(`/api/visit?key=${encodeURIComponent(key)}`);
    if (state.visitVersion !== visitVersion) return;
    state.detail = detail;
    state.views = {...{A2C:'', A3C:'', A4C:''}, ...(detail.decision?.views || {})};
    state.sources = {...{A2C:'', A3C:'', A4C:''}, ...(detail.decision?.selection_source || {})};
    state.autoKeepDraft = detail.decision?.status === 'complete';
    $('visitTitle').textContent = `${detail.visit.patient}  /  ${detail.visit.date}`;
    $('visitSubtitle').textContent = `${detail.files.length} cine DICOMs · ${detail.visit.strain_report === 'Yes' ? 'TXT strain report available' : 'No TXT strain report'} · ${detail.visit.tomtec_bookmark === 'Yes' ? 'TOMTEC bookmark present' : 'No TOMTEC bookmark'}`;
    $('visitStatus').textContent = detail.decision ? detail.decision.status : detail.visit.resolved_views === 3 ? 'TOMTEC resolved' : 'Unreviewed';
    $('visitStatus').className = `status-pill ${detail.decision?.status || ''}`;
    const mismatch = detail.folder_contains_study_dates?.length ? `The visit folder contains DICOMs dated ${detail.folder_contains_study_dates.join(', ')} instead of ${detail.visit.date}.` : '';
    const alert = [detail.visit.audit_notes, detail.visit.matching_dicom_folder !== 'Yes' ? 'No matching DICOM folder was identified for this visit.' : '', mismatch].filter(Boolean).join(' ');
    $('auditNote').hidden = !alert;
    $('auditNote').textContent = alert;
    $('reviewer').value = detail.decision?.reviewer || sessionStorage.getItem('ichilovReviewer') || '';
    $('note').value = detail.decision?.note || '';
    $('saveMessage').textContent = '';
    const readonly = detail.visit.tomtec_bookmark === 'Yes' && detail.visit.resolved_views === 3;
    $('saveButton').disabled = readonly;
    $('invalidButton').disabled = readonly;
    $('saveButton').textContent = readonly ? 'Already resolved by TOMTEC' : 'Save review →';
    renderTomtec();
    renderSlots();
    renderClips();
    renderSuggestions();
    loadSuggestions(false);
  } catch (error) {
    $('clipGrid').innerHTML = `<div class="error">${esc(error.message)}</div>`;
  }
}

function renderTomtec() {
  const visit = state.detail.visit;
  const present = visit.tomtec_bookmark === 'Yes';
  $('tomtecPanel').hidden = !present;
  if (!present) return;
  const recovered = VIEWS.filter(view => visit.audit_views[view]).length;
  $('tomtecSummary').textContent = `${recovered} / 3 source clips recovered from the bookmark. Missing or conflicting references remain unresolved.`;
  const readonly = visit.resolved_views === 3;
  $('tomtecViews').innerHTML = VIEWS.map(view => {
    const path = visit.audit_views[view];
    const file = state.detail.files.find(item => item.path.toLowerCase() === (path || '').toLowerCase());
    const issue = (visit.audit_notes || '').split(';').map(item => item.trim()).find(item => item.startsWith(view + ':'));
    return `<div class="tomtec-view ${path ? 'recovered' : 'unresolved'}"><strong>${view} · ${path ? 'Recovered' : 'Unresolved'}</strong><p>${path ? esc(path.split(/[\\/]/).pop()) : esc(issue ? issue.slice(4).trim() : 'No unique source clip recovered')}</p>${path ? `<small>${file ? 'Matching cine available' : 'Referenced clip is not in this visit’s cine library'}</small>` : ''}${file ? `<div><button data-tomtec-preview="${file.id}">▶ Review cine</button>${!readonly ? `<button data-tomtec-use="${file.id}" data-view="${view}">Use linked clip</button>` : ''}</div>` : ''}</div>`;
  }).join('');
  $('tomtecViews').querySelectorAll('[data-tomtec-preview]').forEach(button => button.addEventListener('click', () => openViewer(button.dataset.tomtecPreview)));
  $('tomtecViews').querySelectorAll('[data-tomtec-use]').forEach(button => button.addEventListener('click', () => assign(button.dataset.view, button.dataset.tomtecUse, 'tomtec')));
}

function renderSlots() {
  if (!state.detail) return;
  $('viewSlots').innerHTML = VIEWS.map((view, index) => {
    const selected = fileById(state.views[view]);
    const origin = state.sources[view] === 'tomtec' ? 'TOMTEC-linked choice' : state.sources[view] === 'suggested' ? 'accepted model suggestion' : 'manual choice';
    return `<div class="view-slot ${selected ? 'filled' : ''}"><span class="slot-number">0${index + 1}</span><div><strong>${view}</strong><small>${selected ? `${esc(selected.name)} · ${origin}` : 'No clip selected by you'}</small></div>${selected ? `<button class="clear-view" data-view="${view}" title="Clear ${view}">×</button>` : '<span class="slot-empty">—</span>'}</div>`;
  }).join('');
  $('viewSlots').querySelectorAll('.clear-view').forEach(button => button.addEventListener('click', () => {
    state.views[button.dataset.view] = '';
    state.sources[button.dataset.view] = '';
    renderSlots(); renderClips();
    selectionChanged();
  }));
}

function renderClips() {
  if (!state.detail) return;
  let files = [...state.detail.files];
  if ($('sortClips').value === 'frames') files.sort((a,b) => b.frames-a.frames || a.instance-b.instance);
  if ($('sortClips').value === 'size') files.sort((a,b) => b.size-a.size || a.instance-b.instance);
  if ($('sortClips').value === 'suggested' && state.suggestions?.status === 'ready') {
    const rank = id => {
      const chosen = Object.values(state.suggestions.selected).includes(id);
      const alternatives = Object.values(state.suggestions.alternatives).flat().findIndex(item => item.id === id);
      return chosen ? 0 : alternatives >= 0 ? alternatives + 1 : 100;
    };
    files.sort((a,b) => rank(a.id)-rank(b.id) || a.instance-b.instance);
  }
  const visible = $('showAll').checked ? files : files.slice(0, state.shown);
  $('clipCount').textContent = files.length;
  $('clipHelp').textContent = files.length ? 'Open a clip to play its full cine loop. You can slow playback or inspect individual frames.' : 'No multi-frame DICOMs have a matching DICOM study date. Check the visit note and source folder.';
  $('clipGrid').innerHTML = visible.map(file => {
    const selected = VIEWS.filter(view => state.views[view] === file.id);
    const prediction = state.suggestions?.status === 'ready' ? state.suggestions.predictions[file.id] : null;
    const modelLabel = prediction?.status === 'ok' ? `${prediction.predicted_view} ${Math.round(prediction.confidence*100)}%${prediction.bmode_candidate ? '' : ' · Doppler/complex'}` : '';
    const mismatch = file.folder_date && file.folder_date.replaceAll('_','-') !== file.study_date;
    return `<article class="clip-card ${selected.length ? 'chosen' : ''}"><button class="clip-image" data-open="${file.id}" aria-label="Play cine ${esc(file.name)}"><img loading="lazy" src="${frameUrl(file, Math.floor(file.frames/2), 420)}" alt="DICOM frame from ${esc(file.name)}"><span class="open-icon">▶</span><span class="clip-action">PLAY CINE</span></button><div class="clip-info"><div class="clip-title"><strong>${esc(file.name)}</strong><span>#${esc(file.instance)}</span></div><p>${file.frames} frames · ${esc(file.rows)} × ${esc(file.columns)} · ${(file.size/1048576).toFixed(1)} MB</p>${modelLabel ? `<p class="model-label">Model: ${esc(modelLabel)}</p>` : ''}<p class="clip-path">${esc(file.series_description || file.image_comments || file.manufacturer || 'Ultrasound cine')}</p>${mismatch ? '<p class="mismatch">Folder date differs from DICOM study date</p>' : ''}<div class="assign-buttons">${VIEWS.map(view => `<button data-file="${file.id}" data-view="${view}" class="${selected.includes(view) ? 'active' : ''}">${view}</button>`).join('')}</div></div></article>`;
  }).join('') || '<div class="no-clips">No playable cine loops found for this visit.</div>';
  $('moreClips').hidden = $('showAll').checked || visible.length >= files.length;
  $('clipGrid').querySelectorAll('[data-open]').forEach(button => button.addEventListener('click', () => openViewer(button.dataset.open)));
  $('clipGrid').querySelectorAll('[data-file][data-view]').forEach(button => button.addEventListener('click', () => assign(button.dataset.view, button.dataset.file)));
  $('clipGrid').querySelectorAll('img').forEach(img => img.addEventListener('error', () => {img.style.display='none'; img.parentElement.classList.add('image-error');}));
}

async function loadSuggestions(start) {
  const key = state.key;
  if (!key) return;
  try {
    const result = await getJson(`/api/suggestions?key=${encodeURIComponent(key)}${start ? '&start=1' : ''}`);
    if (state.key !== key) return;
    state.suggestions = result;
    renderSuggestions();
    if (result.status === 'ready') {
      if ($('sortClips').value === 'instance') $('sortClips').value = 'suggested';
      renderClips();
    }
    if (result.status === 'queued' || result.status === 'running') {
      state.suggestionTimer = setTimeout(() => loadSuggestions(false), 1200);
    }
  } catch (error) {
    if (state.key !== key) return;
    $('suggestionStatus').textContent = error.message;
    $('analyzeButton').disabled = false;
  }
}

function renderSuggestions() {
  const result = state.suggestions;
  const status = result?.status || 'not_started';
  $('analyzeButton').hidden = status === 'ready' || status === 'no_cines';
  $('analyzeButton').disabled = status === 'queued' || status === 'running';
  $('suggestionCards').hidden = status !== 'ready';
  $('useSuggestions').hidden = status !== 'ready' || !Object.values(result.selected).some(Boolean);
  if (status === 'not_started') $('suggestionStatus').textContent = 'Run the local model to see candidate views for this visit.';
  else if (status === 'no_cines') $('suggestionStatus').textContent = 'No matching cine DICOMs are available to analyze.';
  else if (status === 'queued' || status === 'running') $('suggestionStatus').textContent = `Analyzing cine clips locally… ${result.done || 0} / ${result.total || 0}`;
  else if (status === 'error') $('suggestionStatus').textContent = `Analysis failed: ${result.error || 'unknown error'}`;
  else if (status === 'ready') {
    const suggested = Object.values(result.selected).filter(Boolean).length;
    $('suggestionStatus').textContent = `${suggested} / 3 suggested · ${result.analyzed} clips checked${result.errors ? ` · ${result.errors} could not be analyzed` : ''}. Accept three views to save automatically, or replace any suggestion.`;
    $('suggestionCards').innerHTML = VIEWS.map(view => {
      const id = result.selected[view];
      const alternatives = result.alternatives[view] || [];
      const top = alternatives.find(item => item.id === id);
      const file = fileById(id);
      const prediction = result.predictions[id];
      const uncertainty = top?.review_level !== 'strong';
      const details = top && prediction ? `${Math.round(top.probability*100)}% view probability · ${Math.round(top.agreement*100)}% sampled frames agree · ${prediction.quality.duration_seconds}s loop · technical screen ${Math.round(top.quality_proxy*100)}%` : '';
      return `<div class="suggestion-card ${id ? uncertainty ? 'uncertain' : 'strong' : 'uncertain'}"><div class="suggestion-view"><span>${view}</span><span class="confidence">${id ? uncertainty ? 'REVIEW CAREFULLY' : 'STRONG VIEW MATCH' : 'NO RELIABLE MATCH'}</span></div>${id ? `<div class="suggestion-file">${esc(file?.name || id)}</div><p class="suggestion-reason">${esc(details)}</p><div class="suggestion-actions"><button data-preview="${id}">▶ Review cine</button><button data-use="${id}" data-view="${view}">Use this</button></div>` : `<div class="suggestion-empty">No clip passed the minimum view-probability and frame-agreement checks. Review alternatives manually.</div>`}<div class="suggestion-alternatives"><p>OTHER CANDIDATES</p>${alternatives.filter(item => item.id !== id).map(item => `<div class="alternative"><span>${esc(fileById(item.id)?.name || item.id)} · ${Math.round(item.probability*100)}%</span><button data-preview="${item.id}">Preview</button><button data-use="${item.id}" data-view="${view}">Use</button></div>`).join('') || '<span class="subtle">None</span>'}</div></div>`;
    }).join('');
    $('suggestionCards').querySelectorAll('[data-preview]').forEach(button => button.addEventListener('click', () => openViewer(button.dataset.preview)));
    $('suggestionCards').querySelectorAll('[data-use]').forEach(button => button.addEventListener('click', () => assign(button.dataset.view, button.dataset.use, 'suggested')));
  }
}

function assign(view, id, source='manual') {
  if (state.detail.visit.tomtec_bookmark === 'Yes' && state.detail.visit.resolved_views === 3) return;
  if (source !== 'manual' && state.views[view] === id && state.sources[view] === source) return;
  for (const other of VIEWS) if (other !== view && state.views[other] === id) { state.views[other] = ''; state.sources[other] = ''; }
  const clearing = source === 'manual' && state.views[view] === id;
  state.views[view] = clearing ? '' : id;
  state.sources[view] = clearing ? '' : source;
  renderSlots(); renderClips();
  if (state.viewerFile) renderViewerButtons();
  selectionChanged();
}

function openViewer(id) {
  const file = fileById(id);
  if (!file) return;
  stopPlayback();
  state.viewerFile = file;
  $('viewerTitle').textContent = file.name;
  $('viewerMeta').textContent = `${file.frames} frames · instance ${file.instance} · ${file.study_date} · ${file.manufacturer}`;
  $('viewerPath').textContent = `${file.path}  ·  SOP UID ${file.sop_uid}`;
  $('frameSlider').max = file.frames - 1;
  $('frameSlider').value = Math.floor(file.frames/2);
  $('viewerError').hidden = true;
  $('viewerVideo').hidden = true;
  $('viewerImage').hidden = false;
  $('clipStatus').hidden = false;
  $('clipStatus').textContent = 'Preparing full cine loop…';
  $('viewer').showModal();
  showFrame(Number($('frameSlider').value));
  switchMode('cine');
  renderViewerButtons();
}

function showFrame(index) {
  const file = state.viewerFile;
  if (!file) return;
  $('viewerImage').src = frameUrl(file, index, 900);
  $('frameSlider').value = index;
  $('frameLabel').textContent = `${index+1} / ${file.frames}`;
}

function renderViewerButtons() {
  $('viewer').querySelectorAll('.viewer-assign button').forEach(button => button.classList.toggle('active', state.views[button.dataset.view] === state.viewerFile?.id));
}

function stopPlayback() {
  state.clipRequest++;
  $('viewerVideo').pause();
  $('viewerVideo').removeAttribute('src');
  $('viewerVideo').load();
  if (state.clipUrl) URL.revokeObjectURL(state.clipUrl);
  state.clipUrl = null;
}

async function loadCine(file) {
  const request = ++state.clipRequest;
  try {
    const response = await fetch(`/api/clip?id=${encodeURIComponent(file.id)}`);
    if (!response.ok) {
      let message = `Could not prepare cine (${response.status})`;
      try { message = (await response.json()).error || message; } catch (_) {}
      throw new Error(message);
    }
    const blob = await response.blob();
    if (request !== state.clipRequest || state.viewerFile?.id !== file.id) return;
    state.clipUrl = URL.createObjectURL(blob);
    const video = $('viewerVideo');
    video.src = state.clipUrl;
    video.playbackRate = Number($('playbackSpeed').value);
    video.onloadeddata = () => {
      if (request !== state.clipRequest) return;
      video.playbackRate = Number($('playbackSpeed').value);
      $('clipStatus').hidden = true;
      if (state.viewMode === 'cine') {
        $('viewerImage').hidden = true;
        video.hidden = false;
        video.play().catch(() => {});
      }
    };
    video.onerror = () => {
      if (request !== state.clipRequest) return;
      $('clipStatus').hidden = false;
      $('clipStatus').textContent = 'Video playback failed. Try Frame by frame.';
    };
    video.load();
  } catch (error) {
    if (request !== state.clipRequest) return;
    $('clipStatus').hidden = false;
    $('clipStatus').textContent = `${error.message} · Try Frame by frame.`;
  }
}

function switchMode(mode) {
  state.viewMode = mode;
  $('cineButton').classList.toggle('active', mode === 'cine');
  $('framesButton').classList.toggle('active', mode === 'frames');
  $('speedControl').hidden = mode !== 'cine';
  $('frameControls').hidden = mode !== 'frames';
  if (mode === 'frames') {
    $('viewerVideo').pause();
    $('viewerVideo').hidden = true;
    $('viewerImage').hidden = false;
    $('clipStatus').hidden = true;
  } else if (state.clipUrl && $('viewerVideo').readyState >= 2) {
    $('viewerImage').hidden = true;
    $('viewerVideo').hidden = false;
    $('clipStatus').hidden = true;
    $('viewerVideo').play().catch(() => {});
  } else if (state.viewerFile) {
    $('clipStatus').hidden = false;
    if (!state.clipUrl) loadCine(state.viewerFile);
  }
}

function selectionChanged() {
  state.pendingSave = '';
  if (VIEWS.every(view => state.views[view])) state.autoKeepDraft = true;
  if (state.autoKeepDraft) {
    saveDecision({automatic:true});
  } else {
    $('saveMessage').textContent = 'Partial choice; use Save review if you want to keep it.';
  }
}

function saveDecision({invalid=false, automatic=false}={}) {
  if (!state.detail) return saveQueue;
  const reviewer = $('reviewer').value.trim();
  if (!reviewer) {
    state.pendingSave = invalid ? 'invalid' : automatic ? 'auto' : 'manual';
    $('saveMessage').textContent = 'Enter a reviewer name to save this visit.';
    $('reviewer').focus();
    return saveQueue;
  }
  state.pendingSave = '';
  if (invalid) {
    state.autoKeepDraft = false;
    state.views = {A2C:'', A3C:'', A4C:''};
    state.sources = {A2C:'', A3C:'', A4C:''};
    renderSlots(); renderClips();
    if (state.viewerFile) renderViewerButtons();
  }
  const key = state.key;
  const visitVersion = state.visitVersion;
  const version = ++state.saveVersion;
  const views = {...state.views};
  const payload = {visit_key:key, views, selection_source:{...state.sources},
    reviewer, note:$('note').value, invalid};
  state.latestRequestedStatus = invalid ? 'invalid' : VIEWS.every(view => views[view]) ? 'complete' : 'partial';
  $('saveMessage').textContent = automatic ? 'Saving three views automatically…' : 'Saving…';
  $('saveButton').disabled = true;
  $('invalidButton').disabled = true;
  const run = async () => {
    try {
      const decision = await getJson('/api/decision', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
      sessionStorage.setItem('ichilovReviewer', decision.reviewer);
      const visit = state.visits.find(v => v.key === key);
      if (visit) visit.manual_status = decision.status;
      if (state.key === key && state.visitVersion === visitVersion && state.saveVersion === version) {
        state.detail.decision = decision;
        $('visitStatus').textContent = decision.status;
        $('visitStatus').className = `status-pill ${decision.status}`;
        if (invalid) $('note').value = decision.note;
        $('saveMessage').textContent = 'Saved · ' + new Date(decision.updated_at).toLocaleTimeString();
      }
      renderQueue();
    } catch(error) {
      if (state.key === key && state.visitVersion === visitVersion && state.saveVersion === version) {
        $('saveMessage').textContent = `Save failed: ${error.message}`;
      }
    } finally {
      if (state.key === key && state.visitVersion === visitVersion && state.saveVersion === version) {
        $('saveButton').disabled = false;
        $('invalidButton').disabled = false;
      }
    }
  };
  saveQueue = saveQueue.then(run, run);
  return saveQueue;
}

document.addEventListener('DOMContentLoaded', async () => {
  document.querySelectorAll('.filter').forEach(button => button.addEventListener('click', () => {
    state.filter = button.dataset.filter;
    document.querySelectorAll('.filter').forEach(el => el.classList.toggle('active', el === button));
    renderQueue();
  }));
  $('search').addEventListener('input', renderQueue);
  $('sortClips').addEventListener('change', renderClips);
  $('showAll').addEventListener('change', renderClips);
  $('moreClips').addEventListener('click', () => {state.shown += 12; renderClips();});
  $('analyzeButton').addEventListener('click', () => loadSuggestions(true));
  $('useSuggestions').addEventListener('click', () => {
    if (state.detail.visit.tomtec_bookmark === 'Yes' && state.detail.visit.resolved_views === 3) return;
    let changed = false;
    for (const view of VIEWS) {
      const id = state.suggestions?.selected?.[view];
      if (id && !state.views[view]) {state.views[view] = id; state.sources[view] = 'suggested'; changed = true;}
    }
    if (changed) {renderSlots(); renderClips(); selectionChanged();}
  });
  $('saveButton').addEventListener('click', () => saveDecision({
    invalid: state.detail?.decision?.status === 'invalid' && VIEWS.every(view => !state.views[view])
  }));
  $('invalidButton').addEventListener('click', () => saveDecision({invalid:true}));
  $('reviewer').addEventListener('change', () => {
    if (!state.pendingSave || !$('reviewer').value.trim()) return;
    saveDecision({invalid:state.pendingSave === 'invalid', automatic:state.pendingSave === 'auto'});
  });
  $('reviewer').addEventListener('keydown', event => {if (event.key === 'Enter') $('reviewer').blur();});
  $('closeViewer').addEventListener('click', () => $('viewer').close());
  $('viewer').addEventListener('click', event => {
    const bounds = $('viewer').getBoundingClientRect();
    if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) $('viewer').close();
  });
  $('viewer').addEventListener('close', () => {stopPlayback(); state.viewerFile = null; $('viewerImage').src = '';});
  $('frameSlider').addEventListener('input', () => showFrame(Number($('frameSlider').value)));
  $('cineButton').addEventListener('click', () => switchMode('cine'));
  $('framesButton').addEventListener('click', () => switchMode('frames'));
  $('playbackSpeed').addEventListener('change', () => { $('viewerVideo').playbackRate = Number($('playbackSpeed').value); });
  $('viewer').querySelectorAll('.viewer-assign button').forEach(button => button.addEventListener('click', () => assign(button.dataset.view, state.viewerFile.id)));
  try {
    state.visits = await getJson('/api/visits');
    renderQueue();
    const first = queueItems()[0];
    if (first) selectVisit(first.key);
  } catch (error) {
    $('visitList').innerHTML = `<div class="error">${esc(error.message)}</div>`;
  }
});
