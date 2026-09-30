"""Private mobile review of flagged crops, with non-destructive crop revisions."""
from __future__ import annotations
import argparse, csv, io, json, mimetypes, os, sqlite3, subprocess, sys, threading, time, uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import imageio_ffmpeg
import numpy as np
import pydicom
from PIL import Image

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from ichilov3_prepare_selected import padded_crop, _resize_frame, _normalize_frames, atomic_json, digest

ROOT=Path(__file__).resolve().parent
CROPS=Path(r'D:\DS\ichilov3_stage2_padded_20260926')
EMBEDDINGS=Path(r'D:\DS\ichilov3_embeddings_reference_20260926')


def tissue_box(ds):
    boxes=[]
    for r in ds.get('SequenceOfUltrasoundRegions',[]):
        if int(r.get('RegionSpatialFormat',0))==1 and int(r.get('RegionDataType',0))==1:
            b=[max(0,int(r.RegionLocationMinX0)),max(0,int(r.RegionLocationMinY0)),
               min(int(ds.Columns),int(r.RegionLocationMaxX1)+1),min(int(ds.Rows),int(r.RegionLocationMaxY1)+1)]
            if b[2]>b[0] and b[3]>b[1]:boxes.append(b)
    if not boxes:raise ValueError('No valid tissue region; requires a separate manual review')
    return max(boxes,key=lambda b:(b[2]-b[0])*(b[3]-b[1]))


def selected_box(region,rect):
    x0,y0,x1,y1=region
    if rect is None:return region
    if len(rect)!=4 or any(not np.isfinite(v) or not 0<=v<=1 for v in rect):raise ValueError('Rectangle must be within the tissue region')
    a,b,c,d=rect
    if c-a<.1 or d-b<.1:raise ValueError('Crop is too small')
    return [x0+round(a*(x1-x0)),y0+round(b*(y1-y0)),x0+round(c*(x1-x0)),y0+round(d*(y1-y0))]


def rgb(frame):
    if frame.ndim==2:frame=np.repeat(frame[...,None],3,axis=-1)
    if frame.dtype!=np.uint8:raise ValueError('Unvalidated preview pixel depth')
    return np.ascontiguousarray(frame)


def mp4(frames,path,fps):
    first=rgb(frames[0]);h,w=first.shape[:2]
    temporary=path.with_name(path.stem+'.partial.mp4')
    command=[imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-f','rawvideo','-pixel_format','rgb24',
        '-video_size',f'{w}x{h}','-framerate',str(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','veryfast',
        '-crf','21','-pix_fmt','yuv420p','-movflags','+faststart','-y',str(temporary)]
    p=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    try:
        for frame in frames:p.stdin.write(rgb(frame).tobytes())
        p.stdin.close();error=p.stderr.read().decode(errors='replace')
        if p.wait():raise RuntimeError(error[-500:])
        temporary.replace(path)
    finally:
        if p.poll() is None:p.kill();p.wait()


class Store:
    def __init__(self,output,crops=CROPS,embeddings=EMBEDDINGS):
        self.output=output;self.crops=crops;self.embeddings=embeddings
        output.mkdir(parents=True,exist_ok=True)
        self.db=output/'reviews.sqlite3';self.lock=threading.RLock();self.jobs={}
        self.pool=ThreadPoolExecutor(max_workers=2)
        self.repair_lock=threading.Lock()
        self.records=json.loads((crops/'full_manifest.json').read_text())
        self.all={r['file_id']:r for r in self.records}
        self.flagged={k:r for k,r in self.all.items() if r['status']=='needs_review'}
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS decisions (id TEXT PRIMARY KEY, quality TEXT, revision TEXT, updated TEXT, note TEXT)')
            db.execute('CREATE TABLE IF NOT EXISTS history (event INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT, before_json TEXT, after_json TEXT, updated TEXT)')
        self.export_manifest()

    @contextmanager
    def connect(self):
        db=sqlite3.connect(self.db,timeout=20)
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback();raise
        finally:db.close()

    def decisions(self):
        with self.connect() as db:
            return {r[0]:dict(quality=r[1],revision=r[2],updated=r[3],note=r[4]) for r in db.execute('SELECT * FROM decisions')}

    def decide(self,file_id,quality,revision=None,note=''):
        if file_id not in self.flagged:raise ValueError('Unknown review clip')
        if quality not in [None,'good','bad','repaired']:raise ValueError('Unknown decision')
        if revision is not None and (not isinstance(revision,str) or len(revision)!=32 or any(c not in '0123456789abcdef' for c in revision)):raise ValueError('Invalid repair revision')
        if quality=='repaired' and (not revision or not (self.output/'revisions'/file_id/revision/'repair_complete.json').exists()):raise ValueError('Repair is not complete')
        with self.lock:
            before=self.decisions().get(file_id)
            updated=datetime.now(timezone.utc).isoformat()
            after=dict(quality=quality,revision=revision,updated=updated,note=note[:2000]) if quality else None
            with self.connect() as db:
                if after:db.execute('INSERT OR REPLACE INTO decisions VALUES (?,?,?,?,?)',(file_id,quality,revision,updated,note[:2000]))
                else:db.execute('DELETE FROM decisions WHERE id=?',(file_id,))
                event=db.execute('INSERT INTO history(id,before_json,after_json,updated) VALUES (?,?,?,?)',(file_id,json.dumps(before),json.dumps(after),updated)).lastrowid
            self.export_manifest()
            return dict(before=before,decision=after,event=event)

    def export_manifest(self):
        decisions=self.decisions();result=[]
        for original in self.records:
            r=dict(original);d=decisions.get(r['file_id']);quality=d['quality'] if d else ('pending' if r['file_id'] in self.flagged else 'not_required')
            r['review_quality']=quality;r['review_decision']=d
            r['training_eligible_crop']=quality in ['good','repaired','not_required']
            if quality=='repaired':
                revision=self.output/'revisions'/r['file_id']/d['revision']
                revised=json.loads((revision/'crop.json').read_text())
                if revised.get('alternative_source'):
                    r['original_source_path']=r['source_path'];r['original_selection_source']=r['selection_source']
                    for key in ['source_path','source_sha256','source_sop','expected_sop','expected_study','frames','selection_source']:
                        r[key]=revised[key]
                    r['alternative_source']=True
                r['crop_output']=str(revision/'crop.npz')
                r['active_embeddings']={name:str(revision/name/(r['file_id']+'.npz')) for name in ['echoprime','panecho']}
            else:r['active_embeddings']={name:str(self.embeddings/name/(r['file_id']+'.npz')) for name in ['echoprime','panecho']}
            result.append(r)
        atomic_json(self.output/'reviewed_crop_manifest.json',result)
        return result

    def state(self):
        decisions=self.decisions();clips=[]
        for position,r in enumerate(sorted(self.flagged.values(),key=lambda r:(r['patient'],r['visit_date'],r['view'])),1):
            clip={k:r[k] for k in ['file_id','patient','visit_date','view','frames','manufacturer','model','removed_foreground_max']}
            clip['decision']=decisions.get(r['file_id']);clip['review_index']=position;clips.append(clip)
        return dict(clips=clips,total=len(clips),good=sum(d['quality']=='good' for d in decisions.values()),
            bad=sum(d['quality']=='bad' for d in decisions.values()),repaired=sum(d['quality']=='repaired' for d in decisions.values()))

    def active(self,file_id):
        r=self.flagged[file_id];decision=self.decisions().get(file_id)
        if decision and decision.get('revision'):
            return json.loads((self.output/'revisions'/file_id/decision['revision']/'crop.json').read_text())
        return r

    def media(self,file_id,mode):
        if file_id not in self.flagged or mode not in ['current','tissue']:raise ValueError('Invalid clip or mode')
        decision=self.decisions().get(file_id)
        suffix=decision['revision'] if decision and decision.get('revision') else 'base'
        key=f'{file_id}-{mode}-{suffix}';target=self.output/'media'/(key+'.mp4')
        if target.exists():return dict(status='ready',url='/media/'+target.name)
        with self.lock:
            if key not in self.jobs:
                self.jobs[key]=dict(status='working')
                self.pool.submit(self.make_media,key,file_id,mode,target,decision)
            return dict(self.jobs[key])

    def make_media(self,key,file_id,mode,target,decision):
        try:
            r=self.active(file_id);target.parent.mkdir(exist_ok=True)
            if mode=='current':
                source=Path(r['crop_output'])
                if decision and decision.get('revision'):source=self.output/'revisions'/file_id/decision['revision']/'crop.npz'
                with np.load(source) as f:frames=f['frames']
            else:
                ds=pydicom.dcmread(r['source_path']);frames,_,_=_normalize_frames(ds.pixel_array,int(ds.get('SamplesPerPixel',1)))
                x0,y0,x1,y1=tissue_box(ds)
                frames=np.stack([_resize_frame(padded_crop(f,(y0,y1,x0,x1)),518) for f in frames])
            fps=1000/r['frame_time_ms'] if r.get('frame_time_ms') else r.get('cine_rate') or 30
            mp4(frames,target,fps)
            with self.lock:self.jobs[key]=dict(status='ready',url='/media/'+target.name)
        except Exception as e:
            with self.lock:self.jobs[key]=dict(status='error',error=str(e))

    def editor(self,file_id):
        r=self.active(file_id);ds=pydicom.dcmread(r['source_path'],stop_before_pixels=True);box=tissue_box(ds)
        return dict(region=box,width=box[2]-box[0],height=box[3]-box[1])

    def editor_image(self,file_id,index):
        r=self.active(file_id)
        if not 0<=index<r['frames']:raise ValueError('Invalid frame')
        ds=pydicom.dcmread(r['source_path'],stop_before_pixels=True);x0,y0,x1,y1=tissue_box(ds)
        frame=pydicom.pixels.pixel_array(r['source_path'],index=index)[y0:y1,x0:x1]
        data=io.BytesIO();Image.fromarray(rgb(frame)).save(data,format='JPEG',quality=88)
        return data.getvalue()

    def suggestions(self,file_id):
        if file_id not in self.flagged:raise ValueError('Unknown clip')
        path=self.output/'alternatives'/(file_id+'.json')
        if path.exists():return dict(status='ready',candidates=[{k:v for k,v in r.items() if k!='record'} for r in json.loads(path.read_text())])
        key='alternatives-'+file_id
        with self.lock:
            if key not in self.jobs:
                self.jobs[key]=dict(status='working',step='Finding other cines in this visit')
                self.pool.submit(self.make_suggestions,file_id,key,path)
            return dict(self.jobs[key])

    def make_suggestions(self,file_id,key,path):
        try:
            from alternatives import ranked
            def progress(text):
                with self.lock:self.jobs[key]['step']=text
            with self.repair_lock:
                rows=ranked(self.flagged[file_id],self.records,self.output,progress)
            path.parent.mkdir(exist_ok=True);atomic_json(path,rows)
            with self.lock:self.jobs[key]=dict(status='ready',candidates=[{k:v for k,v in r.items() if k!='record'} for r in rows])
        except Exception as e:
            with self.lock:self.jobs[key]=dict(status='error',error=str(e))

    def candidate(self,file_id,candidate_id):
        if file_id not in self.flagged:raise ValueError('Unknown clip')
        path=self.output/'alternatives'/(file_id+'.json')
        if not path.exists():raise ValueError('Classify this visit before selecting a replacement')
        rows=json.loads(path.read_text())
        candidate=next((r for r in rows if r['candidate_id']==candidate_id),None)
        if candidate is None:raise ValueError('Unknown replacement candidate')
        r=candidate['record'];original=self.flagged[file_id]
        if Path(r['source_path']).parent!=Path(original['source_path']).parent or r['expected_study']!=original['expected_study'] or r['view']!=original['view']:raise ValueError('Replacement does not match this visit and view')
        return r

    def candidate_media(self,file_id,candidate_id):
        r=self.candidate(file_id,candidate_id)
        key='alternative-'+candidate_id;target=self.output/'media'/(key+'.mp4')
        if target.exists():return dict(status='ready',url='/media/'+target.name)
        with self.lock:
            if key not in self.jobs:
                self.jobs[key]=dict(status='working')
                def create():
                    try:
                        ds=pydicom.dcmread(r['source_path']);frames,_,_=_normalize_frames(ds.pixel_array,int(ds.get('SamplesPerPixel',1)))
                        x0,y0,x1,y1=tissue_box(ds);frames=np.stack([_resize_frame(padded_crop(f,(y0,y1,x0,x1)),518) for f in frames])
                        target.parent.mkdir(exist_ok=True);mp4(frames,target,1000/r['frame_time_ms'] if r.get('frame_time_ms') else r.get('cine_rate') or 30)
                        with self.lock:self.jobs[key]=dict(status='ready',url='/media/'+target.name)
                    except Exception as e:
                        with self.lock:self.jobs[key]=dict(status='error',error=str(e))
                self.pool.submit(create)
            return dict(self.jobs[key])

    def replace(self,file_id,candidate_id):
        r=self.candidate(file_id,candidate_id);job=uuid.uuid4().hex
        with self.lock:self.jobs[job]=dict(status='working',step='Preparing approved replacement')
        self.pool.submit(self.make_repair,job,file_id,None,r)
        return dict(job=job)

    def repair(self,file_id,rect):
        if file_id not in self.flagged:raise ValueError('Unknown review clip')
        region=self.editor(file_id)['region'];selected_box(region,rect)
        job=uuid.uuid4().hex
        with self.lock:self.jobs[job]=dict(status='working',step='Waiting for repair worker',file_id=file_id)
        self.pool.submit(self.make_repair,job,file_id,rect)
        return dict(job=job)

    def make_repair(self,job,file_id,rect,replacement=None):
        def step(text):
            with self.lock:self.jobs[job]['step']=text
        try:
            # Serialize GPU work; each new revision has independent files.
            with self.repair_lock:
                r=dict(self.active(file_id));step('Restoring pixels from original DICOM')
                if replacement:
                    r.update(replacement);r['source_sha256']=digest(r['source_path']);r['alternative_source']=True
                if digest(r['source_path'])!=r['source_sha256']:raise ValueError('Source hash changed; repair stopped')
                ds=pydicom.dcmread(r['source_path']);frames,n,_=_normalize_frames(ds.pixel_array,int(ds.get('SamplesPerPixel',1)))
                r.update(source_sop=str(ds.SOPInstanceUID),source_study=str(ds.StudyInstanceUID),
                    photometric_interpretation=str(ds.PhotometricInterpretation),
                    frame_time_vector_ms=[float(v) for v in ds.get('FrameTimeVector',[])],
                    heart_rate=float(ds.HeartRate) if ds.get('HeartRate') is not None else None)
                if n!=r['frames']:raise ValueError('Source frame count changed')
                x0,y0,x1,y1=selected_box(tissue_box(ds),rect)
                output=np.stack([_resize_frame(padded_crop(f,(y0,y1,x0,x1)),518) for f in frames])
                if output.shape[-1]==1:output=output[...,0]
                revision=self.output/'revisions'/file_id/job;revision.mkdir(parents=True)
                path=revision/'crop.npz';np.savez_compressed(path,frames=output,source_frame_indices=np.arange(n,dtype=np.int32))
                with np.load(path) as saved:
                    if not np.array_equal(saved['frames'],output):raise ValueError('Repair cache round-trip failed')
                r.update(crop_output=str(path),method='reviewed_tissue_rectangle_no_component_suppression',
                    bbox_y0_y1_x0_x1=[y0,y1,x0,x1],output_shape=list(output.shape),status='processed',qc_flags=[],
                    repair_of=str(self.flagged[file_id]['crop_output']),repair_revision=job,normalized_rect=rect,
                    review_note='Rectangle selected by reviewer; full cine retained; component suppression disabled')
                atomic_json(path.with_suffix('.json'),r)
                from ichilov3_encode_selected import load_encoder, encode
                import torch
                artifacts=PROJECT/'output/dicom_prediction'
                args=argparse.Namespace(output=revision)
                torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
                for name in ['echoprime','panecho']:
                    step('Re-encoding '+name)
                    wp=artifacts/'weights'/('echo_prime_encoder.pt' if name=='echoprime' else 'panecho.pt')
                    model=load_encoder(name,artifacts,device);encode(r,name,model,args,digest(wp),device)
                    del model
                    if device=='cuda':torch.cuda.empty_cache()
                atomic_json(revision/'repair_complete.json',dict(file_id=file_id,revision=job,rectangle=[x0,y0,x1,y1],source_sha256=r['source_sha256']))
                self.decide(file_id,'repaired',job)
                with self.lock:self.jobs[job]=dict(status='ready',file_id=file_id,revision=job)
        except Exception as e:
            with self.lock:self.jobs[job]=dict(status='error',error=f'{type(e).__name__}: {e}')


class Handler(BaseHTTPRequestHandler):
    def send(self,body,kind='application/json',status=200,extra=None):
        if not isinstance(body,bytes):body=json.dumps(body).encode()
        self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(body)))
        self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
        for k,v in (extra or {}).items():self.send_header(k,v)
        self.end_headers();self.wfile.write(body)

    def file(self,path):
        size=path.stat().st_size;start=0;end=size-1;status=200
        request=self.headers.get('Range')
        if request:
            import re
            m=re.fullmatch(r'bytes=(\d+)-(\d*)',request)
            if not m:return self.send(b'Invalid range','text/plain',416)
            start=int(m[1]);end=min(int(m[2]) if m[2] else end,end);status=206
            if start>end:return self.send(b'Invalid range','text/plain',416)
        self.send_response(status);self.send_header('Content-Type',mimetypes.guess_type(str(path))[0] or 'application/octet-stream')
        self.send_header('Content-Length',str(end-start+1));self.send_header('Accept-Ranges','bytes');self.send_header('Cache-Control','private, max-age=3600')
        if status==206:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
        self.end_headers()
        with path.open('rb') as f:
            f.seek(start);remaining=end-start+1
            while remaining:
                block=f.read(min(65536,remaining));self.wfile.write(block);remaining-=len(block)

    def do_GET(self):
        try:
            path=urlsplit(self.path).path;parts=path.strip('/').split('/');store=self.server.store
            if path=='/api/state':return self.send(store.state())
            if len(parts)==4 and parts[:2]==['api','media']:return self.send(store.media(parts[2],parts[3]))
            if len(parts)==3 and parts[:2]==['api','editor']:return self.send(store.editor(parts[2]))
            if len(parts)==3 and parts[:2]==['api','alternatives']:return self.send(store.suggestions(parts[2]))
            if len(parts)==4 and parts[:2]==['api','alternative_media']:return self.send(store.candidate_media(parts[2],parts[3]))
            if len(parts)==4 and parts[:2]==['api','frame']:return self.send(store.editor_image(parts[2],int(parts[3])),'image/jpeg')
            if len(parts)==3 and parts[:2]==['api','job']:return self.send(store.jobs.get(parts[2],dict(status='error',error='Unknown job')))
            if path=='/api/export':
                return self.send(store.export_manifest(),extra={'Content-Disposition':'attachment; filename="reviewed_crop_manifest.json"'})
            if len(parts)==2 and parts[0]=='media' and Path(parts[1]).name==parts[1]:return self.file(store.output/'media'/parts[1])
            if path in ['/','/index.html','/app.js','/cine-buffer.js','/style.css']:
                file=ROOT/('index.html' if path=='/' else path[1:]);return self.send(file.read_bytes(),mimetypes.guess_type(str(file))[0])
            self.send(dict(error='Not found'),status=404)
        except (BrokenPipeError,ConnectionResetError):pass
        except Exception as e:self.send(dict(error=str(e)),status=400)

    def do_POST(self):
        try:
            origin=self.headers.get('Origin')
            if origin and urlsplit(origin).netloc!=self.headers.get('Host'):raise ValueError('Cross-origin write rejected')
            if not self.headers.get('Content-Type','').startswith('application/json'):raise ValueError('JSON required')
            length=int(self.headers.get('Content-Length',0))
            if not 0<length<10000:raise ValueError('Invalid body size')
            body=json.loads(self.rfile.read(length));store=self.server.store
            if self.path=='/api/decision':return self.send(store.decide(body['file_id'],body.get('quality'),body.get('revision'),body.get('note','')))
            if self.path=='/api/repair':return self.send(store.repair(body['file_id'],body.get('rect')))
            if self.path=='/api/replace':return self.send(store.replace(body['file_id'],body['candidate_id']))
            self.send(dict(error='Not found'),status=404)
        except Exception as e:self.send(dict(error=str(e)),status=400)


def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8770)
    p.add_argument('--output',type=Path,default=Path(r'D:\DS\ichilov3_crop_review'))
    args=p.parse_args();store=Store(args.output)
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler);server.store=store
    print(f'Crop reviewer: http://127.0.0.1:{args.port}, {len(store.flagged)} clips',flush=True)
    server.serve_forever()


if __name__=='__main__':main()
