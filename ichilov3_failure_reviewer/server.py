"""Read-only cine review of held-out GLS prediction failures.

Only review decisions and derived MP4 previews are written under the output
directory. Source DICOMs, crops, predictions and prior reviews are untouched.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import mimetypes
import hashlib
import shutil
import sqlite3
import subprocess
import sys
import threading
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import imageio_ffmpeg
import numpy as np
import pandas as pd
import pydicom
from PIL import Image


ROOT = Path(__file__).resolve().parent
DATA = Path(r'D:\DS\ichilov3_temporal_trial_20260927\vendor_failure_analysis')
MANIFEST = Path(r'D:\DS\ichilov3_stage2_padded_20260926\full_manifest.json')
OLD_VIEW_CACHE = [Path(r'D:\us\output\dicom_prediction\ichilov3_manual_selection'),
                  Path(r'D:\DS\ichilov3_crop_review')]
VALID_STATUS = {'no_issue', 'suspected_issue', 'uncertain'}
VALID_REASONS = {'wrong_view', 'crop_or_anatomy', 'timing_or_motion', 'image_quality',
                 'report_or_label', 'wrong_study', 'other'}


def encode_mp4(frames: np.ndarray, target: Path, fps: float):
    height, width = frames.shape[1:3]
    if width % 2 or height % 2:
        raise ValueError('MP4 dimensions must be even')
    temp = target.with_suffix('.partial.mp4')
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error',
               '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', f'{width}x{height}',
               '-framerate', str(min(max(fps, 5), 120)), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
               '-preset', 'veryfast', '-crf', '23', '-pix_fmt', 'yuv420p',
               '-movflags', '+faststart', '-y', str(temp)]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        for frame in frames:
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        proc.stdin.close()
        error = proc.stderr.read().decode(errors='replace')
        if proc.wait():
            raise RuntimeError(error[-800:])
        temp.replace(target)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        if temp.exists():
            temp.unlink()


class Store:
    def __init__(self, out=DATA, manifest_path=MANIFEST, predictions_path=None):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.media_dir = self.out / 'review_media'
        self.media_dir.mkdir(exist_ok=True)
        self.db = self.out / 'cine_reviews.sqlite3'
        self.lock = threading.RLock()
        self.pool = ThreadPoolExecutor(max_workers=2)
        self.rank_pool = ThreadPoolExecutor(max_workers=1)
        self.jobs = {}
        self.candidate_dir = self.out / 'replacement_candidates'
        self.candidate_dir.mkdir(exist_ok=True)
        self.records = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
        self.by_file = {r['file_id']: r for r in self.records}
        assert len(self.by_file) == len(self.records)
        self.by_visit = {}
        for record in self.records:
            self.by_visit.setdefault((record['patient'], str(record['visit_date'])), []).append(record)
        predictions_path = Path(predictions_path or self.out / 'prediction_points.csv')
        pred = pd.read_csv(predictions_path)
        endo = pred[pred.target == 'endo_gls'].set_index('visit_id')
        pred = pred[pred.target == 'mid_gls'].copy()
        assert pred.visit_id.is_unique
        self.visits = []
        for row in pred.itertuples(index=False):
            clips = self.by_visit.get((row.patient_id, str(row.visit_date)))
            if not clips or {c['view'] for c in clips} != {'A2C', 'A3C', 'A4C'}:
                raise ValueError(f'Missing three cines for {row.visit_id}')
            self.visits.append(dict(visit_id=row.visit_id, patient_id=row.patient_id,
                visit_date=str(row.visit_date), manufacturer=row.manufacturer,
                vendor=row.vendor_short, gt=float(row.value), prediction=float(row.prediction),
                signed_error=float(row.error), absolute_error=float(row.absolute_error),
                endo_gt=float(endo.loc[row.visit_id,'value']),
                endo_prediction=float(endo.loc[row.visit_id,'prediction']),
                endo_absolute_error=float(endo.loc[row.visit_id,'absolute_error']),
                all_bookmark=bool(row.all_bookmark),
                clips=[dict(file_id=c['file_id'], view=c['view'], frames=int(c['frames']),
                            model=c.get('model'), selection_source=c['selection_source'],
                            frame_time_ms=c.get('frame_time_ms'),
                            source_name=Path(c['source_path']).name,
                            crop_flagged=c.get('status') == 'needs_review')
                       for c in sorted(clips, key=lambda c: c['view'])]))
        self.visits.sort(key=lambda x: (-x['absolute_error'], x['visit_id']))
        self.visit_ids = {v['visit_id'] for v in self.visits}
        with self.connect() as con:
            con.execute('CREATE TABLE IF NOT EXISTS reviews (visit_id TEXT PRIMARY KEY, status TEXT NOT NULL, reasons TEXT NOT NULL, note TEXT NOT NULL, updated TEXT NOT NULL)')
            con.execute('CREATE TABLE IF NOT EXISTS review_history (event INTEGER PRIMARY KEY AUTOINCREMENT, visit_id TEXT, before_json TEXT, after_json TEXT, updated TEXT)')
            con.execute('CREATE TABLE IF NOT EXISTS replacements (visit_id TEXT NOT NULL, view TEXT NOT NULL, original_file_id TEXT NOT NULL, candidate_id TEXT NOT NULL, source_path TEXT NOT NULL, source_sop TEXT NOT NULL, score REAL NOT NULL, updated TEXT NOT NULL, PRIMARY KEY (visit_id,view))')
            con.execute('CREATE TABLE IF NOT EXISTS replacement_history (event INTEGER PRIMARY KEY AUTOINCREMENT, visit_id TEXT, view TEXT, before_json TEXT, after_json TEXT, updated TEXT)')

    @contextmanager
    def connect(self):
        con = sqlite3.connect(self.db, timeout=20)
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def reviews(self):
        with self.connect() as con:
            rows = con.execute('SELECT visit_id,status,reasons,note,updated FROM reviews').fetchall()
        return {r[0]: dict(status=r[1], reasons=json.loads(r[2]), note=r[3], updated=r[4]) for r in rows}

    def replacements(self):
        with self.connect() as con:
            rows=con.execute('SELECT visit_id,view,original_file_id,candidate_id,source_path,source_sop,score,updated FROM replacements').fetchall()
        result={}
        for row in rows:
            result.setdefault(row[0],{})[row[1]]=dict(original_file_id=row[2],candidate_id=row[3],
                source_path=row[4],source_sop=row[5],score=row[6],updated=row[7],
                source_name=Path(row[4]).name)
        return result

    def state(self):
        reviews = self.reviews()
        replacements=self.replacements()
        return {'visits': [{**v, 'review': reviews.get(v['visit_id']),
                            'replacements':replacements.get(v['visit_id'],{})} for v in self.visits],
                'counts': {'visits': len(self.visits), 'philips': sum(v['vendor']=='Philips' for v in self.visits),
                           'reviewed': len(reviews)}, 'reason_options': sorted(VALID_REASONS)}

    def save_review(self, payload):
        visit_id = payload.get('visit_id')
        status = payload.get('status')
        reasons = payload.get('reasons', [])
        note = payload.get('note', '')
        if visit_id not in self.visit_ids or status not in VALID_STATUS:
            raise ValueError('Invalid visit or status')
        if not isinstance(reasons, list) or len(reasons) > len(VALID_REASONS) or set(reasons) - VALID_REASONS:
            raise ValueError('Invalid reasons')
        if not isinstance(note, str) or len(note) > 2000:
            raise ValueError('Invalid note')
        if status == 'no_issue' and reasons:
            raise ValueError('Clear reasons when marking no issue')
        if status == 'no_issue' and self.replacements().get(visit_id):
            raise ValueError('Revert replacement cines before marking no visible issue')
        updated = datetime.now(timezone.utc).isoformat()
        after = dict(status=status, reasons=sorted(set(reasons)), note=note, updated=updated)
        with self.lock, self.connect() as con:
            row = con.execute('SELECT status,reasons,note,updated FROM reviews WHERE visit_id=?',(visit_id,)).fetchone()
            before = None if not row else dict(status=row[0], reasons=json.loads(row[1]), note=row[2], updated=row[3])
            con.execute('INSERT OR REPLACE INTO reviews VALUES (?,?,?,?,?)',
                        (visit_id, status, json.dumps(after['reasons']), note, updated))
            con.execute('INSERT INTO review_history (visit_id,before_json,after_json,updated) VALUES (?,?,?,?)',
                        (visit_id,json.dumps(before),json.dumps(after),updated))
        return after

    def export_csv(self):
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['visit_id','patient_id','visit_date','vendor','mid_gls_gt','mid_gls_prediction',
                         'absolute_error','status','reasons','note','updated',
                         'replacement_A2C_path','replacement_A2C_sop','replacement_A3C_path',
                         'replacement_A3C_sop','replacement_A4C_path','replacement_A4C_sop'])
        reviews = self.reviews()
        replacements=self.replacements()
        for v in self.visits:
            r = reviews.get(v['visit_id'], {})
            repl=replacements.get(v['visit_id'],{})
            writer.writerow([v['visit_id'],v['patient_id'],v['visit_date'],v['vendor'],v['gt'],
                             v['prediction'],v['absolute_error'],r.get('status',''),
                             ';'.join(r.get('reasons',[])),r.get('note',''),r.get('updated',''),
                             *[value for view in ['A2C','A3C','A4C'] for value in
                               [repl.get(view,{}).get('source_path',''),repl.get(view,{}).get('source_sop','')]]])
        return output.getvalue().encode('utf-8-sig')

    def selected_record(self,visit_id,view):
        visit=next((v for v in self.visits if v['visit_id']==visit_id),None)
        if visit is None or view not in {'A2C','A3C','A4C'}: raise ValueError('Unknown visit or view')
        file_id=next(c['file_id'] for c in visit['clips'] if c['view']==view)
        return self.by_file[file_id]

    def candidate_cache(self,record):
        return self.candidate_dir / f"{record['file_id']}.json"

    def alternatives(self,visit_id,view):
        record=self.selected_record(visit_id,view)
        cache=self.candidate_cache(record)
        if cache.is_file():
            try:
                rows=json.loads(cache.read_text(encoding='utf-8'))
                return {'status':'ready','candidates':self.public_candidates(rows),'current':Path(record['source_path']).name}
            except (ValueError,OSError): pass
        key=('rank',record['file_id'])
        with self.lock:
            if key not in self.jobs:
                self.jobs[key]={'status':'working','step':'Ranking cines in the same study'}
                self.rank_pool.submit(self.make_alternatives,record,key,cache)
            return dict(self.jobs[key])

    @staticmethod
    def public_candidates(rows):
        return [dict(candidate_id=r['candidate_id'],rank=i,name=r['name'],score=r['score'],
                     view_probability=r['probability'],frames=r['record']['frames'],
                     duration_seconds=r['quality']['duration_seconds'],warnings=r['warnings'])
                for i,r in enumerate(rows,1)]

    def make_alternatives(self,record,key,cache):
        try:
            # Reuse immutable classifier predictions if present. Missing cines
            # are classified into this app's own cache, without editing old work.
            sys.path.insert(0,str(ROOT.parent))
            from ichilov3_view_selector.suggestions import Suggester
            from ichilov3_crop_reviewer.alternatives import ranked
            suggester=Suggester(self.out)
            dest=self.out/'view_predictions';dest.mkdir(exist_ok=True)
            for source in Path(record['source_path']).parent.iterdir():
                if not source.is_file(): continue
                item={'id':hashlib.sha256(str(source).casefold().encode()).hexdigest()[:24],
                      'path':str(source)}
                destination=suggester._cache_path(item)
                if destination.exists(): continue
                for old in OLD_VIEW_CACHE:
                    cached=old/'view_predictions'/destination.name
                    if cached.is_file():
                        shutil.copy2(cached,destination)
                        break
            def progress(message):
                with self.lock:self.jobs[key]['step']=message
            rows=ranked(record,self.records,self.out,progress)
            temp=cache.with_suffix('.tmp')
            temp.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
            temp.replace(cache)
            with self.lock:self.jobs[key]={'status':'ready','candidates':self.public_candidates(rows),
                                           'current':Path(record['source_path']).name}
        except Exception as exc:
            with self.lock:self.jobs[key]={'status':'error','error':f'{type(exc).__name__}: {exc}'}

    def candidate(self,visit_id,view,candidate_id):
        record=self.selected_record(visit_id,view)
        cache=self.candidate_cache(record)
        if not cache.is_file(): raise ValueError('Rank alternatives first')
        rows=json.loads(cache.read_text(encoding='utf-8'))
        candidate=next((r for r in rows if r['candidate_id']==candidate_id),None)
        if candidate is None:raise ValueError('Unknown candidate')
        alternative=candidate['record']
        if (Path(alternative['source_path']).resolve().parent!=Path(record['source_path']).resolve().parent
            or alternative['expected_study']!=record['expected_study'] or alternative['view']!=view):
            raise ValueError('Candidate is not from this study and view')
        source=Path(alternative['source_path'])
        if not source.is_file():raise ValueError('Candidate DICOM is missing')
        ds=pydicom.dcmread(source,stop_before_pixels=True)
        if str(ds.SOPInstanceUID)!=alternative['expected_sop'] or str(ds.StudyInstanceUID)!=record['expected_study']:
            raise ValueError('Candidate DICOM identity has changed')
        return candidate

    def set_replacement(self,visit_id,view,candidate_id):
        record=self.selected_record(visit_id,view)
        candidate=self.candidate(visit_id,view,candidate_id) if candidate_id else None
        updated=datetime.now(timezone.utc).isoformat()
        with self.lock,self.connect() as con:
            old=con.execute('SELECT candidate_id,source_path,source_sop,score FROM replacements WHERE visit_id=? AND view=?',(visit_id,view)).fetchone()
            before=None if old is None else dict(candidate_id=old[0],source_path=old[1],source_sop=old[2],score=old[3])
            if candidate:
                alt=candidate['record']
                duplicate=con.execute('SELECT view FROM replacements WHERE visit_id=? AND view!=? AND source_sop=?',
                                      (visit_id,view,alt['expected_sop'])).fetchone()
                if duplicate:raise ValueError('Candidate already replaces another view')
                review=con.execute('SELECT status FROM reviews WHERE visit_id=?',(visit_id,)).fetchone()
                if not review or review[0]!='suspected_issue':raise ValueError('Save Suspected issue review first')
                con.execute('INSERT OR REPLACE INTO replacements VALUES (?,?,?,?,?,?,?,?)',
                            (visit_id,view,record['file_id'],candidate_id,alt['source_path'],
                             alt['expected_sop'],float(candidate['score']),updated))
                after=dict(candidate_id=candidate_id,source_path=alt['source_path'],
                           source_name=Path(alt['source_path']).name,source_sop=alt['expected_sop'],
                           score=float(candidate['score']),original_file_id=record['file_id'],updated=updated)
            else:
                con.execute('DELETE FROM replacements WHERE visit_id=? AND view=?',(visit_id,view))
                after=None
            con.execute('INSERT INTO replacement_history (visit_id,view,before_json,after_json,updated) VALUES (?,?,?,?,?)',
                        (visit_id,view,json.dumps(before),json.dumps(after),updated))
        return after

    def alternative_media_status(self,visit_id,view,candidate_id):
        candidate=self.candidate(visit_id,view,candidate_id)
        target=self.media_dir/f'{candidate_id}-alternative.mp4'
        if target.exists() and target.stat().st_size:
            return {'status':'ready','url':f'/media/{target.name}'}
        key=('alternative_media',candidate_id)
        with self.lock:
            if key not in self.jobs:
                self.jobs[key]={'status':'working'}
                self.pool.submit(self.make_media,candidate['record'],key,target,False)
            return dict(self.jobs[key])

    def media_status(self, file_id, mode):
        if file_id not in self.by_file or mode not in {'crop','source'}:
            raise ValueError('Unknown clip or mode')
        target = self.media_dir / f'{file_id}-{mode}.mp4'
        if target.exists() and target.stat().st_size:
            return {'status':'ready', 'url':f'/media/{target.name}'}
        key = (file_id,mode)
        with self.lock:
            if key not in self.jobs:
                self.jobs[key] = {'status':'working'}
                self.pool.submit(self.make_media, self.by_file[file_id], key, target, mode=='crop')
            return dict(self.jobs[key])

    def make_media(self, record, key, target, crop):
        try:
            if crop:
                with np.load(record['crop_output']) as data:
                    frames = data['frames']
            else:
                ds = pydicom.dcmread(record['source_path'])
                frames = ds.pixel_array
                if frames.ndim == 3: frames = np.repeat(frames[...,None],3,axis=-1)
                if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
                    raise ValueError('Unsupported source pixel layout')
                # Full original frame, including everything outside the crop.
                height,width = frames.shape[1:3]
                scale = min(640 / width, 640 / height, 1)
                size = (max(2,round(width*scale/2)*2), max(2,round(height*scale/2)*2))
                frames = np.stack([np.asarray(Image.fromarray(f).resize(size,Image.Resampling.BILINEAR)) for f in frames])
            fps = 1000/record['frame_time_ms'] if record.get('frame_time_ms') else record.get('cine_rate') or 30
            encode_mp4(frames, target, fps)
            with self.lock: self.jobs[key] = {'status':'ready','url':f'/media/{target.name}'}
        except Exception as exc:
            with self.lock: self.jobs[key] = {'status':'error','error':str(exc)}


class Handler(BaseHTTPRequestHandler):
    store: Store
    def log_message(self, fmt, *args):
        print(f'{self.address_string()} {fmt % args}', flush=True)

    def send_bytes(self, body, content_type, code=200, extra=None):
        self.send_response(code)
        self.send_header('Content-Type',content_type)
        self.send_header('Content-Length',str(len(body)))
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        for k,v in (extra or {}).items(): self.send_header(k,v)
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, value, code=200):
        self.send_bytes(json.dumps(value,ensure_ascii=False).encode('utf-8'),'application/json; charset=utf-8',code)

    def do_GET(self):
        parsed=urlsplit(self.path)
        path=parsed.path
        try:
            if path=='/api/state': return self.send_json(self.store.state())
            if path=='/api/export':
                return self.send_bytes(self.store.export_csv(),'text/csv; charset=utf-8',extra={'Content-Disposition':'attachment; filename="ichilov3_cine_failure_reviews.csv"'})
            if path.startswith('/api/alternatives/'):
                parts=path.split('/')
                if len(parts)!=5:raise ValueError('Invalid alternatives request')
                return self.send_json(self.store.alternatives(parts[3],parts[4]))
            if path.startswith('/api/alternative_media/'):
                parts=path.split('/')
                if len(parts)!=6:raise ValueError('Invalid alternative media request')
                return self.send_json(self.store.alternative_media_status(parts[3],parts[4],parts[5]))
            if path.startswith('/api/media/'):
                parts=path.split('/')
                if len(parts)!=5: raise ValueError('Invalid media request')
                return self.send_json(self.store.media_status(parts[3],parts[4]))
            if path.startswith('/media/'):
                name=path.removeprefix('/media/')
                if not name.endswith('.mp4') or '/' in name or '\\' in name or '..' in name:
                    raise ValueError('Invalid media file')
                file=self.store.media_dir/name
                if not file.is_file(): return self.send_json({'error':'Media is not ready'},404)
                total=file.stat().st_size
                start,end=0,total-1
                partial=False
                request=self.headers.get('Range')
                if request:
                    import re
                    match=re.fullmatch(r'bytes=(\d+)-(\d*)',request.strip())
                    if not match: return self.send_json({'error':'Invalid range'},416)
                    start=int(match.group(1));end=min(int(match.group(2)),end) if match.group(2) else end
                    if start> end: return self.send_json({'error':'Invalid range'},416)
                    partial=True
                length=end-start+1
                self.send_response(206 if partial else 200)
                self.send_header('Content-Type','video/mp4')
                self.send_header('Content-Length',str(length))
                self.send_header('Accept-Ranges','bytes')
                self.send_header('Cache-Control','private, max-age=86400')
                if partial: self.send_header('Content-Range',f'bytes {start}-{end}/{total}')
                self.end_headers()
                with file.open('rb') as stream:
                    stream.seek(start)
                    while length:
                        chunk=stream.read(min(length,1024*1024))
                        if not chunk: break
                        try: self.wfile.write(chunk)
                        except (BrokenPipeError,ConnectionResetError): break
                        length-=len(chunk)
                return
            static={'/':'index.html','/index.html':'index.html','/app.js':'app.js','/style.css':'style.css'}
            if path in static:
                file=ROOT/static[path]
                return self.send_bytes(file.read_bytes(),mimetypes.guess_type(file.name)[0] or 'text/plain')
            self.send_json({'error':'Not found'},404)
        except ValueError as exc: self.send_json({'error':str(exc)},400)
        except Exception as exc: self.send_json({'error':str(exc)},500)

    def do_POST(self):
        path=urlsplit(self.path).path
        if path not in {'/api/review','/api/replacement'}: return self.send_json({'error':'Not found'},404)
        try:
            length=int(self.headers.get('Content-Length','0'))
            if length<2 or length>10000: raise ValueError('Invalid request size')
            payload=json.loads(self.rfile.read(length))
            if path=='/api/review':self.send_json({'review':self.store.save_review(payload)})
            else:self.send_json({'replacement':self.store.set_replacement(payload.get('visit_id'),
                                       payload.get('view'),payload.get('candidate_id'))})
        except (ValueError,TypeError,json.JSONDecodeError) as exc: self.send_json({'error':str(exc)},400)
        except Exception as exc: self.send_json({'error':str(exc)},500)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=8771)
    args=parser.parse_args()
    Handler.store=Store()
    server=ThreadingHTTPServer((args.host,args.port),Handler)
    print(f'Ichilov3 failure reviewer: http://{args.host}:{args.port}/ ; {len(Handler.store.visits)} visits',flush=True)
    server.serve_forever()


if __name__=='__main__': main()
