const {test}=require('node:test');
const assert=require('node:assert/strict');
const CineBuffer=require('./cine-buffer.js');
const clip=(id,revision)=>({file_id:id,decision:revision?{revision}:null});

test('buffer deduplicates downloads and retains the next three clips',async()=>{
 const calls=[];const revoked=[];
 const buffer=new CineBuffer(async(c)=>{calls.push(c.file_id);return {url:'/media/'+c.file_id,blob:new Blob(['cine'])};},{createURL:()=>`blob:${calls.length}`,revokeURL:u=>revoked.push(u)});
 const clips=[1,2,3,4].map(id=>clip(id));buffer.retain(clips,clips[0]);
 const first=buffer.get(clips[0]);assert.equal(first,buffer.get(clips[0]));await first;
 await buffer.warm(clips.slice(1));assert.equal(calls.length,4);assert.equal(buffer.ready(clips),4);
 buffer.retain(clips.slice(1),clips[1]);assert.equal(buffer.entries.size,3);assert.equal(revoked.length,1);
});
test('revision changes evict stale media and oversized cines stream normally',async()=>{
 const revoked=[];
 const buffer=new CineBuffer(async c=>({url:'/media/'+(c.decision?.revision||'base'),blob:new Blob(['cine'])}),{createURL:()=> 'blob:base',revokeURL:u=>revoked.push(u)});
 const original=clip('a');buffer.retain([original],original);await buffer.get(original);
 const repaired=clip('a','new');buffer.retain([repaired],repaired);assert.deepEqual(revoked,['blob:base']);assert.equal(buffer.entries.size,0);
 const large=new CineBuffer(async()=>({url:'/large',blob:{size:17*1024*1024}}),{createURL:()=>{throw Error('too large');}});
 assert.equal(await large.get(original),'/large');
});
test('navigation cancels obsolete downloads without retaining object URLs',async()=>{
 let signal;let complete;
 const buffer=new CineBuffer((c,m,s)=>{signal=s;return new Promise(r=>complete=r);},{createURL:()=>{throw Error('cancelled cine should not be retained');}});
 const original=clip('a');buffer.retain([original],original);const loading=buffer.get(original);
 buffer.retain([],null);assert.equal(signal.aborted,true);complete({blob:new Blob(['cine'])});
 await assert.rejects(loading,{name:'AbortError'});assert.equal(buffer.entries.size,0);
});
