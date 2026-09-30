// Keep only the visible cine and a small look-ahead window in browser memory.
class CineBuffer {
  constructor(load, {createURL = b => URL.createObjectURL(b), revokeURL = u => URL.revokeObjectURL(u), onChange = () => {}} = {}) {
    this.load = load; this.createURL = createURL; this.revokeURL = revokeURL;
    this.onChange = onChange; this.entries = new Map(); this.wanted = new Map();
  }
  key(c, mode = 'current') { return `${c.file_id}/${mode}/${c.decision?.revision || 'base'}`; }
  retain(clips, active, mode = 'current') {
    const wanted = new Map(clips.map(c => [this.key(c), {c, mode:'current'}]));
    if(active) wanted.set(this.key(active, mode), {c:active, mode});
    this.wanted = wanted;
    for(const [key, entry] of this.entries) if(!wanted.has(key)) {
      entry.controller.abort(); if(entry.objectURL) this.revokeURL(entry.objectURL);
      this.entries.delete(key);
    }
    this.onChange();
  }
  get(c, mode = 'current') {
    const key = this.key(c, mode);
    if(this.entries.has(key)) return this.entries.get(key).promise;
    const entry = {controller:new AbortController(), ready:false};
    this.entries.set(key, entry);
    entry.promise = this.load(c, mode, entry.controller.signal).then(media => {
      if(entry.controller.signal.aborted) throw new DOMException('Cancelled', 'AbortError');
      // Five entries at most; oversized files use the normal browser streaming cache.
      entry.objectURL = media.blob && media.blob.size <= 16*1024*1024 ? this.createURL(media.blob) : null;
      entry.ready = true; this.onChange(); return entry.objectURL || media.url;
    }).catch(error => {
      if(this.entries.get(key) === entry) this.entries.delete(key);
      this.onChange(); throw error;
    });
    return entry.promise;
  }
  async warm(clips) {
    // Two workers avoid competing with the visible cine for every connection.
    const pending = clips.slice();
    const worker = async () => {
      while(pending.length) {
        const c = pending.shift(); if(!this.wanted.has(this.key(c))) continue;
        try { await this.get(c); } catch(_) { /* Foreground loading can retry. */ }
      }
    };
    await Promise.all([worker(), worker()]);
  }
  ready(clips) { return clips.filter(c => this.entries.get(this.key(c))?.ready).length; }
}
if(typeof module !== 'undefined') module.exports = CineBuffer;
