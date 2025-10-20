# C:\Users\nuri\Desktop\RAG\ai-stack\bridge\bridge.py
import os, time, pathlib, requests, sys
RAG = os.environ.get('RAG_PROXY','http://rag-proxy:8080').rstrip('/')
WATCH = os.environ.get('WATCH_DIR','/data/uploads')
EXTS = {e.lower() for e in os.environ.get('EXT_FILTER','pdf,pptx,xlsx,txt,csv,md').split(',') if e}
POLL = float(os.environ.get('POLL_SEC','2'))
seen=set()
def stable(p):
    try:
        s1=os.path.getsize(p); time.sleep(0.5); s2=os.path.getsize(p)
        return s1==s2
    except:
        return False
def ingest(p):
    fn=pathlib.Path(p).name
    with open(p,'rb') as fh:
        r=requests.post(f'{RAG}/ingest', files={'file':(fn, fh)}, data={'overwrite':'true'}, timeout=600)
    if r.ok:
        print(f'[ingest ok] {fn}', flush=True); return True
    print(f'[ingest fail] {fn}: {r.status_code} {r.text[:200]}', flush=True); return False
print(f'watching {WATCH} ...', flush=True)
while True:
    for root, _, files in os.walk(WATCH):
        for f in files:
            p=os.path.join(root,f)
            if p in seen: continue
            ext=f.rsplit('.',1)[-1].lower() if '.' in f else ''
            if EXTS and ext not in EXTS: continue
            if not stable(p): continue
            try:
                if ingest(p): seen.add(p)
            except Exception as e:
                print('[error]', e, file=sys.stderr, flush=True)
    time.sleep(POLL)