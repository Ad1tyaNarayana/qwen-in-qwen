"""Offline explorer server. Local private bundles are mounted only under /recordings/."""
import argparse
import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit
from common import HERE, ROOT

class Handler(SimpleHTTPRequestHandler):
    recordings = HERE/'data/recordings'
    def translate_path(self,path):
        route=unquote(urlsplit(path).path)
        if route.startswith('/recordings/'):
            root=self.recordings.resolve()
            candidate=(root/route.removeprefix('/recordings/')).resolve()
            if candidate!=root and root not in candidate.parents:
                return str(HERE/'__denied__')
            return str(candidate)
        return super().translate_path(path)
    def end_headers(self):
        self.send_header('Cache-Control','no-cache')
        self.send_header('X-Content-Type-Options','nosniff')
        super().end_headers()
    def list_directory(self,path):
        self.send_error(403,'Directory listing disabled')
        return None
    def do_GET(self):
        if urlsplit(self.path).path=='/api/manifests':
            raw=json.dumps(manifest_entries()).encode()
            self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
        else: super().do_GET()

def manifest_entries():
    directory=ROOT/'experiments/showcase/manifest'
    entries=[]
    # Gallery order: the tiny Qwen3.5 first, other programs on this network next, older-network demos last.
    def rank(p):
        return ({'tiny-qwen35':0,'stories260k-38':1}.get(p.stem, 3 if p.stem.startswith(('doom','phase1')) else 2), p.name)
    for p in sorted(directory.glob('*.json'), key=rank) if directory.exists() else []:
        try:
            value=json.loads(p.read_text())
            entries.append(dict(file=p.name,manifest=value))
        except (ValueError,OSError):
            continue
    return entries

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port',type=int,default=8792)
    p.add_argument('--recordings',type=Path,default=HERE/'data/recordings')
    p.add_argument('--write-static',action='store_true',help='write data/manifests.json for static hosting (GitHub Pages) and exit')
    a=p.parse_args()
    if a.write_static:
        out=HERE/'data/manifests.json'
        out.write_text(json.dumps(manifest_entries(),indent=1)+'\n')
        print(f'wrote {out.relative_to(ROOT)}');raise SystemExit
    Handler.recordings=a.recordings
    server=ThreadingHTTPServer(('127.0.0.1',a.port),partial(Handler,directory=str(HERE)))
    print(f'Qwenputer Explorer http://127.0.0.1:{a.port}',flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
