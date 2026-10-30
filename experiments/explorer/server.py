"""Offline explorer server. Local private bundles are mounted only under /recordings/."""
import argparse
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
            import json
            directory=ROOT/'experiments/showcase/manifest'
            entries=[]
            for p in sorted(directory.glob('*.json')) if directory.exists() else []:
                try:
                    value=json.loads(p.read_text())
                    entries.append(dict(file=p.name,manifest=value))
                except (ValueError,OSError):
                    continue
            raw=json.dumps(entries).encode()
            self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
        else: super().do_GET()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port',type=int,default=8792)
    p.add_argument('--recordings',type=Path,default=HERE/'data/recordings')
    a=p.parse_args()
    Handler.recordings=a.recordings
    server=ThreadingHTTPServer(('127.0.0.1',a.port),partial(Handler,directory=str(HERE)))
    print(f'Qwenputer Explorer http://127.0.0.1:{a.port}',flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
