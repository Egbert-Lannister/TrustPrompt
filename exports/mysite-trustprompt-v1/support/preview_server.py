"""Read-only local preview server. Does not run or alter mysite."""
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import unquote, urlsplit
ROOT = Path(__file__).resolve().parents[1]
STATIC = Path('D:/Python_Script/mysite/static')

class Handler(SimpleHTTPRequestHandler):
    def translate_path(self, path):
        p = unquote(urlsplit(path).path)
        if p in ('/', '/preview.html'): return str(ROOT / 'support/preview.html')
        if p.startswith('/static/'):
            base, rel = STATIC, p[len('/static/'):]
        elif p.startswith('/assets/'):
            base, rel = ROOT/'assets', p[len('/assets/'):]
        else:
            return str(ROOT/'support/not-found')
        target = (base/rel).resolve()
        return str(target if target.is_relative_to(base.resolve()) else ROOT/'support/not-found')

if __name__ == '__main__':
    print('Local preview: http://127.0.0.1:8766/',flush=True)
    ThreadingHTTPServer(('127.0.0.1',8766),Handler).serve_forever()
