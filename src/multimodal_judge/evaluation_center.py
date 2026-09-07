"""Loopback-only evaluation launcher and read-only result browser."""
import json
import subprocess
import sys
import threading
import uuid
from datetime import datetime
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .evaluation import write_json


class EvaluationCenter:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.output = self.root / 'artifacts/evaluation/runs'
        self.output.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.process = None
        self.run_id = None

    def catalog(self):
        def directories(pattern):
            return sorted({str(p.parent.relative_to(self.root)) for p in self.root.glob(pattern)
                           if p.resolve().is_relative_to(self.root)})
        return {'checkpoints': directories('artifacts/training/**/joint_manifest.json'),
                'datasets': directories('data/training_data/**/train.jsonl')}

    def state(self):
        runs = []
        for path in sorted(self.output.glob('*/report.json'), reverse=True):
            report = json.loads(path.read_text())
            runs.append({'id': path.parent.name, 'split': report['split'],
                         'checkpoint': Path(report['checkpoint']).name,
                         'wandb_url': report.get('wandb_url')})
        progress = None
        if self.run_id:
            path = self.output / self.run_id / 'progress.json'
            progress = json.loads(path.read_text()) if path.exists() else {'status': 'starting'}
            code = self.process.poll()
            if code is not None and progress['status'] not in ('finished', 'failed'):
                progress = {'status': 'failed', 'error': 'Evaluation exited; see the linked run log'}
            progress = {**progress, 'id': self.run_id, 'running': code is None}
        return {'runs': runs, 'progress': progress}

    def launch(self, options):
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                raise ValueError('An evaluation is already running')
            catalog = self.catalog()
            if options.get('checkpoint') not in catalog['checkpoints']:
                raise ValueError('Select a checkpoint from the catalog')
            if options.get('data_dir') not in catalog['datasets']:
                raise ValueError('Select a dataset from the catalog')
            if options.get('split') not in ('test', 'validation'):
                raise ValueError('Select test or validation')
            if options.get('wandb_mode') not in ('online', 'offline', 'disabled'):
                raise ValueError('Invalid W&B mode')
            if type(options.get('include_base', False)) is not bool:
                raise ValueError('include_base must be boolean')
            run_id = datetime.now().strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:6]
            destination = self.output / run_id
            # The evaluator owns creating the run directory, so an existing run is never replaced.
            command = [sys.executable, '-m', 'multimodal_judge', 'evaluate',
                       '--checkpoint', options['checkpoint'], '--data-dir', options['data_dir'],
                       '--split', options['split'], '--output-dir', str(destination),
                       '--wandb-mode', options['wandb_mode']]
            if options.get('include_base'):
                command.append('--include-base')
            log_path = self.output / (run_id + '.log')
            with log_path.open('w') as log:
                self.process = subprocess.Popen(command, cwd=self.root, stdout=log,
                                                stderr=subprocess.STDOUT)
            self.run_id = run_id
            return {'id': run_id}


class CenterHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, center, **kwargs):
        self.center = center
        super().__init__(*args, directory=str(center.output), **kwargs)

    def parse_request(self):
        if not super().parse_request():
            return False
        allowed = {f'127.0.0.1:{self.server.server_port}',
                   f'localhost:{self.server.server_port}'}
        if self.headers.get('Host') not in allowed:
            self.send_error(403)
            return False
        return True

    def json_response(self, value, status=200):
        payload = json.dumps(value, allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        route = urlparse(self.path).path
        if route == '/api/catalog':
            return self.json_response(self.center.catalog())
        if route == '/api/state':
            return self.json_response(self.center.state())
        if route == '/':
            payload = Path(__file__).with_name('evaluation_center.html').read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(payload)
            return
        target = Path(self.translate_path(self.path)).resolve()
        if not target.is_relative_to(self.center.output) or not target.is_file():
            return self.send_error(404)
        super().do_GET()

    def do_POST(self):
        if self.path != '/api/evaluate':
            return self.send_error(404)
        origin = self.headers.get('Origin')
        allowed = {f'http://127.0.0.1:{self.server.server_port}',
                   f'http://localhost:{self.server.server_port}'}
        if origin and origin not in allowed:
            return self.send_error(403)
        if self.headers.get('Content-Type') != 'application/json':
            return self.send_error(415)
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 4096:
                raise ValueError('Invalid request size')
            options = json.loads(self.rfile.read(size))
            if not isinstance(options, dict):
                raise ValueError('Expected JSON object')
            self.json_response(self.center.launch(options), 202)
        except (ValueError, OSError) as exc:
            self.json_response({'error': str(exc)}, 400)


def serve_center(root='.', port=8877):
    center = EvaluationCenter(root)
    server = ThreadingHTTPServer(('127.0.0.1', port), partial(CenterHandler, center=center))
    write_json(center.output / 'server.json', {'url': f'http://127.0.0.1:{server.server_port}'})
    print(f'Evaluation Center: http://127.0.0.1:{server.server_port}', flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
