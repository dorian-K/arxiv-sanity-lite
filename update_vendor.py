"""
Copies the frontend libraries into static/vendor, so pages never load anything from a
third-party server (which would hand every visitor's IP address to that server, a
problem under the GDPR). Downloads each package's tarball from the npm registry,
verifies it against the registry's sha512, and extracts only the files the pages use.

    uv run python update_vendor.py   # after changing a version below; commit static/vendor
"""

import io
import os
import json
import base64
import shutil
import hashlib
import tarfile
import urllib.request

PACKAGES = {
    'react': ('18.3.1', ['umd/react.production.min.js', 'LICENSE']),
    'react-dom': ('18.3.1', ['umd/react-dom.production.min.js', 'LICENSE']),
    'htm': ('3.1.1', ['dist/htm.umd.js', 'LICENSE']),
    # the css refers to fonts/ next to it; woff2 is what every current browser picks
    'katex': ('0.16.11', ['dist/katex.min.css', 'dist/katex.min.js', 'dist/contrib/auto-render.min.js', 'LICENSE']),
}
DEST = os.path.join('static', 'vendor')

if __name__ == '__main__':
    shutil.rmtree(DEST, ignore_errors=True)
    for name, (version, files) in PACKAGES.items():
        meta = json.load(urllib.request.urlopen(f'https://registry.npmjs.org/{name}/{version}', timeout=60))
        algo, want = meta['dist']['integrity'].split('-', 1)
        data = urllib.request.urlopen(meta['dist']['tarball'], timeout=120).read()
        got = base64.b64encode(hashlib.new(algo, data).digest()).decode()
        assert algo == 'sha512' and got == want, f'{name}@{version}: integrity mismatch'
        out = os.path.join(DEST, f'{name}-{version}')
        with tarfile.open(fileobj=io.BytesIO(data)) as tar:
            members = {m.name.removeprefix('package/'): m for m in tar.getmembers()}
            wanted = list(files)
            if name == 'katex':
                wanted += [f for f in members if f.startswith('dist/fonts/') and f.endswith('.woff2')]
            for f in wanted:
                target = os.path.join(out, f.removeprefix('dist/'))
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with open(target, 'wb') as fh:
                    fh.write(tar.extractfile(members[f]).read())
        print(f'{name}@{version}: verified, {len(wanted)} files -> {out}')
