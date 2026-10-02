"""Download the pinned official llama.cpp Windows CPU runtime from GitHub."""
import argparse
import hashlib
import json
import platform
import urllib.request
import zipfile
from pathlib import Path

def setup(work_dir):
    root=Path(__file__).resolve().parents[1]
    config=json.loads((root/'configs/llama_runtime.json').read_text())
    if platform.system()!='Windows' or platform.machine().lower() not in ('amd64','x86_64'):
        raise SystemExit('Pinned binary is Windows x86-64 only. Use an official native CPU llama-server build and LLAMA_SERVER on another platform.')
    downloads=work_dir/'downloads'
    downloads.mkdir(parents=True,exist_ok=True)
    archive=downloads/Path(config['url']).name
    if not archive.exists(): urllib.request.urlretrieve(config['url'],archive)
    h=hashlib.sha256(archive.read_bytes()).hexdigest()
    if h!=config['sha256'] or archive.stat().st_size!=config['size_bytes']:
        raise SystemExit('Official runtime archive failed integrity check; remove the failed download and retry')
    destination=(work_dir/'runtime'/f"llama-{config['release']}").resolve()
    destination.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            target=(destination/item.filename).resolve()
            if not target.is_relative_to(destination) or (item.external_attr>>16)&0o170000==0o120000:
                raise SystemExit('Unsafe runtime archive member')
        z.extractall(destination)
    exe=destination/config['executable']
    if not exe.is_file(): raise SystemExit('Expected llama-server executable absent')
    print(str(exe))
    return exe

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work-dir',type=Path,required=True)
    setup(p.parse_args().work_dir.resolve())
