"""Maintainer-only Hugging Face -> checksummed GitHub Release package preparation.

The VDI uses download_models.py; it never runs this upstream preparation tool.
All upstream revisions, filenames and expected weight hashes are committed.
"""
import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def prepare(repo_root, work_dir, download_upstream=False):
    plan = json.loads((repo_root / 'configs/preparation.json').read_text())
    models = {m['model_id']: m for m in json.loads((repo_root / 'configs/models.json').read_text())['models']}
    resources = work_dir / 'licenses'
    resources.mkdir(parents=True, exist_ok=True)
    license_urls = {
        'Apache-2.0': 'https://www.apache.org/licenses/LICENSE-2.0.txt',
        'MIT-BGE': 'https://raw.githubusercontent.com/FlagOpen/FlagEmbedding/master/LICENSE',
        'MIT-E5': 'https://raw.githubusercontent.com/microsoft/unilm/master/LICENSE',
    }
    for key, url in license_urls.items():
        path = resources / (key + '.txt')
        if not path.exists():
            urllib.request.urlretrieve(url, path)
    if download_upstream:
        from huggingface_hub import hf_hub_download, snapshot_download
        for mid, model in models.items():
            if model.get('status') == 'skipped':
                continue
            snapshot_download(**model['upstream'], local_dir=work_dir / 'sources' / mid,
                              allow_patterns=['*.json', '*.txt', '*.py', '*.md', 'LICENSE*', 'NOTICE*'])
        for dependency in plan['code_dependencies'].values():
            snapshot_download(dependency['repo_id'], revision=dependency['revision'],
                              local_dir=work_dir / 'sources' / dependency['source_key'], allow_patterns=['*.py', '*.md'])
        for job in plan['artifacts']:
            for file in job['downloaded_files']:
                hf_hub_download(job['source']['repo_id'], file['path'], revision=job['source']['revision'],
                                local_dir=work_dir / 'sources' / job['source_key'])
    artifacts = []
    for job in plan['artifacts']:
        mid = job['model_id']
        model = models[mid]
        destination = work_dir / 'models' / job['artifact_id']
        destination.mkdir(parents=True, exist_ok=True)
        official = work_dir / 'sources' / mid
        if not official.is_dir():
            raise FileNotFoundError(f'Official metadata missing: {official}; use --download-upstream on maintainer PC')
        for source in sorted(official.rglob('*')):
            relative = source.relative_to(official)
            if not source.is_file() or any(part in ('.cache', 'onnx', 'openvino', '__pycache__') for part in relative.parts):
                continue
            if source.suffix not in ('.json', '.txt', '.py', '.md') and not source.name.startswith(('LICENSE', 'NOTICE')):
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        modifications = []
        dependency = plan['code_dependencies'].get(mid)
        if dependency:
            for source in (work_dir / 'sources' / dependency['source_key']).glob('*.py'):
                shutil.copyfile(source, destination / source.name)
            config_path = destination / 'config.json'
            shutil.copyfile(config_path, destination / 'config.upstream.json')
            config = json.loads(config_path.read_text())
            config['auto_map'] = {key: value.split('--')[-1] for key, value in config['auto_map'].items()}
            write_json(config_path, config)
            modifications.append('config.json auto_map references local bundled code; original config.upstream.json preserved')
        if mid == 'arctic-embed-m-v2':
            path = destination / 'config.json'
            shutil.copyfile(path, destination / 'config.upstream.json')
            config = json.loads(path.read_text())
            config['unpad_inputs'] = False
            config['use_memory_efficient_attention'] = False
            write_json(path, config)
            modifications.append('CPU config disables unpadding/xformers memory-efficient attention; original config preserved')
        for weight in job['downloaded_files']:
            source = work_dir / 'sources' / job['source_key'] / weight['path']
            if not source.exists():
                source = work_dir / 'models' / mid / weight['path']
            if source.stat().st_size != weight['size'] or sha256(source) != weight['sha256']:
                raise ValueError(f'Upstream weight checksum failed: {source}')
            target = destination / weight['path']
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or sha256(target) != weight['sha256']:
                shutil.copyfile(source, target)
        license_key = 'MIT-BGE' if mid == 'bge-base-en-v1.5' else 'MIT-E5' if mid == 'e5-base-v2' else 'Apache-2.0'
        shutil.copyfile(resources / (license_key + '.txt'), destination / 'LICENSE')
        write_json(destination / 'PROVENANCE.json', {'model': model['upstream'], 'weights': job['source'],
                   'code_dependency': dependency, 'modifications': modifications,
                   'license': model['license'], 'upstream_weight_files': job['downloaded_files']})
        files = [{'path': str(p.relative_to(destination)).replace('\\', '/'),
                  'size': p.stat().st_size, 'sha256': sha256(p)}
                 for p in sorted(destination.rglob('*')) if p.is_file() and '.cache' not in p.parts]
        archive = work_dir / 'downloads' / (job['artifact_id'] + '.zip')
        archive.parent.mkdir(parents=True, exist_ok=True)
        # Stored compression bounds archive creation cost and keeps original payload bytes.
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as z:
            for file in files:
                z.write(destination / file['path'], file['path'])
        if archive.stat().st_size >= 2 * 1024**3:
            raise ValueError('GitHub Release asset exceeds 2 GiB; split before publishing')
        artifact = {'artifact_id': job['artifact_id'], 'model_id': mid,
                    'runtime': {'onnx':'onnxruntime','gguf':'llama.cpp'}.get(job['runtime'],job['runtime']),
                    'precision': job['precision'], 'status':'ready', 'source': {'kind':'pinned_upstream', **job['source']},
                    'github': {'repository':plan['github_repository'],'release':plan['release'],'asset':archive.name},
                    'archive_sha256':sha256(archive),'archive_format':'zip','files':files,
                    'size_bytes':archive.stat().st_size, 'entrypoint':job['weight_files'][0],
                    'cpu_features_required':['avx512f','avx512_vnni'] if 'avx512-vnni' in job['artifact_id'] else [],
                    'redistribution':{'allowed':True,'basis':model['license']['spdx'],'notes':'Upstream model card and license bundled; code/config modifications recorded in PROVENANCE.json'},
                    'modifications':modifications,
                    'validation_status':'reference' if job['runtime']=='pytorch' else 'pending_reference_validation'}
        artifacts.append(artifact)
        print('PACKAGED',job['artifact_id'],artifact['size_bytes'],flush=True)
        # Persist completed packages so development validation can start before all archives finish.
        write_json(repo_root / 'configs/artifacts.json', {'schema_version':1,'artifacts':artifacts})
    artifacts.append({'artifact_id':'embeddinggemma-300m-skipped','model_id':'embeddinggemma-300m',
                      'status':'skipped','reason':models['embeddinggemma-300m']['reason'],
                      'redistribution':{'allowed':False,'basis':'Original model gate/access prerequisite'}})
    write_json(repo_root / 'configs/artifacts.json', {'schema_version':1,'artifacts':artifacts})
    return artifacts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--work-dir',type=Path,required=True)
    parser.add_argument('--download-upstream',action='store_true',help='Maintainer PC only; requires access to Hugging Face')
    args = parser.parse_args()
    prepare(args.repo_root.resolve(),args.work_dir.resolve(),args.download_upstream)
