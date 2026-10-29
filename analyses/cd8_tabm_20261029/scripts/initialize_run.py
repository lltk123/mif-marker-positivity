"""Create a new TabM analysis directory by copying source, never fitted artifacts.

Example:
  python initialize_run.py --source-run PROJECT/analyses/old --run-dir PROJECT/analyses/new
The destination must not exist and must be a direct sibling in PROJECT/analyses.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

COMMIT = '28e47ae301c92ec37787dde1ce923a0793f405b4'
WHEEL_SHA256 = '87fd61270118915cf40888f2164f3a1f0353edf702f15fd91e71a147d23c930d'
EMPTY_DIRS = ['data','models','predictions','results','curves','logs','figures','development_review']
FORBIDDEN = {'.pt','.pth','.ckpt','.safetensors','.bin','.parquet','.npy','.npz','.tif','.tiff','.png','.jpg','.jpeg','.zip','.rar'}
SKIP_DIRS = {'__pycache__','.git','.venv','venv','env','.ssh'}


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b''):
            h.update(chunk)
    return h.hexdigest()


def initialize(source_run, run_dir):
    source=Path(source_run).resolve(strict=True)
    requested=Path(run_dir).absolute()
    if requested.exists() or requested.is_symlink():
        raise FileExistsError('Destination must not exist: '+str(requested))
    target=requested.resolve()
    if source.parent.name!='analyses' or target.parent!=source.parent:
        raise ValueError('Source and target must be direct children of the same project analyses directory')
    if target.name in {'','.', '..'}:
        raise ValueError('Invalid run directory')
    manifest_path=source/'upstream/SOURCE_MANIFEST.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    if manifest.get('commit')!=COMMIT:
        raise ValueError('Source official commit differs from the audited version')
    plan={}
    def add(path, relative):
        relative=Path(relative)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe destination path: '+str(relative))
        if path.is_symlink() or not path.is_file() or source not in path.resolve().parents:
            raise ValueError('Only regular source files may be copied: '+str(path))
        if path.suffix.lower() in FORBIDDEN:
            raise ValueError('Fitted/data artifact cannot be copied: '+str(path))
        if relative.as_posix() in plan:
            raise ValueError('Duplicate copy destination: '+str(relative))
        if path.stat().st_size>256*1024**2:
            raise ValueError('Unexpectedly large source/dependency file: '+str(path))
        plan[relative.as_posix()]={'source':path,'sha256':sha256(path),'bytes':path.stat().st_size}
    scripts=source/'scripts'
    if not scripts.is_dir():
        raise FileNotFoundError(scripts)
    for path in sorted(scripts.rglob('*')):
        if any(part in SKIP_DIRS for part in path.relative_to(scripts).parts):
            continue
        if path.is_file() and path.suffix in {'.py','.sh'}:
            add(path,Path('scripts')/path.relative_to(scripts))
    required=['common_tabm.py','prepare_tabm.py','train_tabm.py','resources_tabm.py','run_pipeline.py','write_tabm_report.py']
    if any('scripts/'+name not in plan for name in required):
        raise ValueError('Source run lacks required research scripts')
    upstream=(source/'upstream').resolve()
    for item in manifest['files']:
        relative=Path(item['file'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe upstream manifest entry')
        path=upstream/relative
        if upstream not in path.resolve().parents or sha256(path)!=item['sha256']:
            raise ValueError('Upstream source failed its pinned hash: '+str(path))
        add(path,Path('upstream')/relative)
    add(manifest_path,'upstream/SOURCE_MANIFEST.json')
    for wheel in sorted(upstream.glob('rtdl_num_embeddings-0.0.12-*.whl')):
        if sha256(wheel)!=WHEEL_SHA256:
            raise ValueError('Unexpected embedding dependency wheel hash')
        add(wheel,Path('upstream')/wheel.name)
    dependencies=source/'dependencies'
    if dependencies.exists():
        if dependencies.is_symlink():
            raise ValueError('Dependency directory must not be a symbolic link')
        for path in sorted(dependencies.rglob('*')):
            if any(part in SKIP_DIRS for part in path.relative_to(dependencies).parts):
                continue
            if path.is_file():
                add(path,Path('dependencies')/path.relative_to(dependencies))
    for path in sorted(source.glob('requirements*.txt')):
        add(path,path.name)
    for name in ['OFFICIAL_IMPLEMENTATION_AUDIT.md','QA_TEMPLATE.md']:
        if (source/name).is_file():
            add(source/name,name)
    # Old environment records remain explicitly source provenance, never current state.
    for name in ['DEPENDENCIES.json','HARDWARE.json','README.md']:
        if (source/name).is_file():
            add(source/name,Path('provenance')/('source_'+name))
    target.mkdir(exist_ok=False)
    for name in EMPTY_DIRS+['scripts','upstream','dependencies','provenance']:
        (target/name).mkdir(exist_ok=True)
    records=[]
    for relative,item in sorted(plan.items()):
        destination=target/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        contents=item['source'].read_bytes()
        if hashlib.sha256(contents).hexdigest()!=item['sha256']:
            raise RuntimeError('Source changed during initialization: '+str(item['source']))
        with destination.open('xb') as stream:
            stream.write(contents)
        records.append(dict(file=relative,source=str(item['source']),sha256=item['sha256'],bytes=item['bytes']))
    record=dict(status='initialized_not_trained',source_run=str(source),run_dir=str(target),
        initialized_utc=datetime.now(timezone.utc).isoformat(),official_commit=COMMIT,files=records,
        fitted_data_models_predictions_copied=False,source_environment_records_are_current=False,
        original_project_inputs_reused_read_only=True,PanM_accessed=False)
    (target/'INITIALIZATION_MANIFEST.json').write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    text=(f'# New TabM run\n\nStatus: initialized; no experiment has run.\n\n'
          f'Source: `{source}`. Fixed upstream commit: `{COMMIT}`.\n\n'
          'Only source, pinned dependencies and audit notes were copied. Data caches, models, predictions and results are empty. '
          'Original PhenoBIC features are reused from the existing project; Pan-M remains sealed.\n\n'
          'The pipeline records the actual current software/hardware. Files under provenance/source_* describe the previous run only. '
          'The copied source README preserves the detailed method and setup instructions; its paths and completion statements refer to the source run, not this new run.\n\n'
          'Use the existing compatible Python environment. Optional --font-path supplies a CJK font for both figures and development review. '
          'The original PhenoBIC feature table, keyed label/GMM mapping, extraction code and development images must remain available in this project.\n\n'
          f'```bash\npython "{target}/scripts/run_pipeline.py" --run-dir "{target}" --stage all --font-path /usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc\n```\n')
    (target/'README.md').write_text(text,encoding='utf-8')
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',required=True,type=Path)
    parser.add_argument('--run-dir',required=True,type=Path)
    args=parser.parse_args()
    result=initialize(args.source_run,args.run_dir)
    print(json.dumps({k:v for k,v in result.items() if k!='files'},ensure_ascii=False))
