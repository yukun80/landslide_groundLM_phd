"""共享路径、原子写入、保护清单与原始文件身份检查。"""
import hashlib
import json
import os
from pathlib import Path
from ..description_v2.io import atomic_write_json, write_jsonl_atomic, read_jsonl, sha256_file
from ..train.data.domain import ROOT, SOURCE, DEST as OLD, NAMES

DEFAULT = ROOT / 'datasets/landslide_lens_full_positive_v1'
BENCHMARK = ROOT / 'datasets/landslide_benchmark_500'
SEED = 20261116
SPLITS = ('train', 'val', 'test')
VIEWS = {NAMES[0]: [[0,1,2]], NAMES[1]: [[7,10,11],[3,2,1]], NAMES[2]: [[4,4,4],[3,3,3],[0,1,2]]}


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_meta(path):
    p=Path(path);s=p.stat()
    return {'size':s.st_size,'mtime_ns':s.st_mtime_ns,'sha256':sha256_file(p)}


def source_paths(source, row, source_root=SOURCE):
    """只接受三套来源目录内的真实文件。"""
    base=Path(source_root)/source/('hdf5' if source==NAMES[1] else '')
    out={k:(base/row[k]).resolve() for k in ('image_hdf5','mask_hdf5')}
    for p in out.values():
        if not p.is_relative_to(Path(source_root).resolve()):raise ValueError(f'outside source root: {p}')
    return out


def old_rows():
    return [r for s in SPLITS for r in read_jsonl(OLD/f'{s}.jsonl')]


def pilot_rows(selection):
    """每类4个固定训练样本：从小目标到大目标，不查看测试结果来选规则。"""
    chosen=[]
    for source in NAMES:
        pool=sorted([r for r in selection if r['source']==source and r['split']=='train' and not r['reuse_text']],key=lambda r:(r['positive_pixels']/r['valid_pixels'],r['id']))
        assert len(pool)>=4
        chosen.extend(pool[i] for i in [0,len(pool)//3,2*len(pool)//3,-1])
    return chosen


def protection():
    """记录旧数据所有文件的身份，后续只读核对，不覆盖历史检查。"""
    paths=[p for base in (OLD,BENCHMARK) for p in base.rglob('*') if p.is_file()]
    # 目录中图片/复核证据均保留；在此保存大小、时间，内容摘要清单另核对核心文件。
    stat={str(p.relative_to(ROOT)):[p.stat().st_size,p.stat().st_mtime_ns] for p in paths}
    core=[OLD/f'{s}.jsonl' for s in SPLITS]+[OLD/'display_ranges.json',OLD/'summary.json',BENCHMARK/'manifest.json',BENCHMARK/'samples.jsonl']
    return {'files':stat,'core_sha256':{str(p.relative_to(ROOT)):sha256_file(p) for p in core}}


def check_protection(record):
    for rel,expected in record['files'].items():
        p=ROOT/rel;s=p.stat()
        assert [s.st_size,s.st_mtime_ns]==expected,rel
    for rel,h in record['core_sha256'].items():assert sha256_file(ROOT/rel)==h,rel


def initialize(dest=DEFAULT):
    dest=Path(dest).resolve();dest.mkdir(parents=True,exist_ok=True)
    p=dest/'build.json'
    if p.exists():
        info=json.loads(p.read_text());assert info['seed']==SEED
        check_protection(info['protected']);return info
    info={'seed':SEED,'source_root':str(SOURCE),'dataset_dir':str(dest),'complete':False,
          'protected':protection(),'display_ranges':json.loads((OLD/'display_ranges.json').read_text()),
          'input_views':VIEWS,'source_manifests':{n:file_meta(SOURCE/n/'hdf5/conversion_manifest.jsonl') for n in NAMES},
          'notes':['three original sources only','source-native bands preserved; original model view combinations',
                   'old 1000 folds pinned; independent 500 and their groups test-only','all splits positive only']}
    atomic_write_json(p,info);return info
