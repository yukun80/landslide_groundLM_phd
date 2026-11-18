"""全量文字的追加复查；默认不发布、不训练，旧记录保持只读。"""
import argparse
import fcntl
import json
from pathlib import Path
from xiaomi_lens.full_dataset.common import DEFAULT, atomic_write_json, sha256_file
from xiaomi_lens.full_dataset import quality_review as quality
from xiaomi_lens.full_dataset.quality_generate import run, protocol


def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['inventory','pilot','approve-pilot','run','status','apply-visual-review','publish'])
    p.add_argument('--dataset-dir',type=Path,default=DEFAULT)
    p.add_argument('--input',type=Path)
    p.add_argument('--limit',type=int,default=0)
    a=p.parse_args();dest=a.dataset_dir.resolve()
    if a.stage=='status':print(json.dumps(quality.status(dest),ensure_ascii=False,indent=2));return
    folder=dest/quality.REVIEW;folder.mkdir(parents=True,exist_ok=True)
    with (dest/'review/generation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (dest/'review/local_resume/generation.lock').open('a') as other:
            fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if a.stage=='inventory':print(json.dumps(quality.inventory(dest),ensure_ascii=False,indent=2))
            elif a.stage in ('pilot','run'):run(dest,pilot=a.stage=='pilot',limit=a.limit)
            elif a.stage=='apply-visual-review':
                from xiaomi_lens.full_dataset.quality_delivery import apply_visual
                assert a.input is not None
                apply_visual(dest,json.loads(a.input.read_text()))
            elif a.stage=='publish':
                from xiaomi_lens.full_dataset.quality_delivery import publish
                print(json.dumps(publish(dest),ensure_ascii=False,indent=2))
            elif a.stage=='approve-pilot':
                assert a.input is not None
                review=json.loads(a.input.read_text());m=quality.load(dest)
                entries=review['entries'];assert len(entries)==len(m['pilot_ids'])
                assert {e['id'] for e in entries}==set(m['pilot_ids'])
                for e in entries:
                    assert e['passed'] is True and e['note'].strip() and e['viewed_images']
                    assert e['record_sha256']==sha256_file(folder/'samples'/f"{e['id']}.json")
                    for path,h in e['viewed_images'].items():assert sha256_file(Path(path))==h
                atomic_write_json(folder/'pilot_approval.json',{'passed':True,'protocol_sha256':protocol(),
                    'records_sha256':{e['id']:e['record_sha256'] for e in entries},'visual_review':review})


if __name__=='__main__':main()
