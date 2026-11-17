"""全量数据准备入口；每步可重复运行，原数据及旧数据集只读。"""
import argparse
from pathlib import Path
from xiaomi_lens.full_dataset.common import DEFAULT
from xiaomi_lens.full_dataset.prepare import scan,partition,export

def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['scan','partition','export'])
    p.add_argument('--dataset-dir',type=Path,default=DEFAULT)
    p.add_argument('--workers',type=int,default=6)
    a=p.parse_args()
    import fcntl
    a.dataset_dir.mkdir(parents=True,exist_ok=True)
    with (a.dataset_dir/'prepare-process.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.stage=='scan':scan(a.dataset_dir,a.workers)
        elif a.stage=='partition':partition(a.dataset_dir)
        else:export(a.dataset_dir,a.workers)

if __name__=='__main__':main()
