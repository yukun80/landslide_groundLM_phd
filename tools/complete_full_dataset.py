"""本次全量制作的一次性接续：等待已有进程，补齐文字，再完成全部检查。"""
import argparse
import json
import subprocess
import time
from pathlib import Path

from xiaomi_lens.full_dataset.common import *


def wait_process(pid,expected):
    """只等待指定的已有任务；原进程退出或编号被复用时结束等待。"""
    path=Path(f'/proc/{pid}/cmdline')
    if not pid:return
    try:identity=Path(f'/proc/{pid}/stat').read_text().split()[21]
    except FileNotFoundError:return
    while path.exists():
        try:
            if expected not in path.read_bytes().decode().replace('\0',' ') or Path(f'/proc/{pid}/stat').read_text().split()[21]!=identity:return
        except FileNotFoundError:return
        time.sleep(10)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset-dir',type=Path,default=DEFAULT)
    parser.add_argument('--wait-export-pid',type=int,default=0);parser.add_argument('--wait-caption-pid',type=int,default=0)
    args=parser.parse_args();dest=args.dataset_dir.resolve();initialize(dest)
    import fcntl
    controller_lock=(dest/'review/completion.lock').open('a')
    fcntl.flock(controller_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    logdir=ROOT/'outputs/full-dataset-completion';logdir.mkdir(exist_ok=True)
    def state(step,**extra):
        report={'step':step,'complete':step=='complete','updated_at':time.strftime('%Y-%m-%dT%H:%M:%S'),**extra}
        atomic_write_json(dest/'audit/pipeline_state.json',report);print(json.dumps(report),flush=True)
    def call(wrapper,tool,*arguments):
        name=Path(tool).stem+'-'+('-'.join(a for a in arguments if not a.startswith('--')) or 'run')
        log=logdir/f'{name}-{time.time_ns()}.log'
        with log.open('w') as output:
            result=subprocess.run([str(ROOT/wrapper),tool,*arguments,'--dataset-dir',str(dest)],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'{tool} failed ({result.returncode}): {log}')
    try:
        if json.loads((dest/'build.json').read_text()).get('annotation_provider')=='local_resume':
            from xiaomi_lens.full_dataset.resume_records import load_manifest, progress
            from xiaomi_lens.full_dataset.local_resume import check_gate
            check_gate(dest,load_manifest(dest),require_encoding=True)
            for attempt in range(3):
                if not progress(dest)['remaining']:break
                try:call('scripts/full_dataset.sh','tools/describe_full_dataset.py','run')
                except RuntimeError:
                    if attempt==2:raise
            assert not progress(dest)['remaining'],'local descriptions still missing'
            state('publishing_descriptions',provider='local_resume')
            call('scripts/full_dataset.sh','tools/describe_full_dataset.py','publish')
            state('checking_copies',provider='local_resume')
            call('scripts/domain.sh','tools/check_full_dataset.py','raw')
            state('checking_training_reads',provider='local_resume')
            call('scripts/domain.sh','tools/check_full_dataset.py','encode','--device','cpu')
            state('checking_records',provider='local_resume')
            call('scripts/domain.sh','tools/check_full_dataset.py','final')
            call('scripts/domain.sh','tools/write_full_dataset_validation.py')
            subprocess.run([str(ROOT/'scripts/domain.sh'),'tools/check_full_dataset_docs.py'],cwd=ROOT,check=True)
            progress(dest)
            state('complete',provider='local_resume',selected=18987,formal_training_run=False)
            return
        if json.loads((dest/'build.json').read_text()).get('annotation_provider')=='codex_session':
            from xiaomi_lens.full_dataset.session_review import status,publish
            progress=status(dest)
            if progress['remaining']:
                print('等待当前Codex会话逐图处理；不会启动本地模型或其他在线程序',flush=True)
                return
            call('scripts/domain.sh','tools/review_full_dataset_session.py','publish')
            call('scripts/domain.sh','tools/check_full_dataset.py','raw')
            call('scripts/domain.sh','tools/check_full_dataset.py','encode','--device','cpu')
            call('scripts/domain.sh','tools/check_full_dataset.py','final')
            call('scripts/domain.sh','tools/write_full_dataset_validation.py')
            subprocess.run([str(ROOT/'scripts/domain.sh'),'tools/check_full_dataset_docs.py'],cwd=ROOT,check=True)
            state('complete',selected=18987,formal_training_run=False)
            return
        state('waiting_for_copy');wait_process(args.wait_export_pid,'prepare_full_dataset.py export')
        selected=read_jsonl(dest/'audit/selected.jsonl')
        if not (dest/'prepared.jsonl').exists() or len(read_jsonl(dest/'prepared.jsonl'))!=len(selected):
            state('resuming_copy');call('scripts/domain.sh','tools/prepare_full_dataset.py','export')
        state('checking_copies',selected=len(selected));call('scripts/domain.sh','tools/check_full_dataset.py','raw','--prepared')
        state('generating_descriptions');wait_process(args.wait_caption_pid,'describe_full_dataset.py run')
        missing=[]
        for attempt in range(4):
            missing=[r['id'] for r in selected if not r['reuse_text'] and not (dest/f"review/samples/{r['id']}.json").exists()]
            if not missing:break
            if attempt==3:raise RuntimeError(('descriptions_still_missing',len(missing),missing[:20]))
            state('resuming_descriptions',attempt=attempt+1,missing=len(missing))
            try:call('scripts/full_dataset.sh','tools/describe_full_dataset.py','run')
            except RuntimeError:
                # 逐条证据已经保存；继续核实剩余项，不把报错样本视为完成。
                if attempt==2:raise
        state('publishing_descriptions');call('scripts/full_dataset.sh','tools/describe_full_dataset.py','publish')
        state('checking_final_data');call('scripts/domain.sh','tools/check_full_dataset.py','raw')
        state('checking_training_reads');call('scripts/domain.sh','tools/check_full_dataset.py','encode')
        quality=json.loads((dest/'audit/protected_quality_conflicts.json').read_text())
        if not quality['resolved']:
            state('awaiting_user_quality_choice',remaining_sample_ids=[r['id'] for r in quality['samples']],all_other_steps_finished=True)
            return
        state('final_check');call('scripts/domain.sh','tools/check_full_dataset.py','final')
        call('scripts/domain.sh','tools/write_full_dataset_validation.py')
        subprocess.run([str(ROOT/'scripts/domain.sh'),'tools/check_full_dataset_docs.py'],cwd=ROOT,check=True)
        state('complete',selected=len(selected),formal_training_run=False)
    except Exception as error:
        state('failed',error=repr(error));raise


if __name__=='__main__':main()
