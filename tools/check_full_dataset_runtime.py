"""仅检查本次文字标注的可选加速运算；不更新模型或Conda包。"""
import json
import argparse
import time
from pathlib import Path

import torch
import transformers
import fla
from transformers.models.qwen3_5 import modeling_qwen3_5 as q
from xiaomi_lens.full_dataset.common import ROOT, atomic_write_json, sha256_file


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',action='store_true');options=parser.parse_args()
    torch.manual_seed(20261116)
    torch.set_num_threads(4)
    results=[]
    for length in (1,67,256):
        shape=(2,length,8,128)
        query,key,value=[torch.randn(shape,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
        g=-torch.rand(shape[:-1],device='cuda')
        beta=torch.rand(shape[:-1],device='cuda',dtype=torch.bfloat16)
        state=torch.randn((2,8,128,128),device='cuda')*.01
        args=dict(query=query,key=key,value=value,g=g,beta=beta,initial_state=state,
                  output_final_state=True,use_qk_l2norm_in_kernel=True)
        with torch.inference_mode():
            a=q.torch_chunk_gated_delta_rule(**args)
            accelerated={**args,'q':query,'k':key,'v':value}
            for name in ('query','key','value'):accelerated.pop(name)
            b=q.chunk_gated_delta_rule(**accelerated)
            errors=[float((x.float()-y.float()).abs().max()) for x,y in zip(a,b)]
            for x,y in zip(a,b):torch.testing.assert_close(x.float(),y.float(),atol=.015,rtol=.03)
            recurrent=None
            if length==1:
                c=q.torch_recurrent_gated_delta_rule(**args)
                d=q.fused_recurrent_gated_delta_rule(**accelerated)
                recurrent=[float((x.float()-y.float()).abs().max()) for x,y in zip(c,d)]
                for x,y in zip(c,d):torch.testing.assert_close(x.float(),y.float(),atol=.015,rtol=.03)
        results.append({'length':length,'chunk_max_errors':errors,'recurrent_max_errors':recurrent})
    before=ROOT/'outputs/full-dataset-environment-before.txt'
    after=ROOT/'outputs/full-dataset-environment-after.txt'
    assert before.read_bytes()==after.read_bytes(),'Conda package list changed'
    report={'passed':True,'torch':torch.__version__,'transformers':transformers.__version__,
            'fla':fla.__version__,'fla_file':fla.__file__,'conda_packages_unchanged':True,
            'numerical_checks':results,'scope':'kernel parity only; real generation checked by pilot',
            'package_list_sha256':sha256_file(after),'checked_at':time.strftime('%Y-%m-%dT%H:%M:%S')}
    if options.model:
        from xiaomi_lens.domain_runtime import load,build,seed_all
        from xiaomi_lens.full_dataset.common import old_rows,NAMES
        seed_all();model,processor=load();layers=[m for m in model.modules() if hasattr(m,'chunk_gated_delta_rule')]
        assert layers and all(m.chunk_gated_delta_rule is q.chunk_gated_delta_rule for m in layers)
        real=[]
        for name in NAMES:
            row=next(r for r in old_rows() if r['source']==name and r['split']=='train')
            inputs=build(processor,row)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                fast=model(**inputs,use_cache=False,logits_to_keep=1).logits.float()
                for m in layers:m.chunk_gated_delta_rule=q.torch_chunk_gated_delta_rule
                slow=model(**inputs,use_cache=False,logits_to_keep=1).logits.float()
                for m in layers:m.chunk_gated_delta_rule=q.chunk_gated_delta_rule
                diff=float((fast-slow).abs().max());kl=float(torch.nn.functional.kl_div(fast.log_softmax(-1),slow.softmax(-1),reduction='batchmean'))
                assert kl<.01 and diff<1.0,(name,diff,kl)
                tokens=model.generate(**inputs,max_new_tokens=8,do_sample=False)
                assert tokens.shape[-1]>inputs['input_ids'].shape[-1]
            real.append({'source':name,'id':row['id'],'max_logit_delta':diff,'kl_divergence':kl,'generation_tokens':8})
        report.update(real_model_forward_and_generation=real,scope='kernel parity and real initial-model forward/generation; no training')
    atomic_write_json(ROOT/'outputs/full-dataset-runtime-check.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
