"""Compare a real, three-source batch with the existing single-image path on CPU."""
from contextlib import closing
from threading import enumerate as active_threads

import torch
from transformers import AutoProcessor
from xiaomi_lens.domain_runtime import BASE, build, configure_processor
from xiaomi_lens.description_v2 import select_answer_row
from xiaomi_lens.full_dataset.common import DEFAULT, NAMES,read_jsonl
from xiaomi_lens.train.domain_batch import batch_inputs, training_batches


def main():
    torch.set_num_threads(1)
    proc = configure_processor(AutoProcessor.from_pretrained(BASE,local_files_only=True,use_fast=False))
    rows = read_jsonl(DEFAULT/'train.jsonl')
    selected = [next(r for r in rows if r['source']==name) for name in NAMES]
    for optical in (True,False):
        batch = batch_inputs(proc,selected,optical,'cpu')
        cursor = 0
        for i,row in enumerate(selected):
            single = build(proc,select_answer_row(row,optical),optical,training=True,device='cpu')
            length = single['input_ids'].shape[1]
            for key in ('input_ids','labels','attention_mask','mm_token_type_ids'):
                assert torch.equal(batch[key][i,:length], single[key][0])
            assert not batch['attention_mask'][i,length:].any() and (batch['labels'][i,length:]==-100).all()
            h,w = single['mask_targets'].shape[-2:]
            for key in ('mask_targets','mask_pixel_valid'):
                assert torch.equal(batch[key][i,:,:h,:w],single[key][0])
                assert not batch[key][i,:,h:].any() and not batch[key][i,:,:,w:].any()
            assert torch.equal(batch['mask_image_index'][i],single['mask_image_index'][0])
            assert torch.equal(batch['mask_positions'][i],single['mask_positions'][0])
            n = len(single['image_grid_thw'])
            assert torch.equal(batch['image_grid_thw'][cursor:cursor+n],single['image_grid_thw'])
            start = batch['image_grid_thw'][:cursor].prod(dim=1).sum().item()
            assert torch.equal(batch['pixel_values'][start:start+len(single['pixel_values'])],single['pixel_values'])
            cursor += n
    # Compare every field and all update boundaries, including the two real tail sizes.
    for tail in (1, 3):
        samples = (selected * 7)[:16 + tail]
        for micro in (3, 8):
            baseline = list(training_batches(proc, samples, False, 8, micro, 0, "cpu"))
            ahead = training_batches(proc, samples, False, 8, micro, 1, "cpu")
            seen = []
            for expected, actual in zip(baseline, ahead, strict=True):
                start, chunk, fields = expected
                assert actual[:2] == (start, chunk) and actual[2].keys() == fields.keys()
                seen.extend(r["id"] for r in chunk)
                for key, value in fields.items():
                    other = actual[2][key]
                    if torch.is_tensor(value):
                        assert other.device.type == "cpu" and value.dtype == other.dtype
                        assert torch.equal(value, other), key
                    else:
                        assert value == other
            assert seen == [r["id"] for r in samples]
    threads = set(active_threads())
    with closing(training_batches(proc, selected * 4, False, 8, 8, 1, "cpu")) as ahead:
        next(ahead)
    assert not (set(active_threads()) - threads)
    broken = dict(selected[0], id="prefetch-failure-check", images=["nonexistent.png"])
    try:
        list(training_batches(proc, selected + [broken], False, 8, 3, 1, "cpu"))
    except ValueError as exc:
        assert broken["id"] in str(exc) and isinstance(exc.__cause__, FileNotFoundError)
    else:
        raise AssertionError("missing image should fail")
    assert not (set(active_threads()) - threads)
    print('PASS: image order, pixels, labels, mask alignment and padding in both modes')


if __name__=='__main__':
    main()
