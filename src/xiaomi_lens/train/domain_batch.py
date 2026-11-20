"""Batch already encoded domain samples, preserving per-sample image indices."""
from concurrent.futures import ThreadPoolExecutor
import json
import hashlib
import torch
from torch.nn.functional import pad
from ..domain_runtime import build
from ..description_v2 import select_answer_row


def merged_snapshot(directory):
    summary = json.loads((directory / 'summary.json').read_text())
    receipt = json.loads((directory / 'review/merged_descriptions/verified.json').read_text())
    hashes = {s: hashlib.sha256((directory / f'{s}.jsonl').read_bytes()).hexdigest()
              for s in ('train', 'val', 'test')}
    if (receipt != summary.get('description_merge') or hashes != receipt['split_sha256']
            or hashes != summary['manifest_sha256'] or receipt['actual_reads'] != 2 * summary['total']):
        raise ValueError('合并记录与现用数据不一致')
    return receipt


def batch_inputs(processor, rows, rgb, device="cuda"):
    items = []
    for row in rows:
        try:
            items.append(build(processor, select_answer_row(row, rgb), rgb, training=True, device='cpu'))
        except Exception as exc:
            raise ValueError(f"准备样本 {row['id']} 失败") from exc
    if len(items) == 1:
        return {k: v.to(device) if torch.is_tensor(v) else v for k,v in items[0].items()}
    length = max(item['input_ids'].size(1) for item in items)
    gh, gw = (max(item['mask_canvas_grid'][axis] for item in items) for axis in (0, 1))
    height, width = (max(item['mask_targets'].size(axis) for item in items) for axis in (-2,-1))
    result = {}
    for key in items[0]:
        if key == 'mask_canvas_grid':
            result[key] = (gh, gw)
            continue
        values = [x[key] for x in items]
        if key in ('input_ids','attention_mask','mm_token_type_ids','labels','token_type_ids'):
            fill = processor.tokenizer.pad_token_id if key == 'input_ids' else (-100 if key == 'labels' else 0)
            values = [pad(v,(0,length-v.size(1)),value=fill) for v in values]
        elif key in ('mask_targets','mask_pixel_valid'):
            values = [pad(v,(0,width-v.size(-1),0,height-v.size(-2))) for v in values]
        elif key not in ('pixel_values','image_grid_thw','mask_positions','mask_image_index','mask_valid_mask'):
            raise ValueError(f'Unsupported batch field: {key}')
        result[key] = torch.cat(values).to(device)
    return result

def training_batches(processor, rows, rgb, group, micro_batch, prefetch, device="cuda"):
    """Keep update boundaries intact and prepare at most one future batch on CPU."""
    chunks = (
        (start, rows[start:min(start + micro_batch, base + group, len(rows))])
        for base in range(0, len(rows), group)
        for start in range(base, min(base + group, len(rows)), micro_batch)
    )
    if not prefetch:
        for start, chunk in chunks:
            yield start, chunk, batch_inputs(processor, chunk, rgb, device)
        return

    with ThreadPoolExecutor(max_workers=1) as worker:
        current = next(chunks, None)
        if current is None:
            return
        future = worker.submit(batch_inputs, processor, current[1], rgb, "cpu")
        while current is not None:
            start, chunk = current
            prepared = future.result()
            current = next(chunks, None)
            if current is not None:
                future = worker.submit(batch_inputs, processor, current[1], rgb, "cpu")
            yield start, chunk, {
                k: v.to(device) if torch.is_tensor(v) else v for k, v in prepared.items()
            }
            del prepared
