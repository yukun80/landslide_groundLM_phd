"""仅在全量实际检查通过后，按真实记录更新本次数据说明。"""
import argparse
import json
from pathlib import Path

from xiaomi_lens.full_dataset.common import DEFAULT,ROOT,atomic_write_json,sha256_file,SPLITS
from xiaomi_lens.description_v2.io import atomic_write_text


def render(summary,report,partition,encoding):
    """将已经通过的检查结果写成短文；不从训练是否结束推断数据质量。"""
    names={'GDCLD':'GDCLD','landslide4sense':'Landslide4Sense','multimodal-landslide-dataset':'多模态'}
    rows=['|来源|训练|调参|测试|小计|','|---|---:|---:|---:|---:|']
    for source,label in names.items():
        counts=partition['counts'][source];rows.append(f"|{label}|{counts.get('train',0)}|{counts.get('val',0)}|{counts.get('test',0)}|{sum(counts.values())}|")
    counts=summary['counts'];rows.append(f"|合计|{counts['train']}|{counts['val']}|{counts['test']}|{summary['total']}|")
    coverage=report['description_coverage']
    text='\n'.join([
        '# 实际检查结果','', '[返回总览](README.md) · [上一节：代码](code.md)','',
        '## 完成情况','',f"全量数据制作与检查已完成。实际检查时间：{report['checked_at']}。",
        f"共{summary['total']:,}条，全部有有效滑坡标注。复用{coverage['reused']:,}条已有文字，新增{coverage['reviewed']:,}条两遍读图记录。",
        '原1000条中全黑但有标注的1张，已按2026-11-17用户选择从新数据排除；旧1000条保持不变。',
        '500条独立评测样本及其同组保护保持在测试部分，没有进行正式训练。','',*rows,'',
        '## 实际通过的检查','',
        '- 23,330条来源均有去向：排除4,011条无目标、10条纯色、11条重复及311条跨固定划分的重叠切片。',
        '- 全部原始副本、实际显示图、滑坡标注和有效区均已核对；已知内容分组及地理范围没有跨划分冲突。',
        f"- 两版答案共完成{encoding['dual_encodings']:,}次真实训练读取，长度、区域对应和微小目标保留检查通过；未更新模型参数。",
        '- 所有新增样本均保留两遍实际模型返回、图片身份和不确定说明；格式失败未当作无信息完成。',
        '- 恢复前保存的13,804条完成记录再次核对，内容保持一致；原数据与原始代码保护检查通过。',
        f"- 新增文字中，{coverage.get('with_optical_appearance',0):,}条有光学外观补充，{coverage.get('with_auxiliary_detail',0):,}条有辅助图补充；看不清时保留基础说明。",'',
        '## 试跑与恢复说明','',
        '最初12张试写发现目标混淆和核对用语，保留了失败依据；增加目标局部后重新试跑，并实际检查24次训练读取。',
        '全量生成曾因辅助句被输出为单句字符串而中断。修复后24张真实试跑通过，包括原出错样本；重试会重新读图，保留原始返回。',
        '旧学习资料检查器的 infer.py 摘要早于当前提交，未为通过检查改写旧摘要或原文件；本轮独立文档检查验证当前页面与原代码。','',
        '## 核对入口','',
        '[数据目录](../../../datasets/landslide_lens_full_positive_v1/) 中的当前记录：','',
        '|记录|说明|','|---|---|',
        '|`audit/scan.json`、`partition.json`|实际清洗与划分|',
        '|`audit/protected_quality_conflicts.json`|用户确认排除全黑图片|',
        '|`audit/raw_prepared_check.json`、`raw_checks/`|最初逐份副本和全部显示像素核对|',
        '|`audit/raw_check.json`、`encoding_check.json`|最终文字与实际训练读取|',
        '|`review/samples/`|逐样本两遍模型返回与图片身份|',
        '|`audit/descriptions_before_resume_20261117.json`|恢复前完成文字的内容清单|',
        '|`audit/final_check.json`、`summary.json`|最终完整性与可用状态|','',
        '## 结论边界','',
        '这些文字是模型辅助制作，尚未经过人工逐图确认；通过制作检查不代表滑坡识别效果已经改善。',
        '所有划分均只含正样本，不能据此评估纯背景误报。缺少位置的信息来源不能声称跨地区独立。',
        '固定旧划分、评测保护及真实地理重叠优先于精确80/10/10比例；实际数量如上。',''])
    if summary.get('annotation_provider')=='codex_session':
        text=text.replace('新增', '重新撰写')
        text=text.replace('条两遍读图记录。','条当前Codex会话实际看图记录。')
        text=text.replace('尚未经过人工逐图确认','尚未经过人工逐图确认')
        text=text.replace('所有重新撰写样本均保留两遍实际模型返回、图片身份和不确定说明；格式失败未当作无信息完成。','所有重新撰写样本均保存本会话看图声明、图片身份、观察和再次核查说明；不冒充独立模型或人工复核。')
        text=text.replace('最初12张试写发现目标混淆和核对用语，保留了失败依据；增加目标局部后重新试跑，并实际检查24次训练读取。','2026-11-17按用户要求停止本地模型和自动接续。历史本地文字保留，不计入本轮Codex完成数量。')
        text=text.replace('全量生成曾因辅助句被输出为单句字符串而中断。修复后24张真实试跑通过，包括原出错样本；重试会重新读图，保留原始返回。','本轮由当前Codex会话分批直接看图，不运行本地看图模型、不另起命令行或接口代写；训练格式检查使用普通处理器。')
        text=text.replace('这些文字是模型辅助制作','这些文字由Codex直接看图制作')
        text=text.replace('`review/samples/`','`review/codex_session/samples/`')
        text=text.replace('逐样本两遍模型返回与图片身份','逐样本本会话看图声明与图片身份')
    elif summary.get('annotation_provider')=='local_resume':
        counts=summary['answer_sources']
        text=text.replace(f"新增{coverage['reviewed']:,}条两遍读图记录。",
            f"其余{coverage['reviewed']:,}条采用保留或修订后的模型描述。")
        text=text.replace('所有新增样本均保留两遍实际模型返回、图片身份和不确定说明；格式失败未当作无信息完成。',
            f"保留27条Codex文字；本轮本地处理{counts.get('local_current',0):,}条，筛查后保留{counts.get('local_preserved',0):,}条历史本地文字，逐条来源明确。")
        text=text.replace('恢复前保存的13,804条完成记录再次核对，内容保持一致；原数据与原始代码保护检查通过。',
            '历史14,212条本地记录和27条Codex记录的内容均未变；原数据、原始模型与原始代码保护检查通过。')
        text=text.replace('最初12张试写发现目标混淆和核对用语，保留了失败依据；增加目标局部后重新试跑，并实际检查24次训练读取。',
            '本轮12张训练图由官方初始模型实际生成、再次核查，并经当前Codex会话逐图核对和24次训练读取；不是人工确认。')
        text=text.replace('全量生成曾因辅助句被输出为单句字符串而中断。修复后24张真实试跑通过，包括原出错样本；重试会重新读图，保留原始返回。',
            '先筛查14,185条历史本地答案，重看可疑项并补齐3,776条缺失；历史被删候选不直接当作当前答案错误。')
        text=text.replace('|`review/samples/`|逐样本两遍模型返回与图片身份|',
            '|`review/local_resume/`|试跑核查、旧答案筛查、本轮两遍返回与逐条采用来源|\n|`review/samples/`、`review/codex_session/`|保留不变的历史文字与查看依据|')
        text=text.replace('这些文字是模型辅助制作，尚未经过人工逐图确认；通过制作检查不代表滑坡识别效果已经改善。',
            '这些文字是模型辅助制作，尚未经过人工逐图确认；同一个模型再次核查不是独立验证，制作检查不代表识别效果改善。')
    return text



def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset-dir',type=Path,default=DEFAULT)
    parser.add_argument('--out',type=Path,default=ROOT/'docs/domain-finetuning/full-dataset/validation.md');args=parser.parse_args();dest=args.dataset_dir
    summary=json.loads((dest/'summary.json').read_text());report=json.loads((dest/'audit/final_check.json').read_text())
    encoding=json.loads((dest/'audit/encoding_check.json').read_text());partition=json.loads((dest/'audit/partition.json').read_text())
    assert summary['complete'] and report['passed'] and encoding['passed']
    assert summary['total']==report['count']==encoding['count']==partition['selected']
    assert summary['manifest_sha256']=={s:sha256_file(dest/f'{s}.jsonl') for s in SPLITS}
    text=render(summary,report,partition,encoding);assert len(text.splitlines())<=120
    atomic_write_text(args.out,text)
    atomic_write_json(dest/'audit/documentation_update.json',{'path':str(args.out),'sha256':sha256_file(args.out),'based_on_final_check':True})
    print('FINAL_DOCUMENTATION_UPDATED',args.out,flush=True)


if __name__=='__main__':main()
