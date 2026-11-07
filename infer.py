"""当前项目预测入口；图片按辅助图片在前、光学图片在后排列。"""

import argparse
from pathlib import Path
from xiaomi_lens.domain_runtime import load, predict, seed_all


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", nargs="+", required=True)
    p.add_argument("--checkpoint", help="微调结果文件；省略时使用初始模型")
    p.add_argument("--output", required=True)
    p.add_argument("--rgb-only", action="store_true")
    a = p.parse_args()
    print("使用微调模型" if a.checkpoint else "使用初始模型（尚未领域微调）")
    seed_all()
    model, processor = load(a.checkpoint)
    a.rgb_only = a.rgb_only or getattr(model, "_domain_rgb_only", False)
    if a.rgb_only and len(a.images) > 1:
        print("该模型仅使用最后一张光学图片。")
    _, result = predict(
        model,
        processor,
        {"images": [str(Path(x).resolve()) for x in a.images]},
        a.output,
        a.rgb_only,
    )

    print(result["answer"])
    print("按预测区域计算：" + result["derived_description"])
    for issue in result.get("consistency_issues", []):
        print("需要核对：" + issue)
    print("结果已保存到：" + str(Path(a.output).resolve()))


if __name__ == "__main__":
    main()
