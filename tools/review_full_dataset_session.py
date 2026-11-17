"""当前Codex会话的图片准备与记录入口，不运行任何看图模型。"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from xiaomi_lens.full_dataset.session_review import main
if __name__=='__main__':main()
