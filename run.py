import sys
import os

# Erebus가 이 파일을 robot0Controller.py로 복사해서 실행하므로
# __file__ 기반 경로가 src/를 가리키지 않을 수 있음.
# 아래 경로를 이 PC의 실제 src/ 절대 경로로 유지할 것.
src_dir = r'C:\Users\learnsteam\Desktop\study\src'

# 혹시 직접 실행하는 경우 대비 fallback
if not os.path.isdir(src_dir):
    src_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, src_dir)

import main
