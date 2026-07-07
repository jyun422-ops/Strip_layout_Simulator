# run.py
import os
import sys
from streamlit.web import cli

if __name__ == '__main__':
    # exe로 빌드되었을 때와 파이썬으로 실행할 때의 경로를 맞춰줍니다.
    if getattr(sys, 'frozen', False):
        dir_name = sys._MEIPASS
    else:
        dir_name = os.path.dirname(os.path.abspath(__file__))

    # 우리가 만든 메인 파이썬 파일의 이름을 지정합니다. (예: app.py)
    app_path = os.path.join(dir_name, 'Strip_layout_Simulator.py') 
    
    # 터미널에서 streamlit run app.py 를 입력한 것과 동일한 효과를 냅니다.
    sys.argv = ["streamlit", "run", app_path, "--global.developmentMode=false"]
    sys.exit(cli.main())
