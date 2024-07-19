# 환경 변수 방법
#########################################
'''
1. 시작 - 환경 변수 (친다.) - 계정의 환경 변수 편집
2. 사용자 변수에 '새로만들기' 클릭
3. 변수이름: OPENAI_API_KEY
4. 변수 값: 'sk-블라블라 키값'
--끝-- 

'''

# lang03.py 카피

###########################

import langchain
import openai
print(langchain.__version__)    # 0.3.7
print(openai.__version__)       # 1.54.3

# openai_api_key='' # 키값 넣기

# import os
# os.environ['OPENAI_API_KEY'] = openai_api_key

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model_name='gpt-3.5-turbo',
                 temperature=0,
                #  openai_api_key=openai_api_key,
                #  api_key=openai_api_key
                )

aaa= llm.invoke('비트캠프 윤영선에 대해 알려줘.').content

print(aaa)
