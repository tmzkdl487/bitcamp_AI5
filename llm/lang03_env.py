# lang02.py 카피
# .env 방법
###########################
'''
1. 루트에 .env 파일을 만든다.
2. 파일안에 키를 넣는다.
   .env 파일 내용
    OPENAI_API_KEY='sk-블라블라....'
3. .env가 깃에 자동으로 안올라가도록 .gitignore 파일 안에 .env를 넣는다.
   .gitignore 내용:
   .env 
    
    끝!!!
'''
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
