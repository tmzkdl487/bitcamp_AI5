import langchain
import openai
print(langchain.__version__)    # 0.3.7
print(openai.__version__)       # 1.54.3

openai_api_key='' # 키 값 넣기

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model_name='gpt-3.5-turbo',
                 temperature=0,
                #  openai_api_key=openai_api_key,
                 api_key=openai_api_key
                )

aaa= llm.invoke('비트캠프 윤영선에 대해 알려줘.').content

print(aaa)
