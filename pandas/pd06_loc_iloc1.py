import pandas as pd
print(pd.__version__) # 1.3.4

data = [
    ["삼성", "1000", "2000"],
    ["현대", "1100", "3000"],
    ["LG", "2000", "500"],
    ["아모레", "3500", "6000"],
    ["네이버", "100", "1500"],
]

index = ['031', '059', '033', '045', '023']
columns = ["종목명", "시가", "종가"]

df = pd.DataFrame(data=data, index=index, columns=columns)

print(df)
#      종목명    시가    종가
# 031   삼성   1000  2000
# 059   현대   1100  3000
# 033   LG     2000   500
# 045  아모레   3500  6000
# 023  네이버    100  1500
print("===================================================================")
# print(df[0])    # KeyError: 0
# print(df['031'])    # KeyError: '031'

# print(df["시가"])   # "판다스 열행" 이기때문에 """ 컬럼이 기준 """
# 031    1000
# 059    1100
# 033    2000
# 045    3500
# 023     100
# Name: 시가, dtype: object

###### 아모레 출력하고 싶어
# print(df[3, 0]) # KeyError: (3, 0)
# print(df['045','종목명'])   # KeyError: ('045', '종목명')
# print(df['종목명', '045'])  # KeyError: ('종목명', '045')
# print(df['종목명']['045']) # 아모레라고 잘 나옴. 판다스 열행이라 요거 나와
print("========================================")

######################################################
#### loc : 인덱스 기준으로 행 데이터 추출
#### iloc: 행번호를 기준으로 행 데이터 추출
    # 인트loc =  인트 로케이션!!! 이렇게 외워라!!!!!
######################################################

print("===================== 아모레 뽑자 ========================")

# print(df.iloc['045'])   # TypeError: Cannot index by location index with a non-integer key
# print(df.iloc[3])   # 잘뽑혀
# 종목명     아모레
# 시가     3500
# 종가     6000
# Name: 045, dtype: object

# print(df.loc[3])    # KeyError: 3
# print(df.loc["045"])    # 잘뽑혀
# 종목명     아모레
# 시가     3500
# 종가     6000
# Name: 045, dtype: object

print("===================== 네이버 뽑자 ========================")
# print(df.loc["023"])
# print(df.iloc[4])
# 종목명     네이버
# 시가      100
# 종가     1500
# Name: 023, dtype: object

print("===================== 아모레 종가 뽑자 ========================")
print(df.loc['045']['종가'])        # 6000
print(df.loc['045', '종가'])        # 6000
print(df.loc['045'].loc['종가'])    # 6000

print(df.iloc[3][2])                # 6000 # 판다스1에서는 되고 판다스2에서는 워닝뜨지만 되.
print(df.iloc[3, 2])                # 6000 
print(df.iloc[3].iloc[2])           # 6000 

print(df.loc['045'][2])             # 6000
# print(df.loc['045', 2])             # KeyError: 2

print(df.iloc[3]['종가'])           # 6000
# print(df.iloc[3,'종가'])            # ValueError: Location based indexing can only have [integer, integer slice (START point is INCLUDED, END point is EXCLUDED), listlike of integers, boolean array] types

print(df.loc['045'].iloc[2])       # 6000
print(df.iloc[3].loc['종가'])       # 6000



