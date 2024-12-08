# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib

# aaa = np.array([[-10, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 50],
#                 [100, 200, -30, 400, 500, 600, -70000, 800, 900, 1000, 210, 420, 350]]).T

#### for문 돌려서 맹그러봐 ####

# def outliers(data_out):
#     quartile_1, q2, quartile_3 = np.percentile(data_out,    # percentile 백분율
#                                                [25, 50, 75])
    
#     print("1사분위 : ", quartile_1)  # 4.0
#     print("q2 : ", q2)              # 중위값 : 7.0
#     print("3사분위 : ", quartile_3)  # 10.0
#     iqr = quartile_3 - quartile_1   # 10.0 - 4.0 = 6.0
#     print("iqr : ", iqr)
#     lower_bound = quartile_1 - (iqr * 1.5)
#     upper_bound = quartile_3 + (iqr * 1.5)
#     return np.where((data_out>upper_bound) |
#                     (data_out<lower_bound)), iqr
    
# outliers_loc, iqr = outliers(aaa)
# print("이상치의 위치 : ", outliers_loc)

# import matplotlib.pyplot as plt
# plt.boxplot(aaa)
# plt.axhline(iqr, color='red', label='TQR')
# plt.show()

# 1사분위 :  5.25
# q2 :  11.5
# 3사분위 :  387.5
# iqr :  382.25
# 이상치의 위치 :  (array([6, 9], dtype=int64), array([1, 1], dtype=int64))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

aaa = np.array([[-10, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 50],
                [100, 200, -30, 400, 500, 600, -70000, 800, 900, 1000, 210, 420, 350]]).T

# 한글 폰트 설정
matplotlib.rc('font', family='Malgun Gothic')

def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out, [25, 50, 75])

    print("1사분위 : ", quartile_1)
    print("q2 : ", q2)
    print("3사분위 : ", quartile_3)
    
    iqr = quartile_3 - quartile_1
    print("iqr : ", iqr)

    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)

    return np.where((data_out > upper_bound) | (data_out < lower_bound)), iqr

# 각 열에 대해 이상치 계산
for col_idx in range(aaa.shape[1]):
    print(f"\n열 {col_idx+1} 분석:")
    outliers_loc, iqr = outliers(aaa[:, col_idx])
    print("이상치의 위치 : ", outliers_loc[0])

    # 박스 플롯 그리기
    plt.boxplot(aaa[:, col_idx], vert=False)
    plt.axvline(iqr, color='red', label='IQR')
    plt.title(f"열 {col_idx+1} 박스플롯")
    plt.legend()
    plt.show()


