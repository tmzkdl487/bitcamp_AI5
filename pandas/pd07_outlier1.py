import numpy as np
aaa = np.array([-10, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 50])

def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out,    # percentile 백분율
                                               [25, 50, 75])
    
    print("1사분위 : ", quartile_1)  # 4.0
    print("q2 : ", q2)              # 중위값 : 7.0
    print("3사분위 : ", quartile_3)  # 10.0
    iqr = quartile_3 - quartile_1   # 10.0 - 4.0 = 6.0
    print("iqr : ", iqr)
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out>upper_bound) |
                    (data_out<lower_bound)), iqr
    
outliers_loc, iqr = outliers(aaa)
print("이상치의 위치 : ", outliers_loc)

import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.axhline(iqr, color='red', label='TQR')
plt.show()

# 1사분위 :  4.0
# q2 :  7.0
# 3사분위 :  10.0
# iqr :  6.0
# 이상치의 위치 :  (array([ 0, 12], dtype=int64),)


