from sklearn.datasets import load_boston, load_iris

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

#1. 데이터
datasets = load_iris()

df = pd.DataFrame(datasets.data, columns=datasets.feature_names)
# print(df)
#      sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
# 0                  5.1               3.5                1.4               0.2
# 1                  4.9               3.0                1.4               0.2
# 2                  4.7               3.2                1.3               0.2
# 3                  4.6               3.1                1.5               0.2
# 4                  5.0               3.6                1.4               0.2
# ..                 ...               ...                ...               ...
# 145                6.7               3.0                5.2               2.3
# 146                6.3               2.5                5.0               1.9
# 147                6.5               3.0                5.2               2.0
# 148                6.2               3.4                5.4               2.3
# 149                5.9               3.0                5.1               1.8

# [150 rows x 4 columns]

n_splits=3
kfold = KFold(n_splits=n_splits, shuffle=False,) # random_state=333
# kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=333)

for train_index, val_index in kfold.split(df):
    print("=========================================")
    print(train_index, '\n', val_index)
    print('훈련데이터 개수 : ', len(train_index), " ",
          '검증데이터 갯수', len(val_index))

# ********** shuffle=True하고 print한것.
# ========================================= 
# [  0   1   3   4   5   7   8   9  10  11  12  13  14  16  19  20  22  23
#   25  26  29  30  31  34  35  37  39  40  42  44  45  46  47  48  49  51
#   52  53  54  57  58  59  60  61  62  63  65  66  67  68  71  72  73  76
#   80  81  82  83  84  86  88  90  91  92  93  96  99 101 102 104 106 108
#  109 110 111 112 113 114 115 116 117 118 121 122 123 124 125 127 130 131
#  132 133 134 137 141 142 143 144 145 149]
#  [  2   6  15  17  18  21  24  27  28  32  33  36  38  41  43  50  55  56
#   64  69  70  74  75  77  78  79  85  87  89  94  95  97  98 100 103 105
#  107 119 120 126 128 129 135 136 138 139 140 146 147 148]
# 훈련데이터 개수 :  100   검증데이터 갯수 50
# =========================================
# [  2   3   5   6   7  12  13  15  17  18  19  20  21  23  24  26  27  28
#   29  31  32  33  35  36  37  38  40  41  43  46  47  50  51  54  55  56
#   57  62  63  64  65  66  67  69  70  71  72  74  75  77  78  79  81  83
#   85  87  88  89  90  91  94  95  96  97  98 100 101 103 105 106 107 109
#  110 111 112 116 118 119 120 123 124 125 126 127 128 129 132 135 136 137
#  138 139 140 142 144 145 146 147 148 149]
#  [  0   1   4   8   9  10  11  14  16  22  25  30  34  39  42  44  45  48
#   49  52  53  58  59  60  61  68  73  76  80  82  84  86  92  93  99 102
#  104 108 113 114 115 117 121 122 130 131 133 134 141 143]
# 훈련데이터 개수 :  100   검증데이터 갯수 50
# =========================================
# [  0   1   2   4   6   8   9  10  11  14  15  16  17  18  21  22  24  25
#   27  28  30  32  33  34  36  38  39  41  42  43  44  45  48  49  50  52
#   53  55  56  58  59  60  61  64  68  69  70  73  74  75  76  77  78  79
#   80  82  84  85  86  87  89  92  93  94  95  97  98  99 100 102 103 104
#  105 107 108 113 114 115 117 119 120 121 122 126 128 129 130 131 133 134
#  135 136 138 139 140 141 143 146 147 148]
#  [  3   5   7  12  13  19  20  23  26  29  31  35  37  40  46  47  51  54
#   57  62  63  65  66  67  71  72  81  83  88  90  91  96 101 106 109 110
#  111 112 116 118 123 124 125 127 132 137 142 144 145 149]
# 훈련데이터 개수 :  100   검증데이터 갯수 50

# ********** shuffle=False을 빼고 # random_state=333을 주석처리하고 print한것.
# ========================================= 
# [ 50  51  52  53  54  55  56  57  58  59  60  61  62  63  64  65  66  67
#   68  69  70  71  72  73  74  75  76  77  78  79  80  81  82  83  84  85
#   86  87  88  89  90  91  92  93  94  95  96  97  98  99 100 101 102 103
#  104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121
#  122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139
#  140 141 142 143 144 145 146 147 148 149]
#  [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
#  24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
#  48 49]
# 훈련데이터 개수 :  100   검증데이터 갯수 50
# =========================================
# [  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17
#   18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35
#   36  37  38  39  40  41  42  43  44  45  46  47  48  49 100 101 102 103
#  104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121
#  122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139
#  140 141 142 143 144 145 146 147 148 149]
#  [50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73
#  74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97
#  98 99]
# 훈련데이터 개수 :  100   검증데이터 갯수 50
# =========================================
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
#  24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
#  48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71
#  72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95
#  96 97 98 99]
#  [100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117
#  118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135
#  136 137 138 139 140 141 142 143 144 145 146 147 148 149]
# 훈련데이터 개수 :  100   검증데이터 갯수 50