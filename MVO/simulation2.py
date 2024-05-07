import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import yfinance as yf
import matplotlib

matplotlib.use('Qt5Agg')

# 삼성전자와 SK하이닉스의 티커 심볼 설정
tickers = ['226490.KS', '411060.KS']

# yfinance를 사용하여 주가 데이터 다운로드
data = yf.download(tickers, start='2023-01-01', end='2023-12-31')

# 'Adj Close' 데이터 가져오기
adj_close = data['Adj Close']

# 로그 수익률 계산
returns = np.log(adj_close / adj_close.shift(1)).dropna()

# 공분산 행렬 계산
cov_matrix = returns.cov()
covMat = cov_matrix.to_numpy()

# 한글 폰트 설정
from matplotlib import rc
rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False

# w1 범위를 [-1, 1]로 설정
w1 = np.linspace(-1, 1, 400)
# 각 w1 값에 대해 |w1| + |w2| = 1을 만족하는 w2 값을 계산
w2_pos = 1 - np.abs(w1)  # w2의 양의 값
w2_neg = -1 + np.abs(w1)  # w2의 음의 값

# 각 w1, w2 쌍에 대한 포트폴리오 분산 Z 계산
Z_pos = np.array([np.array([w1[i], w2_pos[i]]).T @ covMat @ np.array([w1[i], w2_pos[i]]) for i in range(len(w1))])
Z_neg = np.array([np.array([w1[i], w2_neg[i]]).T @ covMat @ np.array([w1[i], w2_neg[i]]) for i in range(len(w1))])

# Create the 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface for the quadratic form
ax.scatter(w1, w2_pos, Z_pos, color='blue', label='w2 positive')
ax.scatter(w1, w2_neg, Z_neg, color='red', label='w2 negative')

# Labels and title
ax.set_title('Portfolio Variance with |w1| + |w2| = 1 Constraint')
ax.set_xlabel('w1 axis')
ax.set_ylabel('w2 axis')
ax.set_zlabel('Portfolio Variance')

ax.legend()

plt.show()
