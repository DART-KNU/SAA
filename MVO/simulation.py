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

# 기대 수익률 계산 (평균 일일 로그 수익률)
expected_returns = returns.mean()

# plot 한글 보이게
from matplotlib import rc
rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False

# 누적 로그 수익률 계산
cumulative_log_returns = returns.cumsum().apply(np.exp)

# Simulating the calculation of expected returns (mean of daily returns)
expected_returns_simulated = returns.mean()

# Calculating volatility (standard deviation of returns) for each asset
volatility = returns.std()

# Assuming a risk-free rate of 0.05% for the Sharpe Ratio calculation
risk_free_rate = 0

# Calculating Sharpe Ratio for each asset
sharpe_ratio = (expected_returns_simulated - risk_free_rate) / volatility

covMat = cov_matrix.to_numpy()
ER1 = expected_returns.to_numpy()[0]
ER2 = expected_returns.to_numpy()[1]

# w1과 w2를 생성하되, 절대값의 합이 1을 만족하도록 설정
w1 = np.linspace(-1, 1, 400)
w2 = 1 - np.abs(w1)

# Z 값을 계산하기 위한 배열 생성
Z = np.array([np.array([w1i, w2i]).T @ covMat @ np.array([w1i, w2i]) for w1i, w2i in zip(w1, w2)])

# Create the 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=90, azim=30)

# Plot the surface for the quadratic form
ax.plot(w1, w2, Z, color='blue')

# Labels and title
ax.set_title('Portfolio Variance with Absolute Sum Constraint')
ax.set_xlabel('w1 axis')
ax.set_ylabel('w2 axis')
ax.set_zlabel('Portfolio Variance')

plt.show()