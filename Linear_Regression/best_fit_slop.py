from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')

xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)

# Scatter graph to see the data of xs and ys
# plt.scatter(xs, ys)
# plt.show()


def best_fit_slop_and_intercept(xs, ys):
    m = (((mean(xs) * mean(ys)) - mean(xs*ys)) /
         ((mean(xs) * mean(xs)) - mean(xs*xs)))
    b = mean(ys) - m*mean(xs)
    return m, b

m, b = best_fit_slop_and_intercept(xs, ys)

regression_line = [(m*x)+b for x in xs]


predict_x = 8
predict_y = (m*predict_x)+b

plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y, color = 'g')
plt.plot(xs, regression_line)
plt.show()