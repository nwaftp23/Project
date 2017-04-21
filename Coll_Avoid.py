from CC_functions import *
from Test_policy import *
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


"""Notes changed rewards to 100 from 500 also changed alpha from 300 to 100 changed lambda from 100"""
state = math.floor(div_y*(s_y/SCREEN_HEIGHT))*div_x+math.ceil(div_y*(s_x/SCREEN_WIDTH))
alpha = 500
beta = .95
theta = np.zeros(n)
w = np.random.randn(ns)
var = 10
lamb = 100
print(state)

np.random.seed(11)
#CMDP = PG_CC(state, alpha , beta, theta, lamb)
MDP = PG(state, theta)
#MDP_AC = AC(state,theta,w)
#MDP_ACCC = AC_CC(state, var , theta, w, lamb, alpha, beta)
#theta2=CMDP[0]
theta2 = MDP

m=1000
x = test_policy(state, theta2, m)
num_bins = 20

fig, ax = plt.subplots()

# the histogram of the data
n, bins, patches = ax.hist(x, num_bins, normed=1)

# add a 'best fit' line
ax.set_xlabel('cost')
ax.set_ylabel('Frequency')
ax.set_title(r'CMDP')

# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()
plt.show()
