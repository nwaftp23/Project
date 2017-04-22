from CC_functions import *
from Test_policy import *
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


"""Notes changed rewards to 100 from 500 also changed alpha from 300 to 100 changed lambda from 100"""
state = math.floor(div_y*(s_y/SCREEN_HEIGHT))*div_x+math.ceil(div_y*(s_x/SCREEN_WIDTH))
alpha = 70
beta = .02
theta = np.zeros(n)
w = np.random.randn(ns)
# Assistance for plain PG
theta[0::4]+=1
theta[2::4]+=1
var = 10
lamb = 100
print(state)

np.random.seed(10)
CMDP = PG_CC(state, alpha , beta, theta, lamb)
MDP = PG(state, theta)
#MDP_AC = AC(state,theta,w)
#MDP_ACCC = AC_CC(state, var , theta, w, lamb, alpha, beta)
theta2=CMDP[0]
theta3 = MDP

m=10000
x = test_policy(state, theta2, m)
x2 = test_policy(state, theta3, m)
num_bins = 20

fig, ax = plt.subplots()

# the histogram of the data
n, bins, patches = ax.hist(x, x2, num_bins, normed=1)

# add a 'best fit' line
ax.set_xlabel('cost')
ax.set_ylabel('Frequency')
ax.set_title(r'CMDP')

# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()
plt.show()
