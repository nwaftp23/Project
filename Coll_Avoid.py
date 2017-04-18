from CVaR_functions import *



state = math.floor(div_y*(s_y/SCREEN_HEIGHT))*div_x+math.ceil(div_y*(s_x/SCREEN_WIDTH))
alpha = .90
beta = 350
theta = np.zeros(n)
theta[0::4]=.73
theta[2::4]=.73
w = np.random.randn(ns)
var = 100
lamb = np.random.randn(1)

CMDP = PG_CVAR(state, alpha , beta, theta, var, lamb)
#MDP = PG(state, theta)
#MDP_AC = AC(state,theta,w)
#MDP_Semi = AC_Semi(state, var , theta, w, var, lamb, alpha, beta)
