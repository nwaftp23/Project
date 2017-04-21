import numpy as np
from scipy.optimize import minimize
#import time
from World import *
from copy import copy

A = 4
s_space = 5
ns = div_x*div_y*s_space
n = A*(div_x*div_y*s_space)
s_x = 225
s_y = 250

# Augmented MDP
# Don't Quit You Are Almost there!!
# Believe Like You Once Did!!

#Fuck Augmented bullshit!!!
def convert_aug(sp):
    if sp > -50:
        sk = 0
    elif -200 < sp < -50:
        sk = int(np.floor(sp/-50))
    else:
        sk = 4
    return sk

#State and Action Agregation
def aug_aggre(s, sp , a):
    one_hot = np.zeros(n)
    one_hot[(s-1)*A*s_space+sp*A+a]=1
    return one_hot

def aug_aggre_s(s , sp):
    one_hot = np.zeros(ns)
    one_hot[(s-1)*s_space+sp]=1
    return one_hot

#softmax policy
def aug_softmax(theta, sp , s):
    v_prob=[]
    for a in range(A):
        one_hot = aug_aggre(s, sp , a)
        prob = np.exp(np.dot(theta , one_hot))/(sum(np.exp(theta[(s-1)*A*s_space+sp*A:(s-1)*A*s_space+sp*A+A])))
        v_prob.append(prob)
    act = np.random.choice(np.arange(0,A),p=v_prob)
    w_prob=v_prob[a]
    return act , v_prob


def convert_act(a, player):
    if a==0:
        player.changespeed(-3, 0)
        #time.sleep(3)
        #player.changespeed(3, 0)
    elif a==1:
        player.changespeed(3, 0)
        #time.sleep(3)
        #player.changespeed(-3, 0)
    elif a==2:
        player.changespeed(0, -3)
        #time.sleep(3)
        #player.changespeed(0, 3)
    else:
        player.changespeed(0, 3)
        #time.sleep(3)
        #player.changespeed(0, -3)


def AC(s, theta, w):
    for i in range(100):
        rew=[]
        grad_l=[]
        print('Episode number', i+1)
        state = copy(s)
        pygame.init()
        screen = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])
        pygame.display.set_caption('Collision Avoidance')
        all_sprite_list = pygame.sprite.Group()
        wall_list = pygame.sprite.Group()
        #wall = Wall(10, 100 , 300, 25)
        #wall_list.add(wall)
        #all_sprite_list.add(wall)
        wall = Wall(10, 0, 390, 10)
        wall_list.add(wall)
        all_sprite_list.add(wall)
        wall = Wall(0, 0, 10, 400)
        wall_list.add(wall)
        all_sprite_list.add(wall)
        wall = Wall(10, 390, 390, 10)
        wall_list.add(wall)
        all_sprite_list.add(wall)
        wall = Wall(390, 10, 10, 380)
        wall_list.add(wall)
        all_sprite_list.add(wall)
        # Create the player paddle object
        player = Player(s_x, s_y)
        player.walls = wall_list
        all_sprite_list.add(player)
        clock = pygame.time.Clock()
        grad=np.zeros(n)
        while state != div_x+2:
            a = softmax(theta , state)
            vec = np.zeros(n)
            vec[(state-1)*A:(state-1)*A+A]=a[1]
            grad = aggre(state,a[0])-vec
            convert_act(a[0], player)
            all_sprite_list.update()
            screen.fill(BLACK)
            all_sprite_list.draw(screen)
            pygame.display.flip()
            clock.tick(60)
            step_1 = 10**-4
            step_2 = 10**-4
            delta = player.reward + np.dot(aggre_s(player.state),w)-np.dot(aggre_s(state),w)
            w +=  step_1*delta*aggre_s(state)
            theta -= step_2*delta*grad
            rew.append(player.reward)
            grad_l.append(grad)
            state = player.state
    return theta




def AC_CC(s, sp , theta, w, lamb, alpha, beta):
    for i in range(1000):
        rew =0
        print('Episode number', i+1)
        state = copy(s)
        pygame.init()
        screen = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])
        pygame.display.set_caption('Collision Avoidance')
        all_sprite_list = pygame.sprite.Group()
        wall_list = pygame.sprite.Group()
        a = np.random.uniform(0,1)
        trump = Wall(10, 150 , 200, 15)
        if a <= .15:
            wall_list.add(trump)
            all_sprite_list.add(trump)
        wall = Wall(10, 0, 290, 10)
        wall_list.add(wall)
        all_sprite_list.add(wall)
        wall = Wall(0, 0, 10, 300)
        wall_list.add(wall)
        all_sprite_list.add(wall)
        wall = Wall(10, 290, 290, 10)
        wall_list.add(wall)
        all_sprite_list.add(wall)
        wall = Wall(290, 10, 10, 280)
        wall_list.add(wall)
        all_sprite_list.add(wall)
        # Create the player paddle object
        player = Player(s_x, s_y)
        player.walls = wall_list
        all_sprite_list.add(player)
        clock = pygame.time.Clock()
        grad=np.zeros(n)
        step_1 = .1 / (i+1)
        step_2 = .1 / (i+1)**0.85
        step_3 = .05 / (i+1)**0.7
        lambmax = 5000
        num_sp = copy(sp)
        r=0
        grad_l = []
        delta_l=[]
        sp_l=[]
        state_l=[]
        while state != div_x+2 and r != 500:
            sp = convert_aug(num_sp)
            a = aug_softmax(theta, sp , state)
            convert_act(a[0], player)
            r = copy(player.reward)
            rew += r
            num_sp -= r
            sp_l.append(sp)
            state_l.append(state)
            state_sp = convert_aug(num_sp)
            delta = r + np.dot(aug_aggre_s(player.state , state_sp),w)-np.dot(aug_aggre_s(state, sp),w)
            vec = np.zeros(n)
            vec[(state-1)*s_space*A+sp*A:(state-1)*s_space*A+sp*A+A]=a[1]
            grad_l.append(aug_aggre(state,sp,a[0])-vec)
            state = player.state
            all_sprite_list.update()
            screen.fill(BLACK)
            all_sprite_list.draw(screen)
            pygame.display.flip()
            clock.tick(60)
            delta_l.append(delta)
        w += step_3*sum([delta_l[e]*aug_aggre_s(state_l[e], sp_l[e]) for e in range(len(delta_l))])
        right2 = sum([delta_l[ee]*grad[ee] for ee in range(len(delta_l))])
        t = lambda x : 1/2*np.linalg.norm((theta-step_2*(right2))-x,2)**2
        bnds = tuple([(-5,5)]*n)
        ta = minimize(t, theta, bounds= bnds)
        right3 = -beta + int(num_sp<=0)
        l = lambda x : 1/2*((lamb+step_1*(right3))-x)**2
        la = minimize(l, lamb, bounds=((0,lambmax),))
        theta = ta.x
        lamb = la.x
        eps = 1e-3
        print('Cost of last episode', rew)
        if abs(lamb-lambmax) < eps:
            lambmax=2*lambmax
    return theta , w , lamb
