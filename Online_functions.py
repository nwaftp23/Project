import numpy as np
from scipy.optimize import minimize
#import time
from World import *
from copy import copy

A = 4
s_space = 5
ns = div_x*div_y*s_space
n = A*(div_x*div_y*s_space)
s_x = 325
s_y = 350

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


def AC_SPSA(s, sp , theta, w, var, lamb, alpha, beta):
    for i in range(100):
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
            a = softmax(theta, sp , state)
            convert_act(a[0], player)
            r = copy(player.reward)
            rew.append(r)
            num_sp -= r
            state_sp = convert_aug(sp)
            delta = r + np.dot(aug_aggre_s(player.state , state_sp),w)-np.dot(aug_aggre_s(state, sp),w)
            #step_1
            #step_2
            #step_3
            #step_4
            #Delta =
            vec = np.zeros(n)
            vec[(state-1)*s_space*A+sp*A:(state-1)*s_space*A+sp*A]=a[1]
            grad = aggre(state,a[0])-vec
            right1 = np.dot(w, INSERT)/2*Delta
            v =  lambda x: 1/2*((var-step_3*(lamb+right1))-x)**2
            va = minimize(v, var)
            w += step_4*delta*aug_aggre_s(state, sp)
            right2 = grad*delta
            t = lambda x : 1/2*((theta-step_2*(right2))-x)**2
            ta = minimize(t, theta)
            right3 = var - beta +1/(1-alpha)*max(0,-sp)*int(state==div_x+2)
            l = lambda x : 1/2*((lamb-step_1*(right3))-x)**2
            state = player.state
            all_sprite_list.update()
            screen.fill(BLACK)
            all_sprite_list.draw(screen)
            pygame.display.flip()
            clock.tick(60)
    return theta


def AC_Semi(s, sp , theta, w, var, lamb, alpha, beta):
    for i in range(100):
        rew =[]
        print('Episode number', i+1)
        state = copy(s)
        pygame.init()
        screen = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])
        pygame.display.set_caption('Collision Avoidance')
        all_sprite_list = pygame.sprite.Group()
        wall_list = pygame.sprite.Group()
        a = np.random.uniform(0,1)
        trump = Wall(10, 100 , 300, 15)
        if a < .1:
            wall_list.add(trump)
            all_sprite_list.add(trump)
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
        step_1 = .1 / (i+1)
        step_2 = .1 / (i+1)**0.85
        step_3 = .05 / (i+1)**0.7
        step_4 = .05 / (i+1)**0.55
        lambmax = 5000
        num_sp = sp
        r=0
        while state != div_x+2:
            if r == 100:
                a = np.random.uniform(0,1)
                wall_list.remove(trump)
                all_sprite_list.remove(trump)
                if a < .1:
                    print('yes wall')
                    trump = Wall(10, 100 , 300, 15)
                    wall_list.add(trump)
                    all_sprite_list.add(trump)
            sp = convert_aug(num_sp)
            a = aug_softmax(theta, sp , state)
            convert_act(a[0], player)
            r = copy(player.reward)
            rew.append(r)
            num_sp -= r
            state_sp = convert_aug(num_sp)
            delta = r + np.dot(aug_aggre_s(player.state , state_sp),w)-np.dot(aug_aggre_s(state, sp),w)
            vec = np.zeros(n)
            vec[(state-1)*s_space*A+sp*A:(state-1)*s_space*A+sp*A+A]=a[1]
            grad = aug_aggre(state,sp,a[0])-vec
            w += step_4*delta*aug_aggre_s(state, sp)
            right2 = grad*delta
            t = lambda x : 1/2*np.linalg.norm((theta-step_2*(right2))-x,2)**2
            bnds = tuple([(-50,50)]*n)
            ta = minimize(t, theta, bounds= bnds)
            right3 = var - beta +1/(1-alpha)*max(0,-sp)*int(state==div_x+2)
            l = lambda x : 1/2*((lamb-step_1*(right3))-x)**2
            la = minimize(l, lamb, tol=1e-6, bounds=((0,lambmax),))
            theta = ta.x
            lamb = la.x
            state = player.state
            all_sprite_list.update()
            screen.fill(BLACK)
            all_sprite_list.draw(screen)
            pygame.display.flip()
            clock.tick(60)
        right1 = lamb-lamb/(1-alpha)
        v =  lambda x: 1/2*((var-step_3*(lamb+right1))-x)**2
        va = minimize(v, var, bounds= ((-100,100),))
        var = va.x
        eps = 1e-3
        if abs(lamb-lambmax) < eps:
            lambmax=2*lambmax
    return theta , w , var, lamb
