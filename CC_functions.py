import numpy as np
from scipy.optimize import minimize
#import time
from World import *
from copy import copy

A = 4
ns = div_x*div_y
n = A*(div_x*div_y)
s_x = 225
s_y = 250

#State and Action Agregation
def aggre(s , a):
    one_hot = np.zeros(n)
    one_hot[(s-1)*A+a]=1
    return one_hot

def aggre_s(s):
    one_hot = np.zeros(ns)
    one_hot[s]=1
    return one_hot

#softmax policy
def softmax(theta , s):
    v_prob=[]
    for a in range(A):
        one_hot = aggre(s , a)
        prob = np.exp(np.dot(theta , one_hot))/(sum(np.exp(theta[(s-1)*A:(s-1)*A+A])))
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


def PG_CC(s, alpha , beta, theta, lamb):
    N=100
    eps = 1e-2
    lambmax = 1e9
    for i in range(20):
        rew_l=[]
        grad_l=[]
        for j in range(N):
            print('Episode number', N*i+j+1)
            rew=0
            state = copy(s)
            # Call this function so the Pygame library can initialize itself
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
            step_1 = .01 / (i+1)**0.65
            step_2 = .01 / (i+1)**0.55
            r=0
            while state != 2 and r!= 500:
                # Produces random wall after hit
                a = softmax(theta , state)
                vec = np.zeros(n)
                vec[(state-1)*A:(state-1)*A+A]=a[1]
                grad += aggre(state,a[0])-vec
                test=[int(r != 0) for r in aggre(state,a[0])-vec]
                if sum(test)%4!=0:
                    print('something is wrong')
                    print(aggre(state,a[0])-vec)
                    input()
                convert_act(a[0], player)
                state = player.state
                r = copy(player.reward)
                all_sprite_list.update()
                screen.fill(BLACK)
                all_sprite_list.draw(screen)
                pygame.display.flip()
                clock.tick(60)
                rew += r
            grad_l.append(grad)
            rew_l.append(rew)
        print('Last Episodes cost',rew_l)
        indic=[int(rr >= alpha) for rr in rew_l]
        right2 = 1/N*sum([grad_l[oo]*rew_l[oo] for oo in range(len(grad_l))]) + lamb*sum([grad_l[ok]*indic[ok] for ok in range(len(grad_l))])
        t = lambda x: 1/2*np.linalg.norm(theta-step_2*right2-x,2)**2
        #print(theta)
        #print(step_2*right2)
        #print(theta-step_2*right2)
        bnds = tuple([(-1e8,1e8)]*n)
        ta = minimize(t, theta, bounds=bnds)
        right3 = -beta+1/N*sum(indic)
        l = lambda x: 1/2*(lamb+step_1*right3-x)**2
        la = minimize(l, lamb, bounds=((0,lambmax),))
        theta = ta.x
        lamb = la.x
        if abs(lamb-lambmax) < eps:
            lambmax=2*lambmax
        print(step_1)
        print(step_2)
        #print('Current theta')
        #print(theta)
        print('Current lambda',lamb)
    return theta, lamb


def PG(s, theta):
    for i in range(2000):
        grad_l=[]
        rew=[]
        print('Episode number', i+1)
        state = copy(s)
        # Call this function so the Pygame library can initialize itself
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
        r=0
        step_1 = .00001
        while state != 2 and r!=500:
            a = softmax(theta , state)
            vec = np.zeros(n)
            vec[(state-1)*A:(state-1)*A+A]=a[1]
            grad = aggre(state,a[0])-vec
            convert_act(a[0], player)
            state = player.state
            all_sprite_list.update()
            screen.fill(BLACK)
            all_sprite_list.draw(screen)
            pygame.display.flip()
            clock.tick(60)
            rew.append(-1*player.reward)
            r=player.reward
            grad_l.append(grad)
        T = len(grad_l)
        print(sum(rew))
        for j in range(T):
            theta += step_1*sum(rew[j:T])*grad_l[j]
        #print(theta)
    return theta
