import numpy as np
from scipy.optimize import minimize
#import time
from World import *
from copy import copy

A = 4
ns = div_x*div_y
n = A*(div_x*div_y)
s_x = 325
s_y = 350

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


def PG_CVAR(s, alpha , beta, theta, var, lamb):
    N=1
    for i in range(1000):
        rew=[]
        grad_l=[]
        print('Episode number', i+1)
        for j in range(N):
            state = copy(s)
            # Call this function so the Pygame library can initialize itself
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
            r=0
            while state != 10:
                # Produces random wall after hit
                if r == 100:
                    a = np.random.uniform(0,1)
                    wall_list.remove(trump)
                    all_sprite_list.remove(trump)
                    if a < .1:
                        print('yes wall')
                        trump = Wall(10, 100 , 300, 15)
                        wall_list.add(trump)
                        all_sprite_list.add(trump)
                a = softmax(theta , state)
                vec = np.zeros(n)
                vec[(state-1)*A:(state-1)*A+A]=a[1]
                grad += aggre(state,a[0])-vec
                convert_act(a[0], player)
                state = player.state
                r = copy(player.reward)
                all_sprite_list.update()
                screen.fill(BLACK)
                all_sprite_list.draw(screen)
                pygame.display.flip()
                clock.tick(60)
            grad_l.append(grad)
            rew.append(player.reward)
        indic=[int(r >= var) for r in rew]
        step_1 = 1/((i+1)**(3/4))
        right1 = (lamb/((1-alpha)*N))*sum(indic)
        v =  lambda x: 1/2*((var-step_1*(lamb-right1))-x)**2
        va = minimize(v, var)
        step_2 = 1/((i+1)**(4/5))
        right2 = 1/N*sum([grad_l[i]*rew[i] for i in range(len(grad_l))]) + (lamb/((1-alpha)*N))*sum([grad_l[i]*(rew[i]-var)*indic[i] for i in range(len(grad_l))])
        t = lambda x: 1/2*np.linalg.norm(theta-step_2*right2-x,2)**2
        ta = minimize(t, theta)
        step_3 = 1/((i+1))
        right3 = var-beta+1/((1-alpha)*N)*sum([(rew[i]-var)*indic[i] for i in range(len(grad_l))])
        l = lambda x: 1/2*(lamb-step_3*right3-x)**2
        la = minimize(l, lamb)
        var = va.x
        theta = ta.x
        lamb = la.x
    return theta, var, lamb


def PG(s, theta):
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
        wall = Wall(10, 100 , 300, 25)
        wall_list.add(wall)
        all_sprite_list.add(wall)
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
            state = player.state
            all_sprite_list.update()
            screen.fill(BLACK)
            all_sprite_list.draw(screen)
            pygame.display.flip()
            clock.tick(60)
            rew.append(player.reward)
            grad_l.append(grad)
        T = len(grad_l)
        for j in range(T):
            #step_1 = 0.5/(i+1)
            step_1 = 10**-4
            theta -= step_1*sum(rew[j:T])*grad_l[j]
        print(theta)
    return theta