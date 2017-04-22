from World import *
import numpy as np
from copy import copy
A = 4
ns = div_x*div_y
n = A*(div_x*div_y)
s_x = 225
s_y = 250
trump = Wall(10, 180 , 200, 50)

def softmax(theta , s):
    v_prob=[]
    for a in range(A):
        one_hot = aggre(s , a)
        prob = np.exp(np.dot(theta , one_hot))/(sum(np.exp(theta[(s-1)*A:(s-1)*A+A])))
        v_prob.append(prob)
    act = np.random.choice(np.arange(0,A),p=v_prob)
    w_prob=v_prob[a]
    return act , v_prob


def aggre(s , a):
    one_hot = np.zeros(n)
    one_hot[(s-1)*A+a]=1
    return one_hot

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


def test_policy(s, theta, n):
    rew_l=[]
    for i in range(n):
        print('Episode number', i+1)
        rew=0
        state = copy(s)
        # Call this function so the Pygame library can initialize itself
        pygame.init()
        screen = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])
        pygame.display.set_caption('Collision Avoidance')
        all_sprite_list = pygame.sprite.Group()
        wall_list = pygame.sprite.Group()
        a = np.random.uniform(0,1)
        if a <= .03:
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
        r=0
        while state != 2 and r!= 50:
            a = softmax(theta , state)
            convert_act(a[0], player)
            state = player.state
            r = copy(player.reward)
            all_sprite_list.update()
            screen.fill(BLACK)
            all_sprite_list.draw(screen)
            pygame.display.flip()
            rew += r
        rew_l.append(rew)
    return rew_l
