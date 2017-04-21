from World import *
import numpy as np

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

def test_policy(state, theta, n):
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
        r=0
        while state != 2 and r!= 500:
            a = softmax(theta , state)
            convert_act(a[0], player)
            state = player.state
            r = copy(player.reward)
            all_sprite_list.update()
            screen.fill(BLACK)
            all_sprite_list.draw(screen)
            pygame.display.flip()
            clock.tick(60)
            rew += r
        rew_l.append(rew)
    return rew_l
