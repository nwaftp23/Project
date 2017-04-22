from World import *
import numpy as np


s_x = 225
s_y = 250
pygame.init()
screen = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])
pygame.display.set_caption('Collision Avoidance')
all_sprite_list = pygame.sprite.Group()
wall_list = pygame.sprite.Group()
a = np.random.uniform(0,1)
trump = Wall(10, 150 , 200, 15)
if a <= .9:
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
state=53
tot_reward = 0
j=0
r=0
while state != div_x+2 and r!= 500:


    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                player.changespeed(-3, 0)
            elif event.key == pygame.K_RIGHT:
                player.changespeed(3, 0)
            elif event.key == pygame.K_UP:
                player.changespeed(0, -3)
            elif event.key == pygame.K_DOWN:
                player.changespeed(0, 3)

        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT:
                player.changespeed(3, 0)
            elif event.key == pygame.K_RIGHT:
                player.changespeed(-3, 0)
            elif event.key == pygame.K_UP:
                player.changespeed(0, 3)
            elif event.key == pygame.K_DOWN:
                player.changespeed(0, -3)

    state = player.state
    tot_reward +=  player.reward
    print(tot_reward)
    r=player.reward
    all_sprite_list.update()

    screen.fill(BLACK)

    all_sprite_list.draw(screen)

    pygame.display.flip()
    j +=1
    clock.tick(60)
