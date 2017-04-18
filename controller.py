from World import *

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
player = Player(325, 350)
player.walls = wall_list
all_sprite_list.add(player)
clock = pygame.time.Clock()
state=53
tot_reward = 0
j=0

while state != div_x+2:
    if j==0:
        wall = Wall(10, 100 , 300, 25)
        wall_list.add(wall)
        all_sprite_list.add(wall)
    elif j==150:
        print(j*100)
        wall_list.remove(wall)
        all_sprite_list.remove(wall)


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

    all_sprite_list.update()

    screen.fill(BLACK)

    all_sprite_list.draw(screen)

    pygame.display.flip()
    j +=1
    clock.tick(60)
