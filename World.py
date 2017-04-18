import pygame
import random
import math

# -- Global constants

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (50, 50, 255)

# Screen dimensions
SCREEN_WIDTH = 300
SCREEN_HEIGHT = 300
div_x = 5
div_y = 5


class Player(pygame.sprite.Sprite):
    """ This class represents the bar at the bottom that the player
    controls. """
    def __init__(self, x, y):
        super().__init__()

        # Set height, width
        self.image = pygame.Surface([10, 10])
        self.image.fill(WHITE)
        self.rect = self.image.get_rect()
        self.rect.y = y
        self.rect.x = x
        self.start_y = y
        self.start_x = x
        self.reward = 0
        self.parti_y = div_y
        self.parti_x = div_x
        self.change_x = 0
        self.change_y = 0
        self.walls = None
        self.state = math.floor(self.parti_y*(self.rect.y/SCREEN_HEIGHT))*self.parti_x+math.ceil(self.parti_x*(self.rect.x/SCREEN_WIDTH))
        self.sanity = 0

    def changespeed(self, x, y):
        """ Change the speed of the player. """
        if self.change_x >= 3 and x > 0:
            pass
        elif self.change_x <= -3 and x < 0:
            pass
        else:
            self.change_x += x
            if self.change_x > 3:
                self.change_x = 3
            elif self.change_x < -3:
                self.change_x = -3
        if self.change_y >= 3 and y > 0:
            pass
        elif self.change_y <= -3 and y < 0:
            pass
        else:
            self.change_y += y
            if self.change_y > 3:
                self.change_y =3
            elif self.change_y < -3:
                self.change_y =-3
        #print('current x speed ', self.change_x)
        #print('current y speed ', self.change_y)


    def update(self):
        """ Update the player position. """
        # Move left/right
        self.rect.x += self.change_x

        # Did this update cause us to hit a wall?
        block_hit_list = pygame.sprite.spritecollide(self, self.walls, False)
        for block in block_hit_list:
            # If we are moving right, set our right side to the left side of
            # the item we hit
            self.rect.y = self.start_y
            self.rect.x = self.start_x
            self.reward = 100
            self.sanity = 1
            """if self.change_x > 0:
                self.rect.right = block.rect.left
            else:
                # Otherwise if we are moving left, do the opposite.
                self.rect.left = block.rect.right"""
        # Move up/down
        self.rect.y += self.change_y

        # Check and see if we hit anything
        block_hit_list = pygame.sprite.spritecollide(self, self.walls, False)
        for block in block_hit_list:

            # Reset our position based on the top/bottom of the object.
            self.rect.y = self.start_y
            self.rect.x = self.start_x
            self.reward = 100
            self.sanity = 1
            """if self.change_y > 0:
                self.rect.bottom = block.rect.top
            else:
                self.rect.top = block.rect.bottom"""
        if self.state != div_x+2 and self.sanity != 1:
            self.reward = 1
        if self.state == div_x+2:
            self.reward = -3000
            self.rect.y = self.start_y
            self.rect.x = self.start_x
        self.state = math.floor(self.parti_y*(self.rect.y/SCREEN_HEIGHT))*self.parti_x+math.ceil(self.parti_x*(self.rect.x/SCREEN_WIDTH))
        self.sanity = 0




class Wall(pygame.sprite.Sprite):
    """ Wall the player can run into. """
    def __init__(self, x, y, width, height):
        """ Constructor for the wall that the player can run into. """
        # Call the parent's constructor
        super().__init__()

        # Make a blue wall, of the size specified in the parameters
        self.image = pygame.Surface([width, height])
        self.image.fill(BLUE)

        # Make our top-left corner the passed-in location.
        self.rect = self.image.get_rect()
        self.rect.y = y
        self.rect.x = x
