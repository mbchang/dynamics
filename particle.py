import cPickle as pickle
import pygame
from pygame.locals import *
from numpy import *
from pygame.color import THECOLORS
import pygame.gfxdraw as gfx

class Particle:
    def __init__(self, screen, size, path, background, pcolor, fieldcolor):
        """
            path is a list of tuples: [(x0, y0), (x1, y1), ... (xn, yn)] for t = 0 ... n
        """
        self.screen = screen
        screensize = self.screen.get_size()
        self.screenwidth = screensize[0]
        self.screenheight = screensize[1]
        # Position of particle on the Screen
        # Particle will start at first position of path
        self.path = path
        self.x = path[0][0]  # the first index is the timestep, [0] is x. self.x is initialized to x in first timestep
        self.y = path[0][1]  # the first index is the timestep, [1] is y. self.y is initialized to y in first timestep
        self.frame = 0
        self.fieldwidth = int(size[0])
        self.width = self.fieldwidth - 5
        self.height = int(size[1])

        self.bgcolor = background
        self.pcolor = pcolor
        self.fieldcolor = fieldcolor
        self.rect = pygame.rect.Rect(self.x, self.y, self.width, self.height)

    def draw(self, debug):
        # Erase the previous particle

        # Update frame number or loop back to first

        if self.frame >= len(self.path):
            self.frame = 0

        # Update position
        # Check for collision with the sides:

        nx, ny = self.path[self.frame][0], self.path[self.frame][1]
        self.frame = self.frame + 1
        if debug: print 'x:',nx,'y:',ny

        # Draw the new particle
        self.rect = pygame.rect.Rect(nx, ny, self.width, self.height)
        pygame.draw.circle( self.screen, self.pcolor, (int(nx),int(ny)), self.width , 0 )
        if self.pcolor != THECOLORS["white"]:
             # pygame.draw.circle( self.screen, (0,0,0), (int(nx),int(ny)), self.width , 8 )

            gfx.aacircle(self.screen, int(nx),int(ny), self.fieldwidth , (0,0,0))
            gfx.aacircle(self.screen, int(nx),int(ny), self.fieldwidth+1 , (0,0,0))
            gfx.aacircle(self.screen, int(nx),int(ny), self.width , (0,0,0))
            ##            pygame.draw.circle(self.screen, THECOLORS["black"], (int(nx),int(ny)), self.fieldwidth, 1)
            pygame.draw.circle(self.screen, self.fieldcolor, (int(nx),int(ny)), self.fieldwidth, 5)

##            for a_radius in arange(1,self.fieldwidth-self.width+1):
##
##                gfx.aacircle(self.screen, int(nx),int(ny), self.fieldwidth-a_radius , (255,255,255))

    def setBackgroundColor(self, color):
        self.bgcolor=color
    def setParticleColor(self, color):
        self.pcolor=color
