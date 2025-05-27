import pygame as pg
from OpenGL.GL import *

class App:
    
    def __init__(self):
        

#Init Application
        #Initialize pygame
        pg.init()
        #Initialize window and OpengGL
        pg.display.set_mode((640, 480), pg.OPENGL|pg.DOUBLEBUF)
        self.clock = pg.time.Clock()
        #Inialaze opengl window
        glClearColor(0.1, 0.2, 0.2, 1)

        self.mainloop()

        def mainloop(self):
            running = True
            while (running):
                for event in pg.event.get():
                    if(event.type == pg.quit):
                        running = False

                glClear(GL_COLOR_BUFFER_BIT)
                pg.display.flip()

                self.clock.tick(60)
