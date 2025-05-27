import pygame as pg
from OpenGL.GL import *
import numpy as np

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

#Gameloop
    def mainloop(self):
        running = True
        while (running):
            #Check events
            for event in pg.event.get():
                if(event.type == pg.QUIT):
                        running = False
            #Refresh screen
            glClear(GL_COLOR_BUFFER_BIT)
            pg.display.flip()

            #Framerate lock
            self.clock.tick(60)
        self.quit()

    def quit(self):
        pg.quit


class  Sphere:
     def __init__(self):
          
        self.verticies  = (
            -0.5, -0.5, 0.0, 1.0, 0.0, 0.0,
            0.5, -0.5, 0.0, 0.0, 1.0, 0.0,
            0.0, 0.5, 0.0, 0.0, 0.0, 1.0
        )

        #Numpy arrary datatype converts verticies into a format readable by gpu
        self.verticies = np.arrary(self.vertices, dtype=np.float32)

        self.vertex_count = 3

        self.veo = glGenVertexArrays(1)
        glBindVertexArray(self.veo)
        #Vertex buffer object
        self.vbo = glGenBuffers(1)
        glBinBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self,verticies,nbytes, self,verticies, GLSTATIC_DRAW)

        

if __name__ == "__main__":
     myApp  = App()


