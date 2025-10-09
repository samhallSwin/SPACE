import sys
import numpy as np

from PyQt5.QtWidgets import QApplication, QMainWindow, QOpenGLWidget

from OpenGL.GL import *
from OpenGL.GLU import *

from skyfield.api import load
from skyfield.api import EarthSatellite

from skyfield.constants import ERAD

class Satellite():
    #A class to hold all information relating to a satellite to not overcrowd a dictionary
    def __init__(self, satellite):
        self.satellite = satellite
        self.positions = self.PopulateSatellitePositions()
        self.show = True #When needed its here

        #Constants
        self.color = self.Color()
        self.sphereRadius = 0.03

    def Color(self): #Temp Color for differentiation, will change to label instead and also be able to show or hide label
        tempColor = hash(self.satellite.model.satnum)
        return (tempColor%253/255, tempColor%254/255, tempColor%255/255)

    def PopulateSatellitePositions(self):
        year = self.satellite.model.epochyr
        day, month = divmod(self.satellite.model.epochdays, 12) #Split the days of epoch day into months and days in month
        revolutionsPerDay = self.satellite.model.no_kozai * 229.1831 #MinutesInADay/2Pi
        
        orbitalPeriod = 1 / (revolutionsPerDay - 1) #-1 to make orbits overlap slightly so other orbits can close

        orbitResolution = 1000 #How smooth the orbit is

        ts = load.timescale()
        times = ts.utc(year, month, np.linspace(day, day+orbitalPeriod, orbitResolution))

        positions = []
        for t in times:
            geocentric = self.satellite.at(t)
            x,y,z = (pos/(ERAD/1000) for pos in geocentric.position.km)
            positions.append((x,y,z))

        return positions

    def DrawLabel(self, quadric):
        if not self.show: return
        pass

    def DrawSatellite(self, quadric, position=0):
        if not self.show: return
        x,y,z = self.positions[position%len(self.positions)]
        glColor3f(*self.color)
        glPushMatrix()
        glTranslate(x,y,z)
        gluSphere(quadric, self.sphereRadius, 40, 40)
        glPopMatrix()

    def DrawOrbit(self):
        if not self.show: return
        glColor3f(*self.color)
        glLineWidth(2)
        glBegin(GL_LINE_STRIP)
        for x,y,z in self.positions:
            glVertex3f(x,y,z)
        glEnd()

class Satellites(QOpenGLWidget):#Technically not needed, just here to show the satellites and their orbits, otherwise a list of Satellite objects would do
    def __init__(self, backend, parent=None):
        super().__init__(parent)
        
        self.backend = backend
        self.backend.subscribe(self.update_satellites)

        self.satellites = {}
        self.update_satellites()
        
        self.quadric = gluNewQuadric()
        self.position = 0
        
    def Quadric(self, quad=None):
        if quad:
            self.quadric = quad
        else:
            return self.quadric

    #If sticking with the read_tle_file function from the sat_sim_handler, then this code works, otherwise use other read_file function
    def update_satellites(self):
        
        self.satellites.clear()

        tle_data = self.backend.get_data_from_enabled_tle_slots()

        if not tle_data:
            print("Null passed")
            return

        print(f"SATELLITES - passed in {len(tle_data)} items")
        
        for i in tle_data:
            name = str(i)
            line_1 = tle_data[name][0]
            line_2 = tle_data[name][1]

            satellite = EarthSatellite(line_1, line_2, name, None)
            self.satellites[name] = Satellite(satellite)

    def Draw(self):
        for satellite in self.satellites.values():
            satellite.DrawSatellite(self.quadric, self.position)
            satellite.DrawOrbit()
