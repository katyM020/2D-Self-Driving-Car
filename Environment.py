# Self Driving Car

"""
Numpy library used for working with arrays in a faster way (50x faster )than the traditional python list. In addition to
this, it is one of the python libraries that works with multidimentional array object
"""
import numpy

'''Graphical plotting library'''
import matplotlib.pyplot as plt

"""Kivy here is used to create an application window or in other words it builds a desktop GUI"""
from kivy.app import App
from kivy.graphics import Color, Line
from kivy.uix.widget import Widget
from kivy.clock import Clock
#from kivy.config import Config
from kivy.uix.button import Button
from kivy.vector import Vector
from kivy.properties import (NumericProperty as NumPro, ReferenceListProperty as RefListProp, ObjectProperty as objProp)

'''Importing the Deep Q_Learning object from our AI in Car_AI.py'''
from Car_AI import Dqn

# Config.set('inumpyut', 'mouse', 'mouse,multitouch_on_demand')

'''
Introducing final_X_point_saved and final_Y_point_saved, used to keep the last point in memory when we draw the 
obstacle on the map
'''
final_X_point_saved = 0
final_Y_point_saved = 0
total_number_of_points = 0
magnitude_of_last_drawing = 0

# Getting our AI, which we call "Artificial_brain", and that contains our neural network that represents our Q-function
inputs, outputs, gamma = 5, 3, 0.9
Artificial_brain = Dqn(inputs, outputs, gamma)
rotation_of_the_car = [0, 10, -10]
finale_reward = 0
points = []

# Initializing the map
first_update = True


def init():
    global target_x
    global target_y
    global obstacle
    global first_update
    obstacle = numpy.zeros((length, width))
    target_x = 20
    target_y = width - 20
    first_update = False


# Initializing the last new_car_distance
last_car_distance = 0


# Creating the car class
class Car_Attributes(Widget):
    '''
    Initializing the important parameters of the car as it is part of kivy environment.

    For example the signal here, for each signal is responsible for measuring the density of the obstacle around one of
    the three sensors to avoid it. It is represented here as sensor_one_signal, sensor_two_signal, sensor_three_signal.
    Each sensor in the car which sense the sand in three different directions have an x-axis and y-axis value as the car
    passes through all of the environment so must obtain its coordinates.

    *NumPro() here is one of the kivy properties to validate/check that the value is one of the numeric values.
    *RefListProp() is used when x-axis and y-axis values used NumPro(). When the car change its
    position and this obviously expected through out the whole program, according to this position it will change the
    values of x and y correspondingly.
    '''

    angle = NumPro(0)
    rotation = NumPro(0)

    first_sensor_x_axis = NumPro(0)
    first_sensor_y_axis = NumPro(0)
    first_sensor = RefListProp(first_sensor_x_axis, first_sensor_y_axis)

    second_sensor_x_axis = NumPro(0)
    second_sensor_y_axis = NumPro(0)
    second_sensor = RefListProp(second_sensor_x_axis, second_sensor_y_axis)

    third_sensor_x_axis = NumPro(0)
    third_sensor_y_axis = NumPro(0)
    third_sensor = RefListProp(third_sensor_x_axis, third_sensor_y_axis)

    sensor_one_signal = NumPro(0)
    sensor_two_signal = NumPro(0)
    sensor_three_signal = NumPro(0)

    # the speed of the 2D car on x and y axis
    speed_on_x_axis = NumPro(0)
    speed_on_y_axis = NumPro(0)
    speed = RefListProp(speed_on_x_axis, speed_on_y_axis)

    def rotating_car(self, rotation):
        """
        This rotating_car function is responsible for the speed of the car, the rotation of it when it hits an obstacle.
        Also, the position of the sensor that is illustrated in line 120 --> 122. The vector here represents the
        distance between the car and the sensor itself, the .rotate is the rotation of the sensor. There is one in the
        middle which doesn't rotate and the other sensors rotate on both directions right and left for detecting the
        obstacle
        """
        self.pos = Vector(*self.speed) + self.pos
        self.rotation = rotation
        self.angle += self.rotation

        distance = 50
        shift = 30
        self.first_sensor = Vector(distance, 0).rotate(self.angle) + self.pos
        self.second_sensor = Vector(distance, 0).rotate((self.angle + shift) % 360) + self.pos
        self.third_sensor = Vector(distance, 0).rotate((self.angle - shift) % 360) + self.pos

        """
        The following lines define the area around each sensor that measures the existence of sand, to return 
        numeric value for each sensor depending on the sand intensity in the area around each sensor
        """
        area1 = obstacle[int(self.first_sensor_x_axis) - 20:int(self.first_sensor_x_axis) + 20,
            int(self.first_sensor_y_axis) - 20:int(self.first_sensor_y_axis) + 20]
        area2 = obstacle[int(self.second_sensor_x_axis) - 20:int(self.second_sensor_x_axis) + 20,
            int(self.second_sensor_y_axis) - 20:int(self.second_sensor_y_axis) + 20]
        area3 = obstacle[int(self.third_sensor_x_axis) - 20:int(self.third_sensor_x_axis) + 20,
            int(self.third_sensor_y_axis) - 20:int(self.third_sensor_y_axis) + 20]

        self.sensor_one_signal = int(numpy.sum(area1)) / 1600.
        self.sensor_two_signal = int(numpy.sum(area2)) / 1600.
        self.sensor_three_signal = int(numpy.sum(area3)) / 1600.

        """
        The 3 if conditions that is shown below illustrate that if one of the 3 three sensors detect the obstacle 
        in front of it will assign the signal of the sensor to 1 as to be send signal to the ai to change its direction
        for not losing its rewards
        """
        ProtectBorders = 10
        if self.first_sensor_x_axis > length - ProtectBorders or self.first_sensor_x_axis < ProtectBorders or \
                self.first_sensor_y_axis > width - ProtectBorders or self.first_sensor_y_axis < ProtectBorders:
            self.sensor_one_signal = 1.

        if self.second_sensor_x_axis > length - ProtectBorders or self.second_sensor_x_axis < ProtectBorders or\
                self.second_sensor_y_axis > width - ProtectBorders or self.second_sensor_y_axis < ProtectBorders:
            self.sensor_two_signal = 1.

        if self.third_sensor_x_axis > length - ProtectBorders or self.third_sensor_x_axis < ProtectBorders or \
                self.third_sensor_y_axis > width - ProtectBorders or self.third_sensor_y_axis < ProtectBorders:
            self.sensor_three_signal = 1.


class LeftSide(Widget):
    pass


class Middle(Widget):
    pass


class RightSide(Widget):
    pass


# Creating the game class

"""
In this class getting the car, ball1, ball2, and ball3 object from car.kv
"""


"""
Ths launching car function is responsible initialize the position of the car and the speed of it when kivy application
is launched 
"""

"""
This update function is a crucial one because every variable mentioned below need to be updated in each discrete time
when getting a new state or in other words when getting a new signal from the sensor that there is an obstacle 
"""

class Self_driving_car(Widget):
    vechile = objProp(None)
    leftside = objProp(None)
    middle = objProp(None)
    rightside = objProp(None)


    def launching_car(self):
        self.vechile.center = self.center
        self.vechile.speed = Vector(6, 0)


    def update(self, dt):

        global Artificial_brain
        global finale_reward
        global points
        global last_car_distance
        global target_x
        global target_y
        global length
        global width

        length = self.width
        width = self.height
        if first_update:
            init()

        # it is the difference between the car and the target on the x-axis
        distance_X_target_car = target_x - self.vechile.x

        # it is the difference between the car and the target on the y-axis
        distance_Y_target_car = target_y - self.vechile.y

        direction_of_the_car = Vector(*self.vechile.speed).angle((distance_X_target_car, distance_Y_target_car)) / 180.
        last_signal = [self.vechile.sensor_one_signal, self.vechile.sensor_two_signal, self.vechile.sensor_three_signal,
                       direction_of_the_car, -direction_of_the_car]
        AI_response = Artificial_brain.weightsUpdate(finale_reward, last_signal)
        points.append(Artificial_brain.avg_score())
        last_rotation = rotation_of_the_car[AI_response]
        self.vechile.rotating_car(last_rotation)
        new_car_distance = numpy.sqrt((self.vechile.x - target_x) ** 2 + (self.vechile.y - target_y) ** 2)
        self.leftside.pos = self.vechile.first_sensor
        self.middle.pos = self.vechile.second_sensor
        self.rightside.pos = self.vechile.third_sensor

        """
        In this if condition says if the car is on the obstacle itself will loss it rewards which will be -1. Adding 
        to this the speed of the car will be decreased into 1
        
        in the else that the car ignores the obstacle and passes through it will have it normal speed but will receive
        a bad reward as it is not its main functionality to be done, as it have to defend it so the car will be having
        the ability to learn from it and not passing through it again
        
        in case the car passes through the obstacle but it is near to the target will receive a small reward as it 
        reached it 
        """
        if obstacle[int(self.vechile.x), int(self.vechile.y)] > 0:
            self.vechile.speed = Vector(1, 0).rotate(self.vechile.angle)
            finale_reward = -1

        else:
            self.vechile.speed = Vector(6, 0).rotate(self.vechile.angle)
            finale_reward = -0.2
            if new_car_distance < last_car_distance:
                finale_reward = 0.1

        protectBorders = 10
        if self.vechile.x < protectBorders:
            self.vechile.x = protectBorders
            finale_reward = -1
        if self.vechile.x > self.width - protectBorders:
            self.vechile.x = self.width - protectBorders
            finale_reward = -1

        if self.vechile.y < protectBorders:
            self.vechile.y = protectBorders
            finale_reward = -1
        if self.vechile.y > self.height - protectBorders:
            self.vechile.y = self.height - protectBorders
            finale_reward = -1

        if new_car_distance < 100:
            target_x = self.width - target_x
            target_y = self.height - target_y
        last_car_distance = new_car_distance


# Adding the painting tools

class PainterWidget(Widget):

    def on_touch_down(self, t):
        global magnitude_of_last_drawing, total_number_of_points, final_X_point_saved, final_Y_point_saved
        with self.canvas:
            Color(1, 1, 0.33)
            d = 10.
            t.ud['line'] = Line(width=10, points=(t.x, t.y))
            final_X_point_saved = int(t.x)
            final_Y_point_saved = int(t.y)
            total_number_of_points = 0
            magnitude_of_last_drawing = 0
            obstacle[int(t.x), int(t.y)] = 1

    def on_touch_move(self, t):
        global magnitude_of_last_drawing, total_number_of_points, final_X_point_saved, final_Y_point_saved
        if t.button == 'left':
            t.ud['line'].points += [t.x, t.y]
            x = int(t.x)
            y = int(t.y)
            magnitude_of_last_drawing += numpy.sqrt(
                max((x - final_X_point_saved) ** 2 + (y - final_Y_point_saved) ** 2, 2))
            total_number_of_points += 1.
            dense = total_number_of_points / (magnitude_of_last_drawing)
            t.ud['line'].width = int(20 * dense + 1)
            obstacle[x - 10: x + 10, y - 10: y + 10] = 1
            final_X_point_saved = x
            final_Y_point_saved = y


# Adding the API Buttons (clear, save and load)

class CarApp(App):

    def build(self):
        parent = Self_driving_car()
        parent.launching_car()
        Clock.schedule_interval(parent.update, 1.0 / 60.0)
        self.paintr = PainterWidget()
        Clear_button = Button(text='clear', size =(100, 50))
        Save_button = Button(pos=(parent.width, 0), text='save', size =(100, 50))
        Load_button = Button(pos=(2 * parent.width, 0), text='load', size =(100, 50))
        Clear_button.bind(on_release=self.clear_canvas)
        Save_button.bind(on_release=self.SaveCurrentState)
        Load_button.bind(on_release=self.LoadState)
        parent.add_widget(self.paintr)
        parent.add_widget(Clear_button)
        parent.add_widget(Save_button)
        parent.add_widget(Load_button)
        return parent

    def clear_canvas(self, obj):
        global obstacle
        self.paintr.canvas.clear()
        obstacle = numpy.zeros((length, width))

    @staticmethod
    def SaveCurrentState(obj):
        print("saving Artificial_brain...")
        Artificial_brain.SaveState()
        plt.plot(points)
        plt.show()

    @staticmethod
    def LoadState(obj):
        print("loading last saved Artificial_brain...")
        Artificial_brain.loadLastState()


# Running the whole thing

if __name__ == '__main__':
    CarApp().run()
