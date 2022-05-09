# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 09:29:08 2022

@author: marvi
"""
import pyfirmata
from pyfirmata import Arduino, util
import time

board = Arduino("COM3")
it = util.Iterator(board)
it.start()


#loopTimes = input('How many times would you like the LED to blink: ') 
#print("Blinking " + loopTimes + " times.") 
board.analog[1].enable_reporting()
board.analog[0].enable_reporting()
LED = board.digital[9]
LED.mode = pyfirmata.PWM
while True:
    
    #board.digital[7].write(1)
    #board.digital[8].write(1)
    LED.write(0.03)
    time.sleep(1)
    print(board.analog[1].read())
    #print(signal)
    
    #time.sleep(1) 
    temp_raw = board.analog[0].read()*(144/0.1408)
    temp_1 = (temp_raw*5/1024-0.5)*100
    print(temp_1)
    LED.write(0)
    #board.digital[8].write(0) 
    #board.analogWrite(9,0)
    while temp_1>35:
        print(temp_1)
        print('led to hot')
        time.sleep(3)
        temp_raw = board.analog[0].read()*(144/0.1408)
        temp_1 = (temp_raw*5/1024-0.5)*100
    #board.digital[7].write(0)
    time.sleep(1) 
    