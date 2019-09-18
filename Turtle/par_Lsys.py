#Trying to implement a parametric L-system
"""
Every command is a tuple: (action, arg):
	-(F,2): Forward segment of 2 * l unities, where l is the lenght unit.
	-(+,30): Right deviation of 30 degrees.
	-([, _): Stacking an action. "]" for popping.
	-("S", _): Save position

The complete instruction is a list of tuples.
A turtle finally execute a list of instr.

"""

import turtle
import itertools

def draw_l_system(turtle, 
			instruction_list, 
			positions,
			unit_length=1, 
			unit_angle=1):
    stack = []
    for command in instruction_list:
        turtle.pd()
        if command[0] in ["F"]:
            turtle.forward(command[1]*unit_length)
        elif command[0] == "f":
            turtle.pu()  # pen up - not drawing
            turtle.forward(command[1]*unit_length)
        elif command[0] == "+":
            turtle.right(command[1]*unit_angle)
        elif command[0] == "-":
        	turtle.left(command[1]*unit_angle)
        elif command[0] == "S":
        	positions.append(turtle.position())
        elif command[0] == "[":
            stack.append((turtle.position(), turtle.heading()))
        elif command[0] == "]":
            turtle.pu()  # pen up - not drawing
            position, heading = stack.pop()
            turtle.goto(position)
            turtle.setheading(heading)


screen = turtle.Screen()
alex = turtle.Turtle()
#instr = [ ("F",100), ("+", 30),("["),("+", 60),("F",50),("]"),("F",50), ]
instr = [("S")]
positions = []
draw_l_system(alex, instr, positions)


def predicate(inst_list, R):
    for inst in inst_list:
        if inst[0] == "S":
            yield ("F",1)
            yield ("[")
            yield ("+",1)
            yield ("S")
            yield ("]")
            yield ("[")
        elif inst[0] == "F":
            yield ("F",inst[1]*R)
	  else:
	  	yield inst

















