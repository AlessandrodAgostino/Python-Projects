import numpy

def predicate(inst_list, R):
	for inst in inst_list:
		if inst[0] == "S":
    	yield ("F",1)
    elif inst[0] == "F":
    	yield ("F",inst[1]*R)
    else:
	  	yield inst

instr = [("S")]
R = np.sqrt(2)

#%%
