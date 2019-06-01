%%file my_first_app.py

def mysum(a, b):
    return a+b

print("this will be executed everytime")

if __name__=='__main__':
    print("this will be executed only on the command line")