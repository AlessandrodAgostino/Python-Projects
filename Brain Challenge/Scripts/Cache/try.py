history = []

class foo():
    def __init__(self, history):
        self.history = history

    def append_sth(self, sth):
        self.history.append(sth)

try1 = foo(history)
try1.history
try1.append_sth(3)
try1.history
history
try1.append_sth(3)
try1.history
history

try2 = foo(history)
try2.append_sth(4)
try2.history
history
