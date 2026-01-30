


import threading
import time
import sys
from typing import List, Any
import sys
import os
sys.path.append(os.path.abspath(""))
from briann.utilities import callbacks as bpuc

class Observer():

    def __init__(self, on_separate_thread: bool = False):
        self._listeners = {}
        
        if on_separate_thread: 
            self._original_trace = threading.gettrace()
            threading.settrace(self.trace_function)
        else: 
            self._original_trace = sys.gettrace()
            sys.settrace(self.trace_function)

    def trace_function(self, frame, event, arg, indent=[0]):
        if self._original_trace: self._original_trace(frame, event, arg)

        # Ensure method is on watch list
        identifier = f"{frame.f_code.co_name}({list(frame.f_locals.keys())})"
        
        if identifier in self._listeners.keys():
                
                entries = self._listeners[identifier]
                for entry in entries:

                    # Ensure the function is on exit
                    if event == entry.event:

                        # No argument value constraints
                        if entry.required_argument_values == None:
                            entry.callback(frame.f_locals)

                        # Argument value constraints
                        else:
                            if all([value == frame.f_locals[key] for key, value in entry.required_argument_values.items()]):
                                entry.callback(frame.f_locals)
                    
        return self.trace_function

    def link(self, method_name: str, argument_names: List[str], callback: callable, event, required_argument_values = None):
        
        target = Observer.Entry(method_name=method_name, argument_names=argument_names, callback = callback, event=event, required_argument_values=required_argument_values)

        # Ensure function name is listened to
        if target.identifier not in self._listeners.keys():
            self._listeners[target.identifier] = []
        
        # Add outlet property to function name
        self._listeners[target.identifier].append(target)

    class Entry():

        def __init__(self, method_name, argument_names, callback, event, required_argument_values):
            self.method_name = method_name
            self.argument_names = argument_names
            self.callback = callback
            self.event = event
            self.required_argument_values = required_argument_values
            
            self.identifier = f"{self.method_name}({(self.argument_names)})"

if __name__ == "1__main__":   
    class A():

        def __init__(self, x):
            self._x = x

        @property
        def x(self):
            return self._x
        
        @x.setter
        def x(self, new_value):
            print("x")
            self._x = new_value

        def add(self, num):
            print("add")
            return self.x + num

    def other_function(num):
        print("other_function")

    # Callbacks
    def set_A_x_callback(args):
        print("set_A_x_callback")

    def A_add_callback(args):
        print("A_add_callback")

    def other_callback(args):
        print("Other callback")

    oldtrace = None
    try:
        #import pydevd
        #debugger=pydevd.GetGlobalDebugger()
        #if debugger is not None:
        #    oldtrace = [debugger.trace_dispatch]
        pass
    except ImportError:
        pass

    #if oldtrace is None:
    #    oldtrace = [frame.f_trace]

    # Regular example
    a = A(x=3)
    o = Observer()
    o.link(method_name='x', argument_names=['self', 'new_value'], callback=set_A_x_callback, event='return', required_argument_values={'self':a})
    o.link(method_name='add', argument_names=['self', 'num'], callback=A_add_callback, event='return', required_argument_values={'self':a})
    o.link(method_name='other_function', argument_names=['num'], callback=other_callback, event='return')
    
    a.x = 2
    a.add(11)
    other_function(num=5)

    print("\n\n\nMulti-threading example")
    # Multi-threading example
    a = A(x=3)
    t1 = threading.Thread(target=lambda x, a=a: setattr(a, 'x', x), args=[10])
    t2 = threading.Thread(target=A.add, args=[a,10])
    t3 = threading.Thread(target=other_function, args=[4])

    o = Observer(on_separate_thread=True)
    o.link(method_name='x', argument_names=['self', 'new_value'], callback=set_A_x_callback, event='return', required_argument_values={'self':a})
    o.link(method_name='add', argument_names=['self', 'num'], callback=A_add_callback, event='return', required_argument_values={'self':a})
    o.link(method_name='other_function', argument_names=['num'], callback=other_callback, event='return')
    
    t1.start()
    t2.start()
    t3.start()

    t1.join()
    t2.join()
    t3.join()

    print("Done!")

class A():

    def __init__(self, x):
        self.x=x

    @property
    def x(self): return self._x 

    @x.setter
    def x(self, new_value): 
        print("inside a.x")
        self._x = new_value

class B(A):

    def __init__(self, x):
        super().__init__(x=x)
        bpuc.CallbackManager.add_callback_to_attribute(B, self, 'x', B.callback)

    def callback(obj, name, value):
        print(obj, name, value)

if __name__ == "2__main__":
    a = A(x=2)
    b = B(x=1)

    a.x=5
    b.x = 3

    import torch

    linear = torch.nn.Linear(3,5, bias=False)
    for p in linear.parameters():
        print(p.shape)
    print(linear.bias==None)
    
import json

class One():
    def __init__(self,x):
        self.x=x

class Two():
    def __init__(self, one):
        self.one=one

if __name__ == "__main__":
    one = One(4)
    two = Two(one)

    json.dumps(two)