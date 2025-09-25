import unittest, sys, os
sys.path.append(os.path.abspath(""))
from src.briann.python.utilities import callbacks as bpuc

class Number():

    def __init__(self, value = 0):
        self.value = value

    @property
    def value(self): return self._value

    @value.setter
    def value(self, new_value): self._value = new_value

    def divide(self, denominator, epsilon=1e-8):
        self.value /= (denominator + epsilon)
        return self.value

def set_value_callback(obj, name, value):
    global callback_obj, callback_name, callback_value
    callback_obj = obj; callback_name = name; callback_value = value

def divide_callback(self, denominator, epsilon=1e-8):
    global callback_self, callack_denominator, callback_epsilon
    callback_self = self; callack_denominator = denominator; callback_epsilon = epsilon

class TestCallbackManager(unittest.TestCase):
    
    def test_add_callback_to_attribute_single_object(self):

        # Create instance
        number = Number()
        
        # Add callback to value attribute
        bpuc.CallbackManager.add_callback_to_attribute(target_class=Number, target_instance=number, attribute_name="value", callback=set_value_callback)
    
        # Test whether callback is called on set value
        global callback_obj, callback_name, callback_value
        callback_obj = None; callback_name = None; callback_value = None
        number.value = 10
        self.assertEqual(callback_obj, number)
        self.assertEqual(callback_name, "value")
        self.assertEqual(callback_value, 10)

    def test_add_callback_to_attribute_two_objects_one_set(self):

        # Create instance
        number_1 = Number(value=0)
        number_2 = Number(value=20)
        
        # Add callback to value attribute
        bpuc.CallbackManager.add_callback_to_attribute(target_class=Number, target_instance=number_1, attribute_name="value", callback=set_value_callback)
        bpuc.CallbackManager.add_callback_to_attribute(target_class=Number, target_instance=number_2, attribute_name="value", callback=set_value_callback)
    
        # Test whether callback is called on set value
        global callback_obj, callback_name, callback_value
        callback_obj = None; callback_name = None; callback_value = None
        number_1.value = 10
        self.assertEqual(callback_obj, number_1)
        self.assertEqual(callback_name, "value")
        self.assertEqual(callback_value, 10)

    def test_add_callback_to_attribute_two_objects_both_set(self):

        # Create instance
        number_1 = Number(value=0)
        number_2 = Number(value=20)
        
        # Add callback to value attribute
        bpuc.CallbackManager.add_callback_to_attribute(target_class=Number, target_instance=number_1, attribute_name="value", callback=set_value_callback)
        bpuc.CallbackManager.add_callback_to_attribute(target_class=Number, target_instance=number_2, attribute_name="value", callback=set_value_callback)
    
        # Test whether callback is called on set value of number_1
        global callback_obj, callback_name, callback_value
        callback_obj = None; callback_name = None; callback_value = None
        number_1.value = 10
        self.assertEqual(callback_obj, number_1)
        self.assertEqual(callback_name, "value")
        self.assertEqual(callback_value, 10)

        # Test whether callback is called on set value of number_2
        callback_obj = None; callback_name = None; callback_value = None
        number_2.value = 30
        self.assertEqual(callback_obj, number_2)
        self.assertEqual(callback_name, "value")
        self.assertEqual(callback_value, 30)

    def test_add_callback_to_method_single_object(self):

        # Create instance
        number = Number(value=20)
        
        # Add callback to divide method
        bpuc.CallbackManager.add_callback_to_method(target_instance=number, method_name="divide", callback=divide_callback)
    
        # Test whether callback is called on divide
        global callback_self, callack_denominator, callback_epsilon
        callback_self = None; callack_denominator = None; callback_epsilon = None
        number.divide(denominator=5, epsilon=1e-6)
        self.assertEqual(callback_self, number)
        self.assertEqual(callack_denominator, 5)
        self.assertEqual(callback_epsilon, 1e-6)

    def test_add_callback_to_method_to_one_of_two_objects_call_both(self):

        # Create instance
        number_1 = Number(value=20)
        number_2 = Number(value=50)
        
        # Add callback to divide method
        bpuc.CallbackManager.add_callback_to_method(target_instance=number_1, method_name="divide", callback=divide_callback)
    
        # Test whether callback is called on divide on number_1
        global callback_self, callack_denominator, callback_epsilon
        callback_self = None; callack_denominator = None; callback_epsilon = None
        number_1.divide(denominator=5, epsilon=1e-6)
        self.assertEqual(callback_self, number_1)
        self.assertEqual(callack_denominator, 5)
        self.assertEqual(callback_epsilon, 1e-6)

        # Test whether callback is called on divide on number_2
        callback_self = None; callack_denominator = None; callback_epsilon = None
        number_2.divide(denominator=3, epsilon=1e-6)
        self.assertEqual(callback_self, None)
        self.assertEqual(callack_denominator, None)
        self.assertEqual(callback_epsilon, None)

    def test_add_callback_to_method_to_two_objects_call_both(self):

        # Create instance
        number_1 = Number(value=20)
        number_2 = Number(value=50)
        
        # Add callback to divide method
        bpuc.CallbackManager.add_callback_to_method(target_instance=number_1, method_name="divide", callback=divide_callback)
        bpuc.CallbackManager.add_callback_to_method(target_instance=number_2, method_name="divide", callback=divide_callback)
    
        # Test whether callback is called on divide on number_1
        global callback_self, callack_denominator, callback_epsilon
        callback_self = None; callack_denominator = None; callback_epsilon = None
        number_1.divide(denominator=5, epsilon=1e-6)
        self.assertEqual(callback_self, number_1)
        self.assertEqual(callack_denominator, 5)
        self.assertEqual(callback_epsilon, 1e-6)

        # Test whether callback is called on divide on number_2
        callback_self = None; callack_denominator = None; callback_epsilon = None
        number_2.divide(denominator=3, epsilon=1e-6)
        self.assertEqual(callback_self, number_2)
        self.assertEqual(callack_denominator, 3)
        self.assertEqual(callback_epsilon, 1e-6)

if __name__ == '__main__':
    unittest.main()