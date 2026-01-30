from typing import Any


class CallbackManager():
    """This class allows to add callbacks to attribute setters and method calls."""

    _class_to_regular_setattr = {}
    _obj_to_attribute_name_to_callbacks = {}

    def __setattr__with_callbacks(target_class, obj, name, value) -> None:
        """Sets an attribute and calls any callbacks associated with it.
        
        :param target_class: The class of the instance.
        :type target_class: Any
        :param obj: The instance to set the attribute on.
        :type obj: Any
        :param name: The name of the attribute to set.
        :type name: str
        :param value: The value to set the attribute to.
        :type value: Any
        :rtype: None"""

        # Call regular setattr
        CallbackManager._class_to_regular_setattr[target_class](obj, name, value)

        # If object is inspected
        if obj in CallbackManager._obj_to_attribute_name_to_callbacks.keys():

            # If attribute is inspected
            if name in CallbackManager._obj_to_attribute_name_to_callbacks[obj].keys():

                # Call the callbacks
                for callback in CallbackManager._obj_to_attribute_name_to_callbacks[obj][name]:
                    callback(obj, name, value)

        
    def add_callback_to_attribute(target_class: Any, target_instance: Any, attribute_name: str, callback: callable) -> None:
        """Adds a callback to an attribute setter of a specific instance of a class.
        The callback must have the signature: callback(obj: Any, name: str, value: Any)
        where obj is the instance, name is the attribute name, and value is the new value. Any returned values from the callback are ignored.
        
        :param target_class: The class of the instance.
        :type target_class: Any
        :param target_instance: The instance to inspect.
        :type target_instance: Any
        :param attribute_name: The name of the attribute to inspect.
        :type attribute_name: str
        :param callback: The callback to call when the attribute is set. The callback must have the signature: callback(obj: Any, name: str, value: Any), where obj is the instance on which the attribute setter is called, name is the name of the attribute and value is the new value to be assigned to the attribute.
        :type callback: callable
        :rtype: None
        """

        # Ensure target class is inspected
        if target_class not in CallbackManager._class_to_regular_setattr.keys():
            CallbackManager._class_to_regular_setattr[target_class] = target_class.__setattr__
            target_class.__setattr__ = lambda obj, name, value, target_class=target_class: CallbackManager.__setattr__with_callbacks(target_class=target_class, obj=obj, name=name, value=value)
        
        # Ensure object is inspected
        if target_instance not in CallbackManager._obj_to_attribute_name_to_callbacks.keys(): 
            CallbackManager._obj_to_attribute_name_to_callbacks[target_instance] = {}
        
        # Ensure name is inspected
        if attribute_name not in CallbackManager._obj_to_attribute_name_to_callbacks[target_instance].keys():
            CallbackManager._obj_to_attribute_name_to_callbacks[target_instance][attribute_name] = []

        # Enter callback
        CallbackManager._obj_to_attribute_name_to_callbacks[target_instance][attribute_name].append(callback)

    def add_callback_to_method(target_instance: Any, method_name: str, callback: callable) -> None:
        """Adds a callback to a method of a specific instance of a class.
        
        :param target_instance: The instance to inspect.
        :type target_instance: Any
        :param method_name: The name of the method to inspect.
        :type method_name: str
        :param callback: The callback to call when the method is called. The callback must have the signature: callback(obj: Any, args, kwargs), where obj is the object in which the method is called, and args and kwargs are the positional and key-word arguments of the method, respectively. For arguments that are optional for the method, the default value assigned to the corresponding callback's argument takes priority over that asigned to the method's argument. Any returned values from the callback are ignored.
        :type callback: callable
        :rtype: None
        """
        
        # Ensure method is wrapped
        if not hasattr(target_instance, f"__regular_{method_name}"):
            global new_method, ttarget_instance
            ttarget_instance = target_instance
            def new_method(*args, **kwargs):
                # Call the method
                global ttarget_instance, aargs, kkwargs
                ttarget_instance, aargs, kkwargs = target_instance, args, kwargs
                exec(f"global ttarget_instance, aargs, kkwargs; ttarget_instance.__regular_{method_name}(*aargs, **kkwargs)")

                # Call the callbacks
                args = (target_instance,) + args
                callbacks = getattr(target_instance, f"_{method_name}_callbacks")
                for callback in callbacks: callback(*args, **kwargs)
        
            exec(f"global ttarget_instance, new_method; ttarget_instance.__regular_{method_name} = ttarget_instance.{method_name}; ttarget_instance.{method_name} = new_method")

        # Ensure target instance has callbacks for this method
        if not hasattr(target_instance, f"_{method_name}_callbacks"):
            setattr(target_instance, f"_{method_name}_callbacks", [])

        # Add callback to list
        getattr(target_instance, f"_{method_name}_callbacks").append(callback)
        