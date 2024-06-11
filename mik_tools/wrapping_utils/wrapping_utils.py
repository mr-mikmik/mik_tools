
class AttributeWrapper(object):
    """
    Wrapps all methods
    """
    def __init__(self, wrapped_object):
        self.wrapped_object = wrapped_object

    def __getattr__(self, attr):
        self_attr_name = '_{}_'.format(self.__class__.__name__)
        # print('Getting the attribute for: {}'.format(self_attr_name))
        if self_attr_name in attr:
            attr_name = attr.split(self_attr_name)[1]
            return getattr(self, attr_name)
        elif attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.wrapped_object, attr)

    def unwrap(self):
        if not isinstance(self.wrapped_object, AttributeWrapper):
            return self.wrapped_object
        else:
            return self.wrapped_object.unwrap()


class ClassWrapper(AttributeWrapper):
    """
    Wrapps all methods, item accessing and representations
    """
    def __getitem__(self, item):
        return self.wrapped_object.__getitem__(item)

    def __str__(self):
        return self.wrapped_object.__str__()

    def __repr__(self):
        return self.wrapped_object.__repr__()


class DecoratorWrapper(object):
    def __init__(self, WrapperClass):
        self.WrapperClass = WrapperClass

    def __call__(self, cls):
        class Wrapper(self.WrapperClass):
            def __init__(self, *args, **kwargs):
                # TODO: Extend this so we can provide more arguments than the ones for the wrapper class
                super().__init__(cls(*args, **kwargs))
        return Wrapper


# ======================================================================================================================
#  -------------------------   EXAMPLES   ------------------------------------------------------------------------------
# ======================================================================================================================

"""
def decorator_example(cls):
    class Wrapper(ClassWrapper):
        def __init__(self, *args, **kwargs):
            super().__init__(cls(*args, **kwargs))
            # we could modify attributes here
            # self.wrappred_object.attribute_1 = value_1

        # We can add or edit like this:
        def example_method_1(self, arg_1, arg_2):
            return arg_1 * arg_2

        def example_method_2(self, arg_1):
            original_return = self.wrapped_object.example_method_2(arg_1)
            return original_return * 1
    return Wrapper


# We could also define a general Wrapper and convert it to a decorator
class ExampleWrapper(ClassWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # we could modify attributes here
        # self.wrappred_object.attribute_1 = value_1

    # We can add or edit like this:
    def example_method_1(self, arg_1, arg_2):
        return arg_1 * arg_2

    def example_method_2(self, arg_1):
        original_return = self.wrapped_object.example_method_2(arg_1)
        return original_return * 1


# NOTE THAT docorator_example is equivalent ot decorator_example_2 but we have factored out the wrapper class in case we
# want to use it without the decorator
def decorator_example_2(cls):
    class Wrapper(ExampleWrapper):
        def __init__(self, *args, **kwargs):
            super().__init__(cls(*args, **kwargs))

    return Wrapper


# This is also equivalent:
decorator_example_3 = DecoratorWrapper(ExampleWrapper)

"""