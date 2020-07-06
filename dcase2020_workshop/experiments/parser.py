import importlib
import copy
from typing import Any, Union


def create_objects_from_config(configuration: dict) -> dict:
    configuration = copy.deepcopy(configuration)
    objects = dict()

    unfinished_keys = list(configuration.keys())
    max = len(unfinished_keys) * len(unfinished_keys)
    i = 0

    while len(unfinished_keys) > 0:

        key = unfinished_keys.pop(0)
        value = configuration[key]

        if is_object(value):
            # object, create and save
            obj, create = create_object_or_false(objects, value)
            if create:
                objects[key] = obj
            else:
                unfinished_keys.append(key)
        else:
            # built-in type, save
            objects[key] = value
        i = i + 1
        if i > max:
            raise AttributeError('Cyclic Configuration')

    return objects


def create_object_or_false(objects, object):
    args = object.get('args', list())
    kwargs = object.get('kwargs', dict())

    for i, arg in enumerate(args):
        if is_reference(arg):
            if has_reference(objects, arg):
                args[i] = get_reference(objects, arg)
            else:
                return None, False

    for k in kwargs:
        if is_reference(kwargs[k]):
            if has_reference(objects, kwargs[k]):
                kwargs[k] = get_reference(objects, kwargs[k])
            else:
                return None, False

    module_name, class_name = object['class'].rsplit(".", 1)
    obj = getattr(importlib.import_module(module_name), class_name)(*args, **kwargs)

    return obj, True


def has_reference(objects, arg):
    parts = arg[1:].rsplit('.', 1)
    assert len(parts[0]) > 0
    if objects.get(parts[0]) is None:
        return False
    else:
        return True


def get_reference(objects, arg):
    parts = arg[1:].rsplit('.', 1)
    assert len(parts[0]) > 0
    if len(parts) > 1:
        obj, part = parts
        if part[-2:] == '()':
            return getattr(objects[obj], part[:-2])()
        else:
            return getattr(objects[obj], part)
    else:
        return objects[parts[0]]


def is_object(obj: Union[str, dict]) -> bool:
    return True if type(obj) is dict and obj.get('class') else False


def is_reference(obj: Union[str, dict]) -> bool:
    return True if type(obj) is str and obj.startswith('@') else False
