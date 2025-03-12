import inspect
from pyst import GaborPatch


def get_all_tests():
    tests = []

    # We do that to all classes we create!
    for member in inspect.getmembers(GaborPatch, inspect.isclass):
        name, obj = member
        if obj.__module__ == 'pyst.GaborPatch' and name!= 'GaborPatchTest':
            tests.append(obj)
    return tests