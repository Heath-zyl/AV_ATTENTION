from abc import ABC, abstractmethod

SCHEDULER_FACTORY = {}


def register_scheduler(cls):
    # assert isinstance(name, str) , "register_scheduler name must be str type"
    # assert len(name) > 0 , "register_scheduler name len must not be zero"
    cls_name = cls.register_name
    def register(cls):
        SCHEDULER_FACTORY[cls_name] = cls
    
    return register(cls)

def get_scheduler(name, args):
    assert name in SCHEDULER_FACTORY.keys(), f"{name} is not a scheduler"
    return SCHEDULER_FACTORY[name](args)


class Scheduler(object):
    def __init__(self, args):
        print(self.__class__.__name__)
        print("args: ==>")
        print(args)
        self.args = args
        
    @abstractmethod
    def submit_impl(self):
        return
    
    def submit(self):
        self.submit_impl()
        self.show_cmd()
        return

    @abstractmethod
    def show_cmd(self):
        return