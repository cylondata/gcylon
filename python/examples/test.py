class Person:
    def __init__(self, name):
        self._name = name

    @property
    def _myname(self):
        '''name property docs'''
        print('fetch...')
        return self._name

bob = Person('Bob Smith')
print(bob._myname)
