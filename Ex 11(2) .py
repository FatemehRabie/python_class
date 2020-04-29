#%%
print('___________    Ex 11    _______(part 2)________ ')

  
class time:
    def __init__(obj,h,m,s):
        obj.hour = h
        obj.minute = m
        obj.second = s
    def show(obj,ruz):
        obj.date = ruz
        return str(obj.hour)+':'+str(obj.minute)+':'+str(obj.second)
    def add(obj,hh,mm,ss):
        obj.hour += hh
        obj.minute += mm
        obj.second += ss

print('___________    Fatemeh Rabie    ___________ ')
