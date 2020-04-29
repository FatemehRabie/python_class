#%%
print('___________    Ex 11    _______(part 1)________ ')

class time:
    def __init__(obj,h,m,s):
        obj.hour = h
        obj.minute = m
        obj.second = s
    def show(obj):
        return str(obj.hour)+':'+str(obj.minute)+':'+str(obj.second)
    def number_of_seconds(obj):
        return str (obj.hour*3600+obj.minute*60+obj.second)


print('___________    Fatemeh Rabie    ___________ ')
