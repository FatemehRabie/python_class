#%%
print('_____________________    Ex 10    __________________________ \n')

class coordinate:
    x=0
    y=0


def Distance(p1,p2):  
    dist = ((p2.x - p1.x)**2 + (p2.y - p1.y)**2)**0.5  
    return dist  

p1=coordinate()
p2=coordinate()
p1.x=int(input ('lotfan x noghte aval ra vared konid'))
p1.y=int(input ('lotfan y noghte aval ra vared konid'))
p2.x=int(input ('lotfan x noghte dovom ra vared konid'))
p2.y=int(input ('lotfan y noghte dovom ra vared konid'))
Distance(p1,p2)
print('*The distance between (',p1.x,',',p1.y,') and (',p2.x,',',p2.y,') is',Distance(p1,p2),'*')

print('\n____________________    Fatemeh Rabie    ____________________ ')

