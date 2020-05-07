# %%
print('___________    Ex 14    _______________ ')

import numpy as np
import matplotlib.pyplot as plt

'___________    Ex 14    _______(1)______ '
x=5
print('x=',x)
print ("Square Value of x : ", np.square(x)) 
print ("cube Value of x :", np.power(x,3)) 
print('\n')

teta=30
print('teta=',teta)
print ("sin (",teta,")=", np.sin(teta))
print ("cos of teta : ", np.cos(teta))
print ("radian of teta : ", np.radians(teta))
print('\n')

y=np.linspace(-1,1,500)
print('53 th =',y[53])
print('\n')

pi=np.pi
plt.plot(y,np.sin(2*np.pi*y))
plt.show()

'___________    Ex 14    _______(2)______ '

vec1 = np.array([ -1., 4., -9.])
mat1 = np.array([[ 1., 3., 5.], [7., -9., 2.], [4., 6., 8. ]])

vec2 = (np.pi/4)*vec1
print("vec2=",vec2)

vec2=np.cos(vec2)
print("vec2=",vec2)


vec3 =vec1+2*vec2
print("vec3=",vec3)

vec4=np.multiply(mat1,vec3)
print("vec4=",vec4)


vec5=np.matrix(mat1)*np.matrix(vec4)
print("vec5=",vec5)


print('mat1.transpose =',mat1.transpose())

print('det(mat1)=', np.linalg.det(mat1))

print('trace(mat1)=',np.trace(mat1))

print('min(vec1)=',np.min(vec1))

print('argmin(vec1)=',np.argmin(vec1))

print('min(mat1)=',np.min(mat1))

A=np.array([[17, 24, 1, 8, 15],
            [23, 5, 7, 14, 16],
            [ 4, 6, 13, 20, 22],
            [10, 12, 19, 21, 3],
            [11, 18, 25, 2, 9]])

print(np.sum(A,axis=0))
print(np.sum(A,axis=1))



if min(np.sum(A,axis=0))==max(np.sum(A,axis=0)) and min(np.sum(A,axis=1))==max(np.sum(A,axis=1)):
    if np.sum(np.diag(A))==np.sum(np.fliplr(A)):
        print('magic square')
    else:
        print('not a magic square')  
else:
    print('not a magic square')

np.random.rand(10, 10)

'___________    Ex 14    _______(3)______ '


x = np.linspace(0, 10, 50)
y1= np.exp(-x/10)*np.sin(np.pi*x) 
y2 = x*np.exp(-x/3)
plt.plot(x,y1,x,y2,label = 'f(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show() 




phi=np.linspace(0,2*np.pi,400)
R=1.2
r=R+np.cos(phi)
x1=r*np.cos(phi)
y=r*np.sin(phi)
plt.plot(x1,y,'r')
plt.xlabel('mehvar x ha')
plt.ylabel('mehvar y ha')
plt.show()

r1 = 0.8 + np.cos(phi)
x1 = r1 * np.cos(phi)
y1 = r1 * np.sin(phi)

r2 = 1 + np.cos(phi)
x2 = r2 * np.cos(phi)
y2 = r2 * np.sin(phi)

r3 = 1.2 + np.cos(phi)
x3 = r3 * np.cos(phi)
y3 = r3 * np.sin(phi)

plt.subplot(3,1,1)
plt.plot(x1, y1, 'r')
plt.subplot(3,1,2)
plt.plot(x2, y2, 'r')
plt.subplot(3,1,3)
plt.plot(x3, y3, 'r')

plt.show()






# %%
