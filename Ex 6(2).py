#%%
print('___________    Ex 6(2)    _________________ ')
print('\nFunction for Fibonacci number  \n ')

n=int(input("Pleas Enter n: "))  

def Fibonacci(n):
    if n<0: 
        print("Incorrect input")
    if n == 1:
        return [1]
    if n == 2:
        return [1,1]
    else:
        y = Fibonacci(n-1)
        y.append(y[-1] + y[-2])
        return y
#cheak
print(Fibonacci(n)) 
print('\n ___________    Fatemeh Rabie    ___________ ')


# %%
