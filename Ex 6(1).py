#%%
print('___________    Ex 6(1)    _________________ ')
print('\nFunction for nth Fibonacci number  \n ')

n=int(input("Pleas Enter n: "))  

def Fibonacci(n): 
    if n<0: 
        print("Incorrect input") 
    elif n==0: 
        return 0

    elif n==1: 
        return 1
 
    return Fibonacci(n-1)+Fibonacci(n-2) 
#cheak
print(Fibonacci(n)) 

print('\n ___________    Fatemeh Rabie    ___________ ')

# %%
