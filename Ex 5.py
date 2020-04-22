
# %%
print('_________________    Ex 5    ______1______ ')
print('\n       Specify the number of days in year \n ')

def julian_date(day,month,year):
    days=[31,28,31,30,31,30,31,31,30,31,30,31]
    days_pass=0
    if year% 400==0 or (year%4==0 and year %100!=0):
        days[1]=29
    for month in days[:month-1]:
        days_pass+=month
    days_pass+=day
    return(days_pass)
julian_date(10,10,2000)

# %%
print('_________________    Ex 5    ______2______ ')
def pos(n):
    for i in range(n):
        print(i)

# %%
print('_________________    Ex 5    ______3______ ')
def dev(num1,num2):
    if (num1%num2==0 or num2%num1==0):
        print('bakhshpazir ast')
    else :
        print('bakhshpazir nist')

# %%
print('_________________    Ex 5    ______4_____ ')
def prime(num):
    if num > 1: 
        for i in range(2, num//2): 
    
            if (num % i) == 0: 
                print(num, "is not a prime number") 
                break
        else: 
            print(num, "is a prime number") 
        
    else: 
        print(num, "is not a prime number")