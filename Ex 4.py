
#%%
print('_________________    Ex 4    ______________ ')
print('\n       Specify the number of days in year \n ')
year=int(input('Please enter the year number'))
month=int(input('Please enter the month number'))
while month>12 or month<=0:
    month=int(input('Please be careful , Please enter the month number correctly'))
day=int(input('Please enter the number of the day '))
while day>32 or day<=0:
    day=int(input('Please be careful , Please enter the day number correctly'))

if month==1:
    print('*****************************************************\n')
    print (' On (',year,',',month,',',day,') We are on',day,'th day of the year')
    print('\n*******************************************************')

elif year %400==0  : #sal kabise ast
    c=1
elif (year %4==0 and year %100==0) : #sal kabise nist
    c=0
elif year %4==0  : #sal kabise ast
    c=1
else:
    c=0

DayofMonth=[31,28,31,30,31,30,31,31,30,31,30,31]
i=0
sumdays=0
if 1<month<12:  #bayad havasemon bashe ke mahe akhar dobar hesab nashe baraye hamin mah akhar ra joda karfam
    if c==0:
        while i < month :
            sumdays += DayOfMonth[i]
            i=i+1
    else :              #sale kabise mah dovom 29 ruze ast
        while i < month :
            sumdays += DayOfMonth[i]
            i=i+1
        sumdays=sumdays+1
        
    sumdays=sumdays+day
    print('**************************************************\n')
    print (' On (',year,',',month,',',day,') We are on',sumdays,'th day of the year')
    print('\n****************************************************')
elif month== 12: 
    if c==0:
        while i < month-1 :
            sumdays += DayOfMonth[i]
            i=i+1
    else :              #sale kabise mah dovom 29 ruze ast
        while i < month-1 :
            sumdays += DayOfMonth[i]
            i=i+1
        sumdays=sumdays+1
    sumdays=sumdays+day
    print('*********************************************************\n')
    print (' On (',year,',',month,',',day,') We are on',sumdays,'th day of the year')
    print('\n***********************************************************')

print('______________    Fatemeh Rabie    ______________ ')
# %%
