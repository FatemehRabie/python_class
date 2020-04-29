#%%
print('___________    Ex 7    _______(part 012)________\n ')

x1=int(input("lotfan adade aval ra vared konid"))
x2=int(input("lotfan adade dovom ra vared konid"))
if x1<x2:
    print (x1,',',x2)
else:
    print(x2,',',x1)


#____________________________________________________________________________
#%%
print('___________    Ex 7    _______(part 013)________\n ')

x1=int(input("lotfan yek adad kamtar az 20 vared konid"))
while x1>=20:
    print('Too high')
    x1=int(input("lotfan yek adad kamtar az 20 vared konid"))
print(" thank  you :) ")

#____________________________________________________________________________
#%%
print('___________    Ex 7    _______(part 014)________\n ')

x1=int(input("lotfan yek adad dar baze 10 ta 20 vared konid"))
while x1>20 or x1<10:
    print('Incorrect answer')
    x1=int(input("lotfan yek adad dar baze 10 ta 20 vared konid"))
print(" thank  you :) ")

#____________________________________________________________________________
#%%
print('___________    Ex 7    _______(part 015)________\n ')

str1 = input ("Insert your favorite color: ")
if str1 == "red" or str1=="Red" or str1=="RED":
    print ("I like " + str1 + " too")
else:
    print ("I don't like " + str1 + " , I prefer red")
    
#____________________________________________________________________________
#%%
print('___________    Ex 7    _______(part 016)________\n ')

str1= str (input ("Is it raining? "))
if str1 == "yes" or str1 == "YES":
    str2= str (input ("Is it windy? "))
    if str2== "yes"or str1 == "YES":
        print ("It is too windy for an umbrella")
    else:
        print ("take an umbrella")
else:
    print ("Enjoy your day")

#____________________________________________________________________________
#%%
print('___________    Ex 7    _______(part 017)________\n ')

x = int (input (" please insert your age: "))
if x > 18 or x == 18:
    print ("You can vote")
elif x == 17:
    print ("You can learn to drive")
elif x == 16:
    print ("You can buy a lottery ticket")
else:
    print ("You can go Trick-or-Treating")

#____________________________________________________________________________
#%%
print('___________    Ex 7    _______(part 018)________\n ')

x = int (input ( "insert a number: "))
if x < 10:
    print ("Too low")
elif x > 9 and x <20:
    print ("correct")
else:
    print ("Too high")

#____________________________________________________________________________
# %%
print('___________    Ex 7    _______(part 019)________\n ')

x = int (input ("insert 1 or 2 or 3: "))
if x== 1:
    print ("Thank you")
elif x==2:
    print ("well done")
elif x==3:
    print ("correct")
else:
    print ("Error message")

print('___________    Fatemeh Rabie    ___________ ')


