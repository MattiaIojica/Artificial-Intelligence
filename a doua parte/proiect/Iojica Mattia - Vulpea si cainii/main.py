#Guess the Date
#Instructions for program
print ('Think of a specific date in any year')
print ('e.g., Jan 1 or Feb 29 or Jul 4 or Dec 25')
print ('Truthfully answer "Yes" or "No" to the following questions')
print ('I will determine the date in ten questions or less')

#Return a list of elements, each element is a date in a calendar year
def Calendar(monthNames, numDaysInMonth): #defining Calendar

    if len(monthNames) != len(numDaysInMonth): 
        return []

    dates = []
    idx = 0     #index is set to zero

    while idx < len(monthNames):
        for date in range(1, numDaysInMonth[idx] + 1):
            dates.append(monthNames[idx] + " " + str(date))
        idx = idx + 1
    return dates
monthNames = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]   #list of months
numDaysInMonth = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31] #list of how many days in each month

#This is a binary search
first =Calendar(monthNames,numDaysInMonth) #first defined through the Calendar code block

def guess_game(first = Calendar(monthNames,numDaysInMonth)): #defining guess_game using months list and numDays item in the Calendar code to work the search

    if len(first) == 1:
        return first[0]

    mid = len(first)//2

    if is_earlier(first[mid]):    #list mindpoint
        return guess_game(first[:mid])
    else:
        return guess_game(first[mid:])

#Answer output, what is out putted in the python shell
def is_earlier(guess = 10): #defining is_ealier and setting the number of guesses equal to 10


    answer = input("Is {} earlier then your date? (Yes - earlier /No - not earlier) ".format(guess))#10 or less guesses to to find answer

    if answer.upper() == "Yes": #if true user can put No, no, n
        return True

    else:
        return False #if false user can put Yes, yes, y

guess_game() #intialize quess_game