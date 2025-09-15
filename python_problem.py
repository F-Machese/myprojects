#  Ask the user for the amount in cents that they have
cents = int(input("How many cents do you have? "))

# Number of quarters and remaining cents
quarters = cents // 25
cents = cents % 25

# Number of dimes and remaining cents
dimes = cents // 10
cents = cents % 10

# Number of nickels and remaining cents
nickels = cents // 5
cents = cents % 5

# Number of pennies
pennies = cents

# Print out results
print("Quarters=", quarters)
print("Dimes=", dimes)
print("Nickels=", nickels)
print("Pennies=", pennies)