# Welcome the user
print("Welcome to Brian_Fred DataCorp")
print("Your satisfaction is our priority.")

# Ask the user for their names
firstname = input("\nWhat's your first name? ").title().strip()
lastname = input("What's your last name? ").title().strip()
fullname = firstname +" "+ lastname
 
 # Greet the user
print(f"\nHello, {firstname} {lastname}")

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
print("\nQuarters=", quarters)
print("Dimes=", dimes)
print("Nickels=", nickels)
print("Pennies=", pennies)

# Give the user applause
print(f"\nThank you,{firstname} for choosing Brian_Fred DataCorp")
print("See You Again!!!")