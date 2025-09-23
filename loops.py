# For Loop
# Print numbers from 1 to 10
print("Numbers from 1 to 10 using For Loop:")
for i in range(1, 11):
    print(i)
    print()

# While Loop
# Print numbers from 1 to 10
print
count = 1
while count <= 10:
    print(count)
    count += 1
    print()

# Do-While Loop
# Print numbers from 1 to 10
print("Numbers from 1 to 10 using Do-While Loop:")
num = 1
while True:
    print(num)
    num += 1
    if num > 10:
        break
    print()

# Foreach Loop
# Print each element in a list
print("Names in a list:")
names = ['Machese', 'Fred', 'Isaac']
for name in names:
    print(name)
    print()

# Loop through a list of numbers
print("Numbers in a list:")
numbers = [2, 4, 6, 8, 10]
for n in numbers:
    print(n)
    print()

# Even Numbers Loop
# Print even numbers from 1 to 10
print("Even numbers from 1 to 10:")
for i in range(1, 11):
    if i % 2 == 0:
        print(i)
        print()

# Odd Numbers Loop
# Print odd numbers from 1 to 10
print("Odd numbers from 1 to 10:")
for i in range(1, 11):
    if i % 2 != 0:
        print(i)
        print()

# Prime Numbers Loop
# Print prime numbers from 1 to 10
print("Prime numbers from 1 to 10:")
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True 
for num in range(1, 101):
    if is_prime(num):
        print(num)
        print()

# Infinite Loop (use with caution)
# Uncommentthe to lines below to run and Ctrl+C to stop.
# while True:
#     print("Hello, world.") 

# Loop with Break
# Print numbers from 1 to 10, but stop if number is 5
print("Numbers from 1 to 10 but stop if a number is 5 using Loop with Break:")
for i in range(1, 11):
    if i == 5:
        break
    print(i)
    print()   

# Loop with Continue
# Print numbers from 1 to 10, but skip number 5
print("Numbers from 1 to 10 while skipping 5 using Loop with Continue:")
for i in range(1, 11):
    if i == 5:
        continue
    print(i)
    print()

# Loop with Else
# Print numbers from 1 to 5, and then print a message
print("Numbers from 1 to 5 with a message using Loop with Else:")
for i in range(1, 6):
    print(i)
else:
    print("Loop completed successfully.")
print() 

# Backwards (10 to 1)
# Print numbers from 10 to 1
print("Numbers from 10 to 1:")
for i in range(10, 0, -1):
    print(i)
    print()

# Sum of numbers 1 to 10
# Calculate and print the sum of numbers from 1 to 100
total = 0
for i in range(1, 11):
    total += i
print("The sum from 1 to 10 is:", total)
print()

# Multiplication Table
# Print multiplication table for a given number from 1 to 12
print("Multiplication Table for any number:")
num = int(input("Enter a number: "))

print(f"\nMultiplication Table for {num}")
for i in range(1, 13):
    print(f"{num} x {i} = {num * i}")
print()

# END



