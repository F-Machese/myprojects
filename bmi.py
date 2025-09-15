# Welcome the user
print("Welcome to Brian_Fred DataCorp")

# Ask the user for weight in kilograms
weight = float(input("\nWhat is your weight in kgs? "))

# Ask the user for height in meters
height = float(input("What is your height in meters? "))

# Calculate BMI
bmi = weight / (height ** 2)

# Print the result
print("Your Body Mass Index (BMI) is", round(bmi, 1))

# BMI Classification according to WHO
if bmi < 16.0:
    print("Category: Severely Underweight")

elif 16.0 <= bmi < 18.5:
    print("Category: Underweight")

elif 18.5 <= bmi < 25:
    print("Category: Normal")

elif 25 <= bmi < 30:
    print("Category: Overweight")
   
elif 30 <= bmi < 35:
    print("Category: Moderately Obese")
   
elif 35 <= bmi < 40:
    print("Category: Severely Obese")
   
else:
    print("Category: Morbidly Obese")
   
# Closing remarks
print("\nThank you!!!")
