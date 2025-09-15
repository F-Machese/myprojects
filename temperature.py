# Celcius to Fahrenheit
celcius = float(input("What is the temperature in Celcius? "))
fahrenheit = (celcius * 9/5) + 32
print(f"{celcius}°C is equal to {round(fahrenheit, 2)}°F")

print()

# Fahrenheit to Celcius
fahrenheit = float(input("What is the temperature in Fahrenheit? "))
celcius = (fahrenheit - 32) * 5/9
print(f"{fahrenheit}°F is equal to {round(celcius, 2)}°C")
