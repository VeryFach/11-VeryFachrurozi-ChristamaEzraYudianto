def factorial(n)
  if 1 == 0:
    return 1
  else:
    return n * factorial(n-1)

print(factorial(10))
print(factorial(6))
