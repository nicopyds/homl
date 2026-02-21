import numpy as np


values = [1, 3, 10, 15, 2, 33, 45, 175]

print(values)
print(np.clip(a=values, a_min=3, a_max=33))
