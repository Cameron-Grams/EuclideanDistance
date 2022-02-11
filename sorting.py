import numpy as np

a = [1,4,9,2,0]
b = ['pig', 'dog', 'cat', 'eagle', 'wolverine']

z_l = list(zip(b,a))

ar = np.array(z_l, dtype=[('name', 'S10'), ('rank', 'i8')])

order = np.argsort(ar, order='rank')

srtd = ar[order]

print(f"The collection: {z_l}")

print(f"The ordered names: {srtd['name']}")
print(f"The ranks in order: {srtd['rank']}")

# from bytes to string
strings = [(i[0]).decode('UTF8') for i in srtd]

print(f"The order as strings: {strings}")




