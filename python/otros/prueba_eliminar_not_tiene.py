num = 4
strings = 'Hola mundo'

def function():
    nuevo_num = 4
    nuevo_str = 'Hola otra vez'
    print(globals())
    print(locals())

function()
print('\n\n\n')
print(locals())
print(globals())

print(('yes' if 'num' in locals() else 'No'))
num = 6
print((num+1)%7)

dct = {
    'hola': (1,2,5),
    'Como': (2,4,7),
    'Estas': (3,5,8)
}

totales = [0 for _ in range(len(list(dct.values())[0]))]
for val1, val2, val3 in dct.values():
    totales[0]+=val1
    totales[1]+=val2
    totales[2]+=val3
print(totales)

numero = (3,4,6)
numero += 5,
print(numero[-1])

import scipy.optimize

scipy.optimize.minimize(he)