"""Curso de python por Banco Santander"""
print ('Hola Mundo!')

# ------------- Tipos de datos basicos -------------
# Enteros (int)
edad =25

#Flotantes 
precio = 9.99

#cadenas de texto (strings)
nombre = "Erika"

#Boleanos
es_mayor_de_edad= True
tiene_descuento=False

# Operadores aritmeticos
a= 10
b= 3
Suma = a + b
resta = a - b
multiplicacion= a * b
division = a / b
division_entera = a // b
modulo = a % b
exponenciacion = a ** b

# Operadores Lógicos
"""and, or, not"""
resultado_and= (a>5) and (b<5) #True
resultado_or= (a>15) or (b<5) # True
resultado_not = not (a>5) # False

# Estructuras de control condicionales y bucles
# Estructuras condicionales
"""if, if-else,if,elif-else"""
# if
edad = 18
if edad >=18:
    print ("Eres mayor de edad")

# if-else
edad =15 

if edad >=15:
    print ("Eres mayor de edad")
else:
    print ("Eres menor de edad")

#if-elif-else
edad = 85
if edad <18:
    print ("Eres menor de edad")
elif edad >=18 and edad < 60:
    print ("Eres mayor de edad")
    
else:
    print ("Eres un adulto")


# Bucles / loops
# for
frutas = ['manzana','bananas','naranja']
for fruta in frutas:
    print (fruta)
    
# while
contador = 0
while contador <5:
    print (contador)
    contador +=1

# Ejemplo 
print ("numeros de 1 al 5 multiplicados por 2 con bucle for:")
for numero in range (1,6):
    print (numero *2)
    
print ("\n Números del 1 al 5 multiplicados por 2 bucle while")
contador = 1
while contador <=5:
    print(contador*2)
    contador += 1

# Control de bucles
"""break(para salir prematuramente de un bucle, independientemente de la condición),
continue (saltar el resto del bloque de código dentro de un bucle y pasar a la siguiente iteración.),
pass (no hace nada)"""

# break
print ("\n Números del 0 al 4 con control de bucle break")
contador = 0
while True:
    print (contador)
    contador += 1
    
    if contador == 5:
        break

# continue (Ejercicio: solo se imprimirán los números impares)
print ("\n Números del 1 al 10 con control de bucle continue,si el número es divisible por 2, se ejecuta la instrucción continue")
for i in range (10):
    if i % 2 ==0:
        continue
    print (i)

# pass
print ("\n Números con control de bucle pass")
for i in range (5):
    pass

# ------------- Estructura de datos -------------
"""listas"""

# Listas creación y acesso
frutas = ['manzana','bananas','naranja']

print (frutas [0])
print (frutas [1])
print (frutas [2])

print  (frutas [-1])
print (frutas [-2])
print (frutas [-3])

#Metodo de listas

# append
frutas.append ('pera')
print (frutas)

# insert
frutas.insert (1,'uva')

# renove 
frutas.remove('bananas')
print (frutas)

# pop
fruta_eliminada= frutas.pop (2)
print (frutas)
print (fruta_eliminada)

# sort
frutas.sort()
print (frutas)

# reverse
frutas.reverse()
print (frutas)

# Listas de comprensión
numero =[1,2,3,4,5]
cuadrados = [x ** 2 for x in numero if x % 2 == 0]
print (cuadrados)

pares = [x for x in range (1,11) if x % 2 ==0 ]
print (pares)

# Generador de numeros del 1 al 10 
numeros = (x for x in range (1,11))
print (list (numeros))

# numeros del 1 al 10 en forma de lista
numeros = [x for x in range (1,11)]
for x in numeros:
    print (x)

# ------------ Tuplas ----------

punto = (3,4)
print (punto[0])
print (punto [1])

# Metodos de tuplas
"""count, index,len"""

mi_tupla= (1, 2, 3, 2, 4, 2)
print (mi_tupla.count (2)) # Cuenta cuántos números 2 hay en la tupla
print (mi_tupla.index (2)) # Devuelve el índice de la primera aparición del número 2
print (mi_tupla.index(2,2)) # Busca el número 2 empezando desde el índice 2
print (mi_tupla.index(2,2,4)) # Busca el número 2 entre los índices 2 y 4 (sin incluir el 4)
print (len(mi_tupla)) # Devuelve la longitud de la tupla

# ------------- Diccionarios -------------
"""Son una estructura de datos nmutable, se encierran en llaves{}"""

persona = {"nombre": "Erika","Edad": 32, "Ciudad": "Oaxaca", "Año de aprendizaje": 2025}
print (persona ["nombre"])
print (persona ["Edad"])
print (persona ["Ciudad"])
print (persona ["Año de aprendizaje"])

# Métodos de diccionarios
"""Para manipular y acceder a los elementos son:
keys(): devuelve una vista de todas las claves del diccionario.
values(): devuelve una vista de todos los valores del diccionario.
items(): devuelve una vista de todos los pares clave-valor del diccionario.
update(otro_diccionario): actualiza el diccionario con los pares clave-valor de otro diccionario.
"""
print (persona.keys())
print (persona.values())
print (persona.items())
print (persona.update())
print (persona)

# ------------- Conjuntos (set) --------------
"""Un conjunto es una estructura de datos mutable y no ordenada que permite almacenar una colección de elementos únicos. 
Los conjuntos se encierran entre llaves {} o se crean utilizando la función set()."""

# Creación y operaciones básicas
frutas = {'manzana','banana','naranja'}
numeros = set ([1,2,3,4,5])

union = frutas | numeros
print (union)

conjunto1 = {1,2,3}
conjunto2 = {3,4,5}

union_conjunto= conjunto1 | conjunto2
print (union_conjunto)

interseccion = conjunto1 & conjunto2
print (interseccion)

diferencia = conjunto1 - conjunto2
print (diferencia)

diferencia_simetrica = conjunto1 ^ conjunto2
print (diferencia_simetrica)

# Metodos de conjuntos
"""Python tienen varios métodos incorporados para manipular y acceder a los elementos. 
Algunos métodos comunes son:

add(elemento): agrega un elemento al conjunto.
remove(elemento): elimina un elemento del conjunto. Si el elemento no existe, genera un error.
discard(elemento): elimina un elemento del conjunto si está presente. Si el elemento no existe, no hace nada.
clear(): elimina todos los elementos del conjunto."""
fruta = {'manzana','banana','naranja'}

frutas.add('pera')
print (frutas)

frutas.remove ('banana')
print (frutas)

frutas.discard ('pera')
print (frutas)

frutas.clear ()
print (frutas)