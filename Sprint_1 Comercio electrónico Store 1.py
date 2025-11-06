# Comercio electronico Store 1
# Estos son los datos que el cliente nos proporcionó. Tienen el formato de una lista de Python, con las siguientes columnas de datos:
# - id: el id del producto
# - **user_name:** El nombre del usuario.
# - **user_age:** La edad del usuario.
# - **fav_categories:** Categorías favoritas de los artículos que compró el usuario, como 'ELECTRONICS', 'SPORT' y 'BOOKS' (ELECTRÓNICOS, DEPORTES y LIBROS), etc.
# - **total_spendings:** Una lista de números enteros que indican la cantidad total gastada en cada una de las categorías favoritas.

users = [
    ['32415', ' mike_reed ', 32.0, ['ELECTRONICS', 'SPORT', 'BOOKS'], [894, 213, 173]],
    ['31980', 'kate morgan', 24.0, ['CLOTHES', 'BOOKS'], [439, 390]],
    ['32156', ' john doe ', 37.0, ['ELECTRONICS', 'HOME', 'FOOD'], [459, 120, 99]],
    ['32761', 'SAMANTHA SMITH', 29.0, ['CLOTHES', 'ELECTRONICS', 'BEAUTY'], [299, 679, 85]],
    ['32984', 'David White', 41.0, ['BOOKS', 'HOME', 'SPORT'], [234, 329, 243]],
    ['33001', 'emily brown', 26.0, ['BEAUTY', 'HOME', 'FOOD'], [213, 659, 79]],
    ['33767', ' Maria Garcia', 33.0, ['CLOTHES', 'FOOD', 'BEAUTY'], [499, 189, 63]],
    ['33912', 'JOSE MARTINEZ', 22.0, ['SPORT', 'ELECTRONICS', 'HOME'], [259, 549, 109]],
    ['34009', 'lisa wilson ', 35.0, ['HOME', 'BOOKS', 'CLOTHES'], [329, 189, 329]],
    ['34278', 'James Lee', 28.0, ['BEAUTY', 'CLOTHES', 'ELECTRONICS'], [189, 299, 579]],
]

# -------------------- Paso 1 ------------
# Store 1 tiene como objetivo garantizar la coherencia en la recopilación de datos. Como parte de esta iniciativa, 
# se debe evaluar la calidad de los datos recopilados sobre los usuarios y las usuarias. Te han pedido que revises los datos recopilados y propongas cambios. 
# A continuación verás datos sobre un usuario o una usuaria en particular; revisa los datos e identifica cualquier posible problema.

user_id = '32415'
user_name = ' mike_reed '
user_age = 32.0
fav_categories = ['ELECTRONICS', 'SPORT', 'BOOKS']


# ------------------- Paso 2------------
# Vamos a implementar los cambios que identificamos. Primero, necesitamos corregir los problemas de la variable user_name Como vimos,
# tiene espacios innecesarios y un guion bajo como separador entre el nombre y el apellido; tu objetivo es eliminar los espacios y luego 
# reemplazar el guion bajo con el espacio.

user_name= (user_name.strip()) #strip() eliminar espacios al principio y al final de una cadena.
user_name= user_name.replace ("_", " ") #replace() reemplaza el guion bajo por un espacio en blanco.
print (user_name)

# -------------------- Paso 3 ------------------
# Luego, debemos dividir el user_name (nombre de usuario o usuaria) actualizado en dos 
# # subcadenas para obtener una lista que contenga dos valores: la cadena para el nombre y
# la cadena para el apellido.

user_name = ' mike reed'
user_name= user_name.split() #split() divide la cadena en una lista de subcadenas utilizando el espacio como separador.
print (user_name)

#------------------ Paso 4 ------------------
# Ahora debemos trabajar con la variable user_age. Como ya mencionamos, esta tiene un tipo de datos incorrecto. 
# Arreglemos este problema transformando el tipo de datos y mostrando el resultado final.

user_age = 32.0
user_age=int(user_age) #int() convierte la cadena en un número entero.
print (user_age)

# ---------------- Paso 5 ------------------
# Como sabemos, los datos no siempre son perfectos. Debemos considerar escenarios en los que el valor de user_age no se pueda convertir en un número entero.
# Para evitar que nuestro sistema se bloquee, debemos tomar medidas con anticipación.
# Escribe un código que intente convertir la variable user_age en un número entero y asigna el valor transformado a user_age_int. 
# #Si el intento falla, mostramos un mensaje pidiendo al usuario o la usuaria que proporcione su edad como un valor numérico con el mensaje: 
# Please provide your age as a numerical value. (Proporcione su edad como un valor numérico.)

user_age= 'treinte y dos' #Ejemplo de valor no convertible a int 
# try-except para intentar la conversión; si falla, proporciona un mensaje claro indicando que la entrada debe ser numérica.
try:  
    user_age= int(user_age)# #int() convierte la cadena en un número entero.
except:
    print("Proporcione su edad como valor numerico")

# -------------------- Paso 6 ------------------
# El equipo de dirección de Store 1 te pidió ayudarles a organizar los datos de sus clientes para analizarlos y gestionarlos mejor.
# Debemos ordenar esta lista por ID de usuario de forma ascendente para que sea más fácil acceder a ella y analizarla.

users = users.copy() # Copia la lista original para no modificarla.
users.sort() # Ordena la lista de usuarios por ID de usuario de forma ascendente.
print (users)

#----------------- Paso 7 -------------------
# Tenemos la información de los hábitos de consumo de nuestros usuarios, incluyendo la cantidad gastada en cada una de sus categorías favoritas.
# La dirección está interesada en conocer la cantidad total gastada por el usuario.
# Calculemos este valor y despleguémoslo.

total_spendings = sum(users[0][4]) # Accedemos a la lista de gastos y la sumamos
print (total_spendings) # Imprimimos el total gastado por el usuario.

# ---------------- Paso 9 ------------------
# La dirección también quiere una forma fácil de conocer la cantidad de clientes con cuyos datos contamos.
# Crearemos una cadena formateada que muestre la cantidad de datos de clientes registrados.
# Esta es la cadena final que queremos crear: Hemos registrado datos de X clientes.

user_info= 'Hemos registrado datos de {} clientes'. format (len(users)) #len() devuelve la longitud de la lista de usuarios.
print(user_info)

# ---------------------- Paso 10 ------------------
# Ahora que hemos realizado todos los cambios necesarios, es hora de aplicar estos cambios a la lista de usuarios.      
# Apliquemos ahora todos los cambios a la lista de clientes. Para simplificar las cosas, te proporcionaremos una más corta. Debes:
# 1. Eliminaremos todos los espacios iniciales y finales de los nombres, así como cualquier guion bajo.
# 2. Convertir todas las edades en números enteros.
# 3. Separar todos los nombres y apellidos en una sublista.
# 4. Guarda la lista modificada como una nueva lista llamada users_clean y muéstrala en la pantalla.

users_list = [
    ['32415', ' mike_reed ', 32.0, ['ELECTRONICS', 'SPORT', 'BOOKS'], [894, 213, 173]],
    ['31980', 'kate morgan', 24.0, ['CLOTHES', 'BOOKS'], [439, 390]],
    ['32156', ' john doe ', 37.0, ['ELECTRONICS', 'HOME', 'FOOD'], [459, 120, 99]],]

user_clean=[] # Creamos una lista vacía para almacenar los datos limpios de los usuarios.

# Procesamos la lista de usuarios
user_name_0 =users_list[0][1].strip(). replace ("_", " ") # Eliminamos los espacios iniciales y finales del nombre de usuario y reemplazamos el guion bajo por un espacio.
user_name_0= int(users_list[0][2]) # Convertimos la edad a un número entero.
user_name_0= users_list[0][1].split() # Separamos el nombre y el apellido en una sublista.  
user_clean.append([users_list[0][0], user_name_0, users_list[0][3], users_list[0][4]]) # Agregamos los datos limpios a la lista de usuarios limpios.

user_name_1 =users_list[1][1].strip(). replace ("_", " ") # Eliminamos los espacios iniciales y finales del nombre de usuario y reemplazamos el guion bajo por un espacio.
user_name_1= int(users_list[1][2]) # Convertimos la edad a un número entero.
user_name_1= users_list[1][1].split() # Separamos el nombre y el apellido en una sublista.
user_clean.append([users_list[1][0], user_name_1, users_list[1][3], users_list[1][4]]) # Agregamos los datos limpios a la lista de usuarios limpios.

user_name_2 =users_list[2][1].strip(). replace ("_", " ") # Eliminamos los espacios iniciales y finales del nombre de usuario y reemplazamos el guion bajo por un espacio.
user_name_2= int(users_list[2][2]) # Convertimos la edad a un número entero.    
user_name_2= users_list[2][1].split() # Separamos el nombre y el apellido en una sublista.
user_clean.append([users_list[2][0], user_name_2, users_list[2][3], users_list[2][4]]) # Agregamos los datos limpios a la lista de usuarios limpios.   

print(user_clean) # Imprimimos la lista de usuarios limpios.
# La lista de usuarios limpios ahora contiene los datos corregidos y organizados de los usuarios.
