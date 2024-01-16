import math
import os
import random
from sympy import symbols, lambdify
import matplotlib.pyplot as plt   
import cv2


class DNA:
    num_bits = 1
    rango = 0
    rango_numero = 0
    resolucion = 0
    limite_inf = 0
    limite_sup = 0
    poblacion_inicial = 0
    poblacion_maxima = 0
    tipo_problema = ""
    poblacion_general = []
    prob_mut_individuo = 0
    prob_mut_gen = 0
    num_generaciones = 0 

formula = "(x**2 * cos(x))/100 + x**2 *sin(x)" 
class Individuo:
    identificador = 0
    def __init__(self, binario, i, x, y):
        Individuo.identificador += 1
        self.id = Individuo.identificador
        self.binario = binario
        self.i = i
        self.x = round(x, 4)
        self.y = round(y, 4)
    def __str__(self):
        return f"id: {self.id}, i: {self.i}, num.binario: {self.binario}, posición en X: {self.x}, posición en Y: {self.y}"

#GENERACION DE GRÁFICAS Y FOTOS
class Estadisticas:
    promedio = []
    peor_individuo = []
    mejor_individuo = []

    @classmethod
    def agregar_promedio(cls, generacion, promedio):
        cls.promedio.append((generacion, promedio))

    @classmethod
    def agregar_mejor_individuo(cls, generacion, mejor_individuo):
        cls.mejor_individuo.append((generacion, mejor_individuo))

    @classmethod
    def agregar_peor_individuo(cls, generacion, peor_individuo):
        cls.peor_individuo.append((generacion, peor_individuo))

def ordenar_por_generacion(filename):
    return int(filename.split('_')[-1].split('.')[0])

def crear_video():
    folder_path = 'generation_plots'
    output_folder = 'video_output'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    img_array = []
    for filename in sorted(os.listdir(folder_path), key=ordenar_por_generacion):
        if filename.endswith(".png"):
            file_path = os.path.join(folder_path, filename)
            img = cv2.imread(file_path)
            img_array.append(img)

    height, width, layers = img_array[0].shape
    video_path = os.path.join(output_folder, 'generation_video.avi')
    
    # Ajusta el codec y la tasa de fotogramas según tus necesidades
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), 2, (width, height))

    for i in range(len(img_array)):
        out.write(img_array[i])

    out.release()

    print(f"Video creado en: {video_path}")

    
def plot_generation(generation, individuals):
    x_values = [individuo.x for individuo in individuals]
    y_values = [individuo.y for individuo in individuals]

    plt.scatter(x_values, y_values)
    plt.title(f'Generation {generation}')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    folder_path = 'generation_plots'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    plt.savefig(os.path.join(folder_path, f'generation_{generation}.png'))
    plt.close()
    crear_video()
    
def calcular_valor_x(num_generado):
    valor_x = DNA.limite_inf + num_generado*DNA.resolucion
    return valor_x

def calcular_funcion(funcion, valor_x):
    x = symbols('x')
    expresion = lambdify(x, funcion, 'numpy')
    resultado = expresion(valor_x)
    return resultado



def calcular_datos():
    DNA.rango = DNA.limite_sup - DNA.limite_inf
    saltos = DNA.rango/DNA.resolucion
    puntos = saltos + 1
    num_bits = int(math.log2(puntos) + 1)
    DNA.rango_numero= 2**num_bits -1 
    DNA.num_bits = len(bin(DNA.rango_numero)[2:])
    
def primer_poblacion():
        for i in range(DNA.poblacion_inicial):
            num_generado = (random.randint(0, DNA.rango_numero))
            num_generado_binario = (bin(num_generado)[2:]).zfill(DNA.num_bits)
            valor_x = calcular_valor_x(num_generado)
            valor_y = calcular_funcion(formula, valor_x)
            individuo = Individuo(i=num_generado, binario=num_generado_binario, x=valor_x, y= valor_y)
            DNA.poblacion_general.append(individuo)

def algoritmo_genetico(data):   
 
    DNA.poblacion_inicial = int(data.pob_inicial)
    DNA.poblacion_maxima = int(data.pob_max)
    DNA.resolucion = float(data.resolucion)
    DNA.limite_inf = float(data.lim_inf)
    DNA.limite_sup = float(data.lim_sup)
    DNA.prob_mut_individuo = float(data.mut_ind)
    DNA.prob_mut_gen = float(data.mut_gen)
    DNA.tipo_problema = data.problema
    DNA.num_generaciones = int(data.num_generaciones)
    
    calcular_datos()
    primer_poblacion()
    
    for generacion in range(1, DNA.num_generaciones + 1):
        print(f"\ngeneracion {generacion}:")
        inicializar(generacion)
        plot_generation(generacion, DNA.poblacion_general)

    for generacion, mejor_individuo in Estadisticas.peor_individuo:
        print(f"mejor individuo {generacion}, id: {mejor_individuo.id}, i: {mejor_individuo.i}, num.binario: {mejor_individuo.binario}, el punto en X: {mejor_individuo.x}, el punto en Y: {mejor_individuo.y}")
    plot_stats()


def inicializar(generacion):
    mejor_ind_act, peor_ind_act = optimizar()

    Estadisticas.agregar_mejor_individuo(generacion, mejor_ind_act)
    Estadisticas.agregar_peor_individuo(generacion, peor_ind_act)

    suma_y = sum(individuo.y for individuo in DNA.poblacion_general)
    promedio = suma_y / len(DNA.poblacion_general)
    Estadisticas.agregar_promedio(generacion, promedio)

  
def optimizar():
    bandera = True
    
    if DNA.tipo_problema == "Minimizacion":
        bandera = False
    individuos_ordenados = sorted(DNA.poblacion_general, key=lambda x: x.y, reverse=bandera)
    
    mitad = int(len(individuos_ordenados) / 2)
    
    part_mejor_aptitud = individuos_ordenados[:mitad]
    
    
    part_menor_aptitud = individuos_ordenados[mitad:]
    
    resto_poblacion = []
    for individuo in part_menor_aptitud:
        resto_poblacion.append(individuo)
        
    emparejar(resto_poblacion, part_mejor_aptitud)
    return part_mejor_aptitud[0], resto_poblacion[-1]

def plot_stats():
    generaciones = [generacion for generacion, _ in Estadisticas.mejor_individuo]
    mejores_y = [mejor_individuo.y for _, mejor_individuo in Estadisticas.mejor_individuo]
    peores_y = [peor_individuo.y for _, peor_individuo in Estadisticas.peor_individuo]
    promedio_y = [promedio for _, promedio in Estadisticas.promedio]

    plt.plot(generaciones, mejores_y, label='Mejor Individuo')
    plt.plot(generaciones, peores_y, label='Peor Individuo')
    plt.plot(generaciones, promedio_y, label='Promedio')

    plt.title('Estadísticas de la Población por Generación')
    plt.xlabel('Generación')
    plt.ylabel('Valor de la Función Objetivo')
    plt.legend()

    folder_path = 'stats_plots'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    plt.savefig(os.path.join(folder_path, 'population_stats.png'))
    plt.close()

def emparejar(resto_poblacion, part_mejor_aptitud):
    
    new_poblacion = []
    for individuo in resto_poblacion:
        new_poblacion.append(individuo)
    
    
    for individuo in part_mejor_aptitud:
        new_poblacion.append(individuo)

    for mejor_individuo in part_mejor_aptitud:
        for individuo in resto_poblacion:
            
            new_individuo1, new_individuo2 = cruzar(mejor_individuo, individuo)
            new_poblacion.append(new_individuo1)
            new_poblacion.append(new_individuo2)


def cruzar(mejor_individuo, individuo): 
    
    puntoDeCruza = int(DNA.num_bits / 2)
    
    p1 = mejor_individuo.binario[:puntoDeCruza]
    p2 = mejor_individuo.binario[puntoDeCruza:]
    p3 = individuo.binario[:puntoDeCruza]
    p4 = individuo.binario[puntoDeCruza:]
    
    new_individuo_1 = p1 + p4
    new_individuo_2 = p3 + p2
    
    if(random.randint(1,100))/100 <= DNA.prob_mut_individuo:
        new_individuo_1 = mutar(new_individuo_1)
        
    if(random.randint(1,100))/100 <= DNA.prob_mut_individuo:
        new_individuo_2 = mutar(new_individuo_2)
    
    nuevos_individuos(new_individuo_1, new_individuo_2)
    podar()
    return new_individuo_1, new_individuo_2



def mutar(individuo):
    binarioSeparado = list(individuo)
    
    for i in range(len(binarioSeparado)):
        if (random.randint(1,100))/100 <= DNA.prob_mut_gen:
            binarioSeparado[i] = '1' if binarioSeparado[i] == '0' else '0'
    new_binario = ''.join(binarioSeparado)
    
    return new_binario


def nuevos_individuos(individuo1, individuo2):
    numero_decimal1 = int(individuo1, 2)
    numero_decimal2 = int(individuo2, 2)
    x1 = DNA.limite_inf + numero_decimal1*DNA.resolucion
    x2 = DNA.limite_inf + numero_decimal2*DNA.resolucion
    y1 = calcular_funcion(formula, x1)
    y2 = calcular_funcion(formula, x2)
    
    individuo1 = Individuo(i=numero_decimal1, binario=individuo1, x=x1, y= y1)
    individuo2 = Individuo(i=numero_decimal2, binario=individuo2, x=x2, y= y2)
    
    DNA.poblacion_general.append(individuo1)
    DNA.poblacion_general.append(individuo2)
    


def podar():
    i_conjunta = set()
    
    poblacion_unique = []

    for individuo in DNA.poblacion_general:
        if individuo.i not in i_conjunta:
            i_conjunta.add(individuo.i)
            poblacion_unique.append(individuo)

    DNA.poblacion_general = poblacion_unique


    bandera = True
    if DNA.tipo_problema == "Minimizacion":
        bandera = False
    individuos_ordenados = sorted(DNA.poblacion_general, key=lambda x: x.y, reverse=bandera)


    if len(individuos_ordenados) > DNA.poblacion_maxima:
        DNA.poblacion_general = individuos_ordenados[:DNA.poblacion_maxima]
 
 
 
    print("-------------poblacion despues de podar-----------------")
    for individuo in DNA.poblacion_general:
        print(individuo)

