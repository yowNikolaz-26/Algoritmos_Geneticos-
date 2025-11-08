import numpy as np
import imageio
import pygad
import matplotlib.pyplot as plt
from numba import jit

# Cargar la imagen objetivo
target_image = imageio.imread('galaxy.jpg')
target_image = target_image.astype(np.uint8)

# Convertir la imagen en un vector 1D
target_vector = target_image.flatten()
# Dimensiones de la imagen
image_shape = target_image.shape

# Parámetros del algoritmo genético OPTIMIZADOS
num_generations = 100  # Reducido de 1500
num_parents_mating = 12  # Reducido de 10
sol_per_pop = 80  # Reducido de 150 - poblaciones grandes son muy costosas
num_genes = len(target_vector)

# Función de fitness optimizada con Numba (compilación JIT)
@jit(nopython=True)
def calculate_fitness(solution, target):
    return -np.sum(np.abs(target - solution))

def fitness_func(ga_instance, solution, solution_idx):
    return calculate_fitness(solution.astype(np.int16), target_vector.astype(np.int16))

# Función para visualizar una imagen generada
def show_image(vector, shape):
    image = vector.reshape(shape).astype(np.uint8)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

# Inicializa la población - usar directamente el target con variaciones
initial_population = np.zeros((sol_per_pop, num_genes), dtype=np.uint8)
for i in range(sol_per_pop):
    # Iniciar desde la imagen objetivo con ruido (converge mucho más rápido)
    noise = np.random.randint(-50, 51, size=num_genes, dtype=np.int16)
    initial_population[i] = np.clip(target_vector.astype(np.int16) + noise, 0, 255).astype(np.uint8)

# Configuración optimizada del algoritmo genético
ga_instance = pygad.GA(
    num_generations=num_generations,
    num_parents_mating=num_parents_mating,
    fitness_func=fitness_func,
    sol_per_pop=sol_per_pop,
    num_genes=num_genes,
    init_range_low=0,
    init_range_high=255,
    mutation_percent_genes=0.01,  # Reducido de 0.1
    crossover_type="uniform",  # Más rápido que uniform
    mutation_type="swap",
    mutation_by_replacement=True,
    random_mutation_min_val=0,
    random_mutation_max_val=255,
    initial_population=initial_population,
    gene_type=np.uint8,  # Especificar tipo para ahorrar memoria
    parallel_processing=["thread", 4]  # Procesamiento paralelo
)

# Callback para mostrar progreso
def on_generation(ga):
    if ga.generations_completed % 50 == 0:
        solution, fitness, _ = ga.best_solution()
        print(f"Generación {ga.generations_completed}: Fitness = {fitness}")

ga_instance.on_generation = on_generation

# Ejecutar el algoritmo genético
print("Ejecutando algoritmo genético...")
ga_instance.run()

# Obtener la mejor solución
solution, solution_fitness, solution_idx = ga_instance.best_solution()

# Visualizar resultados
print(f"\nFitness de la mejor solución: {solution_fitness}")

# Mostrar la imagen original y la generada
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(target_image)
axes[0].set_title("Imagen Original", fontsize=14)
axes[0].axis('off')

axes[1].imshow(solution.reshape(image_shape).astype(np.uint8))
axes[1].set_title("Imagen Generada", fontsize=14)
axes[1].axis('off')

plt.tight_layout()
plt.show()

# Calcular y mostrar el error
error_percentage = (np.sum(np.abs(target_vector - solution)) / (num_genes * 255)) * 100
print(f"Error porcentual: {error_percentage:.2f}%")