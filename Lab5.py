import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.cm as cm
import sys

def simulacion_difusion(M, N, T, u0, K, region):
    """
    Simula el proceso de difusión en un grid MxN durante T unidades de tiempo.
    """
    # Inicialización del grid
    u = np.zeros((T+1, M, N))
    u[0] = u0.copy()

    # Vecindad de 8 vecinos
    vecinos = [(-1, -1), (-1, 0), (-1, 1),
               (0, -1),         (0, 1),
               (1, -1),  (1, 0),  (1, 1)]

    for t in range(T):
        u_t = u[t].copy()
        u_t1 = u[t].copy()
        for i in range(M):
            for j in range(N):
                if region[i, j]:
                    suma_vecinos = 0
                    contador_vecinos = 0
                    for dx, dy in vecinos:
                        x, y = i + dx, j + dy
                        if 0 <= x < M and 0 <= y < N and region[x, y]:
                            suma_vecinos += u_t[x, y]
                            contador_vecinos +=1
                    if contador_vecinos > 0:
                        u_t1[i, j] = (1 - K) * u_t[i, j] + (K / contador_vecinos) * suma_vecinos
        u[t+1] = u_t1.copy()
    return u

def simulacion_particulas(M, N, T, u0, K, region, P, Nexp):
    """
    Simula la difusión usando partículas, promediando sobre Nexp experimentos.
    """
    promedio_grid = np.zeros((T+1, M, N))
    for exp in range(Nexp):
        # Inicializar partículas según u0
        indices = np.array(np.nonzero(region)).T
        probs = u0[region]
        probs = probs / probs.sum()
        particulas = np.random.choice(len(indices), size=P, p=probs)
        posiciones = indices[particulas]

        grid_exp = np.zeros((T+1, M, N))
        for t in range(T+1):
            grid_temp = np.zeros((M, N))
            for pos in posiciones:
                grid_temp[pos[0], pos[1]] += 1
            promedio_grid[t] += grid_temp
            # Mover partículas
            nuevas_posiciones = []
            for pos in posiciones:
                if np.random.rand() < K:
                    # Mover a un vecino al azar
                    vecinos = [(-1, -1), (-1, 0), (-1, 1),
                               (0, -1),         (0, 1),
                               (1, -1),  (1, 0),  (1, 1)]
                    np.random.shuffle(vecinos)
                    for dx, dy in vecinos:
                        x, y = pos[0] + dx, pos[1] + dy
                        if 0 <= x < M and 0 <= y < N and region[x, y]:
                            nuevas_posiciones.append([x, y])
                            break
                    else:
                        nuevas_posiciones.append(pos)
                else:
                    # La partícula se queda en el mismo lugar
                    nuevas_posiciones.append(pos)
            posiciones = np.array(nuevas_posiciones)
    # Promediar y normalizar
    promedio_grid /= (Nexp * P)
    return promedio_grid

def crear_region_L(M, N, r, s):
    """
    Crea una región en forma de L de tamaño MxN, removiendo un subgrid de tamaño rxs.
    """
    region = np.ones((M, N), dtype=bool)
    region[M - r:, N - s:] = False
    return region

def graficar_simulacion(u, region, titulo):
    """
    Genera una animación de la simulación y la guarda con el nombre proporcionado.
    """
    T = u.shape[0] - 1
    M, N = u.shape[1], u.shape[2]
    fig, ax = plt.subplots()
    cmap = cm.viridis
    ims = []
    for t in range(T+1):
        im = ax.imshow(u[t], cmap=cmap, interpolation='nearest', animated=True)
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True, repeat_delay=1000)
    plt.title(titulo)
    plt.colorbar(im, ax=ax)
    # Guardar la animación con el nombre del título
    ani.save(f'{titulo}.gif', writer='Pillow')
    plt.close()

def main():
    while True:
        print("\nSeleccione el ejercicio a simular:")
        print("1. Simulación de difusión en grid")
        print("2. Simulación de difusión usando partículas")
        print("3. Experimentos con diferentes regiones y parámetros")
        print("4. Salir")
        opcion = int(input("Ingrese el número de la opción: "))

        if opcion == 1:
            # Parámetros para la simulación
            M, N = 50, 50
            T = 200
            K = 0.5
            u0 = np.zeros((M, N))
            u0[M//2, N//2] = 1  # Distribución inicial en el centro
            region = np.ones((M, N), dtype=bool)  # Región completa
            u = simulacion_difusion(M, N, T, u0, K, region)
            titulo = "Difusion_en_grid_completo"
            graficar_simulacion(u, region, titulo)
        elif opcion == 2:
            # Parámetros para la simulación
            M, N = 50, 50
            T = 200
            K = 0.5
            P = 1000
            Nexp = 100
            u0 = np.zeros((M, N))
            u0[M//2, N//2] = 1  # Distribución inicial en el centro
            region = np.ones((M, N), dtype=bool)  # Región completa
            promedio_grid = simulacion_particulas(M, N, T, u0, K, region, P, Nexp)
            titulo = "Difusion_usando_particulas"
            graficar_simulacion(promedio_grid, region, titulo)
        elif opcion == 3:
            # Experimentos con diferentes regiones y parámetros
            M, N = 50, 50
            T = 100
            K_values = [0.1, 0.5, 0.9]
            u0_options = ['centro', 'esquina']
            for K in K_values:
                for u0_option in u0_options:
                    u0 = np.zeros((M, N))
                    if u0_option == 'centro':
                        u0[M//2, N//2] = 1
                    elif u0_option == 'esquina':
                        u0[0, 0] = 1
                    # Región en forma de L
                    r, s = 20, 20
                    region = crear_region_L(M, N, r, s)
                    u = simulacion_difusion(M, N, T, u0, K, region)
                    titulo = f"Difusion_con_K={K}_u0={u0_option}"
                    graficar_simulacion(u, region, titulo)
        elif opcion == 4:
            print("Saliendo del programa.")
            sys.exit()
        else:
            print("Opción no válida. Por favor, intente de nuevo.")

if __name__ == "__main__":
    main()
