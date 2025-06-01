import numpy as np
import matplotlib.pyplot as plt

def FTBS(initial: tuple, boundary: tuple, k: float, h: float, rho_max: float, v_max: float, tau: float, chi: float, c0: float, l: float = 1, m: float = 1, plot=True):
    """
    Rozwiązuje układ równań modelu PW za pomocą metody FTBS (Forward-Time Backward-Space).

    Parametry:
        initial (tuple):
            - początkowe wartości gęstości
            - początkowe wartości prędkości
        boundary (tuple):
            - warunek brzegowy dla gęstości
            - warunek brzegowy dla prędkości

        k (float): Krok czasowy (Δt)
        h (float): Krok przestrzenny (Δx)
        rho_max (float): Maksymalna wartość gęstości
        v_max (float): Maksymalna wartość prędkości
        tau (float): Czas relaksacji
        chi (float): Stała zapobiegająca zerowaniu się mianownika
        c0 (float): Stała związana z przewidywaniem zmian gęstości
        l (float): Parametr kształtu funkcji prędkości dla stanu równowagi
        m (float): Parametr kształtu funkcji prędkości dla stanu równowagi
        plot (bool): Czy generować wykresy

        Zwraca:
            - rho (ndarray (N+1, T+1)): Rozwiązanie dla gęstości
            - v (ndarray (N+1, T+1)): Rozwiązanie dla prędkości
            - cfl (float): Warunek CFL
    """    
    
    if not len(initial[0]) == len(initial[1]):
        raise ValueError("Warunki początkowe nie są tej samej długości.")
    if not len(boundary[0]) == len(boundary[1]):
        raise ValueError("Warunki brzegowe nie są tej samej długości.")
    check = [k, h, l, m, v_max, rho_max, tau, c0, chi]
    if any(val <= 0 for val in check):
        raise ValueError("Parametry: k, h, l, m, v_max, rho_max, tau, c0, chi powinny być dodatnie.")
    
    x = np.linspace(0, rho_max, 500)
    y = v_max*(1 - (x/rho_max)**l)**m - v_max*m*l*((x/rho_max)**l)*(1 - (x/rho_max)**l)**(m-1)
    cfl = max(abs(y))*k/h
    
    if cfl >= 1:
        raise Warning("Wynik będzie niestabilny. Dostosuj parametry: k, h.")
    
    N = len(initial[0]) - 1
    T = len(boundary[0]) - 1

    rho = np.zeros((N+1, T+1))
    v = np.zeros((N+1, T+1))

    time = np.linspace(0, k*T, T+1)
    space = np.linspace(0, h*N, N+1)

    #initial
    rho[:, 0] = initial[0]
    v[:, 0] = initial[1]
    #boundary
    rho[0, :] = boundary[0]
    v[0, :] = boundary[1]

    r = k/h
    for t in range(1, T+1):
        for s in range(1, N+1):
            rho[s, t] = rho[s, t-1] + r*(v[s-1, t-1]*rho[s-1, t-1] - v[s, t-1]*rho[s, t-1])
        for s in range(1, N):    
            v[s, t] = v[s, t-1] + k/tau*(v_max*(1 - (rho[s, t-1]/rho_max)**l)**m - v[s, t-1]) \
                 + r*v[s, t-1]*(v[s-1, t-1] - v[s, t-1]) - c0*r*(rho[s+1, t-1] - rho[s, t-1])/(chi + rho[s, t-1]) 

    if plot:        
        S, T = np.meshgrid(space[0:-1], time)

        fig = plt.figure(figsize=plt.figaspect(0.5))

        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.plot_surface(S, T, rho.transpose()[:, 0:-1], cmap='viridis', edgecolor='none')
        ax.view_init(elev=30, azim=-155)
        ax.set_title('Gęstość')

        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.plot_surface(S, T, v.transpose()[:, 0:-1], cmap='viridis', edgecolor='none')
        ax.view_init(elev=30, azim=-155)
        ax.set_title('Prędkość')

        plt.show()        

    return rho, v, cfl

def FTCS(initial: tuple, boundary: tuple, k: float, h: float, rho_max: float, v_max: float, tau: float, chi: float, c0: float, l: float = 1, m: float = 1, plot=True):
    if not len(initial[0]) == len(initial[1]):
        raise ValueError("Warunki początkowe nie są tej samej długości.")
    if not len(boundary[0]) == len(boundary[1]):
        raise ValueError("Warunki brzegowe nie są tej samej długości.")
    check = [k, h, l, m, v_max, rho_max, tau, c0, chi]
    if any(val <= 0 for val in check):
        raise ValueError("Parametry: k, h, l, m, v_max, rho_max, tau, c0, chi powinny być dodatnie.")
    
    x = np.linspace(0, rho_max, 500)
    y = v_max*(1 - (x/rho_max)**l)**m - v_max*m*l*((x/rho_max)**l)*(1 - (x/rho_max)**l)**(m-1)
    cfl = max(abs(y))*k/h
    
    if cfl >= 1:
        raise Warning("Wynik będzie niestabilny. Dostosuj parametry: k, h.")
    
    N = len(initial[0]) - 1
    T = len(boundary[0]) - 1

    rho = np.zeros((N+1, T+1))
    v = np.zeros((N+1, T+1))

    time = np.linspace(0, k*T, T+1)
    space = np.linspace(0, h*N, N+1)

    #initial
    rho[:, 0] = initial[0]
    v[:, 0] = initial[1]
    #boundary
    rho[0, :] = boundary[0]
    rho[N, :] = boundary[1]
    v[0, :] = boundary[2]
    v[N, :] = boundary[3]

    r = k/h
    for t in range(1, T+1):
        for s in range(1, N):
            rho[s, t] = 0.5*(rho[s-1, t-1] - rho[s+1, t-1]) + 0.5*r*(v[s-1, t-1]*rho[s-1, t-1] - v[s+1, t-1]*rho[s+1, t-1])
        for s in range(1, N):    
            v[s, t] = v[s, t-1] + k/tau*(v_max*(1 - (rho[s, t-1]/rho_max)**l)**m - v[s, t-1]) \
                 + r*v[s, t-1]*(v[s-1, t-1] - v[s, t-1]) - c0*r*(rho[s+1, t-1] - rho[s, t-1])/(chi + rho[s, t-1]) 

    if plot:        
        S, T = np.meshgrid(space[0:-1], time)

        fig = plt.figure(figsize=plt.figaspect(0.5))

        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.plot_surface(S, T, rho.transpose()[:, 0:-1], cmap='viridis', edgecolor='none')
        ax.view_init(elev=30, azim=-155)
        ax.set_title('Gęstość')

        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.plot_surface(S, T, v.transpose()[:, 0:-1], cmap='viridis', edgecolor='none')
        ax.view_init(elev=30, azim=-155)
        ax.set_title('Prędkość')

        plt.show()        

    return rho, v, cfl

def L1_error(real_sol: float, num_sol: float, h: float, k: float):
    if not real_sol.shape == num_sol.shape:
        raise ValueError("Rozwiązania mają różne wymiary.")
    
    diff = np.abs(real_sol-num_sol)
    return np.sum(diff*k*h)

# TODO metoda 2 z overleaf
