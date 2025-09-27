import numpy as np
from scipy.linalg import solve_continuous_lyapunov, svd
from scipy.integrate import solve_ivp
from scipy.signal import ss2tf
import matplotlib.pyplot as plt

# Poboljšana parametrizacija sustava
def generate_system(mu):
    A_base = np.array([[-1, 1, 0, 0],
                       [0, -2, 1, 0],
                       [0, 0, -3, 1],
                       [0, 0, 0, -4]])
    A = A_base.copy()
    A[0, 0] = -1 * mu  # Parametrizacija samo ključnih elemenata
    B = np.array([[1], [1], [0], [0]])  # Ispravljen ulaz na više stanja
    C = np.array([[1, 0, 1, 1]])        # Čitanje više stanja
    return A, B, C

# Balansirano odsjecanje
def balanced_truncation(A, B, C, r):
    # Rješavanje Lyapunovljevih jednadžbi
    P = solve_continuous_lyapunov(A, -B @ B.T)
    Q = solve_continuous_lyapunov(A.T, -C.T @ C)
    
    # SVD dekompozicija
    U, s, Vh = svd(P @ Q)
    Sigma_sqrt = np.sqrt(np.diag(s))
    
    # Regularizacija
    eps = 1e-12
    Sigma_reg = Sigma_sqrt[:r, :r] + eps * np.eye(r)
    
    # Transformacijske matrice
    T = U[:, :r] @ np.linalg.inv(Sigma_reg)
    T_inv = Sigma_reg @ U[:, :r].T
    
    # Reducirani sustav
    A_r = T_inv @ A @ T
    B_r = T_inv @ B
    C_r = C @ T
    
    # Ispis Hankel singularnih vrijednosti
    print(f"Hankel singular values: {np.sqrt(s[:5])}")
    return A_r, B_r, C_r, s

# Simulacija sustava
def simulate_system(A, B, C, t, u):
    def ode(_, x):
        return A @ x + B.flatten() * u
    
    x0 = np.zeros(A.shape[0])
    sol = solve_ivp(ode, [t[0], t[-1]], x0, t_eval=t)
    return (C @ sol.y).flatten()

# Greedy algoritam za odabir parametara
def greedy_param_selection(param_space, tol=1e-3, max_iter=10, r=2):
    selected_params = []
    reduced_models = []
    
    # Inicijalizacija srednjom vrijednošću
    mu_init = param_space[len(param_space)//2]
    selected_params.append(mu_init)
    A, B, C = generate_system(mu_init)
    A_r, B_r, C_r, _ = balanced_truncation(A, B, C, r)
    reduced_models.append((mu_init, A_r, B_r, C_r))
    
    for it in range(max_iter):
        max_err = 0
        worst_mu = None
        
        # Pronalaženje najveće greške
        for mu in param_space:
            if mu in selected_params:
                continue
            
            A, B, C = generate_system(mu)
            closest_idx = np.argmin(np.abs(np.array(selected_params) - mu))
            _, A_r, B_r, C_r = reduced_models[closest_idx]
            
            t = np.linspace(0, 10, 100)
            y_full = simulate_system(A, B, C, t, 1.0)
            y_red = simulate_system(A_r, B_r, C_r, t, 1.0)
            
            # Relativna greška
            err = np.linalg.norm(y_full - y_red) / np.linalg.norm(y_full)
            if err > max_err:
                max_err = err
                worst_mu = mu
        
        print(f"Iteracija {it+1}: max greška = {max_err:.5f} na mu = {worst_mu}")
        
        # Kriterij zaustavljanja
        if max_err < tol or worst_mu is None:
            break
        
        # Dodavanje novog parametra
        selected_params.append(worst_mu)
        A, B, C = generate_system(worst_mu)
        A_r, B_r, C_r, _ = balanced_truncation(A, B, C, r)
        reduced_models.append((worst_mu, A_r, B_r, C_r))
    
    return selected_params, reduced_models

# Glavni program
if __name__ == "__main__":
    # Parametarski prostor
    param_space = np.linspace(0.5, 1.5, 20)
    
    # Pokretanje greedy algoritma
    selected_params, reduced_models = greedy_param_selection(
        param_space, tol=1e-3, max_iter=5, r=2
    )
    print(f"Odabrani parametri: {selected_params}")
    
    # Testiranje za specifičan mu
    test_mu = 1.2
    A_full, B_full, C_full = generate_system(test_mu)
    
    # Odabir najbližeg modela
    closest_idx = np.argmin(np.abs(np.array(selected_params) - test_mu))
    _, A_r, B_r, C_r = reduced_models[closest_idx]
    
    # Simulacija odziva
    t = np.linspace(0, 10, 100)
    y_full = simulate_system(A_full, B_full, C_full, t, 1.0)
    y_red = simulate_system(A_r, B_r, C_r, t, 1.0)
    
    # Ispis transfer funkcije
    num, den = ss2tf(A_full, B_full, C_full, 0)
    print(f"Transfer funkcija za mu={test_mu}:\nBrojnik: {num}\nNazivnik: {den}")
    
    # Vizualizacija
    plt.figure(figsize=(10, 6))
    plt.plot(t, y_full, label="Puni model", linewidth=2)
    plt.plot(t, y_red, '--', label="Reducirani model", linewidth=2)
    plt.title(f'Usporedba izlaza za μ={test_mu}', fontsize=14)
    plt.xlabel('Vrijeme [s]', fontsize=12)
    plt.ylabel('Izlaz y(t)', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
