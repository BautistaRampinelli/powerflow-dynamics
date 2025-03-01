import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


st.title("Simulaciones Interactivas – Máquinas y Termomecánica")
st.markdown("""
Esta aplicación forma parte de mi portfolio y reúne distintos trabajos prácticos (TP) sobre:
- Transformaciones politrópicas
- Bombas
- Compresores Centrífugos
- Compresores Axiales
- Turbinas Hidráulicas (Pelton)
- Turbinas de Vapor
- Turbinas de Gas (Ciclo Brayton)

Selecciona en la barra lateral el TP que deseas explorar.
""")

# Menú de selección
opcion = st.sidebar.selectbox("Selecciona el Trabajo Práctico:", 
    ["1. Transformaciones Politrópicas",
     "2. Análisis de Bombas",
     "3. Compresores Centrífugos",
     "4. Compresores Axiales",
     "5. Turbinas Hidráulicas",
     "6. Turbinas de Vapor",
     "7. Turbinas de Gas"])

###############################################################################
# 1. Transformaciones Politrópicas
###############################################################################
if opcion == "1. Transformaciones Politrópicas":
    st.header("Transformaciones Politrópicas en Diagrama T–V")
    st.markdown("""
    **Descripción Teórica:**  
    Se muestran tres procesos: isoentrópico, isotérmico e isobárico.  
    Se utiliza la fórmula:  
    - Isoentrópico: *T = T₀ · (V₀/V)^(k-1)*  
    - Isotérmico: *T = T₀*  
    - Isobárico: *T = T₀ · (V/V₀)*
    """)
    
    T0 = st.slider("T₀ (K)", min_value=250.0, max_value=350.0, value=298.15, step=1.0)
    V0 = st.slider("V₀ (m³/kg)", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
    k_val = st.slider("Exponente k", min_value=1.2, max_value=1.67, value=1.4, step=0.01)
    
    V = np.linspace(0.5 * V0, 1.5 * V0, 200)
    T_iso = T0 * (V0 / V) ** (k_val - 1)
    T_isoTherm = np.full_like(V, T0)
    T_isobar = T0 * (V / V0)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(V, T_iso, label=f"Isoentrópico (n={k_val:.2f}, m={k_val-1:.2f})", linewidth=2)
    ax.plot(V, T_isoTherm, label="Isotérmico (n=1, m=0)", linewidth=2)
    ax.plot(V, T_isobar, label="Isobárico (n=0, m=-1)", linewidth=2)
    ax.axvline(x=V0, color='gray', linestyle='--', label="Isovolumétrica (V=V₀)")
    ax.set_xlabel("Volumen específico (m³/kg)")
    ax.set_ylabel("Temperatura (K)")
    ax.set_title("Transformaciones Politrópicas en Diagrama T–V")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    
    st.code(f'''import numpy as np
import matplotlib.pyplot as plt

def plot_polytropic(T0, V0, k):
    V = np.linspace(0.5*V0, 1.5*V0, 200)
    T_iso = T0 * (V0/V)**(k - 1)
    T_isoTherm = np.full_like(V, T0)
    T_isobar = T0 * (V/V0)
    plt.figure(figsize=(8,6))
    plt.plot(V, T_iso, label=f"Isoentrópico (n={{k:.2f}}, m={{k-1:.2f}})")
    plt.plot(V, T_isoTherm, label="Isotérmico (n=1, m=0)")
    plt.plot(V, T_isobar, label="Isobárico (n=0, m=-1)")
    plt.axvline(x=V0, color='gray', linestyle='--', label="Isovolumétrica (V=V₀)")
    plt.xlabel("Volumen específico (m³/kg)")
    plt.ylabel("Temperatura (K)")
    plt.title("Transformaciones Politrópicas en Diagrama T–V")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_polytropic({T0}, {V0}, {k_val})
''', language='python')

###############################################################################
# 2. Análisis de Bombas
###############################################################################
elif opcion == "2. Análisis de Bombas":
    st.header("Análisis de Bombas")
    st.markdown("""
    **Descripción Teórica:**  
    Se analiza la velocidad específica de la bomba, se calcula la velocidad requerida para alcanzar un valor deseado y se estima el número mínimo de rodetes mediante:
    
    \[
    n_q = \frac{n \sqrt{Q}}{H^{3/4}}, \qquad N_{\min} = \left\lceil \left(\frac{n_{q, \text{deseado}}\, H^{3/4}}{n \sqrt{Q}}\right)^{4/3} \right\rceil
    \]
    """)
    
    n_rpm = st.slider("RPM", min_value=500, max_value=3000, value=1250, step=50)
    Q_m3h = st.slider("Caudal (m³/h)", min_value=10.0, max_value=200.0, value=50.0, step=5.0)
    H_bomba = st.slider("Altura útil (m)", min_value=10.0, max_value=200.0, value=80.0, step=5.0)
    n_q_deseado = st.slider("n_q deseado", min_value=4.0, max_value=12.0, value=8.0, step=0.5)
    
    Q = Q_m3h / 3600.0
    n_q = n_rpm * np.sqrt(Q) / (H_bomba ** 0.75)
    n_required = n_q_deseado * (H_bomba ** 0.75) / np.sqrt(Q)
    N_min = int(np.ceil(((n_q_deseado * (H_bomba ** 0.75)) / (n_rpm * np.sqrt(Q))) ** (4/3)))
    
    st.write(f"Velocidad específica: **{n_q:.2f}**")
    st.write(f"Para alcanzar n_q = {n_q_deseado}:")
    st.write(f"- Velocidad requerida: **{n_required:.0f} rpm**")
    st.write(f"- Número mínimo de rodetes: **{N_min}**")
    
    N_vals = np.linspace(1, 10, 100)
    n_q_etapa = n_rpm * np.sqrt(Q) * (N_vals ** 0.75) / (H_bomba ** 0.75)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(N_vals, n_q_etapa, label=r'$n_{q,etapa} = \frac{n\sqrt{Q}\,N^{3/4}}{H^{3/4}}$', linewidth=2)
    ax.axhline(n_q_deseado, color='red', linestyle='--', label=f"n_q deseado = {n_q_deseado}")
    ax.axvline(N_min, color='green', linestyle='--', label=f"N_min = {N_min}")
    ax.set_xlabel("Número de rodetes (N)")
    ax.set_ylabel("n_q por etapa")
    ax.set_title("Evolución de n_q vs. Número de rodetes")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    
    st.code(f'''import numpy as np
def pump_analysis(n, Q_m3h, H, n_q_deseado):
    Q = Q_m3h / 3600.0
    n_q = n * np.sqrt(Q) / (H**0.75)
    n_required = n_q_deseado * (H**0.75) / np.sqrt(Q)
    N_min = int(np.ceil(((n_q_deseado * (H**0.75))/(n * np.sqrt(Q)))**(4/3)))
    print(f"n_q = {{n_q:.2f}}")
    print(f"n_required = {{n_required:.0f}} rpm, N_min = {{N_min}}")
pump_analysis({n_rpm}, {Q_m3h}, {H_bomba}, {n_q_deseado})
''', language='python')

###############################################################################
# 3. Compresores Centrífugos
###############################################################################
elif opcion == "3. Compresores Centrífugos":
    st.header("Compresores Centrífugos")
    st.markdown("""
    **Descripción Teórica:**  
    Se calcula la velocidad del rotor, se estima la velocidad relativa y la componente tangencial, lo que permite determinar:
    - Factor de deslizamiento: \( \sigma = \frac{C_{u2}}{u} \)
    - Trabajo ideal impartido y relación de compresión.
    """)
    
    n_input = st.slider("RPM (bomba)", min_value=5000, max_value=15000, value=12500, step=100)
    C_r2 = st.slider("Velocidad radial a la salida (C₍r2₎, m/s)", min_value=50.0, max_value=200.0, value=110.0, step=5.0)
    theta_deg = st.slider("Ángulo θ (°)", min_value=0.0, max_value=45.0, value=25.5, step=0.5)
    u_max = st.slider("Velocidad máxima (u_max, m/s)", min_value=300.0, max_value=600.0, value=460.0, step=10.0)
    eta_s = st.slider("Eficiencia isoentrópica (ηₛ)", min_value=0.70, max_value=0.95, value=0.80, step=0.01)
    
    # Suponemos un rotor de 40 cm de diámetro (r = 0.20 m)
    r2 = 0.40 / 2.0
    u = u_max
    rpm_calc = u / r2 * (60 / (2 * np.pi))
    theta = np.radians(theta_deg)
    W2 = C_r2 / np.cos(theta)
    W_u2 = W2 * np.sin(theta)
    C_u2 = u - W_u2
    slip_factor = C_u2 / u
    delta_h_ideal = u * C_u2
    c_p = 1005.0
    T_in_comp = 298.15
    T2s = T_in_comp + delta_h_ideal / c_p
    T2_actual = T_in_comp + delta_h_ideal / (eta_s * c_p)
    k_comp = 1.4
    pr = (T2s / T_in_comp) ** (k_comp / (k_comp - 1))
    delta_h_actual = c_p * (T2_actual - T_in_comp)
    C2 = np.sqrt(C_r2**2 + C_u2**2)
    T_exit = T2s - C2**2 / (2 * c_p)
    a_exit = np.sqrt(k_comp * 287 * T_exit)
    Mach_exit = C2 / a_exit
    
    st.write(f"Velocidad del rotor (u): **{u:.1f} m/s**  →  RPM ≈ **{rpm_calc:.0f}**")
    st.write(f"Factor de deslizamiento: **{slip_factor:.3f}**")
    st.write(f"Trabajo ideal: **{delta_h_ideal:.0f} J/kg**")
    st.write(f"T₂s: **{T2s:.1f} K**, Relación de compresión: **{pr:.2f}**")
    st.write(f"Trabajo real: **{delta_h_actual:.0f} J/kg**")
    st.write(f"Velocidad absoluta de salida: **{C2:.1f} m/s**")
    st.write(f"Temperatura estática de salida: **{T_exit:.1f} K**")
    st.write(f"Número de Mach: **{Mach_exit:.2f}**")
    
    # Diagrama de velocidades
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax[0].quiver(0, 0, u, 0, angles='xy', scale_units='xy', scale=1, color='blue', label='u')
    ax[0].quiver(u, 0, -W_u2, C_r2, angles='xy', scale_units='xy', scale=1, color='red', label='W₂')
    ax[0].quiver(0, 0, u - W_u2, C_r2, angles='xy', scale_units='xy', scale=1, color='green', label='C₂')
    ax[0].set_title("Diagrama de Velocidades en el Rotor")
    ax[0].set_xlabel("Componente Tangencial (m/s)")
    ax[0].set_ylabel("Componente Radial (m/s)")
    ax[0].axis('equal')
    ax[0].grid(True)
    ax[0].legend()
    
    keys = ['RPM', 'Slip', 'p₂/p₁', 'Trabajo real', 'Mach']
    values = [rpm_calc, slip_factor, pr, delta_h_actual, Mach_exit]
    ax[1].bar(keys, values, color=['purple', 'orange', 'teal', 'magenta', 'brown'])
    ax[1].set_title("Parámetros Clave")
    ax[1].set_ylabel("Valor")
    for i, v in enumerate(values):
        ax[1].text(i, v * 1.05, f"{v:.2f}", ha='center')
    st.pyplot(fig)
    
    st.code("Código simplificado de compresor centrífugo...", language="python")

###############################################################################
# 4. Compresores Axiales
###############################################################################
elif opcion == "4. Compresores Axiales":
    st.header("Compresores Axiales")
    st.markdown("""
    **Descripción Teórica:**  
    Se modela el comportamiento de un compresor axial mediante el análisis de los triángulos de velocidades de entrada y salida.  
    Se calculan parámetros como la velocidad del rotor, la velocidad relativa y la potencia.
    """)
    
    rpm_ax = st.slider("RPM", min_value=8000, max_value=12000, value=9500, step=100)
    r_mean = st.slider("Radio medio (m)", min_value=0.3, max_value=0.6, value=0.40, step=0.01)
    m_dot_ax = st.slider("Caudal másico (kg/s)", min_value=5.0, max_value=30.0, value=15.0, step=0.5)
    T_in_ax = st.slider("T_in (K)", min_value=280, max_value=320, value=298, step=1)
    C_axial = st.slider("Velocidad axial (m/s)", min_value=100, max_value=200, value=150, step=5)
    beta2_deg = st.slider("β₂ (°)", min_value=10.0, max_value=40.0, value=20.0, step=0.5)
    eta_ax = st.slider("Eficiencia (η)", min_value=0.80, max_value=0.95, value=0.86, step=0.01)
    
    omega = 2 * np.pi * rpm_ax / 60.0
    U = omega * r_mean
    C1 = np.array([C_axial, 0.0])
    U_vec = np.array([0.0, U])
    W1 = C1 - U_vec
    W1_mag = np.linalg.norm(W1)
    beta1 = np.degrees(np.arctan2(abs(U), C_axial))
    
    beta2_rad = np.radians(beta2_deg)
    W2 = C_axial / np.cos(beta2_rad)
    W2_t = -W2 * np.sin(beta2_rad)
    W2_vec = np.array([C_axial, W2_t])
    C2_vec = W2_vec + U_vec
    C2_mag = np.linalg.norm(C2_vec)
    C2_t = C2_vec[1]
    delta_h_ideal_ax = U * C2_t
    c_p_ax = 1005.0
    T0_ideal_ax = T_in_ax + delta_h_ideal_ax / c_p_ax
    pr_ideal_ax = (T0_ideal_ax / T_in_ax) ** (1.4 / (1.4 - 1))
    delta_h_actual_ax = delta_h_ideal_ax / eta_ax
    Power_ax = m_dot_ax * delta_h_actual_ax
    C2 = C2_mag
    T_exit_ax = T0_ideal_ax - C2**2 / (2 * c_p_ax)
    a_exit_ax = np.sqrt(1.4 * 287 * T_exit_ax)
    Mach_exit_ax = C2 / a_exit_ax
    
    st.write(f"Velocidad del rotor (U): **{U:.1f} m/s**")
    st.write(f"|W₁| = **{W1_mag:.1f} m/s**, β₁ ≈ **{beta1:.1f}°**")
    st.write(f"|C₂| = **{C2_mag:.1f} m/s**")
    st.write(f"Trabajo ideal: **{delta_h_ideal_ax:.0f} J/kg**, T₀,ideal = **{T0_ideal_ax:.1f} K**")
    st.write(f"Potencia requerida: **{Power_ax/1e3:.1f} kW**, Mach = **{Mach_exit_ax:.2f}**")
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax[0].quiver(0, 0, C1[0], C1[1], angles='xy', scale_units='xy', scale=1, color='blue', label='C₁ (Entrada)')
    ax[0].quiver(0, 0, 0, U, angles='xy', scale_units='xy', scale=1, color='green', label='U (Rotor)')
    ax[0].quiver(0, 0, C1[0], -U, angles='xy', scale_units='xy', scale=1, color='red', label='W₁')
    ax[0].set_title("Triángulo de Velocidades – Entrada")
    ax[0].axis('equal')
    ax[0].grid(True)
    ax[0].legend()
    
    ax[1].quiver(0, 0, C_axial, W2_t, angles='xy', scale_units='xy', scale=1, color='red', label='W₂')
    ax[1].quiver(0, 0, 0, U, angles='xy', scale_units='xy', scale=1, color='green', label='U')
    ax[1].quiver(0, 0, C_axial, U + W2_t, angles='xy', scale_units='xy', scale=1, color='blue', label='C₂')
    ax[1].set_title("Triángulo de Velocidades – Salida")
    ax[1].axis('equal')
    ax[1].grid(True)
    ax[1].legend()
    st.pyplot(fig)
    
    st.code("Código simplificado de compresor axial...", language="python")

###############################################################################
# 5. Turbinas Hidráulicas (Turbina Pelton)
###############################################################################
elif opcion == "5. Turbinas Hidráulicas":
    st.header("Turbinas Hidráulicas – Turbina Pelton")
    st.markdown("""
    **Descripción Teórica:**  
    Se calcula el caudal del chorro, la potencia ideal y útil, la altura efectiva y el par sobre el rodete usando:
    
    \[
    A = \pi \left(\frac{d}{2}\right)^2, \quad \dot{m} = \rho A c,\quad P_{ideal} = \tfrac{1}{2}\dot{m} c^2,
    \]
    
    y \( P_{útil} = \eta_h \, P_{ideal} \). El par se obtiene de \( T = \frac{P_{útil}}{U_{tan}} \).
    """)
    
    D_runner = st.slider("Diámetro del rodete (m)", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
    d_jet = st.slider("Diámetro del chorro (m)", min_value=0.05, max_value=0.30, value=0.15, step=0.005)
    c_jet = st.slider("Velocidad del chorro (m/s)", min_value=50, max_value=150, value=100, step=1)
    alpha_deg = st.slider("Ángulo de ataque (°)", min_value=0.0, max_value=45.0, value=15.0, step=0.5)
    eta_h = st.slider("Eficiencia hidráulica (ηₕ)", min_value=0.70, max_value=1.0, value=0.85, step=0.01)
    U_tan = st.slider("Velocidad tangencial (m/s)", min_value=30, max_value=100, value=60, step=1)
    
    g = 9.81
    rho = 1000.0
    A_jet = np.pi * (d_jet / 2) ** 2
    m_dot = rho * A_jet * c_jet
    P_ideal = 0.5 * m_dot * c_jet**2
    P_useful = eta_h * P_ideal
    H_eff = c_jet**2 / (2 * g)
    Torque = P_useful / U_tan
    
    st.write(f"Área del chorro: **{A_jet:.4f} m²**")
    st.write(f"Caudal: **{m_dot:.1f} kg/s**")
    st.write(f"Potencia ideal: **{P_ideal/1e6:.3f} MW**")
    st.write(f"Potencia útil: **{P_useful/1e6:.3f} MW**")
    st.write(f"Altura efectiva (H): **{H_eff:.1f} m**")
    st.write(f"Par sobre el rodete: **{Torque/1e5:.3f} × 10⁵ N·m**")
    
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    labels = ['P₍ideal₎ (MW)', 'P₍útil₎ (MW)']
    values = [P_ideal/1e6, P_useful/1e6]
    bars = ax1.bar(labels, values, color=['skyblue', 'seagreen'])
    ax1.set_title("Potencia del Chorro vs. Potencia Útil")
    ax1.set_ylabel("Potencia (MW)")
    for bar in bars:
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05, f"{bar.get_height():.3f}", ha='center')
    st.pyplot(fig1)
    
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    labels2 = ['Head (m)', 'Par (×10⁵ N·m)']
    values2 = [H_eff, Torque/1e5]
    bars2 = ax2.bar(labels2, values2, color=['salmon', 'mediumpurple'])
    ax2.set_title("Altura Efectiva y Par sobre el Rodete")
    for bar in bars2:
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05, f"{bar.get_height():.2f}", ha='center')
    st.pyplot(fig2)
    
    st.code("Código simplificado de análisis de turbina Pelton...", language="python")

###############################################################################
# 6. Turbinas de Vapor
###############################################################################
elif opcion == "6. Turbinas de Vapor":
    st.header("Turbinas de Vapor")
    st.markdown("""
    **Descripción Teórica:**  
    Se analiza el diseño de una etapa de turbina de vapor de acción.  
    Se determina el número mínimo de etapas para que el salto de entalpía por etapa no supere una velocidad máxima en la tobera, se calcula la velocidad del chorro, el área requerida y se estima el ángulo óptimo de álabes.
    """)
    
    m_dot_tv = st.slider("m_dot (kg/s)", min_value=5.0, max_value=20.0, value=10.0, step=0.5)
    p_in_bar = st.slider("p_in (bar)", min_value=8.0, max_value=12.0, value=10.0, step=0.5)
    T_in_C = st.slider("T_in (°C)", min_value=230, max_value=300, value=250, step=1)
    p_out_bar = st.slider("p_out (bar)", min_value=1.0, max_value=4.0, value=2.0, step=0.5)
    Dh_total = st.slider("Δh_total (kJ/kg)", min_value=300, max_value=700, value=500, step=10)
    a_max = st.slider("a_max (m/s)", min_value=400, max_value=800, value=500, step=10)
    C_axial_tv = st.slider("C_axial requerida (m/s)", min_value=20, max_value=50, value=30, step=1)
    r_base = st.slider("Radio base (m)", min_value=0.5, max_value=1.0, value=0.70, step=0.05)
    alpha1_deg = st.slider("Ángulo mínimo α₁ (°)", min_value=12, max_value=40, value=12, step=1)
    c_p_tv = st.slider("c_p (kJ/kgK)", min_value=1.8, max_value=2.5, value=2.1, step=0.05)
    Kf = st.slider("Kf", min_value=0.90, max_value=1.0, value=0.95, step=0.005)
    Km = st.slider("Km", min_value=0.90, max_value=1.0, value=0.95, step=0.005)
    
    T_in_tv = T_in_C + 273.15
    Δh_etapa_max = (a_max ** 2) / (2 * 1000)
    n_min = int(np.ceil(Dh_total / Δh_etapa_max))
    n = n_min
    Δh_stage = Dh_total / n
    r_stage = (p_in_bar / p_out_bar) ** (1 / n)
    p1_in = p_in_bar
    p1_out = p1_in / r_stage
    ΔT_stage = Δh_stage / c_p_tv
    T1_in = T_in_tv
    T1_out = T1_in - ΔT_stage
    c_nozzle = np.sqrt(2 * Δh_stage * 1000)
    alpha1_rad = np.radians(alpha1_deg)
    C_axial_calc = c_nozzle * np.cos(alpha1_rad)
    rho1 = (p1_in * 1e5) / (461.5 * T1_in)
    A_required = m_dot_tv / (rho1 * C_axial_tv)
    h_blade = A_required / (2 * np.pi * r_base)
    beta_opt_deg = alpha1_deg / (Kf * Km)
    
    st.write(f"Número mínimo de etapas: **{n_min}**  →  Se propone n = **{n}**")
    st.write(f"Salto de entalpía por etapa: **{Δh_stage:.1f} kJ/kg**")
    st.write(f"Relación de presión por etapa: **{r_stage:.3f}**")
    st.write(f"Primera etapa: p_in = **{p1_in:.1f} bar**, p_out = **{p1_out:.2f} bar**")
    st.write(f"T₁_in = **{T1_in:.1f} K**, T₁_out = **{T1_out:.1f} K**")
    st.write(f"c_nozzle: **{c_nozzle:.1f} m/s**, C_axial calculado: **{C_axial_calc:.1f} m/s**")
    st.write(f"ρ₁ = **{rho1:.2f} kg/m³**, Área requerida: **{A_required*1e4:.2f} cm²**")
    st.write(f"Altura de álabes: **{h_blade*1e3:.1f} mm**, Ángulo óptimo: **{beta_opt_deg:.1f}°**")
    
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ['Δh_stage (kJ/kg)', 'c_nozzle (m/s)', 'C_axial_calc (m/s)', 'h_blade (mm)', 'β_opt (°)']
    values = [Δh_stage, c_nozzle, C_axial_calc, h_blade*1e3, beta_opt_deg]
    bars = ax.bar(labels, values, color=['cornflowerblue', 'mediumseagreen', 'gold', 'salmon', 'mediumpurple'])
    ax.set_title("Parámetros Clave de la Primera Etapa")
    ax.set_ylabel("Valor")
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.05, f"{bar.get_height():.1f}", ha='center')
    st.pyplot(fig)
    
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.quiver(0, 0, c_nozzle*np.cos(alpha1_rad), c_nozzle*np.sin(alpha1_rad), 
               angles='xy', scale_units='xy', scale=1, color='blue', label='c_nozzle')
    ax2.quiver(0, 0, c_nozzle*np.cos(alpha1_rad), 0, 
               angles='xy', scale_units='xy', scale=1, color='green', label='Componente axial')
    ax2.quiver(0, 0, 0, c_nozzle*np.sin(alpha1_rad), 
               angles='xy', scale_units='xy', scale=1, color='red', label='Componente tangencial')
    ax2.set_xlim(0, c_nozzle*1.1)
    ax2.set_ylim(0, c_nozzle*1.1)
    ax2.set_xlabel("Axial (m/s)")
    ax2.set_ylabel("Tangencial (m/s)")
    ax2.set_title("Triángulo de Velocidades en la Toberá")
    ax2.grid(True)
    ax2.legend()
    st.pyplot(fig2)
    
    st.code("Código simplificado de análisis de turbina de vapor...", language="python")

###############################################################################
# 7. Turbinas de Gas (Ciclo Brayton)
###############################################################################
elif opcion == "7. Turbinas de Gas":
    st.header("Turbinas de Gas – Ciclo Brayton Simple")
    st.markdown("""
    **Descripción Teórica:**  
    Se analiza un ciclo Brayton simple para una turbina de gas.  
    Se definen cuatro estados del ciclo y se calcula:
    - Trabajo del compresor y de la turbina (con eficiencia mecánica y de combustión)
    - Eficiencia térmica y consumo específico de combustible  
    Además, se genera un diagrama T–s del ciclo.
    """)
    
    T_max = st.slider("T_max (K)", min_value=1200.0, max_value=1350.0, value=1200.0, step=10.0)
    
    # Parámetros fijos
    r_comp = 6.0
    n_c = 1.48
    n_t = 1.26
    eta_m = 0.98
    eta_comb = 0.98
    p_loss_comb = 0.98
    
    T1 = 288.0
    p1 = 101.0
    p2 = r_comp * p1
    T2 = T1 * (p2/p1)**((n_c - 1)/n_c)
    p3 = p_loss_comb * p2
    T3 = T_max
    f = (1.15 * (T3 - T2)) / (eta_comb * 43100.0)
    T4 = T3 * (p1/p3)**((n_t - 1)/n_t)
    T2s = T1 * (p2/p1)**((1.4 - 1)/1.4)
    eta_c = (T2s - T1) / (T2 - T1)
    T4s = T3 * (p1/p3)**((1.33 - 1)/1.33)
    eta_t = (T3 - T4) / (T3 - T4s)
    w_c = 1.005 * (T2 - T1)
    w_t_ideal = 1.15 * (T3 - T4)
    w_t = eta_m * w_t_ideal
    w_net = w_t - w_c
    eta_th = w_net / (f * 43100.0)
    sfc = f / w_net
    R_air = 0.287
    Δs_c = 1.005 * np.log(T2/T1) - R_air * np.log(p2/p1)
    s1 = 0.0
    s2 = s1 + Δs_c
    R_gases = 0.285
    Δs_comb = 1.15 * np.log(T3/T2) - R_gases * np.log(p3/p2)
    s3 = s2 + Δs_comb
    Δs_t = 1.15 * np.log(T4/T3) - R_gases * np.log(p1/p3)
    s4 = s3 + Δs_t
    
    st.write("**Estados del ciclo:**")
    st.write(f"Estado 1: T₁ = {T1:.1f} K, p₁ = {p1:.1f} kPa, s₁ = {s1:.3f} kJ/kgK")
    st.write(f"Estado 2 (Compresor): T₂ = {T2:.1f} K, p₂ = {p2:.1f} kPa, s₂ = {s2:.3f} kJ/kgK")
    st.write(f"Estado 3 (Combustor): T₃ = {T3:.1f} K, p₃ = {p3:.1f} kPa, s₃ = {s3:.3f} kJ/kgK")
    st.write(f"Estado 4 (Turbina): T₄ = {T4:.1f} K, p₄ = {p1:.1f} kPa, s₄ = {s4:.3f} kJ/kgK")
    st.write("")
    st.write("**Balances y eficiencias:**")
    st.write(f"Relación combustible/aire (f): **{f:.4f} kg_fuel/kg_air**")
    st.write(f"Trabajo compresor: **{w_c:.1f} kJ/kg**")
    st.write(f"Trabajo turbina (ideal): **{w_t_ideal:.1f} kJ/kg**, real: **{w_t:.1f} kJ/kg**")
    st.write(f"Trabajo neto: **{w_net:.1f} kJ/kg**")
    st.write(f"Eficiencia térmica: **{eta_th*100:.1f} %**")
    st.write(f"Consumo específico de combustible: **{sfc*1e3:.3f} g/kJ**")
    st.write("")
    st.write("**Eficiencias isentrópicas:**")
    st.write(f"Compresor: T₂s = **{T2s:.1f} K**, η_c = **{eta_c*100:.1f} %**")
    st.write(f"Turbina: T₄s = **{T4s:.1f} K**, η_t = **{eta_t*100:.1f} %**")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([s1, s2, s3, s4, s1], [T1, T2, T3, T4, T1], 'o-', color='darkblue', label='Ciclo Brayton')
    ax.text(s1, T1-20, "1", fontsize=12)
    ax.text(s2, T2+10, "2", fontsize=12)
    ax.text(s3, T3+10, "3", fontsize=12)
    ax.text(s4, T4-20, "4", fontsize=12)
    ax.set_xlabel("Entropía (kJ/kgK)")
    ax.set_ylabel("Temperatura (K)")
    ax.set_title("Diagrama T–s del Ciclo Brayton")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)
    
    st.code("Código simplificado de ciclo Brayton en turbina de gas...", language="python")
