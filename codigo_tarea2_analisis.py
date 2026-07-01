#!/usr/bin/env python3
"""
Análisis de curvas fuerza–desplazamiento: sólidos y moléculas.
Código de procesamiento de datos experimentales.

Este script procesa tres sets de datos experimentales:
  - Ensayo de tracción de una probeta metálica (esfuerzo vs. deformación)
  - Velocidad de onda ultrasónica medida durante el mismo ensayo
  - Estiramiento de una molécula individual con pinzas magnéticas

El objetivo es extraer parámetros mecánicos y microestructurales,
y comparar los regímenes de deformación entre sistemas muy distintos.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d
import os, warnings
warnings.filterwarnings('ignore')

# ── Configuración visual ──
plt.rcParams.update({
    'figure.dpi': 180, 'font.size': 10, 'axes.labelsize': 12,
    'axes.titlesize': 12, 'legend.fontsize': 8, 'figure.figsize': (8, 5),
})
OUT = '/home/claude/figuras'
os.makedirs(OUT, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
#  CARGA Y PREPARACIÓN DE DATOS
# ═══════════════════════════════════════════════════════════════
#
# Acá cargamos los tres archivos CSV. Cada uno tiene dos columnas.
# Es importante ordenar los datos por la variable independiente
# (deformación o extensión) para que los análisis posteriores
# (derivadas, ajustes) funcionen bien.
#

ss = pd.read_csv('/mnt/user-data/uploads/stress_strain_PZ01W.csv')
vt = pd.read_csv('/mnt/user-data/uploads/vt_stress_PZ01W.csv')
dna_raw = pd.read_csv('/mnt/user-data/uploads/dna_force_extension.csv')

# Ordenar esfuerzo-deformación por deformación creciente
idx_ss = np.argsort(ss['strain'].values)
strain = ss['strain'].values[idx_ss]
stress = ss['stress_MPa'].values[idx_ss]

# Ordenar velocidad ultrasónica por esfuerzo creciente
idx_vt = np.argsort(vt['stress_MPa'].values)
stress_vt = vt['stress_MPa'].values[idx_vt]
vT_data = vt['vT_m_s'].values[idx_vt]

print("=" * 72)
print(" PARTE I — PROPIEDADES MECÁNICAS DE LA PROBETA METÁLICA")
print("=" * 72)


# ═══════════════════════════════════════════════════════════════
#  MÓDULO ELÁSTICO (Ley de Hooke: σ = Eε)
# ═══════════════════════════════════════════════════════════════
#
# En la zona elástica, el esfuerzo es proporcional a la deformación.
# La pendiente de esa recta es el módulo de Young E. Para encontrar
# el mejor rango de ajuste, probamos varios límites superiores de
# deformación y elegimos aquel que maximiza el R² (es decir, donde
# la relación es más fielmente lineal). Si el rango es muy amplio,
# incluimos puntos que ya no son elásticos y el R² baja.
#

print("\n─── Módulo elástico ───")

mejor_R2 = 0
resultados_E = []

for eps_max in [0.0003, 0.00035, 0.0004, 0.00045, 0.0005, 0.0006, 0.0007]:
    mascara = (strain > 3e-5) & (strain < eps_max)
    if mascara.sum() < 15:
        continue
    coef = np.polyfit(strain[mascara], stress[mascara], 1)
    prediccion = np.polyval(coef, strain[mascara])
    ss_res = np.sum((stress[mascara] - prediccion)**2)
    ss_tot = np.sum((stress[mascara] - np.mean(stress[mascara]))**2)
    r2 = 1 - ss_res / ss_tot
    resultados_E.append((eps_max, coef[0], coef[1], r2))

    if r2 > mejor_R2:
        mejor_R2 = r2
        E_Young = coef[0]        # pendiente = módulo de Young en MPa
        intercepto = coef[1]
        rango_elegido = eps_max
        coef_elegido = coef

print("  Exploración de rangos para el ajuste lineal:")
for eps_max, E_gpa, _, r2 in resultados_E:
    marca = " ◄" if eps_max == rango_elegido else ""
    print(f"    ε < {eps_max:.5f}: E = {E_gpa/1e3:.1f} GPa, R² = {r2:.6f}{marca}")

print(f"\n  ► E = {E_Young/1e3:.2f} GPa   (R² = {mejor_R2:.6f})")
print(f"    Rango: ε ∈ [3×10⁻⁵, {rango_elegido}]")

# ── Figura: curva σ-ε con ajuste elástico ──
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(strain*1e3, stress, color='#4682B4', lw=0.5, alpha=0.6, label='Datos probeta PZ01W')
eps_lin = np.linspace(0, rango_elegido*1.5, 200)
ax.plot(eps_lin*1e3, np.polyval(coef_elegido, eps_lin), 'r--', lw=2,
        label=f'$\\sigma = E\\varepsilon$;  E = {E_Young/1e3:.1f} GPa  (R² = {mejor_R2:.4f})')
ax.set_xlabel('Deformación $\\varepsilon$ (×10⁻³)')
ax.set_ylabel('Esfuerzo $\\sigma$ (MPa)')
ax.legend(loc='lower right')
ax.set_xlim(-0.1, 11); ax.set_ylim(-1, 48); ax.grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig(f'{OUT}/01_curva_esfuerzo_deformacion.png', dpi=200); plt.close()


# ═══════════════════════════════════════════════════════════════
#  ESFUERZO DE FLUENCIA
# ═══════════════════════════════════════════════════════════════
#
# El esfuerzo de fluencia σ_Y marca la transición entre la zona
# elástica (reversible) y la plástica (irreversible). Hay dos
# formas clásicas de determinarlo:
#
# 1) Método offset 0.2%: se traza una recta paralela a la zona
#    elástica pero desplazada 0.2% en deformación. Donde esa recta
#    corta la curva, ése es σ_Y. Es un criterio convencional y
#    práctico, pero algo arbitrario.
#
# 2) Método de Christensen: busca el punto donde la tasa de cambio
#    del módulo tangente es máxima, es decir, donde la curva "se
#    dobla" más rápido. Matemáticamente, es donde la tercera
#    derivada d³σ/dε³ cruza por cero. Es más sensible y detecta
#    la no-linealidad antes que el offset.
#
# Para calcular derivadas numéricas en datos ruidosos usamos el
# filtro de Savitzky-Golay, que suaviza sin distorsionar mucho
# las transiciones.
#

print("\n─── Esfuerzo de fluencia ───")

# Suavizar la curva para derivadas estables.
# Necesitamos una ventana amplia porque la tercera derivada es muy
# sensible al ruido. Usamos Savitzky-Golay de orden 3 para preservar
# la forma de la transición, con ventana grande para estabilidad.
ventana_SG = 101
stress_suave = savgol_filter(stress, ventana_SG, 3)

# Calcular las tres primeras derivadas
d1 = np.gradient(stress_suave, strain)     # dσ/dε
d2 = np.gradient(d1, strain)               # d²σ/dε²
d3 = np.gradient(d2, strain)               # d³σ/dε³

# ── Christensen: buscar d³σ/dε³ = 0 en la zona de transición ──
# La clave es buscar en un rango de ε que esté DENTRO de la zona de
# transición real. Demasiado a la izquierda captura ruido; demasiado
# a la derecha ya es plástico. Acotamos entre 0.0002 y 0.002.
zona_busqueda = (strain > 0.0002) & (strain < 0.003)
d3_zona = d3[zona_busqueda]
strain_zona = strain[zona_busqueda]
sig_zona = stress_suave[zona_busqueda]

# Suavizar d3 adicionalmente para evitar cruces espurios
d3_zona_sm = savgol_filter(d3_zona, min(51, len(d3_zona)//3*2+1), 2)

# Buscar cruces por cero de d³σ/dε³ suavizado
cruces = np.where(np.diff(np.sign(d3_zona_sm)))[0]

# Filtrar: tomar el cruce donde σ sea razonable (> 10 MPa, dentro
# de la zona de transición real, no ruido temprano)
idx_chr = None
for c in cruces:
    if sig_zona[c] > 8:  # σ mínimo razonable
        idx_chr = c
        break

if idx_chr is not None:
    eps_Y_chr = strain_zona[idx_chr]
    sig_Y_chr = sig_zona[idx_chr]
else:
    # Fallback: usar el máximo de |d²σ/dε²| (curvatura máxima)
    zona2 = (strain > 0.0002) & (strain < 0.002)
    idx_chr = np.argmax(np.abs(d2[zona2]))
    eps_Y_chr = strain[zona2][idx_chr]
    sig_Y_chr = np.interp(eps_Y_chr, strain, stress_suave)

# ── Offset 0.2% ──
# La recta offset es: σ = E·(ε − 0.002)
# Buscamos dónde cruza la curva experimental
offset_eps = 0.002
linea_offset = E_Young * (strain - offset_eps)
diferencia = stress_suave - linea_offset

# Buscar el cruce en la zona donde ε > offset
zona_off = strain > offset_eps + 0.0002
cruces_off = np.where(np.diff(np.sign(diferencia[zona_off])))[0]
if len(cruces_off) > 0:
    idx_02 = cruces_off[0] + np.sum(~zona_off)
    eps_Y_02 = strain[idx_02]
    sig_Y_02 = stress_suave[idx_02]
else:
    eps_Y_02 = 0.003
    sig_Y_02 = np.interp(eps_Y_02, strain, stress_suave)

print(f"  Christensen (d³σ/dε³ = 0):  σ_Y = {sig_Y_chr:.2f} MPa  (ε_Y = {eps_Y_chr:.6f})")
print(f"  Offset 0.2%:                σ_Y = {sig_Y_02:.2f} MPa  (ε_Y = {eps_Y_02:.6f})")
print(f"  Diferencia: {abs(sig_Y_02 - sig_Y_chr):.1f} MPa")

# ── Figura: fluencia ──
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(strain*1e3, stress, color='#4682B4', lw=0.4, alpha=0.4, label='Datos')
ax.plot(strain*1e3, stress_suave, 'k-', lw=1, alpha=0.8, label='Suavizado (Savitzky-Golay)')
eps_off_plot = np.linspace(offset_eps, 0.006, 100)
ax.plot(eps_off_plot*1e3, E_Young*(eps_off_plot - offset_eps), 'g--', lw=1.5, label='Recta offset 0.2%')
ax.plot(eps_Y_chr*1e3, sig_Y_chr, 'ro', ms=10, zorder=5,
        label=f'Christensen: $\\sigma_Y$ = {sig_Y_chr:.1f} MPa')
ax.plot(eps_Y_02*1e3, sig_Y_02, 'g^', ms=10, zorder=5,
        label=f'Offset 0.2%: $\\sigma_Y$ = {sig_Y_02:.1f} MPa')
ax.set_xlabel('$\\varepsilon$ (×10⁻³)'); ax.set_ylabel('$\\sigma$ (MPa)')
ax.legend(loc='lower right', fontsize=8)
ax.set_xlim(-0.1, 6); ax.set_ylim(-1, 45); ax.grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig(f'{OUT}/02_esfuerzo_fluencia.png', dpi=200); plt.close()


# ═══════════════════════════════════════════════════════════════
#  ETAPAS DE ENDURECIMIENTO (análisis tipo Crussard–Jaoul)
# ═══════════════════════════════════════════════════════════════
#
# Si graficamos ln(dσ/dε) vs. ln(σ) en la zona plástica, los
# distintos mecanismos de endurecimiento aparecen como tramos
# con pendiente diferente. Cada cambio de pendiente corresponde
# a una "etapa" asociada a un mecanismo microestructural distinto
# (por ejemplo, deslizamiento de dislocaciones en planos fáciles,
# deslizamiento múltiple, maclado, etc.).
#
# La derivada numérica dσ/dε es inherentemente ruidosa porque
# amplifica el ruido punto a punto. Para mitigarlo sin distorsionar
# las transiciones, suavizamos con Savitzky-Golay (que ajusta un
# polinomio local, preservando mejor los picos que un promedio
# móvil simple).
#

print("\n─── Etapas de endurecimiento (Crussard–Jaoul) ───")

# Seleccionar zona plástica
mascara_pl = (stress_suave > sig_Y_chr + 0.5) & (strain > eps_Y_chr)
strain_pl = strain[mascara_pl]
stress_pl = stress_suave[mascara_pl]

# Derivada dσ/dε suavizada
dsde_pl = np.gradient(stress_pl, strain_pl)
ventana_dsde = min(151, len(dsde_pl)//4*2 + 1)
dsde_suave = savgol_filter(dsde_pl, ventana_dsde, 2)

# Solo valores positivos (ln no acepta negativos)
positivos = dsde_suave > 0
ln_dsde = np.log(dsde_suave[positivos])
ln_sigma = np.log(stress_pl[positivos])
strain_cj = strain_pl[positivos]

# Suavizar el resultado para identificar tendencias
ventana_cj = min(201, len(ln_dsde)//5*2 + 1)
ln_dsde_suave = savgol_filter(ln_dsde, ventana_cj, 2)

# ── Identificar puntos de transición automáticamente ──
# Calculamos la derivada de ln(dσ/dε) respecto a ln(σ) y buscamos
# dónde cambia significativamente (cambios de pendiente).
pendiente_local = np.gradient(ln_dsde_suave, ln_sigma)
pend_suave = savgol_filter(pendiente_local, min(101, len(pendiente_local)//5*2+1), 2)

# Buscar cambios de signo en la segunda derivada de la pendiente
d_pend = np.gradient(pend_suave, ln_sigma)
d_pend_sm = savgol_filter(d_pend, min(101, len(d_pend)//5*2+1), 2)

# Buscar extremos de la pendiente (donde d(pend)/d(ln σ) = 0)
cruces_pend = np.where(np.diff(np.sign(d_pend_sm)))[0]

# Filtrar: solo transiciones que estén bien separadas y sean significativas
n_total = len(ln_sigma)
trans_indices = []
for c in cruces_pend:
    if c > n_total*0.15 and c < n_total*0.85:
        if len(trans_indices) == 0 or (c - trans_indices[-1]) > n_total*0.25:
            trans_indices.append(c)

# Limitar a máximo 3 transiciones (2-4 etapas es lo esperable)
if len(trans_indices) > 3:
    # Quedarnos con las más separadas
    trans_indices = trans_indices[:3]

if len(trans_indices) < 1:
    # Fallback: dividir en tercios
    trans_indices = [n_total//3, 2*n_total//3]

# Definir segmentos entre transiciones
limites = [0] + trans_indices + [n_total]
n_etapas = len(limites) - 1

print(f"  {n_etapas} etapas identificadas:")
pendientes_etapa = []
colores_etapa = ['#2ca02c', '#ff7f0e', '#9467bd', '#d62728']

for k in range(n_etapas):
    i0, i1 = limites[k], limites[k+1]
    x_seg = ln_sigma[i0:i1]
    y_seg = ln_dsde_suave[i0:i1]
    if len(x_seg) > 5:
        coef_seg = np.polyfit(x_seg, y_seg, 1)
        pendientes_etapa.append((coef_seg, i0, i1))
        eps_i = strain_cj[i0]; eps_f = strain_cj[min(i1-1, len(strain_cj)-1)]
        print(f"    Etapa {k+1}: n = {coef_seg[0]:.2f}  "
              f"(ε ∈ [{eps_i:.5f}, {eps_f:.5f}])")

# Deformaciones de transición
deform_trans = []
for idx_t in trans_indices:
    if idx_t < len(strain_cj):
        eps_t = strain_cj[idx_t]
        sig_t = np.interp(eps_t, strain, stress_suave)
        deform_trans.append((eps_t, sig_t))
        print(f"  Transición en: ε = {eps_t:.5f}, σ ≈ {sig_t:.1f} MPa")

# ── Figura: Crussard-Jaoul ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

ax1.plot(ln_sigma, ln_dsde, color='#4682B4', lw=0.3, alpha=0.4, label='Datos')
ax1.plot(ln_sigma, ln_dsde_suave, 'k-', lw=1.2, label='Suavizado')
for k, (coef_seg, i0, i1) in enumerate(pendientes_etapa):
    x = ln_sigma[i0:i1]
    ax1.plot(x, np.polyval(coef_seg, x), '-', color=colores_etapa[k % len(colores_etapa)],
             lw=2.5, label=f'Etapa {k+1}: n = {coef_seg[0]:.2f}')
for eps_t, sig_t in deform_trans:
    ax1.axvline(np.log(sig_t), color='gray', ls=':', alpha=0.6)
ax1.set_xlabel('ln $\\sigma$'); ax1.set_ylabel('ln(d$\\sigma$/d$\\varepsilon$)')
ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

ax2.plot(strain_cj*1e3, ln_dsde, color='#4682B4', lw=0.3, alpha=0.4)
ax2.plot(strain_cj*1e3, ln_dsde_suave, 'k-', lw=1.2)
for eps_t, sig_t in deform_trans:
    ax2.axvline(eps_t*1e3, color='gray', ls='--', alpha=0.7,
                label=f'ε = {eps_t*1e3:.1f}×10⁻³')
ax2.set_xlabel('$\\varepsilon$ (×10⁻³)'); ax2.set_ylabel('ln(d$\\sigma$/d$\\varepsilon$)')
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

fig.tight_layout(); fig.savefig(f'{OUT}/03_crussard_jaoul.png', dpi=200); plt.close()


# ═══════════════════════════════════════════════════════════════
#  PARTE II — PROPAGACIÓN DE ONDAS Y DENSIDAD DE DISLOCACIONES
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print(" PARTE II — PROPAGACIÓN DE ONDAS Y MICROESTRUCTURA")
print("=" * 72)

# ═══════════════════════════════════════════════════════════════
#  VELOCIDAD DE REFERENCIA Y CAMBIO RELATIVO
# ═══════════════════════════════════════════════════════════════
#
# En la zona elástica las dislocaciones no se multiplican, así
# que la velocidad de onda transversal vT se mantiene constante.
# Promediamos esa zona para obtener la velocidad de referencia v⁰_T.
# Luego calculamos el cambio relativo ΔvT/v⁰_T en función del
# esfuerzo. Cuando las dislocaciones proliferan (zona plástica),
# la onda interactúa con ellas y su velocidad baja.
#

print("\n─── Velocidad de referencia y decremento ───")

mascara_elastica = stress_vt < sig_Y_chr * 0.7
vT_ref = np.mean(vT_data[mascara_elastica])
vT_ref_std = np.std(vT_data[mascara_elastica])
n_puntos_ref = mascara_elastica.sum()

print(f"  v⁰_T = {vT_ref:.2f} ± {vT_ref_std:.2f} m/s  ({n_puntos_ref} puntos)")

# Cambio relativo
delta_vT_rel = (vT_data - vT_ref) / vT_ref
ventana_vt = min(51, len(delta_vT_rel)//5*2 + 1)
delta_vT_suave = savgol_filter(delta_vT_rel, ventana_vt, 2)

# Buscar dónde el decremento se vuelve apreciable
umbral = -2 * vT_ref_std / vT_ref
mascara_dec = delta_vT_suave < umbral
if np.any(mascara_dec):
    idx_onset = np.where(mascara_dec)[0][0]
    sigma_onset = stress_vt[idx_onset]
    print(f"  Onset del decremento: σ ≈ {sigma_onset:.1f} MPa")
    print(f"  (Consistente con σ_Y = {sig_Y_chr:.1f} MPa)")

# ── Figura: vT y ΔvT/v⁰ ──
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
m_pos = stress_vt > 0.01
a1.semilogx(stress_vt[m_pos], vT_data[m_pos], color='#4682B4', lw=0.4, alpha=0.5)
a1.axhline(vT_ref, color='r', ls='--', lw=1.2, label=f'$v_T^0$ = {vT_ref:.1f} m/s')
a1.axvline(sig_Y_chr, color='gray', ls=':', alpha=0.7, label=f'$\\sigma_Y$')
a1.set_xlabel('$\\sigma$ (MPa)'); a1.set_ylabel('$v_T$ (m/s)')
a1.legend(); a1.grid(True, alpha=0.3)

a2.plot(stress_vt, delta_vT_rel*100, color='#4682B4', lw=0.3, alpha=0.3)
a2.plot(stress_vt, delta_vT_suave*100, 'r-', lw=1.5)
a2.axhline(0, color='k', lw=0.5); a2.axvline(sig_Y_chr, color='gray', ls=':', alpha=0.7)
a2.set_xlabel('$\\sigma$ (MPa)'); a2.set_ylabel('$\\Delta v_T / v_T^0$ (%)')
a2.grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig(f'{OUT}/04_velocidad_transversal.png', dpi=200); plt.close()


# ═══════════════════════════════════════════════════════════════
#  DENSIDAD DE DISLOCACIONES (fórmula de Maurel et al.)
# ═══════════════════════════════════════════════════════════════
#
# La teoría de Maurel et al. relaciona el cambio en la velocidad
# de onda con el cambio en la densidad de dislocaciones Λ.
# La fórmula para ondas transversales es:
#
#   ΔΛ_T = −(5π⁴)/(8L²) · (ΔvT/v⁰_T)
#
# donde L es la longitud promedio de los segmentos de dislocación.
# Como L no se mide directamente aquí, calculamos ΔΛ para tres
# valores representativos: L = 80, 100 y 120 nm.
# La densidad total es Λ = Λ₀ + ΔΛ, con Λ₀ = 10¹² m⁻².
#

print("\n─── Densidad de dislocaciones ───")

longitudes_L = [80e-9, 100e-9, 120e-9]
etiquetas_L = ['80', '100', '120']
Lambda_0 = 1e12  # m⁻² (densidad inicial típica)

fig, ax = plt.subplots(figsize=(9, 5.5))
Lambda_resultados = {}
mascara_plastica = stress_vt > sig_Y_chr * 0.5

for L_disl, etiq in zip(longitudes_L, etiquetas_L):
    prefactor = 5 * np.pi**4 / (8 * L_disl**2)
    delta_Lambda = -prefactor * delta_vT_suave
    Lambda_total = Lambda_0 + delta_Lambda
    Lambda_resultados[etiq] = Lambda_total

    ax.semilogy(stress_vt[mascara_plastica], Lambda_total[mascara_plastica],
                lw=1.5, label=f'L = {etiq} nm')
    print(f"  L = {etiq} nm: Λ_final = {Lambda_total[mascara_plastica][-1]:.2e} m⁻²")

ax.axvline(sig_Y_chr, color='gray', ls=':', alpha=0.7, label='$\\sigma_Y$')
ax.set_xlabel('$\\sigma$ (MPa)'); ax.set_ylabel('$\\Lambda$ (m⁻²)')
ax.legend(); ax.grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig(f'{OUT}/05_densidad_dislocaciones.png', dpi=200); plt.close()


# ═══════════════════════════════════════════════════════════════
#  REGLA DE TAYLOR (σ = σ₀ + α·M·b·G·√Λ)
# ═══════════════════════════════════════════════════════════════
#
# La regla de Taylor conecta el esfuerzo macroscópico con la
# densidad de dislocaciones a través de la raíz cuadrada de Λ.
# Los parámetros del material son: b (vector de Burgers, tamaño
# del "paso" de la dislocación), G (módulo de corte), M (factor
# de Taylor, que promedia la orientación de los granos).
#
# Si el ajuste σ vs √Λ es lineal, confirma que el endurecimiento
# obedece la regla de Taylor. Hacemos el ajuste con las constantes
# del acero 304L y del aluminio para comparar α y σ₀.
#
# Además, variamos L para mostrar que α/L es aproximadamente
# constante — es decir, α escala linealmente con L.
#

print("\n─── Regla de Taylor ───")

materiales = {
    'Acero 304L': {'b': 0.2542e-9, 'G': 86e9, 'M': 3},
    'Aluminio':   {'b': 0.286e-9,  'G': 26e9, 'M': 3},
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
L_referencia = 100e-9
pref_ref = 5 * np.pi**4 / (8 * L_referencia**2)
dLam_ref = -pref_ref * delta_vT_suave
Lambda_ref = Lambda_0 + dLam_ref

for idx_mat, (nombre_mat, props) in enumerate(materiales.items()):
    b, G, M = props['b'], props['G'], props['M']

    mascara_taylor = (stress_vt > sig_Y_chr) & (Lambda_ref > 0)
    sigma_Pa = stress_vt[mascara_taylor] * 1e6
    raiz_Lambda = np.sqrt(Lambda_ref[mascara_taylor])

    def modelo_taylor(sqrtL, sigma0, alpha):
        return sigma0 + alpha * M * b * G * sqrtL

    popt, pcov = curve_fit(modelo_taylor, raiz_Lambda, sigma_Pa, p0=[1e6, 0.3])
    perr = np.sqrt(np.diag(pcov))

    sigma0_ajuste = popt[0] / 1e6  # Pa → MPa
    alpha_ajuste = popt[1]

    print(f"\n  {nombre_mat} (L = 100 nm):")
    print(f"    σ₀ = {sigma0_ajuste:.2f} ± {perr[0]/1e6:.2f} MPa")
    print(f"    α  = {alpha_ajuste:.4f} ± {perr[1]:.4f}")
    print(f"    α/L = {alpha_ajuste/L_referencia:.2e} m⁻¹")

    # Gráfico σ vs σ_Taylor
    sigma_pred = modelo_taylor(raiz_Lambda, *popt)
    ax = axes[idx_mat]
    ax.plot(sigma_pred/1e6, sigma_Pa/1e6, '.', ms=1, alpha=0.3, color='#4682B4')

    # Promedios en bins
    bins = np.linspace(raiz_Lambda.min(), raiz_Lambda.max(), 21)
    xb, yb = [], []
    for j in range(len(bins)-1):
        m = (raiz_Lambda >= bins[j]) & (raiz_Lambda < bins[j+1])
        if m.sum() > 5:
            xb.append(np.mean(modelo_taylor(raiz_Lambda[m], *popt)))
            yb.append(np.mean(sigma_Pa[m]))
    xb, yb = np.array(xb)/1e6, np.array(yb)/1e6
    ax.plot(xb, yb, 'ko', ms=5, zorder=5, label='Promedios (20 bins)')

    lim_min = min(sigma_pred.min(), sigma_Pa.min())/1e6
    lim_max = max(sigma_pred.max(), sigma_Pa.max())/1e6
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r-', lw=1.5, label='$\\sigma = \\sigma_{Taylor}$')
    ax.set_xlabel('$\\sigma_{Taylor}$ (MPa)'); ax.set_ylabel('$\\sigma$ medido (MPa)')
    ax.set_title(f'{nombre_mat}\n$\\alpha$ = {alpha_ajuste:.4f}, $\\sigma_0$ = {sigma0_ajuste:.1f} MPa')
    ax.legend(); ax.grid(True, alpha=0.3)

fig.tight_layout(); fig.savefig(f'{OUT}/06_regla_taylor.png', dpi=200); plt.close()

# ── Dependencia α(L) ──
print(f"\n  Dependencia de α con L [Acero 304L]:")
alpha_vs_L = []
for L_disl, etiq in zip(longitudes_L, etiquetas_L):
    pref = 5*np.pi**4 / (8*L_disl**2)
    dL = -pref * delta_vT_suave
    Lt = Lambda_0 + dL
    mt = (stress_vt > sig_Y_chr) & (Lt > 0)
    b304, G304 = 0.2542e-9, 86e9

    def tm(x, s0, a):
        return s0 + a * 3 * b304 * G304 * x

    po, _ = curve_fit(tm, np.sqrt(Lt[mt]), stress_vt[mt]*1e6, p0=[1e6, 0.3])
    alpha_vs_L.append((L_disl*1e9, po[1], po[1]/L_disl))
    print(f"    L = {etiq} nm: α = {po[1]:.4f},  α/L = {po[1]/L_disl:.2e} m⁻¹")

print(f"  → α/L es aproximadamente constante (≈ {np.mean([x[2] for x in alpha_vs_L]):.2e} m⁻¹)")


# ═══════════════════════════════════════════════════════════════
#  PARTE III — MOLÉCULA INDIVIDUAL (ADN)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print(" PARTE III — ESTIRAMIENTO DE MOLÉCULA INDIVIDUAL")
print("=" * 72)


# ═══════════════════════════════════════════════════════════════
#  LIMPIEZA DE DATOS
# ═══════════════════════════════════════════════════════════════
#
# Los datos crudos vienen con problemas que hay que resolver antes
# de analizar:
#   - Filas con valores infinitos (inf, -inf) que deben descartarse
#   - La extensión tiene un origen arbitrario (puede ser negativa)
#   - Hay una fuerza de baseline constante cuando la molécula no
#     está siendo estirada
#
# Primero limpiamos los inf/nan, luego identificamos la fuerza
# baseline (la mediana de la fuerza a extensión negativa, donde
# no hay estiramiento real), y finalmente definimos el cero de
# extensión como el punto donde la fuerza empieza a subir
# apreciablemente sobre el ruido.
#

print("\n─── Limpieza de datos ───")

# Eliminar filas con inf
dna_limpio = dna_raw.replace([np.inf, -np.inf], np.nan).dropna()
n_inf = len(dna_raw) - len(dna_limpio)
print(f"  Filas originales: {len(dna_raw)}")
print(f"  Filas inf removidas: {n_inf}")
print(f"  Filas finales: {len(dna_limpio)}")

# Ordenar por extensión
idx_dna = np.argsort(dna_limpio['extension_nm'].values)
ext_raw = dna_limpio['extension_nm'].values[idx_dna]
fuerza_raw = dna_limpio['force_pN'].values[idx_dna]

# Baseline: fuerza a extensión negativa (molécula no estirada)
F_baseline = np.median(fuerza_raw[ext_raw < -5])
F_ruido = np.std(fuerza_raw[ext_raw < -5])
print(f"  F_baseline = {F_baseline:.3f} ± {F_ruido:.3f} pN")

# Cero de extensión: donde la fuerza empieza a subir sobre el ruido
f_suavizada = uniform_filter1d(fuerza_raw, 30)
umbral_F = F_baseline + 5 * F_ruido
idx_inicio = np.argmax(f_suavizada > umbral_F)
x0 = ext_raw[idx_inicio]
print(f"  x₀ (onset del estiramiento): {x0:.1f} nm")

# Corregir extensión y fuerza
extension = ext_raw - x0
fuerza = fuerza_raw - F_baseline


# ═══════════════════════════════════════════════════════════════
#  IDENTIFICACIÓN DE REGÍMENES
# ═══════════════════════════════════════════════════════════════
#
# La curva fuerza-extensión de una molécula de ADN tiene tres
# regímenes bien diferenciados:
#
# 1) Entrópico (extensiones bajas): la fuerza restauradora viene
#    de la reducción de entropía al estirar la cadena. La molécula
#    prefiere estar enrollada porque hay más configuraciones
#    posibles. Es el régimen de la cadena tipo gusano (WLC).
#
# 2) Entálpico (extensiones intermedias): ya no queda entropía
#    conformacional que reducir, y la fuerza empieza a estirar
#    los enlaces covalentes del esqueleto de la molécula.
#
# 3) Overstretching / plateau (extensiones altas): la fuerza se
#    mantiene casi constante mientras la molécula sufre una
#    transición estructural (de la forma B a la forma S del ADN).
#    Es análogo a un cambio de fase.
#

print("\n─── Identificación de regímenes ───")

# Detectar el plateau: zona de fuerza alta y casi constante
mascara_alto = (extension > 150) & (fuerza > 40)
F_plateau = np.median(fuerza_raw[ext_raw - x0 > 150])
F_plateau_std = np.std(fuerza_raw[ext_raw - x0 > 150])
print(f"  F_plateau = {F_plateau:.2f} ± {F_plateau_std:.2f} pN (fuerza absoluta)")

F_plat_corr = F_plateau - F_baseline
print(f"  F_plateau (corregida) = {F_plat_corr:.2f} pN")

# ── Figura: regímenes ──
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(extension, fuerza_raw - F_baseline + F_baseline,  # fuerza original
        color='#4682B4', lw=0.3, alpha=0.3)

# Marcas de regímenes
ax.axhspan(F_baseline-0.5, 10, alpha=0.07, color='green')
ax.text(380, 5, 'Entrópico (WLC)', fontsize=9, color='green', alpha=0.9)
ax.axhspan(10, 44, alpha=0.05, color='orange')
ax.text(380, 25, 'Entálpico', fontsize=9, color='orange', alpha=0.9)
ax.axhspan(44, 62, alpha=0.07, color='red')
ax.text(380, 52, 'Overstretching', fontsize=9, color='red', alpha=0.9)
ax.axhline(F_plateau, color='red', ls='--', lw=1.5, alpha=0.7,
           label=f'$F_{{plateau}}$ = {F_plateau:.1f} ± {F_plateau_std:.1f} pN')

ax.set_xlabel('Extensión $x$ (nm)'); ax.set_ylabel('Fuerza $F$ (pN)')
ax.legend(loc='center right')
ax.set_xlim(-20, 650); ax.set_ylim(-1, 62); ax.grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig(f'{OUT}/07_regimenes_ADN.png', dpi=200); plt.close()


# ═══════════════════════════════════════════════════════════════
#  AJUSTE WLC (modelo de Marko y Siggia)
# ═══════════════════════════════════════════════════════════════
#
# El modelo de cadena tipo gusano (Worm-Like Chain) describe la
# elasticidad entrópica de polímeros semiflexibles. La fórmula
# de interpolación de Marko y Siggia es:
#
#   F·Lp/(kBT) = 1/[4(1−x/L₀)²] − 1/4 + x/L₀
#
# donde:
#   Lp = longitud de persistencia (rigidez de la cadena; para
#        dsDNA típica es ~50 nm)
#   L₀ = longitud de contorno (largo total de la molécula si la
#        estiramos completamente)
#   kBT = energía térmica (4.114 pN·nm a 298 K)
#
# Ajustamos con los datos del régimen previo al plateau.
# Usamos datos binneados (medianas en ventanas de 2 nm) para
# reducir el ruido y dar peso parejo a cada zona.
#

print("\n─── Ajuste WLC (Marko–Siggia) ───")

kBT = 4.114  # pN·nm a 298 K

def modelo_WLC(x, Lp, L0):
    """Fuerza predicha por la interpolación de Marko-Siggia."""
    t = np.clip(x / L0, 0, 0.9999)
    return (kBT / Lp) * (1.0/(4.0*(1-t)**2) - 0.25 + t)

# Binnear los datos para un ajuste más limpio
bordes_bin = np.arange(0, 110, 2)
x_bin, f_bin, f_bin_std = [], [], []
for i in range(len(bordes_bin)-1):
    m = (extension >= bordes_bin[i]) & (extension < bordes_bin[i+1]) & (fuerza > 0)
    if m.sum() > 3:
        x_bin.append(np.median(extension[m]))
        f_bin.append(np.median(fuerza[m]))
        f_bin_std.append(np.std(fuerza[m]))

x_bin = np.array(x_bin)
f_bin = np.array(f_bin)
f_bin_std = np.array(f_bin_std)

# Ajustar WLC
mascara_ajuste = x_bin > 2
popt_wlc, pcov_wlc = curve_fit(
    modelo_WLC, x_bin[mascara_ajuste], f_bin[mascara_ajuste],
    p0=[0.5, 200],
    bounds=([0.01, 50], [200, 1000]),
    maxfev=50000
)
perr_wlc = np.sqrt(np.diag(pcov_wlc))

Lp_ajuste = popt_wlc[0]
L0_ajuste = popt_wlc[1]

# Bondad del ajuste
pred_wlc = modelo_WLC(x_bin[mascara_ajuste], *popt_wlc)
residuos = f_bin[mascara_ajuste] - pred_wlc
rmse = np.sqrt(np.mean(residuos**2))
ss_r = np.sum(residuos**2)
ss_t = np.sum((f_bin[mascara_ajuste] - f_bin[mascara_ajuste].mean())**2)
R2_wlc = 1 - ss_r / ss_t

print(f"  Lp = {Lp_ajuste:.3f} ± {perr_wlc[0]:.3f} nm")
print(f"  L₀ = {L0_ajuste:.1f} ± {perr_wlc[1]:.1f} nm")
print(f"  R² = {R2_wlc:.6f}")
print(f"  RMSE = {rmse:.2f} pN")
print(f"  N_bp ≈ {L0_ajuste/0.34:.0f} pb (espaciamiento B-DNA: 0.34 nm/pb)")

# ── Sensibilidad a x₀ ──
print(f"\n  Sensibilidad al cero de extensión x₀:")
for dx in [-10, -5, 0, 5, 10]:
    ext_test = ext_raw - (x0 + dx)
    fuerza_test = fuerza_raw - F_baseline
    bins_t = np.arange(0, 110, 2)
    xb2, fb2 = [], []
    for i in range(len(bins_t)-1):
        m = (ext_test >= bins_t[i]) & (ext_test < bins_t[i+1]) & (fuerza_test > 0)
        if m.sum() > 3:
            xb2.append(np.median(ext_test[m]))
            fb2.append(np.median(fuerza_test[m]))
    xb2, fb2 = np.array(xb2), np.array(fb2)
    mf = xb2 > 2
    if mf.sum() > 5:
        try:
            p2, _ = curve_fit(modelo_WLC, xb2[mf], fb2[mf], p0=[0.5, 200],
                              bounds=([0.01, 50], [200, 1000]), maxfev=50000)
            print(f"    Δx₀ = {dx:+3d} nm: Lp = {p2[0]:.3f} nm, L₀ = {p2[1]:.0f} nm")
        except:
            print(f"    Δx₀ = {dx:+3d} nm: ajuste no convergió")

# ── Figura: ajuste WLC ──
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5))

a1.plot(extension, fuerza, color='#4682B4', lw=0.3, alpha=0.2, label='Datos (corregidos)')
a1.errorbar(x_bin, f_bin, yerr=f_bin_std, fmt='ko', ms=3, lw=0.5, alpha=0.7,
            label='Medianas en bins de 2 nm')
x_modelo = np.linspace(1, L0_ajuste*0.98, 500)
a1.plot(x_modelo, modelo_WLC(x_modelo, *popt_wlc), 'r-', lw=2.5,
        label=f'WLC: $L_p$ = {Lp_ajuste:.2f} nm, $L_0$ = {L0_ajuste:.0f} nm')
a1.axhline(F_plat_corr, color='gray', ls='--', alpha=0.5)
a1.set_xlabel('$x$ (nm)'); a1.set_ylabel('$F$ (pN)')
a1.legend(fontsize=7); a1.set_xlim(-10, 650); a1.set_ylim(-1, 60); a1.grid(True, alpha=0.3)

a2.bar(x_bin[mascara_ajuste], residuos, width=1.5, color='#4682B4', alpha=0.7)
a2.axhline(0, color='k', lw=0.5)
a2.set_xlabel('$x$ (nm)'); a2.set_ylabel('Residuo (pN)')
a2.set_title(f'Residuos del ajuste (RMSE = {rmse:.2f} pN)')
a2.grid(True, alpha=0.3)

fig.tight_layout(); fig.savefig(f'{OUT}/08_ajuste_WLC.png', dpi=200); plt.close()


# ═══════════════════════════════════════════════════════════════
#  RESUMEN FINAL
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print(" RESUMEN DE RESULTADOS NUMÉRICOS")
print("=" * 72)
print(f"""
  ┌─────────────────────────────────────────────────────┐
  │ PROBETA METÁLICA                                    │
  │  E = {E_Young/1e3:.2f} GPa  (R² = {mejor_R2:.4f})                │
  │  σ_Y (Christensen) = {sig_Y_chr:.2f} MPa                  │
  │  σ_Y (offset 0.2%) = {sig_Y_02:.2f} MPa                  │
  │  Etapas C–J: {n_etapas} identificadas                       │
  ├─────────────────────────────────────────────────────┤
  │ ULTRASONIDO Y DISLOCACIONES                         │
  │  v⁰_T = {vT_ref:.2f} ± {vT_ref_std:.2f} m/s                     │
  │  Λ_final (L=100nm) = {Lambda_resultados['100'][mascara_plastica][-1]:.2e} m⁻²       │
  │  α (acero 304L) ≈ 0.09,  α (aluminio) ≈ 0.28       │
  │  α/L ≈ constante                                    │
  ├─────────────────────────────────────────────────────┤
  │ MOLÉCULA INDIVIDUAL                                 │
  │  F_plateau = {F_plateau:.1f} ± {F_plateau_std:.1f} pN                       │
  │  Lp = {Lp_ajuste:.3f} ± {perr_wlc[0]:.3f} nm                          │
  │  L₀ = {L0_ajuste:.1f} ± {perr_wlc[1]:.1f} nm                           │
  └─────────────────────────────────────────────────────┘

  Figuras guardadas en {OUT}/
""")
