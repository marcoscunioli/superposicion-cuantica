# superpos_qiskit_streamlit.py
# 
# Demostraciones interactivas del módulo "Superposición con Qiskit" en Streamlit (Qiskit 1.x)
# Autoría visible al inicio, explicación pedagógica y guardado automático de figuras PNG.
# 
# Pasos incluidos (en pestañas):
# 1) Moneda cuántica (H + medida)
# 2) Valores esperados <Z>, <X> sin medir (Statevector)
# 3) Dos Hadamard seguidas (interferencia H·H = I)
# 4) Fase π entre dos H (cambia el resultado a '1')
# 5) Elegir fase φ para ~25%/75% (φ = 2π/3 por defecto, ajustable)
# 6) Visualización en esfera de Bloch
# 
# Requisitos de instalación (en consola):
#   pip install streamlit qiskit qiskit-aer
# Ejecución:
#   streamlit run superpos_qiskit_streamlit.py

from __future__ import annotations

import io
import math
from datetime import datetime
from pathlib import Path

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from qiskit.quantum_info import Statevector, Pauli
from qiskit_aer import AerSimulator

# =================== Configuración de página ===================
st.set_page_config(
    page_title="Superposición con Qiskit — Demostrador",
    page_icon="🧪",
    layout="centered",
)

# ============== Banner con autoría (pedido por el usuario) ==============
BANNER = "Script desarrollado por Marcos Sebastian Cunioli - Especialista en Ciberseguridad"
st.success(BANNER)

# =================== Parámetros y utilidades ===================
if "STAMP" not in st.session_state:
    st.session_state.STAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
STAMP = st.session_state.STAMP

OUTPUT_DIR = Path(st.sidebar.text_input("Carpeta para guardar PNGs", value="figuras"))
SAVE_PNGS = st.sidebar.checkbox("Guardar figuras como PNG", value=True)
DEFAULT_SHOTS = st.sidebar.slider("Shots (repeticiones por corrida)", 100, 50000, 1000, step=100)

st.sidebar.caption(
    f"Las imágenes se guardarán con timestamp {STAMP} en: {OUTPUT_DIR.resolve()}"
)

@st.cache_resource(show_spinner=False)
def get_simulator() -> AerSimulator:
    return AerSimulator()


def ensure_outdir(directory: Path) -> None:
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        st.warning(f"No se pudo crear la carpeta '{directory}': {e}")


def save_fig(fig, base_name: str) -> None:
    """Guardar figura PNG en OUTPUT_DIR con timestamp."""
    ensure_outdir(OUTPUT_DIR)
    fname = OUTPUT_DIR / f"{STAMP}_{base_name}.png"
    try:
        fig.savefig(fname, bbox_inches="tight", dpi=200)
        st.info(f"🖼️ Guardado: {fname}")
    except Exception as e:
        st.warning(f"No se pudo guardar {fname}: {e}")
    finally:
        plt.close(fig)


def draw_circuit_text(qc: QuantumCircuit) -> str:
    """Render del circuito en formato texto para mostrar en código."""
    try:
        return qc.draw(output="text").__str__()
    except Exception:
        return "(No fue posible dibujar el circuito en texto)"


def run_and_plot_counts(qc: QuantumCircuit, shots: int, title: str, base_name: str | None = None):
    """Compila, ejecuta en AerSimulator y muestra/guarda histograma de counts."""
    sim = get_simulator()
    qct = transpile(qc, sim)
    result = sim.run(qct, shots=shots).result()
    counts = result.get_counts()

    st.code(draw_circuit_text(qc), language="text")

    # Histograma
    try:
        fig = plot_histogram(counts, title=title)
        st.pyplot(fig)
        if SAVE_PNGS and base_name:
            save_fig(fig, base_name)
        else:
            plt.close(fig)
    except Exception as e:
        st.warning(f"No se pudo mostrar/guardar el histograma: {e}")

    return counts


# =================== Cabecera didáctica ===================
st.title("Superposición cuántica: demostrador interactivo")

st.markdown(
    """
    **¿Qué vas a ver aquí?**

    La **superposición cuántica** es la capacidad de un qubit de estar en una combinación de \n
    los estados `|0⟩` y `|1⟩` al mismo tiempo. La puerta **Hadamard (H)** crea esa mezcla: \n
    transforma `|0⟩` en `|+⟩` (50% `0`, 50% `1`). Si luego **medimos**, la superposición colapsa \n
    a un resultado clásico (0 o 1), pero **si no medimos**, podemos analizar el estado con \n
    vectores de estado y expectativas como ⟨Z⟩ o ⟨X⟩.

    Este demostrador te guía paso a paso con ideas clave:

    - **Azar controlado:** con H sobre `|0⟩` obtienes resultados 50/50.
    - **Interferencia:** aplicar H dos veces (H·H) equivale a no hacer nada (identidad).
    - **Fase:** insertar una **fase φ** entre dos H cambia las probabilidades de medir 0/1.
    - **Geometría:** la **esfera de Bloch** te ayuda a visualizar la posición del estado.

    Ajusta *shots*, cambia la fase y observa cómo responden las probabilidades. 
    """
)

# =================== Verificación de paquetes ===================
with st.expander("Verificación rápida de paquetes"):
    try:
        import qiskit as _qiskit
        import qiskit_aer as _qa
        import qiskit_ibm_runtime as _qir
        st.success("✅ Qiskit importado correctamente")
        st.write(f"qiskit-aer versión: {_qa.__version__}")
        st.write(f"qiskit-ibm-runtime versión: {_qir.__version__}")
    except Exception as e:
        st.error("⚠️ Problema importando paquetes. Asegúrate de instalar: 'pip install qiskit qiskit-aer'.")
        st.exception(e)

# =================== Pestañas por paso ===================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1) Moneda cuántica",
    "2) ⟨Z⟩ y ⟨X⟩ (sin medir)",
    "3) Dos H seguidas",
    "4) Fase π entre H",
    "5) Fase ajustable (25/75)",
    "6) Esfera de Bloch",
])

# ---- Paso 1 ----
with tab1:
    st.subheader("Paso 1 — Moneda cuántica (H + medida)")
    st.write("Aplicamos **H** a `|0⟩` y luego medimos: deberías ver ~50% de `0` y ~50% de `1`.")
    shots = st.number_input("Shots", min_value=100, max_value=100000, step=100, value=DEFAULT_SHOTS)

    if st.button("Ejecutar Paso 1", key="p1"):
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.measure_all()
        run_and_plot_counts(qc, shots=shots, title="H |0⟩ → 50/50", base_name="p1_moneda")

# ---- Paso 2 ----
with tab2:
    st.subheader("Paso 2 — Valores esperados ⟨Z⟩ y ⟨X⟩ sin medir")
    st.write(
        "Con **H** sobre `|0⟩` obtenemos `|+⟩`. Sin medir, calculamos expectativas: ⟨Z⟩≈0 y ⟨X⟩≈+1."
    )

    if st.button("Calcular expectativas", key="p2"):
        qc = QuantumCircuit(1)
        qc.h(0)
        psi = Statevector.from_instruction(qc)
        expZ = float(np.real(psi.expectation_value(Pauli("Z"))))
        expX = float(np.real(psi.expectation_value(Pauli("X"))))
        st.code(draw_circuit_text(qc), language="text")
        st.metric("⟨Z⟩ (esperado ~0)", f"{expZ:.6f}")
        st.metric("⟨X⟩ (esperado ~+1)", f"{expX:.6f}")

# ---- Paso 3 ----
with tab3:
    st.subheader("Paso 3 — Dos Hadamard seguidas (H·H = I)")
    st.write("Aplicar **H** dos veces equivale a la identidad: volvemos a `|0⟩`.")
    shots = st.number_input("Shots", min_value=100, max_value=100000, step=100, value=DEFAULT_SHOTS, key="shots3")

    if st.button("Ejecutar Paso 3", key="p3"):
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.h(0)
        qc.measure_all()
        run_and_plot_counts(qc, shots=shots, title="H H |0⟩ → |0⟩", base_name="p3_hh")

# ---- Paso 4 ----
with tab4:
    st.subheader("Paso 4 — Fase π entre dos H (resultado favorece '1')")
    st.write("Insertamos **fase π** entre dos H: la interferencia ahora favorece medir `1`.")
    shots = st.number_input("Shots", min_value=100, max_value=100000, step=100, value=DEFAULT_SHOTS, key="shots4")

    if st.button("Ejecutar Paso 4", key="p4"):
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.p(math.pi, 0)
        qc.h(0)
        qc.measure_all()
        run_and_plot_counts(qc, shots=shots, title="H – P(π) – H → |1⟩", base_name="p4_fase_pi")

# ---- Paso 5 ----
with tab5:
    st.subheader("Paso 5 — Elegí la fase φ para ajustar las probabilidades")
    st.write(
        "Con **H – RZ(φ) – H** podemos mover la probabilidad de `0`/`1`. \n"
        "Por ejemplo, con φ=2π/3 se espera ~25% `0` / ~75% `1`."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        # Fase en radianes con control deslizante (0 a 2π)
        phi = st.slider(
            "Fase φ (radianes)",
            min_value=0.0,
            max_value=float(2 * math.pi),
            value=float(2 * math.pi / 3),
            step=0.01,
        )
    with col_b:
        shots = st.number_input("Shots", min_value=100, max_value=200000, step=100, value=20000)

    if st.button("Ejecutar Paso 5", key="p5"):
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.rz(phi, 0)
        qc.h(0)
        qc.measure_all()
        counts = run_and_plot_counts(qc, shots=shots, title=f"H – RZ({phi:.2f}) – H", base_name="p5_fase")
        total = sum(counts.values()) or 1
        p0 = counts.get('0', 0) / total
        p1 = counts.get('1', 0) / total
        st.write(f"Proporciones observadas: **P(0)≈{100*p0:.2f}%** • **P(1)≈{100*p1:.2f}%** ")
        st.caption("Teórico: P(0)=cos²(φ/2), P(1)=sin²(φ/2)")

# ---- Paso 6 ----
with tab6:
    st.subheader("Paso 6 — Visualización en la esfera de Bloch")
    st.write("Explora dos estados clave: `|+⟩` (tras aplicar H) y el estado tras **√X (sx)**.")

    if st.button("Mostrar Bloch de H y √X", key="p6"):
        # Estado A: |+> con H
        qc_h = QuantumCircuit(1)
        qc_h.h(0)
        sv_h = Statevector.from_instruction(qc_h)
        st.markdown("**Circuito H:**")
        st.code(draw_circuit_text(qc_h), language="text")
        try:
            fig_h = plot_bloch_multivector(sv_h)
            st.pyplot(fig_h)
            if SAVE_PNGS:
                save_fig(fig_h, "p6_bloch_h")
            else:
                plt.close(fig_h)
        except Exception as e:
            st.warning(f"Bloch H: {e}")

        # Estado B: sobre ecuador con √X (sx)
        qc_sx = QuantumCircuit(1)
        qc_sx.sx(0)
        sv_sx = Statevector.from_instruction(qc_sx)
        st.markdown("**Circuito √X (sx):**")
        st.code(draw_circuit_text(qc_sx), language="text")
        try:
            fig_sx = plot_bloch_multivector(sv_sx)
            st.pyplot(fig_sx)
            if SAVE_PNGS:
                save_fig(fig_sx, "p6_bloch_sx")
            else:
                plt.close(fig_sx)
        except Exception as e:
            st.warning(f"Bloch √X: {e}")

# =================== Cierre ===================
st.divider()
st.markdown(
    """
    **Sugerencias para explorar**
    - Cambiá la fase φ y compará los resultados con la predicción teórica `cos²(φ/2)` / `sin²(φ/2)`.
    - Incrementá *shots* para observar cómo las frecuencias se acercan a las probabilidades teóricas.
    - Reemplazá `RZ(φ)` por `P(φ)` y verificá que el efecto estadístico sea equivalente.
    """
)
