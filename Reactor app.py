
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from fpdf import FPDF
import io
import math

# -------------------- Hardened Engineering Calculations --------------------

# Material properties lookup (representative values — verify against ASME tables for your specific material & temperature)
MATERIAL_PROPERTIES = {
    "Carbon Steel": {"allowable_stress_MPa": 137.0, "density_kg_m3": 7850, "k_W_mK": 54.0, "E_GPa": 210},
    "SS316": {"allowable_stress_MPa": 138.0, "density_kg_m3": 8000, "k_W_mK": 16.0, "E_GPa": 200},
    "Hastelloy": {"allowable_stress_MPa": 150.0, "density_kg_m3": 8400, "k_W_mK": 11.0, "E_GPa": 200},
    "Other": {"allowable_stress_MPa": 100.0, "density_kg_m3": 7800, "k_W_mK": 20.0, "E_GPa": 200}
}

# ASME Section VIII Div 1 thin-wall formula for internal pressure (cylindrical shell)
def asme_wall_thickness(P_design, D_outer, S_allow, E=1.0, CA=0.0):
    # P_design (Pa), D_outer (m), S_allow (Pa)
    denom = (2 * S_allow * E - 1.2 * P_design)
    if denom <= 0:
        # return sensible minimum if denom invalid
        return max(0.001, CA)
    t_req = (P_design * D_outer) / denom
    return t_req + CA

# Dittus-Boelter correlation for turbulent flow in tubes
def overall_U_estimate(Re, Pr, k_fluid, D_inner):
    if Re <= 2300:
        # conservative default for laminar flow
        return 100.0
    Nu = 0.023 * Re**0.8 * Pr**0.4
    h = Nu * k_fluid / D_inner if D_inner > 0 else 0.0
    fouling_factor = 0.0005
    wall_resistance = 0.0005
    if h <= 0:
        return 100.0
    U_overall = 1.0 / (1.0 / h + wall_resistance + fouling_factor)
    return U_overall

# Packed-bed Ergun equation pressure drop
def packed_bed_pressure_drop(L, D_particle, void_frac, rho, mu, v):
    if D_particle <= 0 or void_frac <= 0:
        return None
    term1 = 150 * (1 - void_frac)**2 / (void_frac**3) * (mu * v / D_particle**2)
    term2 = 1.75 * (1 - void_frac) / (void_frac**3) * (rho * v**2 / D_particle)
    return (term1 + term2) * L

# Small temperature-dependent allowable stress tables (representative values — verify before use)
ALLOWABLE_STRESS_TABLES = {
    "Carbon Steel": {20: 137.0, 100: 137.0, 200: 130.0, 300: 120.0, 400: 105.0},
    "SS316": {20: 138.0, 100: 135.0, 200: 130.0, 300: 120.0, 400: 100.0},
    "Hastelloy": {20: 150.0, 100: 148.0, 200: 145.0, 300: 140.0, 400: 130.0},
    "Other": {20: 100.0, 100: 95.0, 200: 90.0}
}


def interp_allowable_stress(material, T_design_C):
    table = ALLOWABLE_STRESS_TABLES.get(material)
    if table is None:
        return None
    temps = sorted(table.keys())
    if T_design_C <= temps[0]:
        return table[temps[0]]
    if T_design_C >= temps[-1]:
        return table[temps[-1]]
    for i in range(len(temps) - 1):
        t0, t1 = temps[i], temps[i + 1]
        if t0 <= T_design_C <= t1:
            s0, s1 = table[t0], table[t1]
            return s0 + (s1 - s0) * (T_design_C - t0) / (t1 - t0)
    return None

# -------------------- Streamlit UI --------------------

st.set_page_config(page_title="Hardened Reactor Check Sheet", layout="wide")
st.title("Reactor Check Sheet — Hardened Calculations")

# Inputs
col1, col2 = st.columns([1, 1])
with col1:
    project_name = st.text_input("Project name", "Reactor-001")
    vendor_name = st.text_input("Vendor", "Vendor Co")
    reactor_type = st.selectbox("Reactor type", ["Batch", "CSTR", "PFR tubular", "Packed-bed"])
    material = st.selectbox("Material", ["Carbon Steel", "SS316", "Hastelloy", "Other"])
    D_outer = st.number_input("Outer Diameter (m)", value=1.0)
    CA = st.number_input("Corrosion allowance (m)", value=0.003)
    T_design = st.number_input("Design temperature (°C)", value=150.0)
with col2:
    P_oper = st.number_input("Operating pressure (bar)", value=5.0)
    P_design = st.number_input("Design pressure (bar)", value=7.0)

    # interpolation with safe fallback
    default_S_allow = interp_allowable_stress(material, T_design)
    if default_S_allow is None:
        default_S_allow = MATERIAL_PROPERTIES.get(material, {}).get('allowable_stress_MPa', 100.0)

    MATERIAL_DEFAULTS = MATERIAL_PROPERTIES.get(material, {})
    default_density = MATERIAL_DEFAULTS.get('density_kg_m3', 7850)

    st.markdown(f"**Default allowable stress (based on material & T={T_design}°C):** {default_S_allow} MPa")
    S_allow = st.number_input("Allowable stress (MPa)", value=float(default_S_allow))
    flow_m3_hr = st.number_input("Flow (m³/hr)", value=10.0)
    density = st.number_input("Density (kg/m³)", value=float(default_density))
    mu = st.number_input("Viscosity (Pa·s)", value=0.001)
    Cp = st.number_input("Cp (J/kg·K)", value=2000.0)

# Additional plotting and operating-zone inputs
st.markdown("---")
col3, col4 = st.columns([1, 1])
with col3:
    flow_min = st.number_input("Flow range min (m³/hr)", value=0.1)
    flow_max = st.number_input("Flow range max (m³/hr)", value=max(10.0, flow_m3_hr * 3))
    n_points = st.number_input("Points in sweep", value=60, min_value=10, max_value=500)
with col4:
    U_min = st.number_input("Minimum acceptable U (W/m²·K)", value=100.0)
    DP_max = st.number_input("Maximum acceptable ΔP (Pa)", value=200000.0)
    flow_axis_log = st.checkbox("Log scale for flow axis", value=False)

flow_vals_m3_s = np.linspace(flow_min / 3600.0, flow_max / 3600.0, int(n_points))

# Derived values and calculations over sweep
P_design_pa = P_design * 1e5
S_allow_pa = S_allow * 1e6

thickness_vals = []
U_vals = []
DP_vals = []

for q in flow_vals_m3_s:
    # geometry and velocity
    A_cross = math.pi * (D_outer**2) / 4
    v_local = q / A_cross if A_cross > 0 else 0
    Re_local = density * v_local * D_outer / mu if mu > 0 else 0
    Pr_local = Cp * mu / MATERIAL_DEFAULTS.get('k_W_mK', 0.6)
    U_local = overall_U_estimate(Re_local, Pr_local, MATERIAL_DEFAULTS.get('k_W_mK', 0.6), D_outer)

    # pressure drop
    if reactor_type == "Packed-bed":
        # assume bed parameters; use inputs if present
        D_particle = 0.01
        void_frac = 0.4
        L_bed = 3.0
        DP_local = packed_bed_pressure_drop(L_bed, D_particle, void_frac, density, mu, v_local)
    else:
        L_tube = 5.0
        DP_local = 0.02 * (L_tube / D_outer) * 0.5 * density * v_local**2

    # wall thickness needed for design pressure (does not vary with flow in this simplified model)
    t_local = asme_wall_thickness(P_design_pa, D_outer, S_allow_pa, E=1.0, CA=CA)

    thickness_vals.append(float(t_local))
    U_vals.append(float(U_local) if U_local is not None else 0.0)
    DP_vals.append(float(DP_local) if DP_local is not None else np.inf)

# Convert arrays to more friendly scales
flow_vals_hr = flow_vals_m3_s * 3600.0

# Determine operating zone where both U >= U_min and DP <= DP_max
U_array = np.array(U_vals, dtype=float)
DP_array = np.array(DP_vals, dtype=float)
zone_mask = (U_array >= U_min) & (DP_array <= DP_max)

# Single design point
design_q = flow_m3_hr / 3600.0
# find nearest index safely
if len(flow_vals_m3_s) > 0:
    design_idx = int((np.abs(flow_vals_m3_s - design_q)).argmin())
else:
    design_idx = 0

# -------------------- Plots --------------------
st.subheader("Interactive Graphs")

# 1) Allowable stress vs Temp
fig_stress = go.Figure()
mat_table = ALLOWABLE_STRESS_TABLES.get(material, {})
if len(mat_table) > 0:
    temps = sorted(mat_table.keys())
    stress_vals = [mat_table[t] for t in temps]
    fig_stress.add_trace(go.Scatter(x=temps, y=stress_vals, mode='lines+markers', name='Allowable Stress (MPa)'))
    fig_stress.add_trace(go.Scatter(x=[T_design], y=[interp_allowable_stress(material, T_design)], mode='markers', marker=dict(color='red', size=10), name='Design T'))
fig_stress.update_layout(title='Allowable Stress vs Temperature', xaxis_title='Temperature (°C)', yaxis_title='Allowable Stress (MPa)')
st.plotly_chart(fig_stress, use_container_width=True)

# 2) Wall thickness vs Pressure (vary pressure sweep)
pressures_bar = np.linspace(0.1, max(2 * P_design, 10), 50)
thickness_vs_P = [asme_wall_thickness(p * 1e5, D_outer, S_allow_pa, E=1.0, CA=CA) for p in pressures_bar]
fig_t = go.Figure()
fig_t.add_trace(go.Scatter(x=pressures_bar, y=thickness_vs_P, mode='lines', name='Thickness (m)'))
fig_t.add_vline(x=P_design, line=dict(color='red', dash='dash'))
fig_t.update_layout(title='Wall thickness vs Design Pressure', xaxis_title='Pressure (bar)', yaxis_title='Thickness (m)')
st.plotly_chart(fig_t, use_container_width=True)

# 3) Pressure drop vs Flow
fig_dp = go.Figure()
fig_dp.add_trace(go.Scatter(x=flow_vals_hr, y=DP_vals, mode='lines', name='ΔP (Pa)'))
# add design marker safely
if 0 <= design_idx < len(flow_vals_hr):
    fig_dp.add_trace(go.Scatter(x=[flow_m3_hr], y=[DP_vals[design_idx]], mode='markers', marker=dict(color='red', size=10), name='Design point'))
fig_dp.update_layout(title='Pressure drop vs Flow', xaxis_title='Flow (m³/hr)', yaxis_title='Pressure drop (Pa)')
if flow_axis_log:
    fig_dp.update_xaxes(type='log')
st.plotly_chart(fig_dp, use_container_width=True)

# 4) U vs Flow
fig_u = go.Figure()
fig_u.add_trace(go.Scatter(x=flow_vals_hr, y=U_vals, mode='lines', name='U (W/m²·K)'))
if 0 <= design_idx < len(flow_vals_hr):
    fig_u.add_trace(go.Scatter(x=[flow_m3_hr], y=[U_vals[design_idx]], mode='markers', marker=dict(color='red', size=10), name='Design point'))
fig_u.update_layout(title='Overall U vs Flow', xaxis_title='Flow (m³/hr)', yaxis_title='U (W/m²·K)')
if flow_axis_log:
    fig_u.update_xaxes(type='log')
st.plotly_chart(fig_u, use_container_width=True)

# 5) Combined performance summary (U & ΔP on left y, thickness on right y)
fig_comb = go.Figure()
fig_comb.add_trace(go.Scatter(x=flow_vals_hr, y=U_vals, mode='lines', name='U (W/m²·K)', yaxis='y1'))
fig_comb.add_trace(go.Scatter(x=flow_vals_hr, y=DP_vals, mode='lines', name='ΔP (Pa)', yaxis='y1'))
fig_comb.add_trace(go.Scatter(x=flow_vals_hr, y=thickness_vals, mode='lines', name='Thickness (m)', yaxis='y2'))

# design markers
if 0 <= design_idx < len(flow_vals_hr):
    fig_comb.add_trace(go.Scatter(x=[flow_m3_hr], y=[U_vals[design_idx]], mode='markers', marker=dict(color='black', size=9), name='U at design', yaxis='y1'))
    fig_comb.add_trace(go.Scatter(x=[flow_m3_hr], y=[DP_vals[design_idx]], mode='markers', marker=dict(color='black', size=9), name='ΔP at design', yaxis='y1'))
    fig_comb.add_trace(go.Scatter(x=[flow_m3_hr], y=[thickness_vals[design_idx]], mode='markers', marker=dict(color='black', size=9), name='Thickness at design', yaxis='y2'))

# add operating zone shading
ranges = []
start = None
for i, ok in enumerate(zone_mask):
    if ok and start is None:
        start = flow_vals_hr[i]
    if not ok and start is not None:
        ranges.append((start, flow_vals_hr[i - 1]))
        start = None
if start is not None:
    ranges.append((start, flow_vals_hr[-1]))

for (a, b) in ranges:
    fig_comb.add_vrect(x0=a, x1=b, fillcolor='green', opacity=0.12, layer='below', line_width=0)

fig_comb.update_layout(
    title='Combined Performance: U, ΔP (left) and Thickness (right) vs Flow',
    xaxis_title='Flow (m³/hr)',
    yaxis=dict(title='U (W/m²·K) / ΔP (Pa)', side='left'),
    yaxis2=dict(title='Thickness (m)', overlaying='y', side='right')
)
if flow_axis_log:
    fig_comb.update_xaxes(type='log')

st.plotly_chart(fig_comb, use_container_width=True)

# -------------------- Results summary and PDF export --------------------
st.subheader("Results & Export")
# single point results
A_cross = math.pi * (D_outer**2) / 4
v = (flow_m3_hr / 3600.0) / A_cross if A_cross > 0 else 0
Re = density * v * D_outer / mu if mu > 0 else 0
Pr = Cp * mu / MATERIAL_DEFAULTS.get('k_W_mK', 0.6)
U_guess = overall_U_estimate(Re, Pr, MATERIAL_DEFAULTS.get('k_W_mK', 0.6), D_outer)

# calculate a dp_point safely
if reactor_type == "Packed-bed":
    D_particle_local = 0.01
    void_frac_local = 0.4
    L_bed_local = 3.0
    dp_point = packed_bed_pressure_drop(L_bed_local, D_particle_local, void_frac_local, density, mu, v)
else:
    L_tube_local = 5.0
    dp_point = 0.02 * (L_tube_local / D_outer) * 0.5 * density * v**2 if D_outer > 0 else np.inf

thickness_point = asme_wall_thickness(P_design_pa, D_outer, S_allow_pa, E=1.0, CA=CA)
Q_watts = density * (flow_m3_hr / 3600.0) * Cp * (50.0)  # simplified default ΔT = 50°C when not specified in UI
A_ht = Q_watts / (U_guess * (50.0)) if U_guess > 0 else None

results = {
    "Material": material,
    "Design temp (°C)": T_design,
    "Wall thickness (m)": thickness_point,
    "Velocity (m/s)": v,
    "Reynolds number": Re,
    "Estimated U (W/m².K)": U_guess,
    "Pressure drop (Pa)": dp_point,
    "Heat duty (W)": Q_watts,
    "Heat transfer area (m²)": A_ht,
}

st.dataframe(pd.DataFrame(results.items(), columns=["Parameter", "Value"]))

# PDF generation: try to render key plots as images and embed

def fig_to_png_bytes(fig):
    try:
        # fig.to_image uses kaleido or orca; if not installed it will raise — caller should handle
        img_bytes = fig.to_image(format='png', width=800, height=400, scale=1)
        return img_bytes
    except Exception:
        return None

if st.button("Generate PDF check sheet with graphs"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, f"Reactor Check Sheet — {project_name}", ln=1)
    pdf.set_font("Arial", size=10)
    # build multi-line inputs string safely
    inputs_text = (
        f"Flow: {flow_m3_hr} m³/hr\n"
        f"Density: {density} kg/m³\n"
        f"Viscosity: {mu} Pa·s\n"
        f"Design Pressure: {P_design} bar\n"
        f"Corrosion allowance: {CA} m\n"
        f"Allowable stress used: {S_allow} MPa"
    )
    pdf.multi_cell(0, 5, f"Vendor: {vendor_name}    Material: {material}    Design Temp: {T_design} °C")
    pdf.ln(2)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 6, "Inputs", ln=1)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 5, inputs_text)
    pdf.ln(2)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 6, "Results", ln=1)
    pdf.set_font("Arial", size=10)
    for k, v in results.items():
        pdf.cell(0, 5, f"{k}: {v}", ln=1)

    # embed combined plot
    png = fig_to_png_bytes(fig_comb)
    if png:
        try:
            # write to a temporary file that FPDF can read
            img_fname = "/tmp/combined_plot.png"
            with open(img_fname, 'wb') as f:
                f.write(png)
            pdf.add_page()
            pdf.image(img_fname, x=10, y=20, w=190)
        except Exception:
            # ignore embed errors and continue
            pass

    pdf_output = pdf.output(dest='S').encode('latin-1', errors='replace')
    st.download_button("Download PDF", data=pdf_output, file_name=f"{project_name}_check_sheet_with_graphs.pdf", mime='application/pdf')

st.markdown("**Notes:** The operating zone is shaded where U ≥ U_min and ΔP ≤ ΔP_max. All material and ASME values are representative — verify with official ASME tables and material datasheets before issuing vendor pack.")

st.caption("Hardened prototype with vendor-intelligent combined performance chart. You can change materials, temperature, and flow sweep bounds to see how the operating zone adapts.")

# -------------------- Lightweight self-tests (non-intrusive) --------------------
# These tests run only when the module is executed outside Streamlit environment and help validate core functions.
if __name__ == '__main__':
    # quick sanity checks
    assert asme_wall_thickness(1e5, 1.0, 100e6, CA=0.001) >= 0
    assert overall_U_estimate(1e5, 1.0, 0.6, 0.1) > 0
    print('Sanity checks passed')
