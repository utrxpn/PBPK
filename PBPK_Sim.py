import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import plotly.graph_objects as go

# Drug release models
def zero_order(t,k):
    #where t is time and k is rate of release
    drug_released = t*k
    return drug_released

def first_order(t,k,C0):
    #where t is time, k is rate of release, and C0 is initial concentration
    drug_released = C0 * (1- np.exp(-k*t))
    return drug_released

def higuchi(t,k):
    #where t is time and k is rate of release
    drug_released = np.sqrt(t) * k
    return drug_released

def korsmeyer_peppas(t,k,n):

    #where t is time, k is rate of release, and n is the coefficent
    drug_released = k*t**n
    return drug_released

def zero_order_rate(t,k):
    rate = k
    return rate

def first_order_rate(t,k,C0):
    rate = C0 * k * np.exp(-t*k)
    return rate

def higuchi_rate(t,k):
    if t == 0:
        rate = 0
    else:
        rate = k / (2 * np.sqrt(t))
    return rate

def korsmeyer_peppas_rate(t,k,n):
    if t == 0:
        rate = 0
    else:
        rate = n* k * t**(n-1)
    return rate

def pbpk_model(t,y,params, release_func, release_params):
    #where t is time, y is the list of concentrations, and params is a dictionary of rate constants
    C_plasma, C_muscle, C_fat, C_liver = y 

    #getting the amount of drug released per time t, dependent on model and rate
    input_rate = release_func(t, *release_params)

    #getting the parameters
    kmuscle = params["k_muscle"]
    kfat = params["k_fat"]
    kliver = params["k_liver"]
    kelim = params["k_elim"]
    kmb = params["k_muscle_back"]
    kfb = params["k_fat_back"]

    #calculating the rate of change for each 'compartment' / tissue 
    dc_plasma = input_rate + C_plasma*(- kfat - kliver - kmuscle) + kmb*(C_muscle) + kfb*(C_fat) 
    dc_muscle = C_muscle*(-kmb) + kmuscle*(C_plasma)
    dc_fat = C_fat*(-kfb) + kfat*(C_plasma)
    dc_liver = C_liver*(-kelim) + kliver*(C_plasma)

    return [dc_plasma, dc_muscle, dc_fat, dc_liver]

#Pharmodynamic Layer: what the drug does in the tissue ie. modeling the effect

#modeling this using the hill equation

def hill_eq(C, Emax, EC50, n):
    #E is effect at concentration percent, EMAX = max possible effect, ec50 = concentration that produces 50% max effect, n is hill coefficent, C = drug concentration current

    E = Emax * C**n / (EC50**n + C**n)
    return E

def cell_viability(C, EC50, n):
    # what fraction of cells would survive from a cancer drug or a drug with toxicity
    variability = 100 * EC50**n / (EC50**n + C**n)
    return variability

def therapeutic_window(C_array, Emax, EC50_effect, EC50_toxic, n):


    effect = hill_eq(C_array, Emax, EC50_effect, n) 
    toxicity = hill_eq(C_array, Emax, EC50_toxic, n)
    
    return effect, toxicity

def radial_diffusion(r,t,M,D):
    #uses ficks law to demonstrate diffusion from a point source
    if t == 0:
        C = 0
    else:
        C = (M / (4 * np.pi * D * t)**(1.5)) * np.exp(-r**2 / (4*D*t))
    return C


def starling_flux(tissue):
    x = tissue_presets[tissue]
    J = x["Lp"] * (x["delta_P"] - x["sigma"] * x["delta_pi"])
    return J

def vascular_diffusion(X,Y,t,M,D,J,n_points = 50):
    vessel_x = np.linspace(-2,2,n_points)
    vessel_y = np.zeros(n_points)
    effective_M = max(M*J, 0)
    c_total = np.zeros_like(X)

    for i in range(n_points):
        r = np.sqrt((X- vessel_x[i])**2 + (Y- vessel_y[i])**2)
        c_total += radial_diffusion(r,t,effective_M, D)
    return c_total




st.title("PBPK/PD Drug Simulation")
tab1, tab2, tab3 = st.tabs(['Simulation', "Spatial Diffusion", "Vascular Network Simulation"])


release_model = st.sidebar.selectbox(
    "Release Model", 
    ["First Order", "Zero Order", "Higuchi", "Korsmeyer-Peppas"]
)

drug_presets = {
    "Custom": {
        "k_muscle": 0.05, "k_fat": 0.02, "k_liver": 0.10,
        "k_elim": 0.15, "k_muscle_back": 0.03, "k_fat_back": 0.01,
        "EC50_effect": 20, "EC50_toxic": 50, "Emax": 100, "dwtr": 0.015
    },
    "Ibuprofen": {
        "k_muscle": 0.08, "k_fat": 0.05, "k_liver": 0.18,
        "k_elim": 0.22, "k_muscle_back": 0.04, "k_fat_back": 0.02,
        "EC50_effect": 15, "EC50_toxic": 80, "Emax": 95, "dwtr":0.033
    },
    "Acetaminophen": {
        "k_muscle": 0.06, "k_fat": 0.02, "k_liver": 0.25,
        "k_elim": 0.30, "k_muscle_back": 0.03, "k_fat_back": 0.01,
        "EC50_effect": 10, "EC50_toxic": 40, "Emax": 90, "dwtr":0.039
    },
    "Warfarin": {
        "k_muscle": 0.03, "k_fat": 0.08, "k_liver": 0.12,
        "k_elim": 0.05, "k_muscle_back": 0.02, "k_fat_back": 0.04,
        "EC50_effect": 5, "EC50_toxic": 25, "Emax": 85, "dwtr":0.0096
    }
}

tissue_presets = {
    "Skeletal Muscle": {
        "cof": 1.85,
        "Lp": 0.0108,      # mm/hour/mmHg
        "sigma": 0.85,   # 0-1
        "delta_P": 22, # mmHg
        "delta_pi": 19.5, # mmHg
    },
    "Fat (interstitial)": {
        "cof": 1.60,
        "Lp": 0.0036,
        "sigma": 0.92,
        "delta_P": 22,
        "delta_pi": 19.5,
    },
    "Liver": {
        "cof": 1.20,
        "Lp": 1.1,
        "sigma": 0.075,
        "delta_P": 22,
        "delta_pi": 4,
    },
    "Tumor": {
        "cof": 2.75,
        "Lp": 0.536,
        "sigma": 0.325,
        "delta_P": -2,
        "delta_pi": 7.5
    },
    "Custom": {
        "cof": 1.00,
        "Lp": 0.05,
        "sigma": 0.5,
        "delta_P": 20,
        "delta_pi": 25
    }
}

def branching_network(X, Y, t, M, D, J, vessels, n_points=20):
    
    c_total = np.zeros_like(X)
    effective_M = max(M * J, 0)
    
    for (x0, y0, x1, y1) in vessels:
        # generate n_points along this vessel segment
        seg_x = np.linspace(x0, x1, n_points)
        seg_y = np.linspace(y0, y1, n_points)
        
        for i in range(n_points):
            r = np.sqrt((X - seg_x[i])**2 + (Y - seg_y[i])**2)
            c_total += radial_diffusion(r, t, effective_M, D)
    
    return c_total

def plot_vessel_network(vessels, t, M, D, J, n_points=50):
    fig = go.Figure()
    effective_M = max(M * J, 0)
    
    for (x0, y0, x1, y1) in vessels:
        # generate points along segment
        seg_x = np.linspace(x0, x1, n_points)
        seg_y = np.linspace(y0, y1, n_points)
        
        # concentration at each point along vessel (r=0, at the source)
        # use a small r to avoid division issues
        r = 0.01
        seg_c = [radial_diffusion(r, t, effective_M, D) for _ in range(n_points)]
        
        # add vessel as 3D line
        fig.add_trace(go.Scatter3d(
            x=seg_x,
            y=seg_y,
            z=seg_c,
            mode='lines',
            line=dict(
                color=seg_c,
                colorscale='Hot',
                width=10
            ),
            showlegend=False
        ))
    
    fig.update_layout(
        title="3D Vessel Network",
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Concentration (mg/mm³)"
        ),
        width=700,
        height=600
    )
    return fig


drug_choice = st.sidebar.selectbox("Drug Preset", list(drug_presets.keys()), key = "durg_choice")
preset = drug_presets[drug_choice]
k = st.sidebar.slider("Release Rate (k)", min_value = 0.01, max_value = 1.0, value = 0.3, step = 0.01, key = "drug_release_rate")
C0 = st.sidebar.slider("Initial Concentration", min_value = 10, max_value = 200, value = 100, step = 10, key = "initial_con")
sim_hours = st.sidebar.slider("Simulation Hours", min_value=1.00, max_value=72.00, value=24.00, step=0.25, key = "sim_hrs")
Emax = st.sidebar.slider("Max Effect", min_value = 50, max_value = 100, value =preset['Emax'], step = 5, key = "max_effectivity")
EC50_effect = st.sidebar.slider("Concentration to get 50% Effect", min_value = 0, max_value = 50, value = preset['EC50_effect'], step = 5,key = "con_to_get_50_effect")
EC50_toxic = st.sidebar.slider("Toxic Effect", min_value = 20, max_value = 100, value = preset['EC50_toxic'], step = 5, key = "toxicity")
n = st.sidebar.slider("N Coefficient (Korsmeyer-Peppas only)", min_value = 0.1, max_value = 1.0, value = 0.75, step = 0.05, key = "n_coeff")
n_hill = st.sidebar.slider("N Hill", min_value = 0.5, max_value = 3.0, value = 1.5, step = 0.1, key = "n_hill")

params = {
    "k_muscle":      preset["k_muscle"],
    "k_fat":         preset["k_fat"],
    "k_liver":       preset["k_liver"],
    "k_elim":        preset["k_elim"],
    "k_muscle_back": preset["k_muscle_back"],
    "k_fat_back":    preset["k_fat_back"],
    "dwtr":          preset["dwtr"]
}

vessels = [
    (-2, 0, 2, 0),      # main trunk horizontal
    (1, 0, 2, 1),       # right branch up
    (1, 0, 2, -1),      # right branch down
    (2, 1, 2, 1.5),     # secondary branch
    (2, -1, 2, -1.5),   # secondary branch
    (-1, 0, -2, 0.8),   # left branch up
    (-1, 0, -2, -0.8),  # left branch down
]
#Logic block
if release_model == "First Order":
    release_fxn = first_order_rate
    release_params = (k, C0)
elif release_model == "Zero Order":
    release_fxn = zero_order_rate
    release_params = (k,)
elif release_model == "Higuchi":
    release_fxn = higuchi_rate
    release_params = (k,)
elif release_model == "Korsmeyer-Peppas":
    release_fxn = korsmeyer_peppas_rate
    release_params = (k, n)


y0 = [0,0,0,0]
t_tst = (0,sim_hours)
t_testing = np.linspace(0,sim_hours,1000)

#solving the system 
solution = solve_ivp(pbpk_model ,t_tst, y0, method ='RK45', t_eval = t_testing, args = (params, release_fxn, release_params ))
effect, toxicity = therapeutic_window(solution.y[0], Emax, EC50_effect, EC50_toxic, n_hill)


       
with tab1:
    st.header("Simulation in 2D")

    fig, axes = plt.subplots(4,2,figsize=(14,16))

    axes[0,0].plot(solution.t, solution.y[0], color = 'blue', label = 'Plasma Concentration')
    axes[0,0].set_xlabel("Time")
    axes[0,0].set_ylabel("Concentration")
    axes[0,0].set_title("Plasma Concentration")
    axes[0,0].legend()
    axes[0,1].plot(solution.t, solution.y[1], color = 'black', label = 'Muscle Concentration')
    axes[0,1].set_xlabel('Time')
    axes[0,1].set_ylabel("Concentration")
    axes[0,1].set_title("Muscle Concentration")
    axes[0,1].legend()
    axes[1,0].plot(solution.t, solution.y[2], color = 'yellow', label = 'Fat Concentration')
    axes[1,0].set_xlabel('Time')
    axes[1,0].set_ylabel("Concentration")
    axes[1,0].set_title("Fat Concentration")
    axes[1,0].legend()
    axes[1,1].plot(solution.t, solution.y[3], color = 'deeppink', label = 'Liver Concentration')
    axes[1,1].set_xlabel('Time')
    axes[1,1].set_ylabel("Concentration")
    axes[1,1].set_title("Liver Concentration")
    axes[1,1].legend()
    axes[2,0].plot(solution.t, effect, color = 'green', label = 'Therapeutic Effect')
    axes[2,0].set_xlabel('Time')
    axes[2,0].set_ylabel("Effect %")
    axes[2,0].set_title("Therapeutic Effect")
    axes[2,0].legend()
    axes[2,1].plot(solution.t, cell_viability(solution.y[0], EC50_toxic, n_hill), color = 'red', label = 'Toxic Effect')
    axes[2,1].set_xlabel('Time')
    axes[2,1].set_ylabel("Effect %")
    axes[2,1].set_title("Toxic Effect")
    axes[2,1].legend()
    axes[3,0].plot(solution.t, toxicity, color = 'red', label = 'Toxicity')
    axes[3,0].plot(solution.t, effect, color = 'green', label = 'Therapeutic Effect')
    axes[3,0].fill_between(solution.t, toxicity, effect, alpha=0.4, color = 'aqua', label = "Therapeutic Window")
    axes[3,0].set_xlabel('Time')
    axes[3,0].set_ylabel("Effect %")
    axes[3,0].set_title("Combined Therapeutic Window")
    axes[3,0].legend()
    axes[3,1].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)

with tab2:
    st.header("Spatial Drug Diffusion")
    mass_drug = st.slider("Drug Mass (mg)", min_value=10, max_value=500, value=100, step = 10, key = "mass_of_drug")
    tissue_choice = st.selectbox("Preset Tissues", list(tissue_presets.keys()), key = "tissue_choice")
    tissue_d = st.slider("Tissue Diffusivity", min_value=0.01, max_value=1.0, value=0.1, step=0.01, key = "Tissue_diffusivity")
    
    pre = tissue_presets[tissue_choice]
    tis_params = params["dwtr"]
    if tissue_choice == "Custom":
        tissue_d = st.slider("Tissue Diffusivity", min_value=0.01, max_value=1.0, value=0.1, step=0.01, key = "tiss_diffusivity")
        H = tissue_d
    else:
        H=(tis_params/(pre["cof"])**2)


    xg = np.linspace(-1,1,200)
    yg = np.linspace(-1,1,200)
    X, Y = np.meshgrid(xg,yg)
    R = np.sqrt(X**2 +Y**2)


    tdiffusion = st.slider("Time in Hours", min_value=0.1, max_value=24.0, value=1.0, step=0.1, key = "time_of_diffusion")
    C_grid = radial_diffusion(R,tdiffusion, M=mass_drug, D=H)

    fig_diffusion = go.Figure(data=go.Heatmap(
        z = C_grid,
        x = xg,
        y = yg,
        colorscale="Hot",
        colorbar=dict(title="Concentration (mg/mm³)")
    ))

    fig_diffusion.update_layout(
        title = "Drug Diffusion in Tissue (2D)",
        xaxis_title = "Distance (mm)",
        yaxis_title = "Distance (mm)",
        width = 600,
        height = 600
    )

    st.plotly_chart(fig_diffusion)

    fig_3d = go.Figure(data=go.Surface(
    z=C_grid,
    x=xg,
    y=yg,
    colorscale='Hot',
    colorbar=dict(title="Concentration (mg/mm³)")
    ))

    fig_3d.update_layout(
        title="3D Drug Diffusion Surface",
        scene=dict(
            xaxis_title="Distance (mm)",
            yaxis_title="Distance (mm)",
            zaxis_title="Concentration"
        ),
        width=700,
        height=600
    )

    st.plotly_chart(fig_3d)

with tab3:
    st.header('Vascular Drug Diffusion')
    m_drug = st.slider("Drug Mass (mg)", min_value=10, max_value=500, value=100, step = 10, key = "m_drug_slider")
    t_choice = st.selectbox("Preset Tissues", list(tissue_presets.keys()), key = "tis_choice")
    t_d = st.slider("Tissue Diffusivity", min_value=0.01, max_value=1.0, value=0.1, step=0.01, key = "tis_diffusion")
    t_vasc = st.slider("Time in Hours", min_value=0.1, max_value=24.0, value=1.0, step=0.1, key= "t_vasc")
    
    pre = tissue_presets[t_choice]

    if t_choice == "Tumor":
        st.warning("Tumor tissue has elevated interstitial pressure (negative Starling flux). Passive vascular delivery is negligible — active targeting required.")

    xg = np.linspace(-2,2,200)
    yg = np.linspace(-2,2,200)
    X, Y = np.meshgrid(xg,yg)

    d_vasc = preset["dwtr"] / pre["cof"]**2

    J = starling_flux(t_choice)

    vd_grid = vascular_diffusion(X, Y, t_vasc, m_drug, d_vasc, J)
    bd_grid = branching_network(X,Y, t_vasc, m_drug, d_vasc, J, vessels)

    fig_vasc_heatmap = go.Figure(data=go.Heatmap(
        z = vd_grid,
        x = xg,
        y = yg,
        colorscale="Hot",
        colorbar=dict(title="Concentration (mg/mm³)")
    ))

    fig_vasc_heatmap.update_layout(
        title = "Vascular Drug Diffusion",
        xaxis_title = "Distance (mm)",
        yaxis_title = "Distance (mm)",
        width = 600,
        height = 600
    )

    st.plotly_chart(fig_vasc_heatmap)

    fig_vasc_3d = go.Figure(data=go.Surface(
    z=vd_grid,
    x=xg,
    y=yg,
    colorscale='Hot',
    colorbar=dict(title="Concentration (mg/mm³)")
    ))

    fig_vasc_3d.update_layout(
        title="3D Vascular Drug Diffusion",
        scene=dict(
            xaxis_title="Distance (mm)",
            yaxis_title="Distance (mm)",
            zaxis_title="Concentration"
        ),
        width=700,
        height=600
    )

    st.plotly_chart(fig_vasc_3d)


    fig_branched = go.Figure(data=go.Heatmap(
            z = bd_grid,
            x = xg,
            y = yg,
            colorscale="Hot",
            colorbar=dict(title="Concentration (mg/mm³)")
        ))
    
    fig_branched.update_layout(
        title = "Branched Vascular Drug Diffusion",
        xaxis_title = "Distance (mm)",
        yaxis_title = "Distance (mm)",
        width = 600,
        height = 600
    )

    st.plotly_chart(fig_branched)

    fig_vessels = plot_vessel_network(vessels, t_vasc, m_drug, d_vasc, J)
    st.plotly_chart(fig_vessels)


    

    


    