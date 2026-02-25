import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# sys.path.append(r'd:\mass\libs')

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import roots_legendre

# Determine the directory of the current script
base_dir = os.path.dirname(os.path.abspath(__file__))

# Redirect output to log files
log_file = open(os.path.join(base_dir, "training_log.txt"), "w", buffering=1)
err_file = open(os.path.join(base_dir, "error_log.txt"), "w", buffering=1)
sys.stdout = log_file
sys.stderr = err_file


# Set random seed for reproducibility
# torch.manual_seed(42)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

Tc = 0.155  # GeV

# HotQCD Table II Data (50 points)
hotqcd_data = np.array([
    [0.155, 0.020019, 0.000418, 0.002684, 2.43], [0.160, 0.024583, 0.000530, 0.003404, 2.76],
    [0.165, 0.029871, 0.000665, 0.004263, 3.07], [0.170, 0.035859, 0.000829, 0.005266, 3.34],
    [0.175, 0.042503, 0.001025, 0.006413, 3.56], [0.180, 0.049751, 0.001256, 0.007700, 3.74],
    [0.185, 0.057554, 0.001524, 0.009124, 3.88], [0.190, 0.065880, 0.001832, 0.010685, 3.97],
    [0.195, 0.074712, 0.002183, 0.012386, 4.03], [0.200, 0.084050, 0.002580, 0.014230, 4.05],
    [0.205, 0.093911, 0.003025, 0.016227, 4.05], [0.210, 0.104316, 0.003520, 0.018386, 4.03],
    [0.215, 0.115294, 0.004069, 0.020720, 3.99], [0.220, 0.126874, 0.004674, 0.023238, 3.94],
    [0.225, 0.139085, 0.005338, 0.025956, 3.88], [0.230, 0.151953, 0.006066, 0.028883, 3.82],
    [0.235, 0.165502, 0.006859, 0.032034, 3.76], [0.240, 0.179755, 0.007722, 0.035419, 3.69],
    [0.245, 0.194730, 0.008658, 0.039051, 3.63], [0.250, 0.210447, 0.009671, 0.042941, 3.57],
    [0.255, 0.226921, 0.010764, 0.047101, 3.50], [0.260, 0.244169, 0.011941, 0.051543, 3.44],
    [0.265, 0.262205, 0.013207, 0.056278, 3.38], [0.270, 0.281046, 0.014564, 0.061318, 3.32],
    [0.275, 0.300703, 0.016018, 0.066675, 3.26], [0.280, 0.321191, 0.017573, 0.072361, 3.20],
    [0.285, 0.342524, 0.019232, 0.078388, 3.14], [0.290, 0.364715, 0.021000, 0.084768, 3.08],
    [0.295, 0.387777, 0.022880, 0.091514, 3.02], [0.300, 0.411722, 0.024879, 0.098638, 2.96],
    [0.305, 0.436561, 0.026999, 0.106152, 2.91], [0.310, 0.462307, 0.029246, 0.114069, 2.85],
    [0.315, 0.488971, 0.031624, 0.122402, 2.79], [0.320, 0.516563, 0.034137, 0.131163, 2.74],
    [0.325, 0.545096, 0.036791, 0.140365, 2.69], [0.330, 0.574580, 0.039590, 0.150022, 2.63],
    [0.335, 0.605026, 0.042538, 0.160145, 2.58], [0.340, 0.636446, 0.045642, 0.170750, 2.53],
    [0.345, 0.668855, 0.048904, 0.181851, 2.48], [0.350, 0.702266, 0.052332, 0.193461, 2.43],
    [0.355, 0.736695, 0.055929, 0.205598, 2.38], [0.360, 0.772158, 0.059700, 0.218276, 2.33],
    [0.365, 0.808673, 0.063652, 0.231513, 2.29], [0.370, 0.846259, 0.067789, 0.245327, 2.24],
    [0.375, 0.884938, 0.072117, 0.259735, 2.20], [0.380, 0.924729, 0.076640, 0.274757, 2.15],
    [0.385, 0.965655, 0.081366, 0.290412, 2.11], [0.390, 1.007739, 0.086299, 0.306719, 2.07],
    [0.395, 1.051025, 0.091445, 0.323711, 2.03], [0.400, 1.095468, 0.096811, 0.341376, 1.99]
])

T_raw_np = hotqcd_data[:, 0]
s_raw_np = hotqcd_data[:, 1] / (T_raw_np ** 3)
delta_raw_np = hotqcd_data[:, 4]

# P_data at Tc (T=0.155 is at index 0)
P_data_Tc_val = hotqcd_data[0, 2] 
P_data_Tc_tensor = torch.tensor(P_data_Tc_val, dtype=torch.float32, device=device)

# Use original data points for training
T_train_phys = torch.tensor(T_raw_np, dtype=torch.float32).unsqueeze(1).to(device)
T_train_phys.requires_grad = True
T_train_norm = T_train_phys / Tc

s_target = torch.tensor(s_raw_np, dtype=torch.float32).unsqueeze(1).to(device)
delta_target = torch.tensor(delta_raw_np, dtype=torch.float32).unsqueeze(1).to(device)

# Integration Setup
N_INT = 60
P_UPPER = 15.0
pts, wts = roots_legendre(N_INT)
p_nodes = torch.tensor(0.5 * P_UPPER * (pts + 1), dtype=torch.float32).to(device)
p_weights = torch.tensor(0.5 * P_UPPER * wts, dtype=torch.float32).to(device)

# ==========================================
# Network Architecture (Double Branch ResNet)
# ==========================================
class ResNetBranch(nn.Module):
    def __init__(self, hidden_dim=32, out_scale=1.5):
        super().__init__()
        self.fc_in = nn.Linear(1, hidden_dim)
        self.silu = nn.SiLU()
        self.res_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(8)]) # 8 hidden layers
        self.fc_out = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid() 
        self.out_scale = out_scale
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
        # Initialize output layer with small weights
        nn.init.normal_(self.fc_out.weight, std=0.01)

    def forward(self, x):
        # Normalize input T/Tc (approx 1.0 to 2.6) to range [-1, 1]
        x_norm = (x - 1.8) / 0.8
        x = self.silu(self.fc_in(x_norm))
        for layer in self.res_layers:
            residual = x
            x = self.silu(layer(x))
            x = x + residual
        return self.sigmoid(self.fc_out(x)) * self.out_scale

class QuasiParticleMassNet(nn.Module):
    def __init__(self, hidden_dim=32, out_scale=1.5):
        super().__init__()
        self.branch1 = ResNetBranch(hidden_dim, out_scale)
        self.branch2 = ResNetBranch(hidden_dim, out_scale)

    def forward(self, T):
        # Sum of two branches
        return self.branch1(T) + self.branch2(T)

model_mg = QuasiParticleMassNet(hidden_dim=32, out_scale=1.5).to(device)
model_mud = QuasiParticleMassNet(hidden_dim=32, out_scale=1.5).to(device)
model_ms = QuasiParticleMassNet(hidden_dim=32, out_scale=1.5).to(device)

# No specific initialization for mass relationship
# Using default Kaiming/Normal initialization defined in ResNetBranch


# ==========================================
# Thermodynamics Calculation
# ==========================================
def calculate_thermodynamics(T_norm, T_phys, m_mg, m_mud, m_ms):
    batch_size = T_phys.shape[0]

    mg = m_mg(T_norm)
    mud = m_mud(T_norm)
    ms = m_ms(T_norm)

    mass_list = [mg, mud, ms]
    dof_list = [16, 24, 12]
    is_boson_list = [True, False, False]

    lnZ_total = torch.zeros((batch_size, 1), device=device)
    epsilon_id_total = torch.zeros((batch_size, 1), device=device)

    T_exp = T_phys.expand(-1, N_INT)
    P_sq = (p_nodes ** 2).expand(batch_size, -1)

    # Step 1: Calculate dm/dT using Autograd (For B(T) calculation)
    grad_mg = torch.autograd.grad(mg, T_phys, grad_outputs=torch.ones_like(mg), create_graph=True)[0]
    grad_mud = torch.autograd.grad(mud, T_phys, grad_outputs=torch.ones_like(mud), create_graph=True)[0]
    grad_ms = torch.autograd.grad(ms, T_phys, grad_outputs=torch.ones_like(ms), create_graph=True)[0]
    
    grad_mass_list = [grad_mg, grad_mud, grad_ms]

    # Step 2: Calculate Integrals
    # We need:
    # I1 = integral(p^2 * ln(1 +/- exp(-E/T))) -> for lnZ -> P_id
    # I2 = integral(p^2 * E * dist) -> for epsilon_id
    # I3 = integral(p^2 * (m/E) * dist) -> for B(T) calculation

    # B(T) integrand accumulator: dB/dT = sum( dPid/dm * dm/dT )
    integrand_B = torch.zeros((batch_size, 1), device=device)

    for m, dm_dT, dof, is_boson in zip(mass_list, grad_mass_list, dof_list, is_boson_list):
        m_exp = m.expand(-1, N_INT)
        E = torch.sqrt(P_sq + m_exp ** 2)
        term = torch.clamp(E / T_exp, max=50.0)
        exp_neg = torch.exp(-term)

        prefactor = dof / (2 * np.pi ** 2)

        if is_boson:
            ln_term = -torch.log(1.0 - exp_neg + 1e-10)
            dist = exp_neg / (1.0 - exp_neg + 1e-10)
        else:
            ln_term = torch.log(1.0 + exp_neg + 1e-10)
            dist = exp_neg / (1.0 + exp_neg + 1e-10)

        # Ideal Gas Contributions
        lnZ_total += prefactor * torch.sum(P_sq * ln_term * p_weights, dim=1, keepdim=True)
        epsilon_id_total += prefactor * torch.sum(P_sq * E * dist * p_weights, dim=1, keepdim=True)
        
        # B(T) Calculation Components
        # dPid/dm = - integral( p^2/E * f(E) * m * dof/2pi^2 )
        # integral_J here computes: (dof/2pi^2) * integral( p^2 * (m/E) * dist )
        integral_J = prefactor * torch.sum(P_sq * (m_exp / E) * dist * p_weights, dim=1, keepdim=True)
        
        # dB/dT = sum( dPid/dm * dm/dT ) = sum( - integral_J * dm/dT )
        integrand_B += - integral_J * dm_dT

    # Step 3: Calculate Physical Quantities
    # P_id = T * lnZ_total
    P_id = T_phys * lnZ_total
    
    # B(T) = integral( dB/dT ) dT + B0
    # B0 = P_id(Tc) - P_data(Tc)
    # We assume the first point of T_phys corresponds to Tc (0.155 GeV)
    B0 = P_id[0] - P_data_Tc_tensor
    
    # Cumulative trapezoidal integration
    B_T_integral = torch.zeros_like(T_phys)
    B_T_integral[1:] = torch.cumsum(
        0.5 * (integrand_B[1:] + integrand_B[:-1]) * (T_phys[1:] - T_phys[:-1]),
        dim=0
    )
    B_T = B_T_integral + B0
    
    # epsilon_real = epsilon_id + B(T)
    epsilon_real = epsilon_id_total + B_T
    
    # P_real = P_id - B(T)
    P_real = P_id - B_T
    
    # Trace Anomaly
    trace_anomaly = epsilon_real - 3 * P_real
    
    # Entropy Density s = (epsilon + P) / T
    # s_real = (epsilon_real + P_real) / T_phys = (epsilon_id + P_id) / T_phys = s_id
    s_real = (epsilon_real + P_real) / T_phys

    return s_real, trace_anomaly, P_real, epsilon_real, B_T, torch.cat(mass_list, dim=1)

# ==========================================
# Loss Function
# ==========================================
def get_loss(s_real, s_target, delta_real, delta_target, masses, T_phys, B_p):
    # Normalize by powers of T for fitting
    s_pred_dimless = s_real / (T_phys ** 3)
    delta_pred_dimless = delta_real / (T_phys ** 4)
    
    # Calculate Losses (Balanced weights)
    # MSE Losses
    loss_s = torch.mean((s_pred_dimless - s_target) ** 2)
    loss_delta = torch.mean((delta_pred_dimless - delta_target) ** 2)


    # Constraints
    # L1: mg/mud -> 1.5 at high T
    # L2: (ms - mud)/0.09 -> 1.0 at high T
    mg, mud, ms = masses[:, 0:1], masses[:, 1:2], masses[:, 2:3]
    mask = (T_phys > 0.38).float() # Apply constraints at high T

    # L1: |mg / mud - 1.5|
    # L2: |(ms - mud) / 0.09 - 1.0|
    # Use L1 norm (abs) as per user instruction
    term_L1 = torch.abs(mg / (mud + 1e-8) - 1.5)
    term_L2 = torch.abs((ms - mud) / 0.09 - 1.0)
    
    # User modification: Sum instead of Mean
    L1_sum = torch.sum(mask * term_L1)
    L2_sum = torch.sum(mask * term_L2)
    
    # Combined constraint loss: (beta1 * L1_sum + beta2 * L2_sum)^2
    # Weights apply to the SUM of errors
    

    # Mass hierarchy constraint: ms > mud
    # If ms <= mud, apply penalty. If ms > mud, penalty is 0.
    loss_order = torch.sum(torch.relu(mud - ms))
    loss_constraint = (0.01 * L1_sum + 0.0006 * L2_sum+0.01*loss_order) ** 2
    
    # Total Loss
    total_loss = loss_s + loss_delta + loss_constraint 
    
    return total_loss, loss_s, loss_delta, L1_sum, L2_sum, loss_order, B_p[0]

# ==========================================
# Training
# ==========================================
# Define Integration Constant B0 is now calculated dynamically, not a parameter

# Optimizer (Adam) - B0 removed from parameters
optimizer = torch.optim.Adam(
    list(model_mg.parameters()) +
    list(model_mud.parameters()) +
    list(model_ms.parameters()),
    lr=1e-3, # Modification 4: Keep learning rate unchanged
    weight_decay=1e-4
)

# Literature uses Adam (which is adaptive). No explicit scheduler mentioned.
# We remove ReduceLROnPlateau to strictly follow the literature.

# Checkpointing
def save_checkpoint(epoch):
    checkpoint = {
        'epoch': epoch,
        'model_mg_state_dict': model_mg.state_dict(),
        'model_mud_state_dict': model_mud.state_dict(),
        'model_ms_state_dict': model_ms.state_dict(),
        # 'B0': B0.item(), # B0 is no longer a parameter
        'optimizer_state_dict': optimizer.state_dict(),
    }
    filename = os.path.join(base_dir, f"checkpoint_{epoch}.pth")
    torch.save(checkpoint, filename)
    torch.save(checkpoint, os.path.join(base_dir, "checkpoint_latest.pth"))
    print(f"Checkpoint saved to {filename} and checkpoint_latest.pth")

# Plotting and Saving
def save_plots(epoch=None):
    model_mg.eval()
    model_mud.eval()
    model_ms.eval()

    T_plot_phys = torch.linspace(0.155, 0.40, 100).reshape(-1, 1).to(device)
    T_plot_phys.requires_grad = True
    # Pass B0 to calculation
    s_v, d_v, p_v, e_v, B_v, m_v = calculate_thermodynamics(T_plot_phys / Tc, T_plot_phys, model_mg, model_mud, model_ms)

    Tp = T_plot_phys.detach().cpu().numpy() / Tc

    fig, axes = plt.subplots(3, 2, figsize=(15, 18))

    # Data for plotting
    s_pred = (s_v / (T_plot_phys ** 3)).detach().cpu().numpy()
    d_pred = (d_v / (T_plot_phys ** 4)).detach().cpu().numpy()
    p_pred = (p_v / (T_plot_phys ** 4)).detach().cpu().numpy()
    e_pred = (e_v / (T_plot_phys ** 4)).detach().cpu().numpy()
    m_pred = m_v.detach().cpu().numpy()
    B_pred = B_v.detach().cpu().numpy()

    # Plot 1: Entropy Density
    axes[0, 0].plot(Tp, s_pred, 'k-', lw=2, label='PINN')
    axes[0, 0].scatter(hotqcd_data[:, 0] / Tc, hotqcd_data[:, 1] / (hotqcd_data[:, 0]**3), c='r', marker='+', label='HotQCD')
    axes[0, 0].set_title(r"Entropy Density $s/T^3$")
    axes[0, 0].set_xlabel(r"$T/T_c$")
    axes[0, 0].set_ylabel(r"$s/T^3$")
    axes[0, 0].legend()

    # Plot 2: Trace Anomaly
    axes[0, 1].plot(Tp, d_pred, 'k-', lw=2, label='PINN')
    axes[0, 1].scatter(hotqcd_data[:, 0] / Tc, hotqcd_data[:, 4], c='r', marker='+', label='HotQCD')
    axes[0, 1].set_title(r"Trace Anomaly $\Delta/T^4$")
    axes[0, 1].set_xlabel(r"$T/T_c$")
    axes[0, 1].set_ylabel(r"$\Delta/T^4$")
    axes[0, 1].legend()

    # Plot 3: Pressure
    axes[1, 0].plot(Tp, p_pred, 'k-', lw=2, label='PINN')
    axes[1, 0].scatter(hotqcd_data[:, 0] / Tc, hotqcd_data[:, 2] / (hotqcd_data[:, 0]**4), c='r', marker='+', label='HotQCD')
    axes[1, 0].set_title(r"Pressure $P/T^4$")
    axes[1, 0].set_xlabel(r"$T/T_c$")
    axes[1, 0].set_ylabel(r"$P/T^4$")
    axes[1, 0].legend()

    # Plot 4: Energy Density
    axes[1, 1].plot(Tp, e_pred, 'k-', lw=2, label='PINN')
    axes[1, 1].scatter(hotqcd_data[:, 0] / Tc, hotqcd_data[:, 3] / (hotqcd_data[:, 0]**4), c='r', marker='+', label='HotQCD')
    axes[1, 1].set_title(r"Energy Density $\epsilon/T^4$")
    axes[1, 1].set_xlabel(r"$T/T_c$")
    axes[1, 1].set_ylabel(r"$\epsilon/T^4$")
    axes[1, 1].legend()

    # Plot 5: Masses
    axes[2, 0].plot(Tp, m_pred[:, 1], 'b-', lw=2, label='$m_{u/d}$')
    axes[2, 0].plot(Tp, m_pred[:, 2], 'orange', lw=2, label='$m_s$')
    axes[2, 0].plot(Tp, m_pred[:, 0], 'g--', lw=2, label='$m_g$')
    axes[2, 0].set_title("Quasi-particle Masses")
    axes[2, 0].set_xlabel(r"$T/T_c$")
    axes[2, 0].set_ylabel(r"$Mass$ [GeV]")
    axes[2, 0].legend()

    # Plot 6: B(T)
    T_phys_np = Tp * Tc
    axes[2, 1].plot(Tp, B_pred / (T_phys_np**4), 'm-', lw=2, label='$B(T)/T^4$')
    axes[2, 1].set_title(r"Bag Constant $B(T)/T^4$")
    axes[2, 1].set_xlabel(r"$T/T_c$")
    axes[2, 1].set_ylabel(r"$B(T)/T^4$")
    axes[2, 1].legend()

    plt.tight_layout()
    filename = os.path.join(base_dir, "result_plot.png")
    if epoch is not None:
        filename = os.path.join(base_dir, f"result_plot_epoch_{epoch}.png")
    
    plt.savefig(filename)
    if epoch is None:
        print(f"Plots saved to {os.path.join(base_dir, 'result_plot.png')}")
    else:
        print(f"Plots saved to {filename}")
    plt.close(fig)

    # Revert to training mode
    model_mg.train()
    model_mud.train()
    model_ms.train()

epochs = 100000
print("Starting training...")

T_train_data = T_train_phys.clone().detach()

for epoch in range(epochs + 1):
    optimizer.zero_grad()
    
    # Use original data for training
    T_train_phys_iter = T_train_data.clone().detach().requires_grad_(True)
    T_train_norm_iter = T_train_phys_iter / Tc

    # Pass B0 to calculation
    s_p, d_p, p_p, e_p, B_p, m_p = calculate_thermodynamics(
        T_train_norm_iter, T_train_phys_iter, model_mg, model_mud, model_ms
    )

    total_loss, ls, ld, L1_sum, L2_sum, loss_order, current_B0 = get_loss(
        s_real=s_p,
        s_target=s_target,
        delta_real=d_p,
        delta_target=delta_target,
        masses=m_p,
        T_phys=T_train_phys_iter,
        B_p=B_p
    )

    total_loss.backward()

    # Gradient clipping (Relaxed to 10.0)
    torch.nn.utils.clip_grad_norm_(model_mg.parameters(), 10.0)
    torch.nn.utils.clip_grad_norm_(model_mud.parameters(), 10.0)
    torch.nn.utils.clip_grad_norm_(model_ms.parameters(), 10.0)

    optimizer.step()

    if epoch % 100 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        grad_norm = 0.0
        for p in model_mg.parameters():
            if p.grad is not None:
                grad_norm += p.grad.norm().item()

        print(f"Epoch {epoch:5d} | Loss: {total_loss.item():.6f} | "
              f"s_MSE: {ls.item():.6f} | Delta_MSE: {ld.item():.6f} | "
              f"LR: {current_lr:.1e} | Grad: {grad_norm:.4f} | "
              f"B_mean: {B_p.mean().item():.4f} | B0_calc: {current_B0.item():.4f} | "
              f"Mass_mean: {m_p.mean().item():.4f} | "
              f"L1_sum: {L1_sum.item():.4f} | L2_sum: {L2_sum.item():.4f} ")
        sys.stdout.flush()

    if epoch % 1000 == 0:
        save_plots(epoch)

# ==========================================
# Plotting and Saving
# ==========================================
print("Training finished. Generating plots...")
save_plots()
save_checkpoint(epochs)
