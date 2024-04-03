import matplotlib.pyplot as plt
import uproot
import numpy as np
from numpy import linalg as LA
from scipy.integrate import odeint


#Definition of ODE function for polarisation
def ds(s, t,boost,gamma) :
    # Compute energy and define variables in natural units

    e = 1.602176634 * 1e-19  # As
    hbar = 6.582119570 * 1e-6  # Mevs
    tau_lambdab = 2.63 * 1e-10  # s
    c_gauss = 299792458  # m/S
    s_lambdab = 1 / 2
    mlambdab = 1115.7  # Mev/c2
    mu_lambdab = 1 / (2 * mlambdab)  # [e m]


    B = np.array([0, 20, 0]) #T
    E = np.array([0,0,0])
    omega_MDM = (mu_lambdab) * ( B - (gamma/(gamma + 1))*(np.dot(boost, B))*boost - np.cross(boost,E) )
    omega_EDM = (mu_lambdab) * ( B - (gamma/(gamma + 1))*(np.dot(boost, E))*boost + np.cross(boost,B) )
    sum = omega_MDM + omega_EDM
    ds_dt = np.cross(s,sum) #s-1
    return ds_dt

#Definition of ODE for position
def dpos(pos, t,boost,c):
    c_gauss = 299792458  # m/S
    dx = boost[0] * c_gauss
    dy = boost[1] * c_gauss
    dz = boost[2] * c_gauss
    dpos_dt = [dx, dy, dz]
    return(dpos_dt)

def dp(p, t,boost,c):
    B = np.array([0, 20, 0]) #T
    B = np.array([0,1.248*1e14,0])#Mev / Am2
    E = np.array([0,0,0])
    c = 299792458 #m/S
    e = 1.602176634 * 1e-19  # As
    dx = 0
    dy = 0
    dz = 0
    dp_dt = [dx, dy, dz]
    dp_dt = c * e * (E + c * np.cross(boost,B))
    return(dp_dt)

def plot_histo(plot_momentum_hist=False, plot_beta_hist=False, plot_gamma_hist=False, plot_energy_hist=False, plot_theta_hist=False,theta = None):

    if plot_momentum_hist:
        save_path = "momentum_histogram.png"
        # Plot theta histogram and save as .png
        fig = plt.figure(figsize=(15, 15))
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.hist(p_tot * 1e-6, bins=100)
        ax1.set_xlabel('Momentum [TeV/c]')
        plt.savefig(save_path)
        plt.close()

    if plot_beta_hist:
        save_path = "beta_histogram.png"
        # Plot theta histogram and save as .png
        fig = plt.figure(figsize=(15, 15))
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.hist(beta, bins=100)
        ax1.set_xlabel('beta []')
        plt.savefig(save_path)
        plt.close()

    if plot_gamma_hist:
        save_path = "gamma_histogram.png"
        # Plot theta histogram and save as .png
        fig = plt.figure(figsize=(15, 15))
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.hist(gamma, bins=100)
        ax1.set_xlabel('gamma []')
        plt.savefig(save_path)
        plt.close()

    if plot_energy_hist:
        save_path = "energy_gauss_histogram.png"
        # Plot theta histogram and save as .png
        fig = plt.figure(figsize=(15, 15))
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.hist(E_gauss * 1e-6, bins=100)
        ax1.set_xlabel('Energy gauss [TeV]')
        plt.savefig(save_path)
        plt.close()

    if plot_theta_hist:
        save_path = "theta_histogram_notnormed.png"
        # Plot theta histogram and save as .png
        fig2 = plt.figure(figsize=(15, 15))
        ax1 = fig2.add_subplot(1, 1, 1)
        ax1.hist(theta, bins=100)
        ax1.set_xlabel('Theta (radians)')
        plt.savefig(save_path)
        plt.close()

def boost_to_cm(pz_lab):
    # Constants
    sqrt_s = 7e6  # Center-of-mass energy [MeV]
    m_p = 1115.7  # Proton mass in MeV/c^2

    # Compute Lorentz factor
    gamma = np.sqrt(sqrt_s) / (2 * m_p)

    # Compute velocity of center-of-mass frame
    beta_cm = np.sqrt(1 - 1 / gamma**2)

    # Compute energy in the laboratory frame
    E_lab = np.sqrt(pz_lab**2 + m_p**2)

    # Compute momentum in the center-of-mass frame
    pz_cm = gamma * (pz_lab - beta_cm * E_lab)

    return pz_cm


def plot_pos_et_s(pos_f,sf):
    fig, axs = plt.subplots(2,3,figsize=(15,10))
    ax0 = axs[0,0]
    ax1 = axs[0,1]
    ax2 = axs[0,2]
    ax3 = axs[1,0]
    ax4 = axs[1,1]
    ax5 = axs[1,2]
    axs[0,0].hist(pos_f[:,0],bins = 50)
    ax0.set_xlabel('x [m]')
    axs[0,1].hist(pos_f[:,1],bins = 50)
    ax1.set_xlabel('y [m]')
    axs[0,2].hist(pos_f[:,0],bins = 50)
    ax2.set_xlabel('z [m]')
    axs[1,0].hist(sf[:,0],bins = 50)
    ax3.set_xlabel('sx[]')
    axs[1,1].hist(sf[:,1],bins = 50)
    ax4.set_xlabel('sy []')
    axs[1,2].hist(sf[:,2],bins = 50)
    ax5.set_xlabel('sz []')

    fig.tight_layout()
    plt.savefig("test.png" )

def solve(gamma, boost, true_tau,s_i, pos_0, p_vec):
    #Initialisation of a time array used for integration and computation of tau_ for each event
    tf = gamma * true_tau #s
    t = np.linspace(0,tf,num = 10) #s
    sf = np.empty_like(s_i) #[]
    pos_f = np.empty_like(pos_0) #[m]
    p_final = np.empty_like(p_vec) #[m]
    filename = 'test_results_9.txt'
    theta = np.zeros(len(p_vec))
    norm_i = []
    norm_f = []
    p_tmp = []
    norm_tmp = []
    with open(filename,'w') as file:
        for nb in range(len(p_vec[:,0])) : 
            s = odeint(ds, s_i[nb], t[:,nb],args = (np.asarray(boost[nb]), gamma[nb]))
            sf[nb] = s[-1] 
            pos_part = odeint(dpos,pos_0[nb],t[:,nb],args =(np.asarray(boost[nb]),1))
            pos_f[nb] = pos_part[-1]
            p = odeint(dp, p_vec[nb], t[:,nb], args=(np.asarray(boost[nb]),1))
            p_final[nb] = p[-1]
            file.write(f"Event n°{nb} \n")
            file.write(f"Polarisation vector at t_f : \n")
            file.write(f"{s[-1]} \n")
            file.write(f"Position of the particle at t_f: \n {pos_part[-1]} \n")

            p_tmp = np.dot(p_vec[nb],p_final[nb])
            norm_i = np.sqrt(p_vec[nb,0]**2 +p_vec[nb,1]**2 + p_vec[nb,2]**2)
            norm_f = np.sqrt(p_final[nb,0]**2 +p_final[nb,1]**2 + p_final[nb,2]**2)
            norm_tmp=norm_i*norm_f
            rat = p_tmp/norm_tmp
            # print (f'ratio = ',rat)
            theta[nb] = np.arccos(rat)
            # print(f'theta = ',theta[nb])
    
    return sf, pos_f, p_final, theta

def plot_xF(xF,p_tot_s):
    save_path = "xF_vs_p_tot_s.png"
    plt.figure(figsize=(15, 15))
    plt.xscale('log')
    plt.plot( xF, p_tot_s, 'o', markersize=3)
    plt.ylabel('p_tot_s')
    plt.xlabel('xF')
    plt.title('p_tot_s as a function of xF')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def main(px = None, py = None, pz = None, true_tau = None, pt = None):

    mlambdab = 1115.7  # Mev/c2
    p_vec = np.stack((px, py, pz), axis=1)  # MeV/c
    p_tot = np.sqrt(px**2 + py**2 + pz**2)
    Ec = np.sqrt(p_tot ** 2  + mlambdab ** 2) - mlambdab
    En = np.sqrt((p_tot)**2 + (mlambdab)**2)  # NU with c=1
    gamma = En / mlambdab
    # for comparison
    #gamma_c = 1 + Ec/mlambdab
    #beta = np.sqrt(1 - 1 / gamma ** 2)

    # Compute boost factors and initialize boost variable
    boost = np.stack((px, py, pz), axis=1) / En[:, None]  # 
    beta = np.sqrt(boost[:, 0]**2 + boost[:, 1]**2 + boost[:, 2]**2)
    
    #Compute the initial conditions for all events using the fact that the initial polarisation is perpendicular to the production plane
    p_tot_s = np.sqrt(px**2 + py**2)
    s_i = np.stack((-py/p_tot_s, px/p_tot_s, np.zeros_like(p_tot)), axis=1)
    pos_0 = np.zeros_like(s_i)

    sf, pos_f, p_final, theta = solve(gamma, boost, true_tau,s_i, pos_0, p_vec)

    # plot_pos_et_s(pos_f,sf)
    # plot_histo(plot_theta_hist = True,theta = theta)
    # Calculate the magnitude of the momentum vector
    p_vec_magnitude = np.linalg.norm(p_vec, axis=1)

    # Perform element-wise division
    p_vec_normed = p_vec / p_vec_magnitude[:, np.newaxis]
    
    for n in range(len(p_vec_magnitude)):
        dot = np.dot(s_i[n],p_vec_normed[n].T)
    sT = s_i - (dot) * p_vec_normed
    sT_norm = np.linalg.norm(sT,axis=1)
    print(sT)
    #Boost back initial momentum into CM frame
    pz_cm = boost_to_cm(pz)
    print(pz)

    sT = pt / p_tot_s #Try with the pT value from the root files 
    
    #Calculate delta x distributions 
    xF = pz_cm / ((7*1e6)/2)
    
    #Determine predicted polarisation
    plot_xF(xF,sT)

    # Plot polarisation distribution as a function of xF




if __name__ == "__main__" : 
    # Import data from a root file
    file = uproot.open("DV_Lb_Lcmunu_pKpi_FixedTarget_NoCut_pSet22_MCDT_9.root")
    tree = file["MCDecayTreeTuple/MCDecayTree"]

    px = tree["Lambda_b0_TRUEP_X"].array(library="np")
    py = tree["Lambda_b0_TRUEP_Y"].array(library="np")
    pz = tree["Lambda_b0_TRUEP_Z"].array(library="np")
    true_tau = tree["Lambda_b0_TRUETAU"].array(library='np') * 1e-7  # s
    pT = tree["Lambda_b0_TRUEPT"].array(library='np')
    print(pT)
    main(px = px, py = py, pz = pz, true_tau = true_tau,pt = pT)
