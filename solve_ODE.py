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
    dx = 0
    dy = 0
    dz = 0
    dp_dt = [dx, dy, dz]
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
        save_path = "theta_histogram.png"
        # Plot theta histogram and save as .png
        fig2 = plt.figure(figsize=(15, 15))
        ax1 = fig2.add_subplot(1, 1, 1)
        ax1.hist(theta, bins=100)
        ax1.set_xlabel('Theta (radians)')
        plt.savefig(save_path)
        plt.close()

def main():
    # Import data from a root file
    file = uproot.open("DV_Lb_Lcmunu_pKpi_FixedTarget_NoCut_pSet22_MCDT_9.root")
    tree = file["MCDecayTreeTuple/MCDecayTree"]

    px = tree["Lambda_b0_TRUEP_X"].array(library="np")
    py = tree["Lambda_b0_TRUEP_Y"].array(library="np")
    pz = tree["Lambda_b0_TRUEP_Z"].array(library="np")
    true_tau = tree["Lambda_b0_TRUETAU"].array(library='np') * 1e-7  # s
    mlambdab = 1115.7  # Mev/c2
    p_vec = np.stack((px, py, pz), axis=1)  # MeV/c
    p_tot = np.sqrt(px**2 + py**2 + pz**2)
    Ec = np.sqrt(p_tot ** 2  + mlambdab ** 2) - mlambdab
    En = np.sqrt((p_tot)**2 + (mlambdab)**2)  # NU with c=1
    # E_gauss = np.sqrt((p_tot * c_gauss)**2 + (mlambdab * (c_gauss**2))**2)  # MeV
    gamma = En / mlambdab
    gamma_c = 1 + Ec/mlambdab
    # beta = np.sqrt(1 - 1 / gamma ** 2)
    # Compute boost factors and initialize boost variable
    boost = np.stack((px, py, pz), axis=1) / En[:, None]  # 
    beta = np.sqrt(boost[:, 0]**2 + boost[:, 1]**2 + boost[:, 2]**2)
    
    #Compute the initial conditions for all events using the fact that the initial polarisation is perpendicular to the production plane
    p_tot_s = np.sqrt(px**2 + py**2)
    s_i = np.stack((px/p_tot_s, py/p_tot_s, np.zeros_like(p_tot)), axis=1)
    pos_0 = np.zeros_like(s_i)

    #Initialisation of a time array used for integration and computation of tau_ for each event
    tf = gamma * true_tau #s
    t = np.linspace(0,tf,num = 10) #s
    sf = np.empty_like(s_i) #[]
    pos_f = np.empty_like(pos_0) #[m]
    p_final = np.empty_like(p_vec) #[m]
    p_vec_normed = p_vec / p_tot[:, np.newaxis]
    filename = 'test_results_9.txt'
    with open(filename,'w') as file:
        for nb in range(len(p_tot)) : 
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

    theta = np.zeros(len(p_vec))
    print(len(p_vec))
    print(theta)
    norm_i = []
    norm_f = []
    p_tmp = []
    norm_tmp = []
    p_vec_normed = p_vec / p_tot[:, np.newaxis]
    p_f_normed = p_final / p_tot[:, np.newaxis]
    # for n in range(1, len(p_vec),2000):
    for n in range(len(p_vec)): 
        p_tmp = np.dot(p_vec_normed[n],p_f_normed[n])
        norm_i = np.sqrt(p_vec[n,0]**2 +p_vec[n,1]**2 + p_vec[n,2]**2)
        norm_f = np.sqrt(p_final[n,0]**2 +p_final[n,1]**2 + p_final[n,2]**2)
        norm_tmp=norm_i*norm_f
        rat = p_tmp/norm_tmp
        theta[n] = np.arccos(p_tmp/norm_tmp)

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

    plot_histo(plot_theta_hist = True,theta = theta)

if __name__ == "__main__" : 
    main()