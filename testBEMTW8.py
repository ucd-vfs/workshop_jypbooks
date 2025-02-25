import numpy as np
import matplotlib.pyplot as plt

#Initial Values
sigma = 0.1
C_La = 2 * np.pi
C_Treq = 0.008
C_d0 = 0.01 #we're assuming a constant drag polar, in this case
N_b = 4 #number of blades
N_ds = 51 #number of discretized sections
r = np.linspace(1/(2*N_ds), 1 - 1/(2*N_ds), N_ds) #making our nondimension blade position matrix
delta_r = 1 / N_ds #this is the length of each discretized section

def theta(t_0, t_tw, r): #we'll come back to this t_tw
    theta = t_0 + t_tw * r
    return theta

def inflow(sigma, C_La, F, theta, r): #I would use lambda as my variable here, but its a reserved term in python
    inflow = ((sigma * C_La) / (16 * F)) * (np.sqrt(1 + ((32 * F) / (sigma * C_La)) * theta * r) - 1)
    return inflow

def prandtls(f):
    F = 2 / np.pi * np.arccos(np.e ** (-f))
    return F

def sprandtls(N_b, r, inflow):
    f = (N_b/2) * ((1 - r)/inflow)
    return f

def secthrust(sigma, C_La, theta, r, inflow, delta_r):
    dC_T = (sigma * C_La)/2 * (theta * r ** 2 - inflow * r) * delta_r
    return dC_T

def seclift(C_La, theta, inflow, r):
    dC_L = C_La * (theta - (inflow/r))
    return dC_L

def collective(theta_0i, C_Treq, C_Ti, sigma, C_La):
    theta_0 = theta_0i + (6 * (C_Treq - C_Ti))/(sigma * C_La) + (3 * np.sqrt(2))/4 * (np.sqrt(C_Treq) - np.sqrt(C_Ti))
    return theta_0

def colini(C_Treq, sigma, C_La, t_tw):
    colini = (6 * C_Treq)/(sigma * C_La) - 3/4 * t_tw + 3/4 * np.sqrt(2 *C_Treq)
    return colini

def profpow(sigma, C_d0, r, delta_r):
    C_P0 = (sigma / 2) * C_d0 * (r ** 3) * delta_r
    return C_P0

def collectiveiteration(t_tw): #notice how this whole function calculates everything based on a given twist angle
    #iteration loop:
    #initialization of loop:
    Fold = 1
    t_0old = colini(C_Treq, sigma, C_La, t_tw)
    res =  1 # this is to make sure the loop doesnt exit after first iteration
    while res >= 1e-5:
        #here's that nested loop
        while res >= 1e-5:
            inflow_val = inflow(sigma, C_La, Fold, theta(t_0old, t_tw, r), r)
            f_val = sprandtls(N_b, r, inflow_val)
            F = prandtls(f_val)
            res = abs(F - Fold).max()
            Fold = F
        inflow_val = inflow(sigma, C_La, F, theta(t_0old, t_tw, r), r)
        Cl_val = seclift(C_La, theta(t_0old, t_tw, r), inflow_val, r)
        dCt_val = secthrust(sigma, C_La, theta(t_0old, t_tw, r), r, inflow_val, delta_r)
        Ct_val = np.sum(dCt_val)
        Cpi_val = dCt_val * inflow_val
        Cp0_val = profpow(sigma, C_d0, r, delta_r)
        t_0val = collective(t_0old, C_Treq, Ct_val, sigma, C_La)
        res = abs(t_0val - t_0old)
        t_0old = t_0val
    return inflow_val, Cl_val, dCt_val, Cpi_val, Cp0_val

for t_tw in [0, -5, -10, -15, -20, -25]:
    inflow_ratio, lift, thrust, inducedp, profp = collectiveiteration(np.deg2rad(t_tw))
    plt.figure(1)
    plt.plot(r, inflow_ratio, label=f"twist angle = {t_tw}°")
    
    plt.figure(2)
    plt.plot(r, lift, label=f"twist angle = {t_tw}°")

    plt.figure(3)
    plt.plot(r, thrust, label=f"twist angle = {t_tw}°")

    plt.figure(4)
    plt.plot(r, inducedp, label=f"twist angle = {t_tw}°")

    plt.figure(5)
    plt.plot(r, profp, label=f"twist angle = {t_tw}°")

plt.figure(1)
plt.xlabel("Non-dimensionalized Radial position (r)")
plt.ylabel("Inflow ratio ($\\lambda$)")
plt.title("Variation of Inflow Across Blade for Select Twist Angles")
plt.legend(title="Twist Angles")

plt.figure(2)
plt.xlabel("Non-dimensionalized Radial position (r)")
plt.ylabel("Coefficient of Lift (Cl)")
plt.title("Variation of Lift Across Blade for Select Twist Angles")
plt.legend(title="Twist Angles")

plt.figure(3)
plt.xlabel("Non-dimensionalized Radial position (r)")
plt.ylabel("Coefficient of Thrust (Ct)")
plt.title("Variation of Thrust Across Blade for Select Twist Angles")
plt.legend(title="Twist Angles")

plt.figure(4)
plt.xlabel("Non-dimensionalized Radial position (r)")
plt.ylabel("Coefficient of Induced Power (Cpi)")
plt.title("Variation of Induced Power Across Blade for Select Twist Angles")
plt.legend(title="Twist Angles")

plt.figure(5)
plt.xlabel("Non-dimensionalized Radial position (r)")
plt.ylabel("Coefficient of Profile Power (Cp0)")
plt.title("Variation of Profile Power Across Blade for Select Twist Angles")
plt.legend(title="Twist Angles")

plt.show()