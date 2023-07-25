import numpy as np   
import matplotlib.pyplot as plt
import scipy.special as sci
import sys
from scipy.interpolate import UnivariateSpline

#----------------------------------
#Input parameters
#----------------------------------

rho_m = 8000 # Metal density, kg / m^3
r_min = 0.  # Minimum radius for pdf, in meters
gamma = 1.3082 #Geometric constant such that sigma_r = r_eff for a spherical clump
Shape = 4 #Choose shape for mass as a function of time, 1 = Sigmoid, 2 = Sphere, 3 = two clumps with sigmoids, 4 = data from Miki's simulation files
Clumps = 0 #1 = divide into 2 clumps, 0 = do not divide into clumps
Mmars = 6.41e23 #kg

#----------------------------------
#Choose shape of Mass as a function of time
#----------------------------------

if Shape == 1 : 
    # Sigmoid function
    t_max = 10000 # Maximum time in seconds
    Dt = t_max / 1000 # Time step in seconds
    Time = np.arange(0,t_max,Dt) # Time vector
    Time_c = t_max / 12. # Characteristic time
    U = 10.e3 * np.ones(len(Time)) # Velocity of metal entering the mantle in m / s
    r_eff = 1.e6 # Effective spherical radius of total metal mass entering the mantle, in meters
    M_tot = rho_m*4./3.*np.pi*(r_eff)**3 # Total metal mass entering the mantle, in kg
    Mass = M_tot/(1.+np.exp(-(Time/Time_c)+5)) # Mass as a function of time, in kg
    Dr = r_eff / 100. # Radius step for pdf, in meters

if Shape == 2 : 
    # Sphere entering magma ocean
    r_eff = 1.e6 # Effective spherical radius of total metal mass entering the mantle, in meters
    U = 10.e3  # Velocity of metal entering the mantle, in m/s
    t_max = 2.*r_eff/np.max(U) # Maximum time, in seconds
    Dt = t_max/1000 # Time step, in seconds
    Time = np.arange(0,t_max+Dt,Dt) # Time vector, in seconds
    U = U * np.ones(len(Time))  # Velocity of metal entering the mantle, in m/s
    z = np.max(U)*Time #Height, in meters
    s = (r_eff**2-(r_eff-z)**2)**0.5 # Cylindrical radius, in meters
    DMass = rho_m*np.pi*s**2 * U * Dt # Mass increment in Dt, in kg
    Mass = np.zeros(len(DMass)) # Mass vector, in kg
    for i in range(len(DMass)-1) : 
        Mass[i+1] = Mass[i]+DMass[i]
    M_tot = Mass[-1]
    Dr = r_eff / 100.  # Radius step for pdf, in meters

if Shape == 3 : 
    # Two clumps with sigmoid functions
    t_max = 10000  # Maximum time, in s
    Dt = t_max / 1000 # Time step, in s
    Time = np.arange(0,t_max,Dt) # Time vector, in s
    Time_c1 = t_max / 40. # Characteristic time for Clump 1 , in s
    Time_c2 = t_max / 40. # Characteristic time for Clump 2, in seconds
    shift = 20 # Shift in time between clump 1 and clump 2, dimensionless
    U = 10.e3 * np.ones(len(Time))# Velocity of metal entering the mantle, in m/s
    r_eff = 1.e6 # Effective spherical radius of total metal mass entering the mantle, in meters
    M_tot = rho_m*4./3.*np.pi*(r_eff)**3  # Total metal mass entering the mantle, in kg
    Mass1 = 1./(1.+np.exp(-(Time/Time_c1)+5)) # Mass as a function of time for clump 1
    Mass2 = 1./(1.+np.exp(-(Time/Time_c2)+5+shift))# Mass as a function of time for clump 2
    Mass = Mass1+Mass2
    Mass = M_tot / Mass[-1]*Mass #Mass as a function of time, in kg
    Dr = r_eff / 100.  # Radius step for pdf, in meters
    
if Shape == 4 : # read from a file
    data = np.loadtxt('Accretion_Ratios/sphg9__01_HTR-accRatios.txt', skiprows=1,usecols=[0,1,2])
    Time_data = data[:, 0] * 3600 # Time converted into seconds 
    Mass_data = data[:, 1] #unit less
    Velocity_data = data[:,2] #Velocity in m / s
    Time = Time_data
    #Time = np.linspace(Time_data[0],Time_data[0]  )
    

# Define the window size for the moving average filter
    window_size = 1

# Apply the moving average filter
    Mass = np.convolve(Mass_data, np.ones(window_size)/window_size, mode='same')
    # Making it smooth so that we can compute dM/dt later
    
    Uimp = 6000.0 #Impact speed in m /s
    U = Velocity_data # Velocity of metal entering the mantle in m / s
    U[U==0] = Uimp#If U = 0, set U to impact velocity
    #r_eff = 1.e6 # Effective spherical radius of total metal mass entering the mantle, in meters
    M_tot = Mass_data[-1]* 0.3 *  0.295 * Mmars #MN: I am hard-cording this here, which is not quite right
    r_eff = (M_tot/(4.0/3.0 * np.pi * rho_m))**0.333333  
    Dr = r_eff / 20.  # Radius step for pdf, in meters    
    #Dt = max(Time) / 1000

    
    #M_tot = rho_m*4./3.*np.pi*(r_eff)**3 # Total metal mass entering the mantle, in kg
    t_max = max(Time)
    
    Mass = Mass * M_tot #Making the mass dimensional
    
    
    #plt.figure(1)
    #plt.scatter(Time,Mass_data,label='Mass/M_tot')
    #plt.plot(Time,Mass/M_tot,label='Mass/M_tot')
    #plt.xlabel('Time (s)')
    #plt.show()
    #plt.close()
    #sys.exit()

#----------------------------------
#Compute radius and stretching
#----------------------------------


#Compute dM/dt

dM_dt = np.gradient(Mass, Time)

#Dt += [Dt[-1]] #Dt.append(Dt[-1])



#print(len(dM_dt), len(Dt))
#sys.exit()

valid_indices = np.where(dM_dt >= 0) #removing dM/dt < 0 (this happens at the end. There are some zeros though)
#valid_indices = [1,2,3,4,5,6,7,8,9,10]

Mass  = Mass[valid_indices]
Time  = Time[valid_indices]
dM_dt = dM_dt[valid_indices]
U     = U[valid_indices]



#Compute typical width of plume entering the mantle
Plume_Radius = (dM_dt/(rho_m*np.pi*U))**0.5

print(np.average(Plume_Radius)*1e-3,r_eff*1e-3, np.average(Plume_Radius)/r_eff)
#sys.exit()




#Function to compute probability density function (pdf) for plume radius 
def Compute_PDF(M,dM_dt,Dt,Radius,r_min,r_max,Dr) : 
    r = np.arange(r_min,2.*r_max,Dr) #Radius
    p = np.zeros(len(r)) # p = pdf
    Mass_tot    = M[-1]- M[0] # Total mass
    for i_r in range(len(r)-1):
        X = np.where(Radius>=r[i_r])
        X1 = np.where(Radius[X]<r[i_r+1])
        DMass = np.sum(dM_dt[X][X1]*Dt) #Mass between radius r and r + Dr
        p[i_r] = DMass / (Dr*Mass_tot)
    return p,r

#Function to compute probability density function (pdf) for plume radius 
def Compute_PDF_Shape4(M,dM_dt,Dt,Radius,r_min,r_max,Dr) : 
    r = np.arange(r_min,2.*r_max,Dr)# Radius
    p = np.zeros(len(r)) # p = pdf
    Mass_tot = np.sum(dM_dt*Dt) + M[0]
    for i_r in range(len(r)-1):
        X = np.where(Radius>=r[i_r])
        X1 = np.where(Radius[X]<r[i_r+1])
        DMass = np.sum(dM_dt[X][X1]*Dt[X][X1]) #Mass between radius r and r + Dr
        p[i_r] = DMass / (Dr*Mass_tot)
    return p,r




#Function that computes moment of order n for pdf p
def Compute_moment_n(p,r,Dr,n) : 
    Mn = np.sum(p*r**n*Dr)
    return Mn



#Computer characteristic length scale sigma_r and stretching
if Shape ==4:
    dM_dt = dM_dt[0:-1]
    Dt = Time[1:]-Time[:-1] 
    Mass = Mass[0:-1]
    Time = Time[0:-1]
    U    = U[0:-1]
    Plume_Radius = Plume_Radius[0:-1]
    
    [p,r]  = Compute_PDF_Shape4(Mass,dM_dt,Dt,Plume_Radius,r_min,r_eff,Dr)
else:
    #p = pdf for plume radius     
    [p,r]  = Compute_PDF(Mass,dM_dt,Dt,Plume_Radius,r_min,r_eff,Dr) #p = pdf for plume radius 
M2 = Compute_moment_n(p,r,Dr,2) #2nd moment of the pdf
sigma_r = gamma*M2**0.5 # Characteristic width (or radius) of the plume over the entire time window, in meters
sigma_dMdt = np.sum(dM_dt**2.*Dt)/M_tot #Mean accretion rate
sigma_l = 4./3.*r_eff**3./sigma_r**2 # Equivalent to r_max in Arnav thesis


Stretching_r = sigma_l/sigma_r # Stretching parameter
Stretching = sigma_l/r_eff # Stretching parameter

print('Characteristic plume radius / effective core radius = '+str(sigma_r/r_eff))
print('Stretching = '+str(Stretching))
print('Stretching = '+str(Stretching_r))
print('Mean dM/dt = '+str(sigma_dMdt))



if Clumps == 1 : 
    plt.figure(3)
    plt.plot(Time,Mass/M_tot)
    plt.xlabel('Time (s)')
    plt.ylabel('Mass/Total mass')
    plt.show()
    X = plt.ginput(1)
    time_between_clumps = X[0][0]
    #time_between_clumps = 30000

    junk = np.where(Time>time_between_clumps)
    i    = junk[0][0]
    Mass_fraction_clump1 = X[0][1]
    Mass_fraction_clump2 = 1. - Mass_fraction_clump1
    
    # Clump 1 
    #[p,r]  =               Compute_PDF_Shape4(Mass,dM_dt,Dt,Plume_Radius,r_min,r_eff,Dr, M_tot)
    [p_clump1,r_clump1]  = Compute_PDF_Shape4(Mass[0:i],dM_dt[0:i],Dt[0:i],Plume_Radius[0:i],r_min,r_eff,Dr, M_tot)
    M2_clump1 = Compute_moment_n(p_clump1,r_clump1,Dr,2) #2nd moment
    reff_clump1 = (Mass_fraction_clump1 * M_tot /(rho_m*4./3.*np.pi))**(1./3)
    sigma_r_clump1 = gamma*M2_clump1**0.5
    sigma_l_clump1 = 4./3.*r_eff**3.*Mass_fraction_clump1/sigma_r_clump1**2
    Stretching_clump1 = sigma_l_clump1/sigma_r_clump1
    Stretching_clump1 = sigma_l_clump1/reff_clump1
    print('Characteristic plume radius for clump 1 / effective radius = '+str(sigma_r_clump1/reff_clump1))
    print('Stretching of clump1 = '+str(Stretching_clump1))
    
    # Clump 2
    
    [p_clump2,r_clump2]  = Compute_PDF_Shape4(Mass[i+1:-1],dM_dt[i+1:-1],Dt[i+1:-1],Plume_Radius[i+1:-1],r_min,r_eff,Dr, M_tot)
    M2_clump2 = Compute_moment_n(p_clump2,r_clump2,Dr,2) #2nd moment
    reff_clump2 = (Mass_fraction_clump2 * M_tot /(rho_m*4./3.*np.pi))**(1./3)
    sigma_r_clump2 = gamma*M2_clump2**0.5
    sigma_l_clump2 = 4./3.*r_eff**3.*Mass_fraction_clump2/sigma_r_clump2**2
    Stretching_clump2 = sigma_l_clump2/sigma_r_clump2
    Stretching_clump2 = sigma_l_clump2/reff_clump2

    print('Characteristic plume radius for clump 2 / effective radius = '+str(sigma_r_clump2/reff_clump2))
    print('Stretching of clump2 = '+str(Stretching_clump2))
   # print('Stretching of clump2 = '+str(Stretching_clump2))
#Plots

plt.figure(1)
plt.plot(Time,Mass/M_tot,'o-',label='Mass/M_tot')
plt.plot(Time,U/np.max(U),label='Velocity / Max velocity')
#plt.plot(Time,dM_dt/(M_tot/t_max),label='dM/dt')
plt.plot(Time,Plume_Radius/r_eff,label='Radius/r_max')
plt.xlabel('Time (s)')
plt.legend()
plt.show()
#plt.close()


#X = np.where(p>0)
plt.figure(2)
plt.plot(r,p)
plt.xlabel('Plume radius (m)')
plt.ylabel('Probability density function')
plt.show()
#plt.close()

