import math
import numpy as np
from matplotlib import pyplot as plt

dmSquared_21 = 7.53e-5
dmSquared_32 = 2.44e-3
dmSquared_31 = dmSquared_21 + dmSquared_32

theta_12 = 33.36*math.pi/180
theta_23 = 40.0*math.pi/180
theta_13 = 8.66*math.pi/180

c12 = math.cos(theta_12)
s12 = math.sin(theta_12)

c13 = math.cos(theta_13)
s13 = math.sin(theta_13)

c23 = math.cos(theta_23)
s23 = math.sin(theta_23)

U = [[c12*c13, s12*c13, s13],
     [-s12*c23-c12*s23*s13, c12*c23-s12*s23*s13, s23*c13],
     [s12*s23-c12*c23*s13, -c12*s23-s12*c23*s13, c23*c13]]

#UH = np.transpose(U)
#print U
#print UH

#print np.dot(U, UH)

#print 'dmSquared_21 = ', dmSquared_21
#print 'dmSquared_31 = ', dmSquared_31
#print 'dmSquared_32 = ', dmSquared_32



dmSquares = [[0, dmSquared_21, dmSquared_31],
             [dmSquared_21, 0, dmSquared_32],
             [dmSquared_31, dmSquared_32, 0]]

class nu:
    """
    Holds a name and an index corresponding to matrices...
    """
    def __init__(self, index, label):
        self.index = index
        self.name = label    


def luminosityCalcs():
    aveEnergy = 10
    """
    luminosity in ergs per second
    1 erg = 1e-7 joules
    1 joule = 6.24150934e18 electron volts
    1 erg = 6.24150934e11 electron volts
    1 erg = 6.24150934e5 MeV

    So 1000e50 ergs / sec = 6241.50934e55 MeV / sec
    But average neutrino energy = 10 MeV / neutrino
    => 1000e50 ergs / sec = 6.24e57 neutrinos / sec
    """

    nu_e = nu(0, r'$\nu_e$')
    nu_e_bar = nu(0, r'$\bar{\nu_e}$')
    nu_x = nu(0, r'$\nu_x$')
    aveEnergy = []
    lum_nu_e = [0,500,5000,1000,560,510,400,270,170,120,110,105,100,95,90,85,80,75,70]
    lum_nu_e_bar = [0,0,0,120,450,600,460,300,180,130,120,115,110,105,100,95,90,85,80]
    lum_nu_x = [0,0,200,260,410,490,400,270,170,120,110,105,100,95,90,85,80,75,70]
    E_nu_e = [10,9,12,8.8,9.2,11,12,12.2,12.2,12.7,12.8,12.8,12.8,12.8,12.9,12.9,12.9,12.9,12.9]
    E_nu_e_bar = [10,11,11,11.2,11.8,13.8,14.8,15.2,15.4,15.8,16.0,16.0,16.0,16.0,16.1,16.2,16.3,16.4,16.5]
    E_nu_x = [14,15,15,15,15.6,17.8,18.7,19.3,20.0,20.2,20.6,21.4,22.2,23.0,23.3,23.6,23.9,24.2,24.5]
    time = [0,0.03,0.04,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5]
    plt.figure()
    plt.plot(time, lum_nu_e, label = nu_e.name)
    plt.plot(time, lum_nu_e_bar, label = nu_e_bar.name)
    plt.plot(time, lum_nu_x, '-', label = nu_x.name)
    plt.title('Supernova neutrino luminosity as a function of time')
    plt.ylabel('Luminosity ($10^{50}$ erg / sec)')
    plt.xlabel('Time (s)')
    plt.legend()

    plt.figure()
    plt.plot(time, E_nu_e, label = nu_e.name)
    plt.plot(time, E_nu_e_bar, label = nu_e_bar.name)
    plt.plot(time, E_nu_x, label = nu_x.name)
    plt.title(r'Average $\nu$ energy as a function of time')
    plt.ylabel('Energy (MeV)')
    plt.xlabel('Time (s)')
    ax = plt.gca()
    ax.set_ylim(5, 27)
    plt.legend()

    erg_to_MeV = 6241.50934e5
    num_prod_e = [erg_to_MeV*lum/E for lum, E in zip(lum_nu_e, E_nu_e)]
    num_prod_e_bar = [erg_to_MeV*lum/E for lum, E in zip(lum_nu_e_bar, E_nu_e_bar)]
    num_prod_x = [erg_to_MeV*lum/E for lum, E in zip(lum_nu_x, E_nu_x)]
    plt.figure()
    plt.plot(time, num_prod_e, label = nu_e.name)
    plt.plot(time, num_prod_e_bar, label = nu_e_bar.name)
    plt.plot(time, num_prod_x, label = nu_x.name)    
    plt.title(r'Supernova $\nu$ emission as a function of time')
    plt.ylabel(r'Number of $\nu$s')
    plt.xlabel('Time (s)')
    ax = plt.gca()
    ax.set_yscale('log')
    plt.legend()

def main():

    plt.ion()
    #luminosityCalcs()
    oldmain()
    #mean = 10
    #sigma = 5
    #vals = [generateEnergyDistribution(mean, sigma) for i in xrange(100000)]
    #plt.hist(vals, bins=100)
        
def oldmain():
    """"
    L/E needs to be (km)/(GeV)
    Or (m)/(MeV)
    """
    plt.ion()

    d_earth_sun = 1.496e11 # meters
    r_sun = 696.3e6 # meters
    r_earth = 6.371e3 # meters

    #E = 10 # MeV
    #E = 0.3 # MeV
    E = 15 # MeV
    LNominalMeters = 1.5428e21 #meters

    nu_e = nu(0, r'$\nu_e$')
    nu_mu = nu(1, r'$\nu_{\mu}$')
    nu_tau = nu(2, r'$\nu_{\tau}$')
    
    #Losc = 983459.230639132 #156522.39788557024 #49822.626656169865
    #Losc = 100*math.pi*E/(1.27*min(dmSquared_21, dmSquared_32))
    #Losc = 1.54283879e+21 # 50 kpc
    Losc = 1e10
    
    #plotOscillations(nu_e, Losc, E)#, savePlot=True)
    #plotOscillations(nu_mu, Losc, E)
    #plotOscillations(nu_tau, Losc, E)
    
    plotOscillations2(nu_e, Losc, E, deltaE = E*1e-2)
    #plotOscillations2(nu_mu, Losc, E, deltaE = E*1e-3)    

    p_e_e = transitionProb(d_earth_sun, E, nu_e, nu_e)    
    p_e_mu = transitionProb(d_earth_sun, E, nu_e, nu_mu)
    p_e_tau = transitionProb(d_earth_sun, E, nu_e, nu_tau)

    print p_e_e, p_e_mu, p_e_tau, sum([p_e_e, p_e_mu, p_e_tau])

    p_e_e = transitionProb(d_earth_sun+r_earth, E, nu_e, nu_e)    
    p_e_mu = transitionProb(d_earth_sun+r_earth, E, nu_e, nu_mu)
    p_e_tau = transitionProb(d_earth_sun+r_earth, E, nu_e, nu_tau)

    print p_e_e, p_e_mu, p_e_tau, sum([p_e_e, p_e_mu, p_e_tau])




def generateEnergyDistribution(mean, sigma):
    """
    Starting with simple gaussian distribution, just for now"
    """
    x = np.random.standard_normal()
    x = sigma*x + mean
    return x
                        

def cumulativeMeanArray(arr):
    s = 0.0
    cumArr = []
    for val in arr:
        s += val
        cumArr.append(s)

    cumMeanArr = [c/(n+1) for c, n in zip(cumArr, xrange(len(cumArr)))]
    return cumMeanArr

def integrateOverDistance(arr, distance, dx):

    numPoints = distance/dx
    numPoints = int(numPoints)

    ringBuffer = np.zeros(numPoints)
    newArr = []
    ringSum = 0.0
    for index, val in enumerate(arr):
        index2 = index % numPoints

        normFactor = numPoints
        if index >= numPoints:
            ringSum -= ringBuffer[index2]
        else:
            normFactor = index + 1

        ringBuffer[index2] = val
        ringSum += ringBuffer[index2]
        #ringSum2 = sum(ringBuffer)
        #print index, index2, val, ringBuffer[index2], normFactor, ringSum, ringSum2
        
        newArr.append(ringSum/normFactor)

        #if index >= 2000:
        #    break
    return newArr

def plotOscillations2(nu_init, Lmax, E, deltaE = 0, drawPlot=True, savePlot=False):

    numPoints = 1000
    numEnergyPoints = 1000
    
    nu_e = nu(0, r'$\nu_e$')
    nu_mu = nu(1, r'$\nu_{\mu}$')
    nu_tau = nu(2, r'$\nu_{\tau}$')

    logLMax = math.log10(Lmax)
    Ls = [pow(10, i*logLMax/numPoints) for i in xrange(numPoints)]
    #print Ls
    #print Ls

    probs_e = [0 for L in Ls]
    probs_mu = [0 for L in Ls]
    probs_tau = [0 for L in Ls]
    
    Es = [generateEnergyDistribution(E, deltaE) for i in xrange(numEnergyPoints)]
    plt.figure()
    plt.hist(Es, bins=10)
    
    for Eval in Es:
        probs_e2 = [transitionProb(L, Eval, nu_init, nu_e) for L in Ls]
        probs_mu2 = [transitionProb(L, Eval, nu_init, nu_mu) for L in Ls]
        probs_tau2 = [transitionProb(L, Eval, nu_init, nu_tau) for L in Ls]
        probs_e = [a+b for a, b in zip(probs_e, probs_e2)]
        probs_mu = [a+b for a, b in zip(probs_mu, probs_mu2)]
        probs_tau = [a+b for a, b in zip(probs_tau, probs_tau2)]

    probs_e = [a/numEnergyPoints for a in probs_e]
    probs_mu = [a/numEnergyPoints for a in probs_mu]
    probs_tau = [a/numEnergyPoints for a in probs_tau]
        
    #probsSum = [a+b+c for a, b, c in zip(probs_e, probs_mu, probs_tau)]

    dx = Ls[1]-Ls[0]

    if drawPlot or True:
        plt.figure()
        plt.plot(Ls, probs_e, 'b', label = nu_init.name+r'$\rightarrow$'+nu_e.name)
        plt.plot(Ls, probs_mu, 'g', label = nu_init.name+r'$\rightarrow$'+nu_mu.name)
        plt.plot(Ls, probs_tau, 'r', label = nu_init.name+r'$\rightarrow$'+nu_tau.name)
        plt.title('Oscillation probablilties for a ' + nu_init.name + ' of energy ' + str(E) + ' MeV')
        plt.xlabel('Distance, L (m)')
        plt.ylabel('Oscillation probabililty')
        plt.ylim([-0.1, 1.1])
        plt.legend()
        ax = plt.gca()
        ax.set_xscale('log')
        if savePlot:
            plt.savefig('oscillation.eps', format='eps', dpi=1000)


    return Ls, probs_e, probs_mu, probs_tau



def plotOscillations(nu_init, Lmax, E, drawPlot=True, savePlot=False):

    numPoints = 100000
    nu_e = nu(0, r'$\nu_e$')
    nu_mu = nu(1, r'$\nu_{\mu}$')
    nu_tau = nu(2, r'$\nu_{\tau}$')

    Ls = [i*Lmax/numPoints for i in xrange(numPoints)]
    probs_e = [transitionProb(L, E, nu_init, nu_e) for L in Ls]
    probs_mu = [transitionProb(L, E, nu_init, nu_mu) for L in Ls]
    probs_tau = [transitionProb(L, E, nu_init, nu_tau) for L in Ls]
    #probsSum = [a+b+c for a, b, c in zip(probs_e, probs_mu, probs_tau)]

    #mean_e = cumulativeMeanArray(probs_e)
    #mean_mu = cumulativeMeanArray(probs_mu)
    #mean_tau = cumulativeMeanArray(probs_tau)
    dx = Ls[1]-Ls[0]
    #mean_e = integrateOverDistance(probs_e, 1e5, dx)
    #mean_e = integrateOverDistance(probs_e, 3*math.pi*E/(1.27*dmSquared_21), dx)
    #mean_mu = integrateOverDistance(probs_mu, 3*math.pi*E/(1.27*dmSquared_21), dx)
    #mean_tau = integrateOverDistance(probs_tau, 3*math.pi*E/(1.27*dmSquared_21), dx)
    mean_e = integrateOverDistance(probs_e, 1e5, dx)
    mean_mu = integrateOverDistance(probs_mu, 1e5, dx)
    mean_tau = integrateOverDistance(probs_tau, 1e5, dx)    

    if drawPlot:
        plt.figure()
        plt.plot(Ls, probs_e, 'b', label = nu_init.name+r'$\rightarrow$'+nu_e.name)
        plt.plot(Ls, probs_mu, 'g', label = nu_init.name+r'$\rightarrow$'+nu_mu.name)
        plt.plot(Ls, probs_tau, 'r', label = nu_init.name+r'$\rightarrow$'+nu_tau.name)
        #plt.plot(Ls, probsSum, 'cyan', label = 'Sum')
        plt.title('Oscillation probablilties for a ' + nu_init.name + ' of energy ' + str(E) + ' MeV')
        plt.xlabel('Distance, L (m)')
        plt.ylabel('Oscillation probabililty')
        plt.ylim([-0.1, 1.1])
        plt.legend()
        ax = plt.gca()
        #ax.set_xscale('log')
        if savePlot:
            plt.savefig('oscillation.eps', format='eps', dpi=1000)


        plt.figure()
        plt.plot(Ls, mean_e, 'b', label = nu_init.name+r'$\rightarrow$'+nu_e.name)
        plt.plot(Ls, mean_mu, 'g', label = nu_init.name+r'$\rightarrow$'+nu_mu.name)
        plt.plot(Ls, mean_tau, 'r', label = nu_init.name+r'$\rightarrow$'+nu_tau.name)
        #plt.plot(Ls, probsSum, 'cyan', label = 'Sum')
        plt.title('Oscillation probablilties integrated over $10^{5}$m for a ' + str(E) + ' MeV ' + nu_init.name)
        plt.xlabel('Distance, L (m)')
        plt.ylabel('Oscillation probabililty')
        plt.ylim([-0.1, 1.1])
        plt.legend()
        ax = plt.gca()
        #ax.set_xscale('log')

        if savePlot:
            plt.savefig('integrated_oscillation.eps', format='eps', dpi=1000)
        
    return Ls, probs_e, probs_mu, probs_tau




    
def transitionProb(L, E, nu_a, nu_b):
    """
    L should be in km (m)
    E should be in GeV (MeV)
    """

    a = nu_a.index
    b = nu_b.index
    if a > 2 or b > 2 or a < 0 or b < 0:
        return 0

    
    prob = delta(a, b)
    for j in range(3):
        for i in range(j):

            oscPart = pow(math.sin(1.27*dmSquares[i][j]*L/E), 2)
            #print i, j, '...', i+1, j+1, dmSquares[i][j], math.asin(oscPart)*180/math.pi
            prob -= 4*U[a][i]*U[b][i]*U[a][j]*U[b][j]*oscPart

            #if a==0 and b==1:
            #    print i, j, L , E, oscPart
    

    return prob

def delta(a, b):
    d = 0
    if a==b:
        d = 1
    return d



if __name__ == '__main__':
    main()
