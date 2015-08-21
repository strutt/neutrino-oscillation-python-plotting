import math
import numpy as np
from matplotlib import pyplot as plt


dmSquared_21 = 7.53e-5
dmSquared_32 = 2.44e-3
dmSquared_31 = dmSquared_21 + dmSquared_32
#dmSquared_31 = dmSquared_32 - dmSquared_21

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
    

def main():
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
    E = 10 # MeV
    LNominalMeters = 1.5428e21 #meters

    nu_e = nu(0, r'$\nu_e$')
    nu_mu = nu(1, r'$\nu_{\mu}$')
    nu_tau = nu(2, r'$\nu_{\tau}$')

    #Losc = 983459.230639132 #156522.39788557024 #49822.626656169865
    Losc = 100*math.pi*E/(1.27*min(dmSquared_21, dmSquared_32))
    
    plotOscillations(nu_e, Losc, E)
    #plotOscillations(nu_mu, Losc, E)
    #plotOscillations(nu_tau, Losc, E)

    p_e_e = transitionProb(d_earth_sun, E, nu_e, nu_e)    
    p_e_mu = transitionProb(d_earth_sun, E, nu_e, nu_mu)
    p_e_tau = transitionProb(d_earth_sun, E, nu_e, nu_tau)

    print p_e_e, p_e_mu, p_e_tau, sum([p_e_e, p_e_mu, p_e_tau])

    p_e_e = transitionProb(d_earth_sun+r_earth, E, nu_e, nu_e)    
    p_e_mu = transitionProb(d_earth_sun+r_earth, E, nu_e, nu_mu)
    p_e_tau = transitionProb(d_earth_sun+r_earth, E, nu_e, nu_tau)

    print p_e_e, p_e_mu, p_e_tau, sum([p_e_e, p_e_mu, p_e_tau])


    
    
                        

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

def plotOscillations(nu_init, Lmax, E, drawPlot=True):

    numPoints = 100000
    nu_e = nu(0, r'$\nu_e$')
    nu_mu = nu(1, r'$\nu_{\mu}$')
    nu_tau = nu(2, r'$\nu_{\tau}$')

    Ls = [x*Lmax/numPoints for x in xrange(numPoints)]
    probs_e = [transitionProb(L, E, nu_init, nu_e) for L in Ls]
    probs_mu = [transitionProb(L, E, nu_init, nu_mu) for L in Ls]
    probs_tau = [transitionProb(L, E, nu_init, nu_tau) for L in Ls]
    #probsSum = [a+b+c for a, b, c in zip(probs_e, probs_mu, probs_tau)]

    #mean_e = cumulativeMeanArray(probs_e)
    #mean_mu = cumulativeMeanArray(probs_mu)
    #mean_tau = cumulativeMeanArray(probs_tau)
    dx = Ls[1]-Ls[0]
    mean_e = integrateOverDistance(probs_e, 10*1e5, dx)
    mean_mu = integrateOverDistance(probs_mu, 10*1e5, dx)
    mean_tau = integrateOverDistance(probs_tau, 10*1e5, dx)

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
        ax.set_xscale('log')    

        plt.figure()
        plt.plot(Ls, mean_e, 'b', label = nu_init.name+r'$\rightarrow$'+nu_e.name)
        plt.plot(Ls, mean_mu, 'g', label = nu_init.name+r'$\rightarrow$'+nu_mu.name)
        plt.plot(Ls, mean_tau, 'r', label = nu_init.name+r'$\rightarrow$'+nu_tau.name)
        #plt.plot(Ls, probsSum, 'cyan', label = 'Sum')
        plt.title('Integrated oscillation probablilties for a ' + nu_init.name + ' of energy ' + str(E) + ' MeV')
        plt.xlabel('Distance, L (m)')
        plt.ylabel('Oscillation probabililty')
        plt.ylim([-0.1, 1.1])
        plt.legend()
        ax = plt.gca()
        #ax.set_xscale('log')
        
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
