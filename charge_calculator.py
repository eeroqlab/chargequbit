from matplotlib import colors
from findiff import FinDiff
import sys

from zeroheliumkit.src.settings import *
from zeroheliumkit.fem.fieldreader import FieldAnalyzer
from scipy.sparse import diags
from scipy.sparse.linalg import eigs


class ChargeAnalyzer(FieldAnalyzer):

    def __init__(self, *args):
        super().__init__(*args)
        # constants
        self.hbar = 1.05457182e-34       # [J*s]             plank's constant          
        self.epsilon_0 = 8.854e-12       # [F/m]             vacuum permittivity
        self.epsilon_He = 1.057          # [1]               dielectric constant of helium
        self.qe = 1.602e-19              # [Coulombs]        electron charge
        self.me = 9.1093837e-31          # [kg]              electron mass
        self.g = 9.81                    # [m/s^2]           gravity constant
        self.alpha = 3.58e-4             # [N/p]             surface tension
        self.rho = 146                   # [kg/m^3]          helium density
        self.kb = 1.3806e-23             # [J/K]             boltzman constant

        # length scale / um
        self.l0 = 1e-6

        # energy scale
        self.Escale = self.hbar**2/2/self.me/self.l0**2
    

    def plot_potential2D_electrons(self, x: list, y: list, potential: list, ax=None, electronPotential=None,showPotential=True, **kwargs):
        if ax is None:
            ax = _default_ax()
        
        vmin = np.amin(potential)
        vmax = np.amax(potential)
        if electronPotential is not None:
            norm = colors.TwoSlopeNorm(vcenter=electronPotential,vmin=vmin, vmax=vmax)
            if showPotential:
                im = ax.contourf(x, y, potential, 17, norm=norm, **kwargs)
            ax.contour(x, y, potential, [electronPotential], linestyles='dashed', colors=BLACK)
        else:
            norm = colors.TwoSlopeNorm(vcenter = np.mean([vmin,vmax]), vmin=vmin, vmax=vmax)
            im = ax.contourf(x, y, potential, 17, norm=norm, **kwargs)
          
    def shift_potential(self, x: list, y: list, potential: list):
        potential = np.transpose(potential)
        potentialShift = np.amax(potential)
        y0, x0 = np.where(potential==potentialShift)
        shiftedPotential = potential - potentialShift
        
        return x[x0][0], y[y0][0], shiftedPotential
    
    def induced_charge(self,x: list,y: list,potential: list,gateCouplingConstant: list,depth: float,electronPotential=None,electronNumber=None,verbose=False) -> float:
        if electronNumber is not None:
            electronPotential = self.electron_potential_from_number(x, y, potential, electronNumber, depth)
        if electronPotential is None:
            print('Must supply either an electron potential or a valid electron number!')
            return 0
        
        gateCouplingConstant = np.transpose(gateCouplingConstant)
        potential = np.transpose(potential)
        electronCharge = 1.6e-19
        ep0 = 8.854e-12
        epHe = 1.057
        dA = (x[1] - x[0]) * 1e-6 * (y[1] - y[0]) * 1e-6
        A = 0
        N = 0
        Q = 0
        xs, ys = np.where(potential > electronPotential)
        for di in range(len(xs)):
            vi  = potential[xs[di], ys[di]] 
            ni  = (vi - electronPotential) * (ep0 * epHe)/(electronCharge)/(depth)
            A+= dA
            N+= ni*dA
            Q+= electronCharge * gateCouplingConstant[xs[di], ys[di]] * ni * dA
        if verbose:
            print(f'Area           = {A*1e12:.2f} um^2')
            print(f'Number         = {N:.2f} Electrons')
            print(f'Density        = {N/A*1e-4:.2e} Electrons per cm^2')
            print(f'Potential      = {electronPotential:.2f} V')
            print(f'Induced Charge = {Q/electronCharge:.2f} e')
        
        if electronNumber is not None:
            return Q, electronPotential
        else:
            return Q, None
    
    def barrier_check(self,x: list,y: list,potential: list,electronPotential: float,barrierAxis='x',barrierValue=0):
        barrierAxis='x'
        barrierValue=0
        if barrierAxis.lower() == 'x':
            yind = np.argmin(abs(y-barrierValue))
            barrierLine = potential[yind,:]
            barrierRange = x
        else:
            xind = np.argmin(abs(x-barrierValue))
            barrierLine = potential[:,xind]
            barrierRange = y
        blocked = np.max(barrierLine)<electronPotential
        return blocked
            
    def get_trapfreq_Schrodinger(self, 
                                 X_trap: list, 
                                 Y_trap: list, 
                                 Phi_trap: list, 
                                 extract_states=False):
        '''
        calculates motional frequency from a Shrodinger equation
        '''
        dx = X_trap[1] - X_trap[0]
        dy = Y_trap[1] - Y_trap[0]

        laplace = FinDiff(0, dx, 2) + FinDiff(1, dy, 2)
        hamiltonian = -laplace.matrix(Phi_trap.shape) - np.float64(self.qe/self.Escale)*diags(Phi_trap.reshape(-1))
        energies_unsorted, states_unsorted = eigs(hamiltonian, k=6, which='SR')
        energies_unsorted = energies_unsorted.real
        if extract_states:
            sorted_indices = np.argsort(energies_unsorted)
            energies = np.array([energies_unsorted[i] for i in sorted_indices]) * self.Escale
            states = [states_unsorted[:,i] for i in sorted_indices]
            frequencies = (energies - energies[0]) 
            return energies, states, states_unsorted
        else:
            energies = np.array(energies_unsorted.sort()) * self.Escale
            return frequencies[1:]
    
    def electron_potential_from_number(self, x: list, y: list, potential: list, N: int, depth: float, returnArea = False):
        electronCharge = 1.6e-19
        ep0 = 8.854e-12
        epHe = 1.057
        errorN = 5e-2
        errorPotential = 1e-6
        minPotential = np.amin(potential)
        maxPotential = np.amax(potential)
        guessPotential = np.mean([minPotential, maxPotential])
        while True:
            guessN = 0
            A = 0
            dA = (x[1] - x[0]) * 1e-6 * (y[1] - y[0]) * 1e-6
            xs, ys = np.where(potential > guessPotential)
            for di in range(len(xs)):
                vi  = potential[xs[di], ys[di]] 
                ni  = (vi - guessPotential) * (ep0 * epHe)/(electronCharge)/(depth)
                A     += dA
                guessN+= ni * dA
            if abs(guessN - N) < errorN:
                if returnArea:
                    return A, guessPotential
                else:
                    return guessPotential
            if abs(guessPotential - np.amin(potential)) < errorPotential:
                print('Trap cannot hold this many electrons!')
                if returnArea:
                    return None,None
                else:
                    return None
            if guessN < N:
                maxPotential = guessPotential
                guessPotential = np.mean([guessPotential, minPotential])
            else:
                minPotential = guessPotential
                guessPotential = np.mean([guessPotential, maxPotential])
            if minPotential == guessPotential or maxPotential == guessPotential:
                print('Cannot get elecron number within error: N = ',guessN,', error = ', abs(guessN - N))
                if returnArea:
                    return A, guessN
                else:
                    return guessN