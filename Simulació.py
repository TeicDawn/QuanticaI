#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Codi creat per Martí Berenguer, optimitzat per Adrià Labay, modificat per Roger Balsach, Àlex Gómez i Martí Gimeno.

from __future__ import division, print_function

import numpy as np
import math
import matplotlib.pyplot as plt
import time
import os

current_milli_time = lambda: int(round(time.time() * 1000))

k = 5.12665			  	# Factor d'unitats ( eV^(-1/2)*nm^(-1) )
hbar = 6.582e-4			# h barra (eV · ps)

L = 10.0				# Mitja longitud de la caixa (nm)
xi = -L/2			   	# Posicio inicial del paquet (nm)
l = 0.6				   	# Mitja amplada de la barrera (nm)
sigmax = 0.6			# Incertesa inicial del paquet (nm)

T = 0.5				   	# Energia cinetica (eV)
V0 = 0.0				# Barrera de potencial (eV)

Nx = 1024				# Numero de particions en x
dx = 2 * L / Nx 	   	# Pas en x
N1, N2, N3 = int((L - l) / dx), int(2 * l/ dx), int((L - l) / dx) # Numero de particions abans, dins i despres de la barrera o del pou

dt = 0.0005			   	# Pas de temps (ps)
Nt = 2000				# Numero de passos de temps

N = 2 * 128				# Numero d'estats propis que utilitzarem
dE = 0.0001		  		# Precisio en energies (eV)


FILE_ENERGIES = 'energies_{}.txt'.format([L,l,V0,N,dE])
FILE_PHI = 'phi_{}.txt'.format([L,l,V0,N])



###################################################################################
###################################################################################
##################											   	 ##################
##################					PART I					 	 ##################
##################   resolució de les equacions trascendentals 	 ##################
##################		        per les energies			  	 ##################
##################											     ##################
###################################################################################
###################################################################################

# Equacions trascendentals pels casos parell i senar
# i els diferents valors de l'energia:
#		 0 < E < V0  --> *_l
#		 V0 < E, E>0 --> *_g
#        V0 < E < 0  --> *_n
#
def _even_l(E):
	return (math.sqrt(V0-E))*math.tanh(k*(math.sqrt(V0-E))*l)*math.sin(k*(math.sqrt(E))*(L-l)) + \
		(math.sqrt(E))*math.cos(k*(math.sqrt(E))*(L-l))

def _even_g(E):
	return (math.sqrt(E-V0))*math.sin(k*(math.sqrt(E-V0))*l)*math.sin(k*(math.sqrt(E))*(L-l)) - \
		(math.sqrt(E))*math.cos(k*(math.sqrt(E-V0))*l)*math.cos(k*(math.sqrt(E))*(L-l))

def _even_n(E):
    return math.sqrt(E-V0)*math.tanh(k*math.sqrt(-E)*(l-L))*math.sin(k*math.sqrt(E-V0)*l) + \
        math.sqrt(-E)*math.cos(k*math.sqrt(E-V0)*l)

def _odd_l(E):
	return (math.sqrt(V0-E))*math.sin(k*(math.sqrt(E))*(L-l)) + \
		(math.sqrt(E))*math.tanh(k*(math.sqrt(V0-E))*l)*math.cos(k*(math.sqrt(E))*(L-l))

def _odd_g(E):
	return (math.sqrt(E-V0))*math.cos(k*(math.sqrt(E-V0))*l)*math.sin(k*(math.sqrt(E))*(L-l)) + \
		(math.sqrt(E))*math.sin(k*(math.sqrt(E-V0))*l)*math.cos(k*(math.sqrt(E))*(L-l))
    
def _odd_n(E):
    return math.sqrt(E-V0)*math.tanh(k*math.sqrt(-E)*(l-L))*math.cos(k*math.sqrt(E-V0)*l) - \
        math.sqrt(-E)*math.sin(k*math.sqrt(E-V0)*l)

# Funcions per guardar les energies en un arxiu de text i per llegir-les d'aquest

def save_energies(E):
	with open(FILE_ENERGIES, 'w') as outf:
		for j in range(len(E)):
			outf.write('%d\t%.6g\n' % (j, E[j])) # Compte xifres significatives.

def read_energies(file_name):
	Ep = []
	with open(file_name) as f:
		for line in f:
			Ep.append(float(line.split('\t')[1].strip()))
	return np.array(Ep)

# Funcio per trobar els valors propis de les energies

def find_roots():
	E0 = min(0,V0)
	E = E0 + dE
    
	Ep = [] # energia dels estats
	j = 0 # numero d'estats

    # Per diferenciar el cas del pou al de la barrera
	if V0>0:
		last_even, last_odd = _even_l(0), _odd_l(0)
	elif V0<0:    
		last_even, last_odd = _even_n(E0), _odd_n(E0)

	print('Start root finding...', end=' ')
	start = current_milli_time()

	while E < V0 and j < N:
		e, o = _even_l(E), _odd_l(E)

		if e * last_even < 0: # canvi de signe, arrel trobada
			Ep.append(E)
			j+=1
        
		if o * last_odd < 0: 
			Ep.append(E)
			j+=1

		last_even, last_odd = e, o
		E += dE

	while E<0 and j < N:
		e, o = _even_n(E), _odd_n(E)

		if e * last_even < 0: # canvi de signe, arrel trobada
			Ep.append(E)
			j+=1

		if o * last_odd < 0: # canvi de signe, arrel trobada
			Ep.append(E)
			j+=1

		last_even, last_odd = e, o
		E += dE

	last_even, last_odd = _even_g(max(0,V0)), _odd_g(max(0,V0))
	while j < N:
		e, o = _even_g(E), _odd_g(E)

		if e * last_even < 0: # canvi de signe, arrel trobada
			Ep.append(E)
			j+=1

		if o * last_odd < 0: # canvi de signe, arrel trobada
			Ep.append(E)
			j+=1

		last_even, last_odd = e, o
		E += dE

	print('OK (%.2f s)' % ((current_milli_time() - start) / 1000))

	return sorted(Ep)

# Evalua les energies si no estan desades. Si ja estan calculades, les llegeix del arxiu
if os.path.exists(FILE_ENERGIES):
	print('Reading energies')
	Ep = read_energies(FILE_ENERGIES)
else:
    print('Evaluating energies')
    Ep = find_roots()
    save_energies(Ep)

N=len(Ep)



###################################################################################
###################################################################################
####################									   ########################
####################				 PART II			   ########################
####################   Definicio de les funcions propies   ########################
####################									   ########################
###################################################################################
###################################################################################

# Definició de les funcions d'ona pel cas parell i pel senar

def _phi_even_l(reg, E, x):
	if reg == 1:
		return np.sin(k*(np.sqrt(E))*(x+L))
	elif reg == 2:
		return np.sin(k*(np.sqrt(E))*(L-l))*np.cosh(k*(np.sqrt(V0-E))*x)/(np.cosh(k*(np.sqrt(V0-E))*l))
	elif reg == 3:
		return -np.sin(k*(np.sqrt(E))*(x-L))

def _phi_even_g(reg, E, x):
	if reg == 1:
		return np.sin(k*(np.sqrt(E))*(x+L))
	elif reg == 2:
		return np.sin(k*(np.sqrt(E))*(L-l))*np.cos(k*(np.sqrt(E-V0))*x)/(np.cos(k*(np.sqrt(E-V0))*l))
	elif reg == 3:
		return -np.sin(k*(np.sqrt(E))*(x-L))
    
def _phi_even_n(reg, E, x):
	if reg == 1:
		return np.sinh(k*np.sqrt(-E)*(x+L))
	elif reg == 2:
		return np.sinh(k*(np.sqrt(-E))*(L-l))*np.cos(k*(np.sqrt(E-V0))*x)/(np.cos(k*(np.sqrt(E-V0))*l))
	elif reg == 3:
		return -np.sinh(k*(np.sqrt(-E))*(x-L))

def _phi_odd_l(reg, E, x):
	if reg == 1:
		return np.sin(k*(np.sqrt(E))*(x+L))
	elif reg == 2:
		return -np.sin(k*(np.sqrt(E))*(L-l))*np.sinh(k*(np.sqrt(V0-E))*x)/(np.sinh(k*(np.sqrt(V0-E))*l))
	elif reg == 3:
		return np.sin(k*(np.sqrt(E))*(x-L))

def _phi_odd_g(reg, E, x):
	if reg == 1:
		return np.sin(k*(np.sqrt(E))*(x+L))
	elif reg == 2:
		return -np.sin(k*(np.sqrt(E))*(L-l))*np.sin(k*(np.sqrt(E-V0))*x)/(np.sin(k*(np.sqrt(E-V0))*l))
	elif reg == 3:
		return np.sin(k*(np.sqrt(E))*(x-L))
    
def _phi_odd_n(reg, E, x):
	if reg == 1:
		return np.sinh(k*(np.sqrt(-E))*(x+L))
	elif reg == 2:
		return -np.sinh(k*(np.sqrt(-E))*(L-l))*np.sin(k*(np.sqrt(E-V0))*x)/(np.sin(k*(np.sqrt(E-V0))*l))
	elif reg == 3:
		return np.sinh(k*(np.sqrt(-E))*(x-L))

def phi_odd(reg, E, x):
	if E<V0:
		return _phi_odd_l(reg, E, x)
	elif E>0:
		return _phi_odd_g(reg, E, x)
	else:
		return _phi_odd_n(reg, E, x)

def phi_even(reg, E, x):
	if E<V0:
		return _phi_even_l(reg, E, x)
	elif E>0:
		return _phi_even_g(reg, E, x)
	else:
		return _phi_even_n(reg, E, x)


def evaluate_wave_function(Ep):
	# matriu que contindrà totes les funcions propies, cadascuna en una fila:
	PHI = np.zeros((N, Nx))

	# defineix les tres regions diferents de x
	x1, x2, x3 = np.linspace(-L, -l, N1), np.linspace(-l, l, N2), np.linspace(l, L, N3) 

	for j in range(N): # bucle en tots els estats
		E = Ep[j] 
		if j % 2 == 0:
			PHI[j,:N1] = phi_even(1, E, x1)
			PHI[j,N1:N2 + N1] = phi_even(2, E, x2)
			PHI[j,N1 + N2:N3 + N2 + N1] = phi_even(3, E, x3)
		else:
			PHI[j,:N1] = phi_odd(1, E, x1)
			PHI[j,N1:N2 + N1] = phi_odd(2, E, x2)
			PHI[j,N1 + N2:N3 + N2 + N1] = phi_odd(3, E, x3)

		# normalitzacio de la funció d'ona
		PHI[j] /= np.sqrt(np.sum(PHI[j] * PHI[j]))

	return PHI

start=current_milli_time()
print('Evaluating wave functions')
phi = evaluate_wave_function(Ep)/np.sqrt(dx)

print('OK (%.2f s)' % ((current_milli_time() - start) / 1000))

# plots d'alguns exemples
x = np.linspace(-L, L, Nx)
plt.figure()
for j in [0]: # representa graficament les funcions d'ona corresponents als estats introduits a la llista
	plt.plot(x, phi[j], label=str(j), lw=0.8)
plt.legend()
plt.show()



##################################################################################
##################################################################################
##################											  	##################
##################				   PART III						##################
##################		definicio de la funcio gaussiana		##################
##################			  i aplicacio del kick				##################
##################											  	##################
##################################################################################
##################################################################################
import scipy.integrate as integrate

# Definim ara una funcio gaussiana

def gaussiana(x):
	return np.exp( - (x - xi)**2 / (4 * sigmax**2) )

# La normalitzem de -L a +L
integral = integrate.quad(lambda x: (gaussiana(x))**2, -L, L)
Norm = np.sqrt((integral[0]))

# I ens definim un vector que li direm gauss, on hi posarem els valors de la gaussiana
# a cada posicio de la caixa

gauss = gaussiana(x) / Norm


# Aplicacio del kick, i separacio de la gaussiana en part real i part imaginaria

gaussr = np.cos(k*np.sqrt(T)*x)*gauss
gaussi = np.sin(k*np.sqrt(T)*x)*gauss

# Plots de la part real, la part imaginaria i el modul al quadrat de la gaussiana
plt.figure()
plt.plot(x, gauss, 'r')
plt.plot(x,gaussr, "b")
plt.plot(x,gaussi, "g")
plt.show()



##################################################################################
##################################################################################
##################											  	##################
##################				    PART IV						##################
##################		   descomposició en base pròpia	    	##################
##################											  	##################
##################################################################################
##################################################################################

# càlcul coeficients a t = 0.
# reals.
qr = np.zeros((N, Nx))
Cr = np.zeros((N, Nt))

for i in range(N):
    for j in range(Nx):
        qr[i][j]=phi[i][j]*gaussr[j]
        Cr[i][0]=Cr[i][0]+qr[i][j]*dx

# imaginaris  
qi = np.zeros((N, Nx))
Ci = np.zeros((N, Nt))

for i in range(N):
    for j in range(Nx):
        qi[i][j] = phi[i][j]*gaussi[j]
        Ci[i][0] = Ci[i][0]+qi[i][j]*dx
print("primer checkpoint")
# error per haver fet servir sin\cos.
Dif = np.zeros((Nx))

for i in range(N):
    Dif[i] = math.sqrt(gaussr[i]**2+gaussi[i]**2) - math.sqrt(gauss[i]**2)

# print(np.amax(Dif))
# DifMax = 1.1102230246251565 * 10^(-16)

# comrpovació gaussiana.
# real.
gaussrF = np.zeros((N, Nx))
gaussr2 = np.zeros((Nx))
gaussrD = np.zeros((Nx))

for i in range(N):
    for j in range(Nx):
        gaussrF[i][j] = phi[i][j]*Cr[i][0]
        gaussr2[j] = gaussr2[j]+gaussrF[i][j]

for j in range(Nx):
        gaussrD[j] = gaussr2[j]-gaussr[j]

# imaginària.
gaussiF = np.zeros((N, Nx))
gaussi2 = np.zeros((Nx))
gaussiD = np.zeros((Nx))

for i in range(N):
    for j in range(Nx):
        gaussiF[i][j] = phi[i][j]*Ci[i][0]
        gaussi2[j] = gaussi2[j]+gaussiF[i][j]

for j in range(Nx):
        gaussiD[j] = gaussi2[j]-gaussi[j]
        
print(np.amax(gaussrD),np.amax(gaussiD))
# 0.0015287761883271181 0.0010494002740158714

# comprovació probabilitat.
#Prob = 0
#for i in range(N):
#    Prob = Prob + Cr[i][0]**2 + Ci[i][0]**2

# print(Prob)
# 0.9973613700338806



##################################################################################
##################################################################################
##################											  	##################
##################				    PART V						##################
##################             evolució temporal                ##################  
##################											  	##################
##################################################################################
##################################################################################

omega = np.zeros((N))

for i in range(N):
    omega[i]=Ep[i]/hbar
    
for i in range(N):
    for j in range(Nt):
        Cr[i][j] = math.cos(omega[i]*j*dt)*Cr[i][0] + math.sin(omega[i]*j*dt)*Ci[i][0]
        Ci[i][j] = math.cos(omega[i]*j*dt)*Ci[i][0] - math.sin(omega[i]*j*dt)*Cr[i][0]

# real.
GaussrXT = np.zeros((Nt, Nx))

for z in range(Nt):
    waveFunctionF = np.zeros((N, Nx))
    waveFunction = np.zeros((Nx))
    for i in range(N):
        for j in range(Nx):
            waveFunctionF[i][j] = phi[i][j]*Cr[i][z]
            waveFunction[j] = waveFunction[j] + waveFunctionF[i][j]
    
    for j in range(Nx):
        GaussrXT[z][j] = waveFunction[j]

# imaginària.
GaussiXT = np.zeros((Nt, Nx))

for z in range(Nt):
    waveFunctionF = np.zeros((N, Nx))
    waveFunction = np.zeros((Nx))
    for i in range(N):
        for j in range(Nx):
            waveFunctionF[i][j] = phi[i][j]*Ci[i][z]
            waveFunction[j] = waveFunction[j]+waveFunctionF[i][j]
    
    for j in range(Nx):
        GaussiXT[z][j] = waveFunction[j]

# probabilitat.
PROB = np.zeros((Nt, Nx))

for z in range(Nt):
    for j in range(Nx):
        PROB[z][j] = GaussrXT[z][j]**2 + GaussiXT[z][j]**2


"""Aquest és el codi per fer plots a temps real. La matriu PROB ha de tenir a cada fila un
instant de temps, i per a cada instant de temps, els valors del paquet en cada posició en
columnes. El programa guarda en el directori que especifiqueu (que existeixi) fotos ordenades
de la funció d'ona a cada instant de temps."""


for i in range(Nt): # Nombre de passos de temps que vulguis
    plt.figure("fig"+str(i))
    plt.axis([-L, L, -0.1, 1])
    plt.plot((-l,-l), (0,V0), 'r-', linewidth=3)
    plt.plot((l,l), (0,V0), 'r-', linewidth=3)
#    plt.plot((-l,l), (V0,V0), 'r-', linewidth=3)
    plt.fill([-l,l,l,-l], [0,0,V0,V0], 'r', alpha=0.2)   
    plt.plot(x,PROB[i,:],'k',linewidth=0.8)
    plt.savefig("C:\\Users\TeicD\Desktop\EsaSimuQueMolaV1\\fig"+str(i)) # Directori que existeixi + nombre de cada imatge (preferiblement sense espais ni accents ni ñ)
    plt.close() # Tancar cada figura per no col·lapsar el Spyder
