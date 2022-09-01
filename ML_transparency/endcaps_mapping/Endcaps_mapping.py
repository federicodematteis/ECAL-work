import math 
import numpy as np
import matplotlib.pyplot as plt

#the following script gives a approximative mapping of ecal endcaps i-Rings as a function of pseudorapidity

#angolo finale Endcaps
theta_i = 5.7
#angolo iniziale Endcaps
theta_f = 25.2
#pseudorapidità per l' ultimo ring nelle Endcaps
eta_0 = 2.98

#angular resolution of 0.5 degrees
delta_theta = (theta_i-theta_f)/39

Theta=[]
Eta=[]

#divido l'angolo totale in 39 sezioni (anelli concentrici)
for i in range (0,38)

	#get angle 
	theta = theta_0 + i * delta_theta
	#get pseudorapidity
	eta = (-np.log(np.tan(theta/2)))
	Theta = np.append(Theta, theta)
	Eta = np.append(Eta, eta)
	
print('theta - eta')
print(Theta,Eta)
print(eta)

plt.plot(theta,eta, color='blue', linewidht=0.2, marker='p')
plt.grid()
plt.xlabel('theta')
plt.ylabel('eta')
plt.show()


