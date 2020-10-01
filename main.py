import numpy as np 
import math
import matplotlib.pyplot as plt
import sys
import scipy.interpolate as sp
import pandas as pd
from scipy.special import erf, erfc
from scipy import ndimage	
class superradiance:
	
	def __init__(self):
		self.newton = 6.7071186*10**-57
		self.solar_mass = 2.*10**30*5.6095886*10**35
		self.planck = 2.435*10**18


class spinzero:
	
	def __init__(self):
		self.newton = 6.7071186*10**-57
		self.solar_mass = 2.*10**30*5.6095886*10**35

	def normalised_outer_horizon(self,spin):	
		return 1 + np.sqrt(1-spin**2)

	def numerical_constant(self,n,l):
		
		a = (2.**(4.*l+1.)*math.factorial(n+l))/(n**(2.*l+4)*math.factorial(n-l-1.))
		b = ((math.factorial(l))/(math.factorial(2.*l)*math.factorial(2.*l+1.)))**2.

		
		
		return a*b
		
	def product_function(self,n,l,m,spin,mbh,mu):	
				
		k_vec = np.arange(1,l+1)
		product = [k**2*(1-spin**2)+(spin*m-2*self.normalised_outer_horizon(spin)*self.grav_radius(mbh)*self.energy_levels(mu,mbh,l,n))**2 for k in k_vec]
		return math.prod(product)

	def grav_radius(self,mbh):
		return self.newton*mbh*self.solar_mass

	def energy_levels(self,mu,mbh,l,n):
		return mu*(1-(self.grav_atom(mbh,mu)**2)/(2*(l+n+1)**2))	

	def grav_atom(self,mbh,mu):
			return mu*self.grav_radius(mbh)


	def product_factor(self,m,mbh,spin,mu,l,n):	
			return m*self.horizon_velocity(self.grav_radius(mbh),spin) - self.energy_levels(mu, mbh, l, n)	

	def horizon_velocity(self,rg,spin):	
		return (1./(2.*rg))*(spin/(1.+np.sqrt(1.-spin**2)))	


	def alpha_converter(self,x,y,z):
		if z == 'boson':
			value = x/(self.newton*self.solar_mass*y) 
		if z =='bh':
			value = x/(self.newton*self.solar_mass*y) 
		return value	
	

	def scalar_rate(self,mbh,spin,n,l,m,mu):
		
		
		rate  = 2.*self.normalised_outer_horizon(spin)*self.product_function(n,l,m, spin, mbh,mu)*self.product_factor(m,mbh,spin,mu,l,n)*self.grav_atom(mbh, mu)**(4.*l+5.)*self.numerical_constant(n, l)
			
		return rate
	
	
	def free_field_bounds(self,bh_name,confidence,solar):
		
		if solar == True:
			mu = np.load(f'results_scalar/mu_{bh_name}_sal.npy')
			ex = np.load(f'results_scalar/ex_{bh_name}_sal.npy')
		else:	
			mu = np.load(f'SCALAR/mu_SCALAR_{bh_name}_SUPER_sal.npy')
			ex = np.load(f'SCALAR/ex_SCALAR_{bh_name}_SUPER_sal.npy')

		yreduced = np.array(ex) - confidence
		x_sm = np.array(mu)
		y_sm = np.array(ex)
		x_smooth = np.linspace(x_sm.min(), x_sm.max(), 2000)

		sigma = 0.75
		x_g1d = ndimage.gaussian_filter1d(x_sm, sigma)
		y_g1d = ndimage.gaussian_filter1d(y_sm, sigma)
		
		try:
			idx = np.argwhere(np.diff(np.sign(y_g1d - confidence)) != 0).reshape(-1) + 0
			upper_bound = mu[idx[-1]]	
			lower_bound = mu[idx[0]]
		except IndexError:
			idx = np.argwhere(np.diff(np.sign(y_g1d - 0.25)) != 0).reshape(-1) + 0
			upper_bound = mu[idx[-1]]	
			lower_bound = mu[idx[0]]	
		
		return lower_bound, upper_bound 
	
	def bose_factor(self,mbh,mu,c0,n,f_a):

		
		bose = 10**2*c0*((n**4)/(self.grav_atom(mbh, mu)**3))*(f_a/(2.435*10**18))**2

		bose[bose >1] = 1

		
		return bose	
		
	def scalar_rate_interacting(self,mbh,spin,n,l,m,mu,f_a,bh_name,solar):
		
		c_0=5.
	
		upper_bound, lower_bound = self.free_field_bounds(bh_name,0.68,solar)

		
		if mu > upper_bound or mu < lower_bound:
			rate  = 2.*self.normalised_outer_horizon(spin)*self.product_function(n,l,m, spin, mbh,mu)*self.product_factor(m,mbh,spin,mu,l,n)*self.grav_atom(mbh, mu)**(4.*l+5.)*self.numerical_constant(n, l)
		else:	
			rate  = 2.*self.normalised_outer_horizon(spin)*self.product_function(n,l,m, spin, mbh,mu)*self.product_factor(m,mbh,spin,mu,l,n)*self.grav_atom(mbh, mu)**(4.*l+5.)*self.numerical_constant(n, l)*self.bose_factor(mbh,mu,c_0,n,f_a)
			
		return rate	



class spinone:
	
	
	def __init__(self):
		self.newton = 6.7071186*10**-57
		self.solar_mass = 2.*10**30*5.6095886*10**35


	"""
	
	
	Dominant modes (|nljm>): |1011>,|2111>,|2122>,|2121>,|3211> 
	
	Notation: tilde_horizon  = 
	
	"""
	
	def alpha_converter(self,x,y,z):
		if z == 'boson':
			value = x/(self.newton*self.solar_mass*y) 
		if z =='bh':
			value = x/(self.newton*self.solar_mass*y) 
		return value	
	
	
	def grav_radius(self,mbh):
		return self.newton*mbh*self.solar_mass
	
	def grav_atom(self,mbh,mu):
		return mu*self.grav_radius(mbh)
	
	def energy_levels(self,mu,mbh,l,n):
		return mu*(1-(self.grav_atom(mbh,mu)**2)/(2*(n)**2))	
		
	def horizon_velocity(self,rg,spin):	
		return (1./(2.*rg))*(spin/(1.+np.sqrt(1.-spin**2)))	
	
	def normalised_outer_horizon(self,spin):	
		return 1 + np.sqrt(1-spin**2)
		
	def product_factor(self,m,mbh,spin,mu,l,n):	
		return m*self.horizon_velocity(self.grav_radius(mbh),spin) - self.energy_levels(mu, mbh, l, n)	

	
	def numerical_constant(self,j,n,l):
		
		a = (2.**(2.*l+2.*j+1.)*math.factorial(n+l))/(n**(2.*l+4.)*math.factorial(n-l-1.))
		b = ((math.factorial(l))/(math.factorial(l+j)*math.factorial(l+j+1.)))**2.
		c = (1.+(2.*(1.+l-j)*(1.-l+j))/(l+j))**2.
		return a*b*c
		
		
	
	def product_function(self,n,l,j,m,spin,mbh,mu):	
		
		k_vec = np.arange(1,l+1)
		product = [k**2*(1-spin**2)+(spin*m-2*self.normalised_outer_horizon(spin)*self.grav_radius(mbh)*self.energy_levels(mu,mbh,l,n))**2 for k in k_vec]
		return math.prod(product)
		
		
	def vector_rate(self,mbh,spin,n,l,j,m,mu):
		
		

		rate  = 2.*self.normalised_outer_horizon(spin)*self.product_function(n,l,j,m, spin, mbh,mu)*self.product_factor(m,mbh,spin,mu,l,n)*self.grav_atom(mbh, mu)**(2.*l+2.*j+5.)*self.numerical_constant(j,n,l)
			
		return rate



class spintwo:
	''' Implementation of the work in https://arxiv.org/pdf/2002.04055.pdf
	
	
	'''
	def __init__(self):
		self.newton = 6.7071186*10**-57
		self.solar_mass = 2.*10**30*5.6095886*10**35
		self.constants = [128./45.,10./9.,4./4725.,640./19683.,2.519526329050139e-07,7.498590265030176e-11,2.2317232931637427e-14]
		self.paper_results = [1.0*10**3,2.8*10**4,3.6*10**8,2.4*10**7]

		
	
	
	#def numerical_constant:
	
	def grav_radius(self,mbh):
		return self.newton*mbh*self.solar_mass
	
	def grav_atom(self,mbh,mu):
		return mu*self.grav_radius(mbh)
		
	def rate_factor(self,mu,mbh,l,n,m,spin):
		return (self.energy_levels(mu,mbh,l,n) - m*self.horizon_velocity(self.grav_radius(mbh),spin))
	
	def horizon_velocity(self,rg,spin):	
		return (1./(2.*rg))*(spin/(1.+np.sqrt(1.-spin**2)))	
		
	def energy_levels(self,mu,mbh,l,n):
		return mu*(1-(self.grav_atom(mbh,mu)**2)/(2*(l+n+1)**2))	
		
	def spinfactor(self,mbh,spin,n,l,j,m,mu):

		rg = self.newton*mbh*self.solar_mass
		delta = np.sqrt(1.-spin**2)
		kappa = delta/(1.+delta)

		
		alpha = self.grav_atom(mbh,mu)
		omegar = self.energy_levels(mu,mbh,l,n)
		horizon = self.horizon_velocity(rg,spin)
		
		pre_factor = (1+delta)*delta**(2*j)
		q_vec = np.arange(1,j+1)

		#print(self.grav_radius(mbh),self.energy_levels(mu,mbh,l,n),m,self.horizon_velocity(rg,spin),kappa, 'THIS ONE')
		product_factor = math.prod([1+4*rg**2*((omegar-m*self.horizon_velocity(rg,spin))/(q*kappa))**2 for q in q_vec])

		return pre_factor*product_factor
	
	
	def tensor_rate(self,mbh,spin,n,l,j,m,mu):
	
		rg = self.grav_radius(mbh)
		power_factor = self.spinfactor(mbh,spin,n,l,j,m,mu)/self.spinfactor(mbh,0.,n,l,m,j,mu)
		omegar = self.energy_levels(mu,mbh,l,n)

		if j == 2 and l == 0:
			const = self.constants[0]
		elif j == 1 and l == 1:
			const = self.constants[1]
		elif j == 3 and l == 1:
			const = self.constants[2]
		elif j == 1 and l == 2:
			const = self.constants[3]
		elif j == 4 and l == 2:
			const = self.constants[4]
		elif j == 5 and l == 3:
			const = self.constants[5]
		else:
			const = self.constants[6]
					
				
				

		return -const*power_factor*(self.rate_factor(mu,mbh,l,n,m,spin))*self.grav_atom(mbh,mu)**(2*(l+j)+5)
	
	def ev_to_secs(self,x):
		return x*1.51926757933*10**15
	
	def rate_to_time(self,x):
		return 1./x	
	
	
	
	
class Regge:
	
	def __init__(self):
		print('Initialise Regge Contour finder')
		self.years_to_seconds = 3.154*10**7
		self.ev_to_seconds = 1.51926757933*10**15	
		
		
	def isocountours(self,X,Y,Z,time):
		bound = 1./(time*self.years_to_seconds*self.ev_to_seconds)
		cont = plt.contour(X,Y,Z,[bound],alpha=0)	
		iso = cont.collections[0].get_paths()[0]
		values = iso.vertices	
		x_values = values[:,0]
		y_values = values[:,1]
		return x_values,y_values	
	
	def rect_inter_inner(self,x1,x2):
		n1=x1.shape[0]-1
		n2=x2.shape[0]-1
		X1=np.c_[x1[:-1],x1[1:]]
		X2=np.c_[x2[:-1],x2[1:]]
		S1=np.tile(X1.min(axis=1),(n2,1)).T
		S2=np.tile(X2.max(axis=1),(n1,1))
		S3=np.tile(X1.max(axis=1),(n2,1)).T
		S4=np.tile(X2.min(axis=1),(n1,1))
		return S1,S2,S3,S4

	def rectangle_intersection(self,x1,y1,x2,y2):
		S1,S2,S3,S4=self.rect_inter_inner(x1,x2)
		S5,S6,S7,S8=self.rect_inter_inner(y1,y2)

		C1=np.less_equal(S1,S2)
		C2=np.greater_equal(S3,S4)
		C3=np.less_equal(S5,S6)
		C4=np.greater_equal(S7,S8)

		ii,jj=np.nonzero(C1 & C2 & C3 & C4)
		return ii,jj
	
	def intersection(self,x1,y1,x2,y2):

		ii,jj=self.rectangle_intersection(x1,y1,x2,y2)
		n=len(ii)
		dxy1=np.diff(np.c_[x1,y1],axis=0)
		dxy2=np.diff(np.c_[x2,y2],axis=0)

		T=np.zeros((4,n))
		AA=np.zeros((4,4,n))
		AA[0:2,2,:]=-1
		AA[2:4,3,:]=-1
		AA[0::2,0,:]=dxy1[ii,:].T
		AA[1::2,1,:]=dxy2[jj,:].T

		BB=np.zeros((4,n))
		BB[0,:]=-x1[ii].ravel()
		BB[1,:]=-x2[jj].ravel()
		BB[2,:]=-y1[ii].ravel()
		BB[3,:]=-y2[jj].ravel()

		for i in range(n):
			try:
				T[:,i]=np.linalg.solve(AA[:,:,i],BB[:,i])
			except:
				T[:,i]=np.NaN

		in_range= (T[0,:] >=0) & (T[1,:] >=0) & (T[0,:] <=1) & (T[1,:] <=1)

		xy0=T[2:,in_range]
		xy0=xy0.T
		return xy0[:,0],xy0[:,1]
	
	def find_nearest(self,array,value):
		idx = (np.abs(array-value)).argmin()
		return idx
	
	def imterpolation(self,x,y,new_length):
		nxn = np.linspace(x.min(), x.max(), new_length)
		nyn = sp.interp1d(x, y, kind='cubic')(nxn)
		return nxn,nyn
		
		
	def intersection_points(self,x_set,y_set):
		xin={}
		yin={}
		for i in range(0,len(x_set)-1):
			xin[i]=[]
			yin[i]=[]
			xin[i],yin[i]=self.intersection(x_set[i],y_set[i],x_set[i+1],y_set[i+1])
			
		return xin,yin	
			
	def intersection_index(self,x_set,xin):
		indxn={}
		indx2n={}
		for i in range(0,len(x_set)-1):

			indxn[i]=[]
			indx2n[i]=[]
			try:
				indxn[i] = self.find_nearest(x_set[i],xin[i][0])
				indx2n[i] = self.find_nearest(x_set[i+1],xin[i][0])
			except IndexError:
				del indxn[i]
				del indx2n[i]

				

			
		return indxn, indx2n	
	
	def regge_tragectories(self,nxn,nyn,indxn):
	
		regiesx={}
		regiesy={}	
		for i in range(len(nyn)-(len(nyn)-len(indxn))):
			regiesx[i]=[]
			regiesy[i] = []
			regiesx[i] =  nxn[i][indxn[i]:-1]
			regiesy[i] =  nyn[i][indxn[i]:-1]
		
		return regiesx,regiesy
	
	

	
	def contour_filter(self,bounds):
		inds = []
		
		
		for i in range(len(bounds)):
	
			if np.ptp(bounds[i][1]) == 1.0 or np.std(bounds[i][0]) < 0.001:
				inds.append(i)
		
		
				
		bounds = [v for i, v in enumerate(bounds) if i not in inds]	

		bounds = [v for i, v in enumerate(bounds) if len(v[0])>10 ]	# Bug fixxx
		

			
		return bounds
	
	
	
	def outer_shell(self,nxn,nyn,indxn,indx2n):
		
		
		fxn=[]
		fyn=[]
		

		if len(nxn)==1:
			fxn=np.array(nxn[0])
			fyn=np.array(nyn[0])
			
		elif len(nxn)-len(indxn)>1:
			
			try:	
				fxn.extend(nxn[0][0:indxn[0]])
				for i in range(1,len(nxn)-(len(nxn)-len(indxn))):
					fxn.extend(nxn[i][indx2n[i-1]:indxn[i]])
				
				
				fyn.extend(nyn[0][0:indxn[0]])
				for i in range(1,len(nyn)-(len(nxn)-len(indxn))):
					fyn.extend(nyn[i][indx2n[i-1]:indxn[i]])
				
			
				fxn=np.array(fxn)
				fyn=np.array(fyn)
			
			except KeyError:
				fxn=np.array(nxn[0])
				fyn=np.array(nyn[0])
					
			
		else:
			fxn.extend(nxn[0][0:indxn[0]])
			for i in range(1,len(nxn)-1):
				fxn.extend(nxn[i][indx2n[i-1]:indxn[i]])
			fxn.extend(nxn[-1][list(indx2n.values())[-1]:-1])
			
			fyn.extend(nyn[0][0:indxn[0]])
			for i in range(1,len(nyn)-1):
				fyn.extend(nyn[i][indx2n[i-1]:indxn[i]])
			fyn.extend(nyn[-1][list(indx2n.values())[-1]:-1])	
		
			fxn=np.array(fxn)
			fyn=np.array(fyn)	
		
		return fxn,fyn

	
	
class black_hole:
	
	def __init__(self):
		print('loading black holes')
		
		
	def read_in_bh_data(self):
		black_holes=pd.read_csv('black_hole_data.csv', sep=',',header=None,encoding='latin-1')
		black_holes=pd.DataFrame(black_holes.values[1:], columns=black_holes.iloc[0])
		return black_holes	
		
	def bounds_function(self,fx,fy,x):
		f = sp.interp1d( fx, fy,bounds_error=False)
		y=f(x)
		return(y)		
		
	def grad(self,x,fx,fy):
		f = sp.interp1d( fx, fy,bounds_error=False )
		a=0.01
		xtem = np.linspace(x-a,x+a,99)
		ytem = f(xtem)
		m = (ytem[-1]-ytem[0])/(xtem[-1]-xtem[0])
		return(m)
		
	def grady(self,y,fx,fy):
		f = sp.interp1d( fy, fx ,bounds_error=False)
		a=0.01
		ytem = np.linspace(y-a,y+a,99)
		xtem = f(ytem)
		m = (xtem[-1]-xtem[0])/(ytem[-1]-ytem[0])
		return(m)		
		
	def exclusion_probability(self,bh_data,bh_values,fx,fy):
		
		masss = bh_data.loc[bh_values].mass.astype(float).tolist()
		spins = bh_data.loc[bh_values].spin.astype(float).tolist()

	
		dmasss = []
		dspins = []
		
		for i in bh_values:
			
		
			
			if float(bh_data.loc[i].mass_error_plus) >= float(bh_data.loc[i].mass_error_minus):
				dmasss.append(float(bh_data.loc[i].mass_error_plus))
			else: 
				dmasss.append(float(bh_data.loc[i].mass_error_minus))	
			
			if float(bh_data.loc[i].spin_error_plus) >= float(bh_data.loc[i].spin_error_minus):
				dspins.append(float(bh_data.loc[i].spin_error_plus))
			else:
				dspins.append(float(bh_data.loc[i].spin_error_minus))
	
		
		
		total_exclusion=[]
		normprob=[]	
		for i in range(len(bh_values)):	
			if fx[-1] < masss[i]:
				check = 1
			elif fx[0] < masss[i] < fx[-1]:
				check =2
			else:
				check =3			
			
			if check == 1:
				try:
					yreduced = np.array(fy) - spins[i]
					freduced = sp.UnivariateSpline(fx, yreduced, s=0)
					try :
						xroot = (freduced.roots()[-1])
					except IndexError or ValueError:
						normprob.append(1.0)
						continue	
					zscore=((xroot-masss[i])/dmasss[i])
					prob = 1.-(0.5+0.5*erf(zscore/np.sqrt(2)))
					normprob.append(np.abs(prob))
				except ValueError or  IndexError:
					normprob.append(1.0)	
				
			
			if check == 2:
				ff = (self.bounds_function(fx,fy,masss[i]))
				fdx = (self.grad(masss[i],fx,fy))	
				
				if math.isnan(fdx) == True:
					normprob.append(1.0)
					continue
			
				effvar = (dspins[i]**2+fdx**2*dmasss[i]**2)
				effsd = (np.sqrt(effvar))
				zscore=((ff-spins[i])/effsd)
				prob = 0.5+(0.5*erf(zscore/np.sqrt(2)))
				normprob.append(np.abs(prob))
			
			if check == 3:
				
				try:	
					yreduced = np.array(fy) - spins[i]
					freduced = sp.UnivariateSpline(fx, yreduced, s=0)
					try:
						xroot = (freduced.roots()[0])
					except IndexError or ValueError:
						normprob.append(1.0)
						continue	
					zscore=((xroot-masss[i])/dmasss[i])
					prob = (0.5+0.5*erf(zscore/np.sqrt(2)))
					normprob.append(np.abs(prob))	
				except ValueError or IndexError:
					normprob.append(1.0)
		
		
			
		normprob = np.array(normprob)
		total_exclusion_temp = np.product(normprob)
		total_exclusion_temp = 1.-total_exclusion_temp	
		total_exclusion.append(total_exclusion_temp)
		
		return total_exclusion
			
	
	
	
def main():
	
	spin_switch = 0.5

	#####################################################
	#    CLASSES
	#####################################################
	
	spin0 = spinzero()
	spin1 = spinone()
	spin2 = spintwo()
	regge_values = Regge()
	bhs = black_hole()


	#####################################################
	#    PARAMETER SPACE
	#####################################################
	solar = True
	accuracy = 50
	
	
	bhbh=[12]
	if solar == True:	
		mbh = np.logspace(-3,9.0,500)
		fa = np.logspace(18.25,11.75,accuracy)
	else: 
		mbh = np.logspace(3,12.0,500)
		fa = np.logspace(18.25,14,accuracy)
		
	spin = np.linspace(0,1,300)
	X,Y = np.meshgrid(mbh,spin)
	
	

	time_note = 'sal'
	if time_note == 'hub':
		time_scale = 1*10**10
	if time_note == 'sal':
			time_scale = 4.5*10**7
	if time_note == 'bin':
			time_scale = 4.5*10**6	
	#####################################################
	#    SCALAR FIELDS
	#####################################################
	typee = 'STELLAR' 
	
	if spin_switch == 0:
	
		results = []
		

		for axion in mu:
			print(axion)
			scalar_modes = [[2,1,1],[3,2,2],[4,3,3],[5,4,4],[6,5,5]]
			#scalar_modes = [[2,1,1],[3,2,2]]
			scalar_rates = [spin0.scalar_rate(X,Y,mode[0],mode[1],mode[2],axion) for mode in scalar_modes]
			bounds = [regge_values.isocountours(X, Y, Z, time_scale) for Z in scalar_rates]
			bounds = regge_values.contour_filter(bounds)
			interpolated_bounds = np.array([regge_values.imterpolation(rates[0], rates[1], 1000) for rates in bounds])
			

			xin,yin = regge_values.intersection_points(interpolated_bounds[:,0],interpolated_bounds[:,1])
			in1, in2 = regge_values.intersection_index(interpolated_bounds[:,0], xin)
			reggex,reggey = regge_values.regge_tragectories(interpolated_bounds[:,0], interpolated_bounds[:,1], in1)
			outerx,outery = regge_values.outer_shell(interpolated_bounds[:,0], interpolated_bounds[:,1], in1, in2)
			
		
			
			bh_data = bhs.read_in_bh_data()
			print(bh_data.loc[bbhh].name.tolist()[0],'this?')
			mass = bh_data.loc[bbhh].mass.astype(float).tolist()[0]
			spin = bh_data.loc[bbhh].spin.astype(float).tolist()[0]
			dxtem = bh_data.loc[bbhh].mass_error_minus.astype(float).tolist()[0]
			dytem = bh_data.loc[bbhh].spin_error_minus.astype(float).tolist()[0]
			
			print(mass)
			'''
			for i in bbhh:
				plt.errorbar(mass,spin,xerr=dxtem,yerr=dytem,linewidth=0.5,markersize=2.5)
			plt.plot(outerx,outery)
			plt.xlim(0.08,10**12.5)
			plt.xscale('log')
			plt.show()
			'''
			
			exclusion_probability = bhs.exclusion_probability(bh_data,bbhh, outerx, outery)[0]
			results.append(exclusion_probability)
			
			
			
			print('Axion Mass:',axion,'is excluded at the level:',exclusion_probability)
		
		
		black_hole_name = bh_data.loc[bbhh].name.tolist()[0]
		np.save(f'TOTAL_SUBSET/mu_SCALAR_{typee}_{time_note}.npy',mu)
		np.save(f'TOTAL_SUBSET/ex_SCALAR_{typee}_{time_note}.npy',results)
	
		
	#####################################################
	#    INTERACTING FIELDS
	#####################################################	
		
	if spin_switch == 0.5:
		counter = 0
		for jj in range(len(bhbh)):
			bbhh = [bhbh[jj]]
			bh_data = bhs.read_in_bh_data()
			black_hole_name = bh_data.loc[bbhh].name.tolist()[0]
			print(black_hole_name)
			lower_mass, upper_mass = spin0.free_field_bounds(black_hole_name,0.05,solar)
	
			mu = np.logspace(np.log10(lower_mass)+0.5,np.log10(upper_mass)-0.5,accuracy)
			
			results_master=[]
			for countt in range (len(fa)):
				decay = fa[countt]
				results = []
				for axion in mu:
					counter+=1
					print('The counter: ',counter)
					
					print(axion,countt)
					scalar_modes = [[2,1,1],[3,2,2],[4,3,3],[5,4,4],[6,5,5]]
					scalar_rates = [spin0.scalar_rate_interacting(X,Y,mode[0],mode[1],mode[2],axion,decay,black_hole_name,solar) for mode in scalar_modes]
					bounds = [regge_values.isocountours(X, Y, Z, time_scale) for Z in scalar_rates]
					bounds = regge_values.contour_filter(bounds)
					interpolated_bounds = np.array([regge_values.imterpolation(rates[0], rates[1], 1000) for rates in bounds])
				
		
				
					try:
						xin,yin = regge_values.intersection_points(interpolated_bounds[:,0],interpolated_bounds[:,1])
						in1, in2 = regge_values.intersection_index(interpolated_bounds[:,0], xin)
						reggex,reggey = regge_values.regge_tragectories(interpolated_bounds[:,0], interpolated_bounds[:,1], in1)
						outerx,outery = regge_values.outer_shell(interpolated_bounds[:,0], interpolated_bounds[:,1], in1, in2)
						exclusion_probability = bhs.exclusion_probability(bh_data,bbhh, outerx, outery)[0]
						results.append(exclusion_probability)
					except IndexError:
						exclusion_probability = 0
						results.append(exclusion_probability)
					
					
					print('Axion Mass:',axion,'is excluded at the level:',exclusion_probability)
			
				results_master.append(results)
			
			
			np.save(f'results_interacting_params/mu_{black_hole_name}.npy',mu)
			np.save(f'results_interacting_params/decay_{black_hole_name}.npy',fa)
			np.save(f'results_interacting_ex/ex_{black_hole_name}.npy',results_master)
			

	#####################################################
	#    VECTOR FIELDS
	#####################################################
		
	if spin_switch == 1:
		
		results = []
		
		for vector in mu:
	
			vector_modes = [[1,0,1,1],[2,1,2,2],[3,2,3,3],[4,3,4,4],[5,4,5,5]]
			vector_rates = [spin1.vector_rate(X,Y,mode[0],mode[1],mode[2],mode[3],vector) for mode in vector_modes]
			bounds = [regge_values.isocountours(X, Y, Z, time_scale) for Z in vector_rates]
			bounds = regge_values.contour_filter(bounds)
			
				
			interpolated_bounds = np.array([regge_values.imterpolation(rates[0], rates[1], 1000) for rates in bounds])
			
			xin,yin = regge_values.intersection_points(interpolated_bounds[:,0],interpolated_bounds[:,1])
			in1, in2 = regge_values.intersection_index(interpolated_bounds[:,0], xin)
			reggex,reggey = regge_values.regge_tragectories(interpolated_bounds[:,0], interpolated_bounds[:,1], in1)
			outerx,outery = regge_values.outer_shell(interpolated_bounds[:,0], interpolated_bounds[:,1], in1, in2)
			

			bh_data = bhs.read_in_bh_data()
			
			
			exclusion_probability = bhs.exclusion_probability(bh_data,bbhh, outerx, outery)[0]
			results.append(exclusion_probability)
			
			
			
			print('Vector Mass:',vector,'is excluded at the level:',exclusion_probability)
		
		
		black_hole_name = bh_data.loc[bbhh].name.tolist()[0]
		np.save(f'TOTAL_SUBSET/mu_VECTOR_{typee}_{time_note}.npy',mu)
		np.save(f'TOTAL_SUBSET/ex_VECTOR_{typee}_{time_note}.npy',results)


	#####################################################
	#    TENSOR FIELDS
	#####################################################
			
				
	if spin_switch == 2:
		
		results = []
		
		for tensor in mu:
			print(tensor)
	
			
			tensor_modes = [[1,0,2,2],[2,1,3,3],[3,2,4,4]]
			tensor_rates = [spin2.tensor_rate(X,Y,mode[0],mode[1],mode[2],mode[3],tensor) for mode in tensor_modes]
			bounds = [regge_values.isocountours(X, Y, Z, time_scale) for Z in tensor_rates]
			bounds = regge_values.contour_filter(bounds)
			

			interpolated_bounds = np.array([regge_values.imterpolation(rates[0], rates[1], 1000) for rates in bounds])
			
			xin,yin = regge_values.intersection_points(interpolated_bounds[:,0],interpolated_bounds[:,1])
			in1, in2 = regge_values.intersection_index(interpolated_bounds[:,0], xin)
			reggex,reggey = regge_values.regge_tragectories(interpolated_bounds[:,0], interpolated_bounds[:,1], in1)
			outerx,outery = regge_values.outer_shell(interpolated_bounds[:,0], interpolated_bounds[:,1], in1, in2)
	
			bh_data = bhs.read_in_bh_data()
			
			exclusion_probability = bhs.exclusion_probability(bh_data,bbhh, outerx, outery)[0]
			results.append(exclusion_probability)
			
			print('Tensor Mass:',tensor,'is excluded at the level:',exclusion_probability)
		
		
		black_hole_name = bh_data.loc[bbhh].name.tolist()[0]
		np.save(f'TOTAL_SUBSET/mu_TENSOR_{typee}_{time_note}.npy',mu)
		np.save(f'TOTAL_SUBSET/ex_TENSOR_{typee}_{time_note}.npy',results)
		
		
		'''
		
		##### replication of spin-2 plot!
		xt = spin2.spintwo_rate(mbh,0.5,2,0)
		
		spinz = np.linspace(0.0001,0.6,1000)
		xtt=[]
		for i in spinz:
			xtt.append(np.abs(spin2.spintwo_rate(15,i,2,0))*15*7524835207.481591)
			
		plt.ylim(10**(-12),10**(-8))
		plt.xlim(0,0.6)
		plt.yscale('log')
		plt.plot(spinz,xtt)
		plt.show()
		################################
		'''
		
		
		
	
	

	
	'''
	########### VECTOR FIELDS
	
	x_check_vec = spin1.vector_rate(X,Y,1,0,1,1,mu)
	x_check_vec2 = spin1.vector_rate(X,Y,2,1,2,2,mu)
	x_check_vec3 = spin1.vector_rate(X,Y,3,2,3,3,mu)
	x_check_vec4 = spin1.vector_rate(X,Y,4,3,4,4,mu)
		
	
	x_test_v,y_test_v = regge_values.isocountours(X, Y, x_check_vec, 10**-23)
	x_test_v2,y_test_v2 = regge_values.isocountours(X, Y, x_check_vec2, 10**-23)
	x_test_v3,y_test_v3 = regge_values.isocountours(X, Y, x_check_vec3, 10**-23)
	x_test_v4,y_test_v4 = regge_values.isocountours(X, Y, x_check_vec4, 10**-23)

		
	new_x_v,new_y_v = regge_values.imterpolation(x_test_v, y_test_v, 1000)
	new_x_v2,new_y_v2 = regge_values.imterpolation(x_test_v2, y_test_v2, 1000)
	new_x_v3,new_y_v3 = regge_values.imterpolation(x_test_v3, y_test_v3, 1000)
	new_x_v4,new_y_v4 = regge_values.imterpolation(x_test_v4, y_test_v4, 1000)
	
	x_set_vec = [new_x_v,new_x_v2,new_x_v3,new_x_v4]
	y_set_vec = [new_y_v,new_y_v2,new_y_v3,new_y_v4]
	
	xin_vec,yin_vec = regge_values.intersection_points(x_set_vec,y_set_vec)
	in1_vec, in2_vec = regge_values.intersection_index(x_set_vec, xin_vec)
	reggex_vec,reggey_vec = regge_values.regge_tragectories(x_set_vec, y_set_vec, in1_vec)
	outerx_vec,outery_vec = regge_values.outer_shell(x_set_vec, y_set_vec, in1_vec, in2_vec)



	########### TENSOR FIELDS

	
	x_check_ten = spin2.spintwo_rate(X,Y,2,0)
	x_check_ten2 = spin2.spintwo_rate(X,Y,1,1)
	x_check_ten3 = spin2.spintwo_rate(X,Y,3,1)
	x_check_ten4 = spin2.spintwo_rate(X,Y,1,2)

	x_test_t,y_test_t = regge_values.isocountours(X, Y, x_check_ten, 10**-23)
	x_test_t2,y_test_t2 = regge_values.isocountours(X, Y, x_check_ten2, 10**-23)
	x_test_t3,y_test_t3 = regge_values.isocountours(X, Y, x_check_ten3, 10**-23)
	x_test_t4,y_test_t4 = regge_values.isocountours(X, Y, x_check_ten4, 10**-23)
	
	new_x_t,new_y_t = regge_values.imterpolation(x_test_t, y_test_t, 1000)
	new_x_t2,new_y_t2 = regge_values.imterpolation(x_test_t2, y_test_t2, 1000)
	new_x_t3,new_y_t3 = regge_values.imterpolation(x_test_t3, y_test_t3, 1000)
	new_x_t4,new_y_t4 = regge_values.imterpolation(x_test_t4, y_test_t4, 1000)

	x_set_ten = [new_x_t,new_x_t3]
	y_set_ten = [new_y_t,new_y_t3]
	

	
	xin_ten,yin_ten = regge_values.intersection_points(x_set_ten,y_set_ten)
	

	in1_ten, in2_ten = regge_values.intersection_index(x_set_ten, xin_ten)
	reggex_ten,reggey_ten = regge_values.regge_tragectories(x_set_ten, y_set_ten, in1_ten)
	outerx_ten,outery_ten = regge_values.outer_shell(x_set_ten, y_set_ten, in1_ten, in2_ten)
		
	'''
	
	


	
	'''
	#plt.contour(X,Y,x_check, [10**-23])

	
	### SPIN-2 TEST
	
	
	
	##### replication of spin-2 plot!
	xt = spin2.spintwo_rate(mbh,0.5,2,0)
	
	spinz = np.linspace(0.0001,0.6,1000)
	xtt=[]
	for i in spinz:
		xtt.append(np.abs(spin2.spintwo_rate(15,i,2,0))*15*7524835207.481591)
		
	plt.ylim(10**(-12),10**(-8))
	plt.xlim(0,0.6)
	plt.yscale('log')
	plt.plot(spinz,xtt)
	plt.show()
	################################
	
	
	
	
	
	###############################
	# COMBINED PLOT 
	###############################
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	

	
	
	#xt2 = spin2.spintwo_rate(mbh,0.5,1,1)/(mu)
	#xt3 = spin2.spintwo_rate(mbh,0.5,3,1)/(mu)
	#xt4 = spin2.spintwo_rate(mbh,0.5,1,2)/(mu)

	
	
	
	
	
	print(x)
	
	plt.plot(mbh[:-2]*6.7071186*10**-57*2.*10**30*5.6095886*10**35*mu,x[:-2])
	plt.plot(mbh[:-2]*6.7071186*10**-57*2.*10**30*5.6095886*10**35*mu,x2[:-2])
	plt.plot(mbh[:-2]*6.7071186*10**-57*2.*10**30*5.6095886*10**35*mu,x3[:-2])
	
	plt.plot(mbh[:-2]*6.7071186*10**-57*2.*10**30*5.6095886*10**35*mu,xv[:-2], linestyle='--')
	plt.plot(mbh[:-2]*6.7071186*10**-57*2.*10**30*5.6095886*10**35*mu,xv2[:-2], linestyle='--')
	plt.plot(mbh[:-2]*6.7071186*10**-57*2.*10**30*5.6095886*10**35*mu,xv3[:-2], linestyle='--')
	plt.plot(mbh[:-2]*6.7071186*10**-57*2.*10**30*5.6095886*10**35*mu,xv4[:-2], linestyle='--')
	plt.plot(mbh[:-2]*6.7071186*10**-57*2.*10**30*5.6095886*10**35*mu,xv5[:-2], linestyle='--')
	
	plt.plot(mbh[:-2]*6.7071186*10**-57*2.*10**30*5.6095886*10**35*mu,xt[:-2], linestyle=':')
	plt.plot(mbh[:-2]*6.7071186*10**-57*2.*10**30*5.6095886*10**35*mu,xt2[:-2], linestyle=':')
	plt.plot(mbh[:-2]*6.7071186*10**-57*2.*10**30*5.6095886*10**35*mu,xt3[:-2], linestyle=':')
	plt.plot(mbh[:-2]*6.7071186*10**-57*2.*10**30*5.6095886*10**35*mu,xt4[:-2], linestyle=':')
	
	
	
	
	plt.hlines(10**-9,10**-2,0.3)
	plt.hlines(10**-12,10**-2,0.3)
	plt.hlines(10**-15,10**-2,0.3)
	plt.hlines(10**-18,10**-2,0.3)
	
	
	plt.yscale('log')
	plt.xscale('log')

	plt.xlim(0.01,0.6)
	plt.ylim(10**-24,10**-1)
	plt.show()
	
	'''
	
	
if __name__ == '__main__':
	main()		
	