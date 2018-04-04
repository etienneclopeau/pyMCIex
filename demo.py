


from numpy import array,cos, sin, asarray
import numpy as np
from scipy.optimize import minimize
import copy
# gestion des intervals
from mpmath import mpi

import logging
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', 
                    level=logging.DEBUG, 
                    datefmt='%I:%M:%S')
logging.info("##################################")
logging.info("####    nouvelle execution    ####")


def getMatRotx(angle):
    angle = np.radians(angle)
    if isinstance(angle, float):
        return np.array([[1.,         0.,          0.],
                          [0., cos(angle), -sin(angle)],
                          [0., sin(angle),  cos(angle)]])
    else:
        n = angle.shape[0]
        return np.array([[np.ones(n) , np.zeros(n), np.zeros(n)],
                         [np.zeros(n),  cos(angle), -sin(angle)],
                         [np.zeros(n),  sin(angle),  cos(angle)]]).swapaxes(0,2)
def getMatRoty(angle):
    angle = np.radians(angle)
    if isinstance(angle, float):
        return np.array([[ cos(angle),   0., sin(angle)],
                          [         0.,   1.,         0.],
                          [-sin(angle),   0., cos(angle)]])
    else:
        n = angle.shape[0]
        return np.array([[ cos(angle), np.zeros(n), sin(angle)],
                          [ np.zeros(n), np.ones(n), np.zeros(n)],
                          [-sin(angle), np.zeros(n), cos(angle)]]).swapaxes(0,2)
def getMatRotz(angle):
    angle = np.radians(angle)
    if isinstance(angle, float):
        return np.array([[ cos(angle), -sin(angle), 0.],
                          [ sin(angle),  cos(angle), 0.],
                          [         0.,          0., 1.]])
    else:
        n = angle.shape[0]
        return np.array([[ cos(angle), -sin(angle), np.zeros(n)],
                          [ sin(angle),  cos(angle), np.zeros(n)],
                          [ np.zeros(n), np.zeros(n), np.ones(n)]]).swapaxes(0,2)

class Ref():
    def __init__(self, name, O, psi, theta, phi, ref = None, depo = None):
        # logging.debug("créé un referentiel. Inputs = %s"%(locals()))

        self.name = name
        self.O = np.array(O, dtype = float)
        self.psi = psi
        self.theta = theta
        self.phi = phi
        self.ref = ref
        self.mat = None

        self.O_n = copy.deepcopy(self.O)
        self.psi_n = copy.deepcopy(self.psi)
        self.theta_n = copy.deepcopy(self.theta)
        self.phi_n = copy.deepcopy(self.phi)

        self.depo = depo
        if self.depo is not None and 'O' in self.depo : self.depo['O'] = np.array(self.depo['O'],dtype = float)

        self.depo_0 = copy.deepcopy(self.depo)

        self.calc_mat()

    def calc_mat(self):
        # premiere rotation d'angle psi autour de X
        # deuxième rotation d'angle theta autour de Y
        # troisième rotation d'angle phi autour de X
        self.mat = getMatRotx(self.phi)@getMatRoty(self.theta)@getMatRotx(self.psi)
        self.Orig = np.zeros(self.O.shape)
        # logging.debug("self.O %s" %self.O)
        if self.ref is not None:
            self.mat = self.ref.mat@self.mat
            self.Orig += self.ref.Orig
        # logging.debug("shape of self.O %s ndim %s"%(self.O.shape,self.O.ndim))
        # logging.debug('shape of self.mat %s ndim %s'%(self.mat.shape,self.mat.ndim))
        if self.mat.ndim > 2 : #and self.O.ndim == 1:
            # logging.debug("shape of self.O correc %s"%(np.expand_dims(self.O,self.O.ndim+1).transpose(0,2,1).shape,))
            temp = (np.expand_dims(self.O,self.O.ndim+1).transpose(0,2,1)@self.mat)
            # logging.debug("shape of temp %s"%(temp.shape,))
            # logging.debug("temp %s"%(temp))
            # logging.debug("shape of self.Orig %s"%(self.Orig,))
            self.Orig  += temp[:,0,:]
        else: 
            self.Orig += self.O@self.mat
        # logging.debug("self;Orig %s" %self.Orig)


    def getParams(self,params = None):
        if params is None: params = list()
        len0 = len(params)
        if self.depo is not None:
            if 'O' in self.depo.keys(): params += [a for a in self.depo['O'].tolist() if a != 0]
            if 'psi' in self.depo.keys() and self.depo['psi'] != 0 : params.append(self.depo['psi'])
            if 'theta' in self.depo.keys() and self.depo['theta'] != 0 : params.append(self.depo['theta'])
            if 'phi' in self.depo.keys() and self.depo['phi'] != 0 : params.append(self.depo['phi'])
        logging.debug("found %s parameters to optimize in ref %s"%(len(params)-len0,self.name))
        self.nbrOfParameters = len(params)-len0
        return params

    def putParams(self, allParamsValues):
        len0 = len(allParamsValues)
        self.O = copy.deepcopy(self.O_n)
        self.psi = copy.deepcopy(self.psi_n)
        self.theta = copy.deepcopy(self.theta_n)
        self.phi = copy.deepcopy(self.phi_n)
        if self.depo is not None:
            if 'O' in self.depo.keys() : 
                for i in range(3): 
                    if self.depo['O'][i] != 0 : 
                        temp = allParamsValues.pop(0)
                        # logging.debug('self.O[i] %s, temp %s'%(self.O[i], temp))
                        self.O[i] = self.O[i] + temp 
                        # logging.debug('insertion incertitude centrage %i dans %s: %s'%(i,self.name, temp))
                        # logging.debug('self.O[i] %s'%(self.O[i]))
            if 'psi' in self.depo.keys() and self.depo['psi'] != 0 : self.psi += allParamsValues.pop(0)
            if 'theta' in self.depo.keys() and self.depo['theta'] != 0 : self.theta += allParamsValues.pop(0)
            if 'phi' in self.depo.keys() and self.depo['phi'] != 0 : self.phi += allParamsValues.pop(0)
        self.calc_mat()
        # logging.debug('nombre de parametres insérés dans %s: %s'%(self.name, len(allParamsValues)-len0))
        if len0 - len(allParamsValues) != self.nbrOfParameters : raise(ValueError)
        # logging.debug('self.o apres dispersion: %s'%(self.O))
        return allParamsValues

    def nominalAgain(self, incertAlso = False):
        self.O = copy.deepcopy(self.O_n)
        self.psi = copy.deepcopy(self.psi_n)
        self.theta = copy.deepcopy(self.theta_n)
        self.phi = copy.deepcopy(self.phi_n)
        if incertAlso:
            self.depo = copy.deepcopy(self.depo_0)
        self.calc_mat()
      
    def findRefs(self, refs= None):
        if refs is None: refs = list()
        if self.ref is not None and self.ref not in refs : 
            refs = self.ref.findRefs(refs)
            refs.append(self.ref)
        return refs

    def make_tirages(self, ntirages):
        self.O = self.O_n * np.ones((ntirages,3))
        self.psi = np.ones(ntirages)*self.psi_n
        self.theta = np.ones(ntirages)*self.theta_n
        self.phi = np.ones(ntirages)*self.phi_n
        if self.depo is not None:
            if 'O' in self.depo.keys() : 
                O = np.zeros((ntirages,3))
                for i in range(3):
                    O[:,i] = np.random.uniform(-self.depo['O'][i], self.depo['O'][i], ntirages)
                self.O = self.O + O
            if 'psi' in self.depo.keys() : 
                self.psi += np.random.uniform(-self.depo['psi'], self.depo['psi'], ntirages)
            if 'theta' in self.depo.keys() : 
                self.theta += np.random.uniform(-self.depo['theta'], self.depo['theta'], ntirages)
            if 'phi' in self.depo.keys() : 
                self.phi += np.random.uniform(-self.depo['phi'], self.depo['phi'], ntirages)

        self.calc_mat()

    def put_tirages(self, tirages):
        ntirages = len(tirages[0])
        len0 = len(tirages)
        self.O = self.O_n * np.ones((ntirages,3))
        self.psi = np.ones(ntirages)*self.psi_n
        self.theta = np.ones(ntirages)*self.theta_n
        self.phi = np.ones(ntirages)*self.phi_n
        if self.depo is not None:
            if 'O' in self.depo.keys() : 
                O = np.zeros((ntirages,3))
                for i in range(3):
                    if self.depo['O'][i] !=0 : O[:,i] = tirages.pop(0)
                self.O = self.O + O
            if 'psi' in self.depo.keys() and self.depo['psi'] != 0 : 
                self.psi += tirages.pop(0)
            if 'theta' in self.depo.keys() and self.depo['theta'] != 0 : 
                self.theta += tirages.pop(0)
            if 'phi' in self.depo.keys() and self.depo['phi'] != 0 : 
                self.phi += tirages.pop(0)

        self.calc_mat()
        if len0 - len(tirages) != self.nbrOfParameters : 
            raise(ValueError("putted %s parames instead of %s in %s"%(len0 - len(tirages),self.nbrOfParameters, self.name)))

    def neutraliseIncert(self):

        if self.depo is not None:
            # logging.debug('neutralise depo %s'%self.name)
            if 'O' in self.depo.keys() : 
                self.depo['O'] = np.zeros(3)
            if 'psi' in self.depo.keys() : 
                self.depo['psi'] = 0.
            if 'theta' in self.depo.keys() : 
                self.depo['theta'] = 0.
            if 'phi' in self.depo.keys() : 
                self.depo['phi'] = 0.
        # logging.debug(self)   


    # def getMat(self):
    #     if self.mat is None: self.calculMat()
    #     return self.mat

class Element():
    def __init__(self, name, m, c, I, ref = None, garde = None, meco = None, disp = None):
        # logging.debug("créé un élément. Inputs = %s"%(locals()))
        self.name = name
        self.m = m
        self.c = array(c,dtype=float)
        self.I = array(I,dtype=float)
        self.m_n = copy.deepcopy(self.m)
        self.c_n = copy.deepcopy(self.c)
        self.I_n = copy.deepcopy(self.I)
        self.ref = ref
        self.garde = garde
        self.meco = meco
        self.disp = disp

        def toFloat( incert):
            if incert is not None:
                if 'm' in incert.keys():
                    incert['m'] = float(incert['m'])
                if 'c' in incert.keys():
                    incert['c'] = np.array(incert['c'], dtype = float)
                if 'I' in incert.keys():
                    incert['I'] = np.array(incert['I'], dtype = float)
        toFloat(self.garde)
        toFloat(self.meco)
        toFloat(self.disp)


        self.garde_0 = garde
        self.meco_0 = meco
        self.disp_0 = disp
        # if garde is not None:
        #     if 'm' in garde.keys(): self.m_g = garde['m']
        #     if 'c' in garde.keys(): self.c_g = garde['c']
        #     if 'I' in garde.keys(): self.I_g = garde['I']
        # if meco is not None:
        #     if 'm' in meco.keys(): self.m_m = meco['m']
        #     if 'c' in meco.keys(): self.c_m = meco['c']
        #     if 'I' in meco.keys(): self.I_m = meco['I']
        # if disp is not None:
        #     if 'm' in disp.keys(): self.m_d = disp['m']
        #     if 'c' in disp.keys(): self.c_d = disp['c']
        #     if 'I' in disp.keys(): self.I_d = disp['I']
           

    def toRef0(self, GMD = False):
        if self.ref is not None:
            #on etend self.c a une nouvelle dimension pour pouvoir traiter d'un coup les tirages.
            #la ligne d'origine:
            #c = self.ref.O + self.ref.mat@self.c
            c = self.ref.Orig + np.squeeze(self.ref.mat@np.expand_dims(self.c,self.c.ndim+1))
            # logging.debug("toRef0 : I shape = %s"%(self.I.shape,))
            # logging.debug("toRef0 : ref.mat shape = %s"%(self.ref.mat.shape,))
            if self.I.ndim == 2:
                I = self.ref.mat.T@self.I@self.ref.mat
            else:
                # si on est en MTC,  il faut transposer uniquement les deux derniers axes au lieu de tt la matrice
                # logging.debug("self.ref.mat shape %s"%(self.ref.mat.shape,))
                # logging.debug("incert['I'] shape %s"%(self.I.shape,))
                I = self.ref.mat.transpose(0,2,1)@self.I@self.ref.mat
            def toRefIncert(incert):
                if incert is not None:
                    incertRef0 =dict()
                    if 'm' in incert.keys(): incertRef0['m'] = incert['m']
                    if 'c' in incert.keys(): incertRef0['c'] = np.squeeze(self.ref.mat@np.expand_dims(incert['c'],incert['c'].ndim+1))
                    if 'I' in incert.keys(): 
                        if incert['I'].ndim == 2:
                            incertRef0['I'] = self.ref.mat.T@incert['I']@self.ref.mat
                        else:
                            # si on est en MTC,  il faut transposer uniquement les deux derniers axes au lieu de tt la matrice
                            logging.debug("incert['I'] shape %s"%(incert['I'].shape,))
                            incertRef0['I'] = self.ref.mat.transpose(0,2,1)@incert['I']@self.ref.mat
                    return incertRef0
                else:
                    return None
            if GMD :
                return Element(self.name, self.m, c, I, garde = toRefIncert(self.garde_0), 
                                                        meco = toRefIncert(self.meco_0),
                                                        disp = toRefIncert(self.disp_0))
            else:
                return Element(self.name, self.m, c, I, garde = self.garde, 
                                                        meco = self.meco,
                                                        disp = self.disp)
        else:
            return self
        
    def getParams(self, params = None):
        # print(params)
        if params is None: params = list()
        len0 = len(params)
        if self.garde is not None:
            if 'm' in self.garde.keys() and self.garde['m'] != 0 : params.append(self.garde['m'])
            if 'c' in self.garde.keys(): params += [a for a in self.garde['c'] if a != 0]
            if 'I' in self.garde.keys(): params += [a for a in self.garde['I'].flatten() if a != 0]
            # if 'c' in self.garde.keys() :logging.debug("test %s"%(self.garde['c']))
        if self.meco is not None:
            if 'm' in self.meco.keys() and self.meco['m'] != 0 : params.append(self.meco['m'])
            if 'c' in self.meco.keys(): params += [a for a in self.meco['c'] if a != 0]
            if 'I' in self.meco.keys(): params += [a for a in self.meco['I'].flatten() if a != 0]
        if self.disp is not None:
            if 'm' in self.disp.keys() and self.disp['m'] != 0 : params.append(self.disp['m'])
            if 'c' in self.disp.keys(): params += [a for a in self.disp['c'] if a != 0]
            if 'I' in self.disp.keys(): params += [a for a in self.disp['I'].flatten() if a != 0]
        logging.debug("found %s parameters to optimize in elemeny %s"%(len(params)-len0,self.name))
        self.nbrOfParameters = len(params)-len0
        return params

    def putParams(self, allParamsValues):
        self.nominalAgain()
        len0 = len(allParamsValues)
        # logging.debug("reste %s parametres à distribuer"%len0)
        def putdisp(m, c, I, incert):
            if incert is not None:
                if 'm' in incert.keys() and incert['m'] != 0 : 
                    m = m + allParamsValues.pop(0)
                    # logging.debug("put masse in %s. incert linked: %s "%(self.name,incert['m']))
                if 'c' in incert.keys():
                    for i in range(3):  
                        if incert['c'][i] != 0 : 
                            c[i] = c[i] + allParamsValues.pop(0) 
                            # logging.debug("put centrage in %s : i %s"%(self.name, i))
                if 'I' in incert.keys():
                    for i in range(3):
                        for j in range(3):
                            if incert['I'][i,j] != 0 :
                                temp = allParamsValues.pop(0) 
                                I[i,j] = I[i,j] + temp
                                # logging.debug("put inertie in %s : %s, %s, %s"%(self.name,temp, i,j))
            return m,c,I
        self.m,self.c,self.I = putdisp(self.m,self.c,self.I,self.garde)
        self.m,self.c,self.I = putdisp(self.m,self.c,self.I,self.meco)
        self.m,self.c,self.I = putdisp(self.m,self.c,self.I,self.disp)
        # logging.debug('nombre de parametre insérés : %s'%(len(allParamsValues)-len0))
        if -len(allParamsValues)+len0 != self.nbrOfParameters : raise(ValueError)
        return allParamsValues

    def nominalAgain(self,incertAlso = False):
        self.m = copy.deepcopy(self.m_n)
        self.c = copy.deepcopy(self.c_n)
        self.I = copy.deepcopy(self.I_n)
        if incertAlso:
            self.garde = copy.deepcopy(self.garde_0)
            self.meco = copy.deepcopy(self.meco_0)
            self.disp = copy.deepcopy(self.disp_0)

        

    def findRefs(self, refs= None):
        if refs is None: refs = list()
        if self.ref is not None and self.ref not in refs : 
            refs = self.ref.findRefs(refs)
            refs.append(self.ref)
        return refs
    
    def make_tirages(self, ntirages):
        self.m = self.m_n * np.ones(ntirages)
        self.c = self.c_n * np.ones((ntirages,3))
        self.I = self.I_n * np.ones((ntirages,3,3))
        if self.garde is not None :
            if 'm' in self.garde.keys(): 
                self.m += np.random.uniform(-self.garde['m'], self.garde['m'], ntirages)
            if 'c' in self.garde.keys(): 
                c = np.zeros((ntirages,3))
                for i in range(3):
                    c[:,i] = np.random.uniform(-self.garde['c'][i], self.garde['c'][i], ntirages)
                self.c += c
            if 'I' in self.garde.keys(): 
                I = np.zeros((ntirages,3,3))
                for i in range(3):
                    for j in range(3):
                        I[:,i,j] = np.random.uniform(-self.garde['I'][i,j], self.garde['I'][i,j], ntirages)
                self.I = self.I + I
        if self.meco is not None :
            if 'm' in self.meco.keys(): 
                self.m += np.random.uniform(-self.meco['m'], self.meco['m'], ntirages)
            if 'c' in self.meco.keys(): 
                c = np.zeros((ntirages,3))
                for i in range(3):
                    c[:,i] = np.random.uniform(-self.meco['c'][i], self.meco['c'][i], ntirages)
                self.c += c
            if 'I' in self.meco.keys(): 
                I = np.zeros((ntirages,3,3))
                for i in range(3):
                    for j in range(3):
                        I[:,i,j] = np.random.uniform(-self.meco['I'][i,j], self.meco['I'][i,j], ntirages)
                self.I = self.I + I
        if self.disp is not None :
            if 'm' in self.disp.keys(): 
                self.m += np.random.uniform(-self.disp['m'], self.disp['m'], ntirages)
            if 'c' in self.disp.keys():
                c = np.zeros((ntirages,3))
                for i in range(3):
                    c[:,i] = np.random.uniform(-self.disp['c'][i], self.disp['c'][i], ntirages)
                self.c += c
            if 'I' in self.disp.keys(): 
                I = np.zeros((ntirages,3,3))
                for i in range(3):
                    for j in range(3):
                        I[:,i,j] = np.random.uniform(-self.disp['I'][i,j], self.disp['I'][i,j], ntirages)
                self.I = self.I + I
        logging.debug("make tirage: shape of m,c I %s %s %s"%(self.m.shape,self.c.shape,self.I.shape))

    def put_tirages(self, tirages):
        #contrairement à make_tirage, cette fonction insere les valeurs déjà choisies pour chaque parametre
        # elle est utilisé pour les optimisation, en permettant de calculer les dérivées en fonction de chaque entrée
        # dans ce cas les différents tirage présentent un petit écart sur un des parametre pour calculer 
        # le gradiant par différence fini.
        # passer par cette fonction permet d'évaluer en une seule fois, l'ensemble des dérivées
        ntirages = len(tirages[0])
        len0 = len(tirages)
        # logging.debug("len0 %s"%len0)
        self.m = self.m_n * np.ones(ntirages)
        self.c = self.c_n * np.ones((ntirages,3))
        self.I = self.I_n * np.ones((ntirages,3,3))

        def put_tirage_incert(incert):
            if incert is not None :
                if 'm' in incert.keys() and incert['m'] != 0: 
                    self.m += tirages.pop(0)
                if 'c' in incert.keys() : 
                    c = np.zeros((ntirages,3))
                    for i in range(3):
                        if incert['c'][i] != 0 : c[:,i] = tirages.pop(0)
                    self.c += c
                if 'I' in incert.keys() : 
                    I = np.zeros((ntirages,3,3))
                    for i in range(3):
                        for j in range(3):
                            if incert['I'][i,j] != 0 : I[:,i,j] = tirages.pop(0)
                    self.I = self.I + I
        put_tirage_incert(self.garde)
        # logging.debug("len0 %s"%(len0 - len(tirages)))
        put_tirage_incert(self.meco)
        # logging.debug("len0 %s"%(len0 - len(tirages)))
        put_tirage_incert(self.disp)
        # logging.debug("len0 %s"%(len0 - len(tirages)))
        if -len(tirages)+len0 != self.nbrOfParameters : 
            raise(ValueError("putted %s parames instead of %s in %s"%(len0 - len(tirages),self.nbrOfParameters, self.name)))

        # logging.debug("put tirage: shape of m,c I %s %s %s"%(self.m.shape,self.c.shape,self.I.shape))


    def neutraliseIncert(self, cara):
        logging.debug("neutralise %s"%cara)
        if cara == 'depo':
            if self.ref is not None:
                self.ref.neutraliseIncert()
            return
        if self.garde is not None:
            if cara in self.garde.keys():
                if cara == 'm':
                    self.garde[cara] = 0
                else :
                    self.garde[cara][:] = 0
        if self.meco is not None:
            if cara in self.meco.keys():
                if cara == 'm':
                    self.meco[cara] = 0
                else :
                    self.meco[cara][:] = 0
        if self.disp is not None:
            if cara in self.disp.keys():
                if cara == 'm':
                    self.disp[cara] = 0
                else :
                    self.disp[cara][:] = 0

            

class Ensemble():
    def __init__(self,*args):
        # logging.debug("créé un enesemble. Inputs = %s"%(locals()))

        self.elements = args
        self.findRefs()

    # def prepareRefs(self):
    #     self.refs = list()
    #     for e in self.elements:
    #         if e.ref not in self.refs: self.refs.append(e.ref)

    #     for ref in self.refs:
    #         ref.calculMat()

    def findRefs(self):
        self.refs = list()
        for e in self.elements:
            self.refs = e.findRefs(self.refs)

    def toRef0(self, GMD = False):
        # self.prepareRefs()
        
        return [e.toRef0(GMD) for e in self.elements]

    def assemble(self):

        elements = self.toRef0()

        ms = array([e.m for e in elements])
        cs = array([e.c for e in elements])
        Is = array([e.I for e in elements])
        if ms.ndim == 2:
            ms = ms.transpose()
            cs = cs.swapaxes(0,1)
            Is = Is.swapaxes(0,1)
        logging.debug("ms: %s"%(ms.shape,))
        logging.debug("cs: %s"%(cs.shape,))
        logging.debug("Is: %s"%(Is.shape,))

        m = np.array(ms.sum(axis = -1))
        logging.debug('m.shape %s'%(m.shape,))
#        c  = np.sum(ms*cs, axis = 0)/m
        logging.debug("np.sum((np.expand_dims(ms,ms.ndim+1)*cs), axis = -1) %s"%(np.sum((np.expand_dims(ms,ms.ndim+1)*cs), axis = -2).shape,))
        c  = np.sum((np.expand_dims(ms,ms.ndim+1)*cs), axis = -2)/np.expand_dims(m,m.ndim+1)
        logging.debug("c %s"%(c.shape,))
        if c.ndim > 1: raise
        # calcul de I de façon itérative
        # I = 0.
        # for me,ce,Ie in zip(ms,cs,Is):
        #     d = ce-c
        #     x ,y, z = d[0], d[1], d[2]
        #     I += Ie + me*array([[y**2+z**2, -x*y      ,  -x*z    ],
        #                        [   -x*y  , x**2+z**2 ,  -y*z    ],
        #                        [   -x*z  ,  -y*z     , x**2+y**2]])
        
        #de facon matricielle
        #d = cs -c
        # en matriciel : d: dim1 = xyz ; dim2 = nelements ; dim3= ntirage
        d = (cs - np.expand_dims(c,c.ndim+1).swapaxes(-2,-1))
        logging.debug("d %s"%(d.shape,))
        if d.ndim == 3 : d = d.swapaxes(0,-1).swapaxes(-2,-1)
        else: d = d.T
#        print('d',d.shape, d)
        x ,y, z = d[0], d[1], d[2]
#        print('x',x.shape, x)
        transportI = array([[y**2+z**2, -x*y      ,  -x*z    ],
                            [   -x*y  , x**2+z**2 ,  -y*z    ],
                            [   -x*z  ,  -y*z     , x**2+y**2]])
        if transportI.ndim == 3 : transportI = transportI.swapaxes(0,2)
        elif transportI.ndim == 4 : transportI = transportI.transpose(2,3,0,1)
        else : raise(ValueError('should not be possible...'))
#        print('transportI ',transportI.shape)
        msItransported = np.expand_dims( np.expand_dims(ms,ms.ndim+1) ,ms.ndim+1) * transportI
#        print('msItransported ',msItransported.shape)
        I = (Is + msItransported).sum(axis = 0)
#        print('I ',I.shape)

        return m,c,I

    def domainAnalitic(self, approxMethod = False):
        logging.warning("le domain analitic ne prend pas en compte les dépositionnements")
        self.nominalAgain()

        me,ce,Ie = self.assemble()
        if not approxMethod :
            raise(NotImplementedError())
        else: #approx method
            elements = self.toRef0(GMD = True)

            def get_gmd(incert):
                m_i_contribs = list()
                cx_i_contribs,cy_i_contribs,cz_i_contribs = list(), list(), list()
                Ixx_i_contribs,Iyy_i_contribs,Izz_i_contribs = list(), list(), list()
                for e in elements:
                    if incert == 'garde':
                        incertitude = e.garde
                    elif incert == 'meco':
                        incertitude = e.meco
                    elif incert == 'disp':
                        incertitude = e.disp
                    else:
                        raise(ValueError)
                    logging.debug("%s %s : %s"%(e.name,incert,incertitude))
                    if  incertitude is not None:
                        if 'm' in incertitude.keys():  e_m_i = incertitude['m']
                        else: e_m_i = 0
                        if 'c' in incertitude.keys():  e_c_i = incertitude['c']
                        else: e_c_i = np.zeros(3)
                        if 'I' in incertitude.keys():  e_I_i = incertitude['I']
                        else: e_I_i = np.zeros((3,3))
                    else:
                        e_m_i = 0
                        e_c_i = np.zeros(3)
                        e_I_i = np.zeros((3,3))

                    logging.debug("garde m,c,I,%s,%s,%s"%(e_m_i,e_c_i,e_I_i))
                    m_i_contribs.append(e_m_i)
                    cx_i_contribs.append(e_m_i*(e.c[0]-ce[0])/me + e.m*e_c_i[0]/me)
                    cy_i_contribs.append(e_m_i*(e.c[1]-ce[1])/me + e.m*e_c_i[1]/me)
                    cz_i_contribs.append(e_m_i*(e.c[2]-ce[2])/me + e.m*e_c_i[2]/me)

                    Ixx_i_contribs.append(e_I_i[0,0] + e_m_i*(e.c[1]-ce[1])**2+e_m_i*(e.c[2]-ce[2])**2 +
                                            e.m*(e.c[1]-ce[1])*e_c_i[1]+e.m*(e.c[2]-ce[2])*e_c_i[2])
                    Iyy_i_contribs.append(e_I_i[1,1] + e_m_i*(e.c[0]-ce[0])**2+e_m_i*(e.c[2]-ce[2])**2 +
                                            e.m*(e.c[0]-ce[0])*e_c_i[0]+e.m*(e.c[2]-ce[2])*e_c_i[2])
                    Izz_i_contribs.append(e_I_i[2,2] + e_m_i*(e.c[0]-ce[0])**2+e_m_i*(e.c[1]-ce[1])**2 +
                                            e.m*(e.c[0]-ce[0])*e_c_i[0]+e.m*(e.c[1]-ce[1])*e_c_i[1])
                logging.debug("contribs cx %s \n cy %s \n cz %s\n Ixx %s \n Iyy %s \n czz %s"%( \
                                        cx_i_contribs,cy_i_contribs,cz_i_contribs, \
                                        Ixx_i_contribs,Iyy_i_contribs,Izz_i_contribs))
                
                m_i = np.asarray(m_i_contribs).sum()
                c_i = np.array([np.asarray(cx_i_contribs).sum(), np.asarray(cy_i_contribs).sum(), np.asarray(cz_i_contribs).sum()])
                Ixx_i, Iyy_i, Izz_i = np.asarray(Ixx_i_contribs).sum(), np.asarray(Iyy_i_contribs).sum(), np.asarray(Izz_i_contribs).sum()
                I_i = np.array([[Ixx_i,0,0],
                                [0,Iyy_i,0],
                                [0,0,Izz_i]])
                return dict(m=m_i,c=c_i,I=I_i)

            garde = get_gmd(incert='garde')
            meco = get_gmd(incert='meco')
            disp = get_gmd(incert='disp')
            return Element('res',me,ce,Ie, garde = garde, meco = meco, disp = disp)
            # return np.asarray(cx_g_contribs).sum(), np.asarray(cy_g_contribs).sum(), np.asarray(cz_g_contribs).sum(), \
            #          np.asarray(Ixx_g_contribs).sum(), np.asarray(Iyy_g_contribs).sum(), np.asarray(Izz_g_contribs).sum()



    def find_parameters(self):
        self.allParams = list()
        for e in self.elements:
            self.allParams = e.getParams(self.allParams)
        for ref in self.refs:
            self.allParams = ref.getParams(self.allParams)
                
    def dispers(self, allParamsValues):
        if isinstance(allParamsValues, np.ndarray): allParamsValues = allParamsValues.tolist()
        for e in self.elements:
            # logging.debug("disperse %s "%(e.name))
            allParamsValues = e.putParams(allParamsValues)
        for ref in self.refs:
            allParamsValues = ref.putParams(allParamsValues)
        if len(allParamsValues) != 0 : raise(ValueError)
    
    def nominalAgain(self,incertAlso = False):
        for e in self.elements:
            e.nominalAgain(incertAlso)
        for ref in self.refs:
            ref.nominalAgain(incertAlso)

    def optimize(self, objective = 'cr_max'):
        logging.debug("lancement d'une optimisation : %s"%(objective))

        self.findRefs()
        self.nominalAgain()
        logging.debug('%s référentiels différents trouvés'%len(self.refs))
        self.find_parameters()
        # self.dispersed = copy.deepcopy(self)

        logging.debug("nombre de parametres a optimiser : %s"%(len(self.allParams)))

        params_nom, params_min, params_max = list(), list(), list()
        for param in self.allParams:
            if isinstance(param, float) or isinstance(param, int) or isinstance(param, np.int64):
                params_nom.append(0.)
                params_min.append(-param)
                params_max.append(+param)
            # elif isinstance (param, np.ndarray) :
            #     garde = param
            #     # c'est un array à 1 ou 2 dimensions
            #     params_nom = np.zeros(garde.shape).flatten().tolist()
            #     params_min = (-garde).flatten().tolist()
            #     params_max = garde.flatten().tolist()
            else:
                # d'autres cas à créé peut être pour avoir plus de possibilité
                # comme la distinction de garde en - different de garde en +
                raise(ValueError("param is not of anticipated type %s, %s"%(param, type(param))))
        bounds = [(vmin,vmax) for vmin, vmax in zip(params_min,params_max)]
        logging.debug("valeurs initiales des parametres : %s"%(params_nom))
        logging.debug("bornes utilisées pour l'optimisation : %s"%(bounds))

        
        def getObjectiveValue(m,c,I):
            if c.ndim == 1:
                #alors on a un seul point de calcul.
                # pour rendre cette fonction compatible de plusieurs points de calculs
                # on ajoute une dimension devant, pour simuler un résultat sur plusieurs points, avec un seul point
                m = np.ones(1)*m
                c = np.expand_dims(c,2).transpose()
                I = np.expand_dims(I,3).transpose(2,0,1)

            if objective == 'cr_max':
                res =  -np.sqrt(c[:,1]**2+c[:,2]**2)
            elif objective == 'cr_min':
                res =  np.sqrt(c[:,1]**2+c[:,2]**2)
            elif objective == 'cx_max':
                res =  -c[:,0]
            elif objective == 'cx_min':
                res =  c[:,0]
            elif objective == 'cy_max':
                res =  -c[:,1]
            elif objective == 'cy_min':
                res =  c[:,1]
            elif objective == 'cz_max':
                res =  -c[:,2]
            elif objective == 'cz_min':
                res =  c[:,2]
            elif objective == 'Ixx_max':
                res =  -I[:,0,0]
            elif objective == 'Ixx_min':
                res =  I[:,0,0]
            elif objective == 'Iyy_max':
                res =  -I[:,1,1]
            elif objective == 'Iyy_min':
                res =  I[:,1,1]
            elif objective == 'Izz_max':
                res =  -I[:,2,2]
            elif objective == 'Izz_min':
                res =  I[:,2,2]
            elif objective == 'Ixy_max':
                res =  -I[:,0,1]
            elif objective == 'Ixy_min':
                res =  I[:,0,1]
            elif objective == 'Ixz_max':
                res =  -I[:,0,2]
            elif objective == 'Ixz_min':
                res =  I[:,0,2]
            elif objective == 'Iyz_max':
                res =  -I[:,1,2]
            elif objective == 'Iyz_min':
                res =  I[:,1,2]
            elif objective == 'm_min':
                res =  m[:] 
            elif objective == 'm_max':
                res =  -m[:]
            else:
                raise(ValueError)

            # logging.debug("res %s, %s"%(res,c.shape[0]))
            if c.shape[0] == 1 :
                # on ne veut pas un tableau en sortie mais jsute un float
                return res[0] 
            else:
                return res




        def costFunction(dispersedParams):
            # logging.debug("dispersion avec ces parametres : %s"%dispersedParams)
            self.dispers(dispersedParams)
            # logging.debug("Inertie : %s"%(self.elements[0].I))
            m,c,I = self.assemble()
            # logging.debug("resultat : %s %s %s"%(m,c,I))

            return getObjectiveValue(m,c,I)

        def testgradiantFunction(dispersedParams):
            epsilon = 1e-8
            ref = costFunction(dispersedParams)
            logging.debug("ref %s"%ref)

            grad = list()
            for i, param in enumerate(dispersedParams):
                dispP = copy.deepcopy(dispersedParams)
                dispP[i] += epsilon
                grad.append((costFunction(dispP)-ref)/epsilon)
            logging.debug("grad %s"%grad)
            return np.array(grad)

        def gradiantFunction(dispersedParams):
            gradtest = testgradiantFunction(dispersedParams)
            # calcule la dérivée par différence finie sur l'ensemble des parametres
            epsilon = 1e-8

            self.nominalAgain()
            paramForGrad = list()
            for i,param in enumerate(dispersedParams):
                paramForGrad.append(param + np.zeros(len(dispersedParams)))
                # paramForGrad[-1][i] += -epsilon
                # paramForGrad[-1][i] += -(params_max[i]-params_min[i])*epsilon
            # logging.debug("jeu de parametre pour gradiant shape: %s %s"%(len(paramForGrad),paramForGrad[0].shape))
            # logging.debug("jeu de parametre pour gradiant : %s"%(paramForGrad[:2]))
            logging.debug("objective %s"%(objective))
            logging.debug("ecart entre dispersedParam et paramForGrad %s"%(np.array(dispersedParams)-np.array(paramForGrad)[:,0]))

            for element in self.elements:
                element.put_tirages(paramForGrad)
            for ref in self.refs:
                ref.put_tirages(paramForGrad)
            if len(paramForGrad) != 0 : raise(ValueError)
            nom_m, nom_c, nom_I = self.assemble()
            nom_objectiv = getObjectiveValue(nom_m, nom_c, nom_I)
            logging.debug("nom_objectiv %s"%nom_objectiv)
            logging.debug("devrait être egal à %s"%costFunction(dispersedParams))

            self.nominalAgain()
            paramForGrad = list()
            for i,param in enumerate(dispersedParams):
                paramForGrad.append(param + np.zeros(len(dispersedParams)))
                paramForGrad[-1][i] += epsilon
                # paramForGrad[-1][i] += (params_max[i]-params_min[i])*epsilon
            # logging.debug("jeu de parametre pour gradiant shape: %s %s"%(len(paramForGrad),paramForGrad[0].shape))
            # logging.debug("jeu de parametre pour gradiant : %s"%(paramForGrad[:2]))

            for element in self.elements:
                element.put_tirages(paramForGrad)
            for ref in self.refs:
                ref.put_tirages(paramForGrad)
            if len(paramForGrad) != 0 : raise(ValueError)
            
            grad_m,grad_c,grad_I = self.assemble()
            grad_objectiv = getObjectiveValue(grad_m,grad_c,grad_I)

            # grad = (grad_objectiv - nom_objectiv)/(2*epsilon)
            grad = (grad_objectiv - nom_objectiv)/(epsilon)
            logging.debug("idem en rapide %s"%grad)
            logging.debug("diff %s"%(grad-gradtest))

            return grad
            # return (grad_objectiv - nom_objectiv)/(epsilon)


            



        
        def callback(dispersedParams):
            self.dispers(dispersedParams)
            m,c,I = self.assemble()
            if 'cr_' in objective:
                logging.info(np.sqrt(c[1]**2+c[2]**2))
            elif 'cx_' in objective:
                logging.info(c[0])
            elif 'cy_' in objective:
                logging.info(c[1])
            elif 'cz_' in objective:
                logging.info(c[2])
            elif 'Ixx_' in objective:
                logging.info(I[0,0])
            elif 'Iyy_' in objective:
                logging.info(I[1,1])
            elif 'Izz_' in objective:
                logging.info(I[2,2])
            elif 'Ixy_' in objective:
                logging.info(I[0,1])
            elif 'Ixz_' in objective:
                logging.info(I[0,2])
            elif 'Iyz_' in objective:
                logging.info(I[1,2])
            elif 'm_' in objective:
                logging.info(m)
            else:
                raise(ValueError)
            
        
        # res = minimize(fun = costFunction, x0 = params_nom, bounds = bounds, callback = callback, method = 'TNC')
        res = minimize(fun = costFunction, jac = gradiantFunction, x0 = params_nom, bounds = bounds, callback = callback, method = 'L-BFGS-B')
        # res = minimize(fun = costFunction, jac = testgradiantFunction, x0 = params_nom, bounds = bounds, callback = callback, method = 'L-BFGS-B')
        # res = minimize(fun = costFunction, jac = gradiantFunction, x0 = params_nom, bounds = bounds, callback = callback, method = 'TNC')
        # res = minimize(fun = costFunction, x0 = params_nom, bounds = bounds, callback = callback, method = 'L-BFGS-B')
        # res = minimize(fun = costFunction, x0 = params_nom, bounds = bounds, callback = None)
        logging.debug(res)
        self.nominalAgain()
        if res['success'] is not True :
            raise(ValueError("l'optimisation n'a pas abouti sur ce parametre: %s"%objective))
        if 'max' in objective :
            logging.info("%s = %s"%(objective, -res['fun']))
            return -res['fun']
        elif 'min' in objective :
            logging.info("%s = %s"%(objective, res['fun']))
            return res['fun']
        else :
            raise(ValueError)
        
    def bornesByOptimization(self, only = None):
        m,c,I = self.assemble()
        res = dict()
        if only is None or only == "min":
            m_min = self.optimize(objective = 'm_min') 
            c_min = np.array([self.optimize(objective = obj) for obj in ['cx_min', 'cy_min', 'cz_min']])
            I_min = np.array([[self.optimize(objective = 'Ixx_min'),self.optimize(objective = 'Ixy_min'),self.optimize(objective = 'Ixz_min')],
                     [0, self.optimize(objective = 'Iyy_min'),self.optimize(objective = 'Iyz_min')],
                     [0,0,self.optimize(objective = 'Izz_min')]])
            I_min[1,0]=I_min[0,1]
            I_min[2,0]=I_min[0,2]
            I_min[2,1]=I_min[1,2]
            logging.debug('c_min %s'%c_min)
            logging.debug('I_min %s'%I_min)
            res["mmin"] = m_min
            res["cmin"] = c_min
            res["Imin"] = I_min
        
        logging.debug('c     %s'%c)
        logging.debug('I     %s'%I)

        if only is None or only == "max":
            m_max = self.optimize(objective = 'm_max') 
            c_max = np.array([self.optimize(objective = obj) for obj in ['cx_max', 'cy_max', 'cz_max']])
            I_max = np.array([[self.optimize(objective = 'Ixx_max'),self.optimize(objective = 'Ixy_max'),self.optimize(objective = 'Ixz_max')],
                     [0, self.optimize(objective = 'Iyy_max'),self.optimize(objective = 'Iyz_max')],
                     [0,0,self.optimize(objective = 'Izz_max')]])
            I_max[1,0]=I_max[0,1]
            I_max[2,0]=I_max[0,2]
            I_max[2,1]=I_max[1,2]
            res["mmax"] = m_max
            res["cmax"] = c_max
            res["Imax"] = I_max
            logging.debug('c_max %s'%c_max)
            logging.debug('I_max %s'%I_max)
 
        return res

    def MTC(self, ntirages = 10):
        
        # self.MTC = copy.deepcopy(self)
        self.findRefs()

        logging.info("start MTC")
        for e in self.elements:
            logging.info("make tirage element %s"%e.name)
            e.make_tirages(ntirages)
        for ref in self.refs:
            logging.info("make tirage ref %s"%ref.name)
            ref.make_tirages(ntirages)
        
        m,c,i = self.assemble()
        logging.debug('nombre de tirage de m: %s'%m.shape)

        self.nominalAgain()
        return m,c,i
        
if __name__ == "__main__":
    ref1 = Ref(name = 'ref1', O = [10.,0.,2.], psi = 150., theta = 10., phi = 0.)
    refe1 = Ref(name = 'refe1', O = [1,1,1], psi = 0., theta = 30., phi = 80., ref = None)
    e1 = Element(name = 'e1', m = 10., c = array([1,2,3]), I = array([[1,0,0],
                                                         [0,1,0],
                                                         [0,0,1]]), ref = None)
    refe2 = Ref(name = 'refe2', O = [1,-10,1], psi = 0., theta = -80., phi = 90.)
    e2 = Element(name = 'e2', m = 10., c = array([0,1,2]), I = array([[1,0,0],
                                                         [0,1,0],
                                                         [0,0,1]]), ref = None)
    
    ensemble = Ensemble(e1,e2)
    print(ensemble.assemble())
    
    
    # principe des intervals.
    # ne fonctionne pas car on redivise pas m sans tenir compte 
    # du fait que m min ne peut pas avoir été obtenu avec ms min...
    # e1 = Element(m = mpi(10,15.), c = array([1,2,3]), I = array([[1,0,0],
    #                                                              [0,1,0],
    #                                                              [0,0,1]]), ref = None)
    
    
    ref1 = Ref(name = 'ref1', O = [10.,0.,2.], psi = 150., theta = 0., phi = 0., depo = dict(O=[1,1,1] ,
                                                                                           psi=1))
    e1 = Element(name = 'e2', m = 1., c = array([1,2,3]), I = array([[1,0,0],
                                                        [0,1,0],
                                                        [0,0,1]]), ref = ref1,
                  garde=dict(m=0.1 , I=np.ones((3,3))*0.1))
    ensemble = Ensemble(e1,e2)
    
    
    print('#######  bornes by optimization #########')
    ensemble.bornesByOptimization()
    # ensemble = Ensemble(e1)
    # ensemble.optimize('cx_max')

    
    # print('#######  MTC #########')
    # m,c,I = ensemble.MTC(ntirages = 50000)
    # print('m ',m.shape,m)
    # print('c ',c.shape, c)
    # print('I ',I.shape, I)
    
    
    # for i in range(50000):
    #     ensemble.assemble()
