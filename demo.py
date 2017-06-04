


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
        self.O = asarray(O)
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
        if self.depo is not None and 'O' in self.depo : self.depo['O'] = np.asarray(self.depo['O'])

        # if depo is not None:
        #     if 'O' in depo.keys(): self.O_d = depo['O']
        #     if 'psi' in depo.keys(): self.psi_d = depo['psi']
        #     if 'theta' in depo.keys(): self.theta_d = depo['theta']
        #     if 'phi' in depo.keys(): self.phi_d = depo['phi']

        self.calc_mat()

    def calc_mat(self):
        # premiere rotation d'angle psi autour de X
        # deuxième rotation d'angle theta autour de Y
        # troisième rotation d'angle phi autour de X
        self.mat = getMatRotx(self.phi)@getMatRoty(self.theta)@getMatRotx(self.psi)
        if self.ref is not None:
            self.O = self.ref.O + self.O
            self.mat = self.ref.mat@self.mat

    def getParams(self,params = None):
        if params is None: params = list()
        len0 = len(params)
        if self.depo is not None:
            if 'O' in self.depo.keys(): params += [a for a in self.depo['O'].tolist() if a != 0]
            if 'psi' in self.depo.keys() and self.depo['psi'] != 0 : params.append(self.depo['psi'])
            if 'theta' in self.depo.keys() and self.depo['theta'] != 0 : params.append(self.depo['theta'])
            if 'phi' in self.depo.keys() and self.depo['phi'] != 0 : params.append(self.depo['phi'])
        logging.debug("found %s parameters to optimize in ref %s"%(len(params)-len0,self.name))
        return params

    def putParams(self, allParamsValues):
        self.O = copy.deepcopy(self.O_n)
        self.psi = copy.deepcopy(self.psi_n)
        self.theta = copy.deepcopy(self.theta_n)
        self.phi = copy.deepcopy(self.phi_n)
        if self.depo is not None:
            if 'O' in self.depo.keys() : 
                for i in range(3): 
                    if self.depo['O'][i] != 0 : self.O[i] += allParamsValues.pop(0) 
            if 'psi' in self.depo.keys() and self.depo['psi'] != 0 : self.psi = self.psi_n + allParamsValues.pop(0)
            if 'theta' in self.depo.keys() and self.depo['theta'] != 0 : self.theta = self.theta_n + allParamsValues.pop(0)
            if 'phi' in self.depo.keys() and self.depo['phi'] != 0 : self.phi = self.phi_n + allParamsValues.pop(0)
        self.calc_mat()
        return allParamsValues

    def nominalAgain(self):
        self.O = copy.deepcopy(self.O_n)
        self.psi = copy.deepcopy(self.psi_n)
        self.theta = copy.deepcopy(self.theta_n)
        self.phi = copy.deepcopy(self.phi_n)
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
                O = np.ones((ntirages,3))
                for i in range(3):
                    O[:,i] = np.random.uniform(-self.depo['O'][i], self.depo['O'][i], ntirages)
                self.O = self.O + O
            if 'psi' in self.depo.keys() : 
                self.psi += np.random.uniform(self.depo['psi'], self.depo['psi'], ntirages)
            if 'theta' in self.depo.keys() : 
                self.theta += np.random.uniform(self.depo['theta'], self.depo['theta'], ntirages)
            if 'phi' in self.depo.keys() : 
                self.phi += np.random.uniform(self.depo['phi'], self.depo['phi'], ntirages)

        self.calc_mat()


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
           

    def toRef0(self):
        if self.ref is not None:
#            print("debug", type(self.ref.mat), selfa.ref.mat, self.c)
            #on etend self.c a une nouvelle dimension pour pouvoir traiter d'un coup les tirages.
            #la ligne d'origine:
#            c = self.ref.O + self.ref.mat@self.c
            c = self.ref.O + np.squeeze(self.ref.mat@np.expand_dims(self.c,self.c.ndim+1))
            # print("debug2",c)
            logging.debug("toRef0 : I shape = %s"%(self.I.shape,))
            logging.debug("toRef0 : ref.mat shape = %s"%(self.ref.mat.shape,))
            if self.I.ndim == 2:
                I = self.ref.mat.T@self.I@self.ref.mat
            else:
#                si on est en MTC,  il faut transposer uniquement les deux derniers axes au lieu de tt la matrice
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
            #                si on est en MTC,  il faut transposer uniquement les deux derniers axes au lieu de tt la matrice
                            incertRef0['I'] = self.ref.mat.transpose(0,2,1)@incert['I']@self.ref.mat
                    return incertRef0
                else:
                    return None
            return Element(self.name, self.m, c, I, garde = toRefIncert(self.garde), 
                                                    meco = toRefIncert(self.meco),
                                                    disp = toRefIncert(self.disp))
        else:
            return self
        
    def getParams(self, params = None):
        # print(params)
        if params is None: params = list()
        len0 = len(params)
        if self.garde is not None:
            if 'm' in self.garde.keys(): params.append(self.garde['m'])
            if 'c' in self.garde.keys(): params += [a for a in self.garde['c'] if a != 0]
            if 'I' in self.garde.keys(): params += [a for a in self.garde['I'].flatten() if a != 0]
            if 'c' in self.garde.keys() :logging.debug("test %s"%(self.garde['c']))
        if self.meco is not None:
            if 'm' in self.meco.keys(): params.append(self.meco['m'])
            if 'c' in self.meco.keys(): params += [a for a in self.meco['c'] if a != 0]
            if 'I' in self.meco.keys(): params += [a for a in self.meco['I'].flatten() if a != 0]
        if self.disp is not None:
            if 'm' in self.disp.keys(): params.append(self.disp['m'])
            if 'c' in self.disp.keys(): params += [a for a in self.disp['c'] if a != 0]
            if 'I' in self.disp.keys(): params += [a for a in self.disp['I'].flatten() if a != 0]
        logging.debug("found %s parameters to optimize in elemeny %s"%(len(params)-len0,self.name))

        return params

    def putParams(self, allParamsValues):
        self.nominalAgain()
        def putdisp(m, c, I, incert):
            if 'm' in incert.keys(): m = m + allParamsValues.pop(0)
            if 'c' in incert.keys():
                for i in range(3):  
                    if incert['c'][i] != 0 : c[i] = c[i] + allParamsValues.pop(0) 
            if 'I' in incert.keys():
                for i in range(3):
                    for j in range(3):
                        if incert['I'][i,j] != 0 :
                            # temp = allParamsValues.pop(0) 
                            I[i,j] = I[i,j] + allParamsValues.pop(0)
                            # logging.debug("put inertie : %s, %s, %s"%(temp, i,j))
            return m,c,I
        self.m,self.c,self.I = putdisp(self.m,self.c,self.I,self.garde)
        self.m,self.c,self.I = putdisp(self.m,self.c,self.I,self.meco)
        self.m,self.c,self.I = putdisp(self.m,self.c,self.I,self.disp)
        return allParamsValues

    def nominalAgain(self):
        self.m = copy.deepcopy(self.m_n)
        self.c = copy.deepcopy(self.c_n)
        self.I = copy.deepcopy(self.I_n)

        

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
                c = np.ones((ntirages,3))
                for i in range(3):
                    c[:,i] = np.random.uniform(-self.garde['c'][i], self.garde['c'][i], ntirages)
                self.c += c
            if 'I' in self.garde.keys(): 
                I = np.ones((ntirages,3,3))
                for i in range(3):
                    for j in range(3):
                        I[:,i,j] = np.random.uniform(-self.garde['I'][i,j], self.garde['I'][i,j], ntirages)
                self.I = self.I + I
        if self.meco is not None :
            if 'm' in self.meco.keys(): 
                self.m += np.random.uniform(-self.meco['m'], self.meco['m'], ntirages)
            if 'c' in self.meco.keys(): 
                c = np.ones((ntirages,3))
                for i in range(3):
                    c[:,i] = np.random.uniform(-self.meco['c'][i], self.meco['c'][i], ntirages)
                self.c += c
            if 'I' in self.meco.keys(): 
                I = np.ones((ntirages,3,3))
                for i in range(3):
                    for j in range(3):
                        I[:,i,j] = np.random.uniform(-self.meco['I'][i,j], self.meco['I'][i,j], ntirages)
                self.I = self.I + I
        if self.disp is not None :
            if 'm' in self.disp.keys(): 
                self.m += np.random.uniform(-self.disp['m'], self.disp['m'], ntirages)
            if 'c' in self.disp.keys():
                c = np.ones((ntirages,3))
                for i in range(3):
                    c[:,i] = np.random.uniform(-self.disp['c'][i], self.disp['c'][i], ntirages)
                self.c += c
            if 'I' in self.disp.keys(): 
                I = np.ones((ntirages,3,3))
                for i in range(3):
                    for j in range(3):
                        I[:,i,j] = np.random.uniform(-self.disp['I'][i,j], self.disp['I'][i,j], ntirages)
                self.I = self.I + I
        logging.debug("make tirage: shape of m,c I %s %s %s"%(self.m.shape,self.c.shape,self.I.shape))
            

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

    def toRef0(self):
        # self.prepareRefs()
        
        return [e.toRef0() for e in self.elements]

    def assemble(self):

        elements = self.toRef0()

        ms = array([e.m for e in elements])
        cs = array([e.c for e in elements])
        Is = array([e.I for e in elements])
#        print("ms: ",ms.shape, ms)
#        print("cs: ",cs.shape, cs)
#        print("Is: ",Is.shape, Is)

        m = np.array(ms.sum(axis = 0))
#        c  = np.sum(ms*cs, axis = 0)/m
        c  = np.sum((np.expand_dims(ms,ms.ndim+1)*cs), axis = 0)/np.expand_dims(m,m.ndim+1)
#        print("c",c.shape,c)

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
        d = (cs - c)
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
            elements = self.toRef0()
            logging.warning("ATTENTION : les gmd ne sont pas renvoyées vers ref0 pour l'instant...")

            cx_g_contribs,cy_g_contribs,cz_g_contribs = list(), list(), list()
            Ixx_g_contribs,Iyy_g_contribs,Izz_g_contribs = list(), list(), list()
            for e in elements:
                logging.debug("%s garde : %s"%(e.name,e.garde))
                if  e.garde is not None:
                    if 'm' in e.garde.keys():  e_m_g = e.garde['m']
                    else: e_m_g = 0
                    if 'c' in e.garde.keys():  e_c_g = e.garde['c']
                    else: e_c_g = np.zeros(3)
                    if 'I' in e.garde.keys():  e_I_g = e.garde['I']
                    else: e_I_g = np.zeros((3,3))
                else:
                    e_m_g = 0
                    e_c_g = np.zeros(3)
                    e_I_g = np.zeros((3,3))

                logging.debug("garde m,g,I,%s,%s,%s"%(e_m_g,e_c_g,e_I_g))
                cx_g_contribs.append(e_m_g*(e.c[0]-ce[0])/me + e.m*e_c_g[0]/me)
                cy_g_contribs.append(e_m_g*(e.c[1]-ce[1])/me + e.m*e_c_g[1]/me)
                cz_g_contribs.append(e_m_g*(e.c[2]-ce[2])/me + e.m*e_c_g[2]/me)

                Ixx_g_contribs.append(e_I_g[0,0] + e_m_g*(e.c[1]-ce[1])**2+e_m_g*(e.c[2]-ce[2])**2 +
                                        e.m*(e.c[1]-ce[1])*e_c_g[1]+e.m*(e.c[2]-ce[2])*e_c_g[2])
                Iyy_g_contribs.append(e_I_g[1,1] + e_m_g*(e.c[0]-ce[0])**2+e_m_g*(e.c[2]-ce[2])**2 +
                                        e.m*(e.c[0]-ce[0])*e_c_g[0]+e.m*(e.c[2]-ce[2])*e_c_g[2])
                Izz_g_contribs.append(e_I_g[2,2] + e_m_g*(e.c[0]-ce[0])**2+e_m_g*(e.c[1]-ce[1])**2 +
                                        e.m*(e.c[0]-ce[0])*e_c_g[0]+e.m*(e.c[1]-ce[1])*e_c_g[1])
            logging.debug("contribs cx %s \n cy %s \n cz %s\n Ixx %s \n Iyy %s \n czz %s"%( \
                                    cx_g_contribs,cy_g_contribs,cz_g_contribs, \
                                    Ixx_g_contribs,Iyy_g_contribs,Izz_g_contribs))
            return np.asarray(cx_g_contribs).sum(), np.asarray(cy_g_contribs).sum(), np.asarray(cz_g_contribs).sum(), \
                     np.asarray(Ixx_g_contribs).sum(), np.asarray(Iyy_g_contribs).sum(), np.asarray(Izz_g_contribs).sum()



    def find_parameters(self):
        self.allParams = list()
        for e in self.elements:
            self.allParams = e.getParams(self.allParams)
        for ref in self.refs:
            self.allParams = ref.getParams(self.allParams)
                
    def dispers(self, allParamsValues):
        if isinstance(allParamsValues, np.ndarray): allParamsValues = allParamsValues.tolist()
        for e in self.elements:
            allParamsValues = e.putParams(allParamsValues)
        for ref in self.refs:
            allParamsValues = ref.putParams(allParamsValues)
        if len(allParamsValues) != 0 : raise(ValueError)
    
    def nominalAgain(self):
        for e in self.elements:
            e.nominalAgain()
        for ref in self.refs:
            ref.nominalAgain()

    def optimize(self, objective = 'cr_max'):
        logging.debug("lancement d'une optimisation : %s"%(objective))

        self.findRefs()
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

        
        def costFunction(dispersedParams):
            logging.debug("dispersion avec ces parametres : %s"%dispersedParams[:3])
            self.dispers(dispersedParams)
            # logging.debug("Inertie : %s"%(self.elements[0].I))
            m,c,I = self.assemble()
            logging.debug("resultat : %s %s %s"%(m,c,I))

            if objective == 'cr_max':
                return -np.sqrt(c[1]**2+c[2]**2)
            elif objective == 'cr_min':
                return np.sqrt(c[1]**2+c[2]**2)
            elif objective == 'cx_max':
                return -c[0]
            elif objective == 'cx_min':
                return c[0]
            elif objective == 'cy_max':
                return -c[1]
            elif objective == 'cy_min':
                return c[1]
            elif objective == 'cz_max':
                return -c[2]
            elif objective == 'cz_min':
                return c[2]
            elif objective == 'Ixx_max':
                return -I[0,0]
            elif objective == 'Ixx_min':
                return I[0,0]
            elif objective == 'Iyy_max':
                return -I[1,1]
            elif objective == 'Iyy_min':
                return I[1,1]
            elif objective == 'Izz_max':
                return -I[2,2]
            elif objective == 'Izz_min':
                return I[2,2]
            else:
                raise(ValueError)
        
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
            else:
                raise(ValueError)
            
        
        res = minimize(fun = costFunction, x0 = params_nom, bounds = bounds, callback = callback)#, method = 'TNC')
        # res = minimize(fun = costFunction, x0 = params_nom, bounds = bounds, callback = None)
        logging.debug(res)
        self.nominalAgain()
        if res['success'] is not True :
            raise(ValueError("l'optimisation n'a pas abouti"))
        if 'max' in objective :
            logging.info("%s = %s"%(objective, -res['fun']))
            return -res['fun']
        elif 'min' in objective :
            logging.info("%s = %s"%(objective, res['fun']))
            return res['fun']
        else :
            raise(ValueError)
        
    def bornesByOptimization(self):
        m,c,I = self.assemble()
        c_min = [self.optimize(objective = obj) for obj in ['cx_min', 'cy_min', 'cz_min']]
        c_max = [self.optimize(objective = obj) for obj in ['cx_max', 'cy_max', 'cz_max']]
        I_min = [self.optimize(objective = obj) for obj in ['Ixx_min', 'Iyy_min', 'Izz_min']]
        I_max = [self.optimize(objective = obj) for obj in ['Ixx_max', 'Iyy_max', 'Izz_max']]
        logging.debug('c_min %s'%c_min)
        logging.debug('c     %s'%c)
        logging.debug('c_max %s'%c_max)
        logging.debug('I_min %s'%I_min)
        logging.debug('I     %s'%I)
        logging.debug('I_max %s'%I_max)

    def MTC(self, ntirages = 10):
        
        # self.MTC = copy.deepcopy(self)
        logging.info("start MTC")
        for e in self.elements:
            logging.info("make tirage element %s"%e.name)
            e.make_tirages(ntirages)
        for ref in self.refs:
            logging.info("make tirage ref %s"%ref.name)
            ref.make_tirages(ntirages)
        
        m,c,i = self.assemble()

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
    
    
    ref1 = Ref(name = 'ref1', O = [10.,0.,2.], psi = 150., theta = 10., phi = 0., depo = dict(O=[1,1,1] ,
                                                                            psi=1))
    e1 = Element(name = 'e2', m = 1., c = array([1,2,3]), I = array([[1,0,0],
                                                        [0,1,0],
                                                        [0,0,1]]), ref = ref1,
                  garde=dict(m=0.1 , I=np.ones((3,3))*0.1))
    ensemble = Ensemble(e1,e2)
    
    
    print('#######  bornes by optimization #########')
    ensemble.bornesByOptimization()
    
    
    print('#######  MTC #########')
    m,c,I = ensemble.MTC(ntirages = 50000)
    print('m ',m.shape,m)
    print('c ',c.shape, c)
    print('I ',I.shape, I)
    
    
    # for i in range(50000):
    #     ensemble.assemble()
