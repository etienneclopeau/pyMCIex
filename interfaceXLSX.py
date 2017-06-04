
from demo import Ref, Element, Ensemble
from openpyxl import load_workbook
from collections import OrderedDict
import numpy as np
import copy
import logging
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', 
                    level=logging.INFO, 
                    datefmt='%I:%M:%S')
wb = load_workbook('assemblage.xlsx')
wb.get_sheet_names()


sheet_elements = wb['éléments']

c_m = 2
c_m_inerte = c_m + 4
c_cx = c_m_inerte + 5
c_cy = c_cx + 4
c_cz = c_cy + 4
c_Ix = c_cz + 5
c_Iy = c_Ix + 4
c_Iz = c_Iy + 4
c_Ixy = c_Iz + 5
c_Ixz = c_Ixy + 4
c_Iyz = c_Ixz + 4

print('chargement des elements')
elements = OrderedDict()
for row in sheet_elements.iter_rows(row_offset = 4):
    if row[1].value is None: continue
    row = [c.value if c.value is not None else 0 for c in row]
    # row = [c.value for c in row]
    
    
    m = row[c_m]
    c = np.array([row[c_cx],row[c_cy],row[c_cz]])/1000.
    I = np.array([[row[c_Ix]  , -row[c_Ixy], -row[c_Ixz] ],
                  [-row[c_Ixy],  row[c_Iy] , -row[c_Iyz] ],
                  [-row[c_Ixz], -row[c_Iyz],  row[c_Iz]  ]])
    garde = dict(m = row[c_m+1],
                 c = np.array([row[c_cx+1],row[c_cy+1],row[c_cz+1]])/1000.,
                 I = np.array([[row[c_Ix+1]  , row[c_Ixy+1], row[c_Ixz+1] ],
                               [row[c_Ixy+1] ,  row[c_Iy+1], row[c_Iyz+1] ],
                               [row[c_Ixz+1] , row[c_Iyz+1], row[c_Iz+1]  ]]),
                 )
    def delIfNone(incert):
        if incert['m'] == 0 : incert.pop('m')
        if (incert['c']==0).all() : incert.pop('c')
        if (incert['I']==0).all() : incert.pop('I')
        return incert
    garde = delIfNone(garde)
    meco = dict(m = row[c_m+2],
                 c = np.array([row[c_cx+2],row[c_cy+2],row[c_cz+2]])/1000.,
                 I = np.array([[row[c_Ix+2]  , row[c_Ixy+2], row[c_Ixz+2] ],
                                 [ row[c_Ixy+2],  row[c_Iy+2] , row[c_Iyz+2] ],
                                 [ row[c_Ixz+2],  row[c_Iyz+2],  row[c_Iz+2]  ]]),
                 )
    meco = delIfNone(meco)

    disp = dict(m = row[c_m+3],
                 c = np.array([row[c_cx+3],row[c_cy+3],row[c_cz+3]])/1000.,
                 I = np.array([[row[c_Ix+3]  ,  row[c_Ixy+3],  row[c_Ixz+3] ],
                                 [ row[c_Ixy+3],  row[c_Iy+3] ,  row[c_Iyz+3] ],
                                 [ row[c_Ixz+3],  row[c_Iyz+3],  row[c_Iz+3]  ]]),
                 )
    disp = delIfNone(disp)

    logging.debug("%s : %s, %s, %s,%s, %s, %s"%(row[1],m,c,I,garde,meco,disp))
    # raise
    if row[1] in elements.keys():
        raise(ValueError("l'element %s a déjà été renseigné"%row[1]))
    else: 
        print(row[1])
        if row[1] != 0 : 
            elements[row[1]] = Element(name = row[1] , m = m, c = c, I = I, garde = garde, meco = meco , disp = disp, )


sheet_referentiel = wb['référentiels']
print("chargement des référentiels")
refs = OrderedDict()
for row in sheet_referentiel.iter_rows(row_offset = 4):
    if row[1].value is None: continue
    row = [c.value if c.value is not None else 0 for c in row]

    if row[1] in refs.keys():
        raise(ValueError("le référentiel %s a déjà été renseigné"%row[1]))
    else:
        print(row[1])
        if row[1] != 0 : 
            if row[8] != 0: ref = refs[row[8]]
            else: ref = None
            refs[row[1]] = Ref(name = row[1], O = row[2:5], psi = row[5], theta = row[6], phi = row[7], ref  = ref,
                       depo = dict( O = row[10:13], psi = row[13], theta = row[14], phi = row[15]))



sheet_constituants = wb['constituants']
print("chargement des constituants")
constituants = OrderedDict()
for row in sheet_constituants.iter_rows(row_offset = 4):
    if row[1].value is None: continue
    row = [c.value if c.value is not None else 0 for c in row]
    if row[1] in constituants.keys():
        raise(ValueError("le constituant %s a déjà été renseigné"%row[1]))
    else:
        print(row[1])
        if row[1] != 0 : 
            constituants[row[1]] = [elements[row[2]], refs[row[3]]]

sheet_etats = wb['états']
print("chargement des états")
etats = OrderedDict()
indexCDP = [c.value for c in sheet_etats['A']].index('CDP')
for col in sheet_etats.iter_cols(min_col = 2):
    col = [c.value if c.value is not None else 0 for c in col]
    if col[2] in etats.keys():
        raise(ValueError("l'état %s a déjà été renseigné"%col[2]))
    else:
        etats[col[2]] = [a  for a in col[3:indexCDP] if a != 0]

print(etats)

print("### construction assemblage ###")

for etat in etats.keys():
    print(  '=====================================')
    print(  '=====================================')

    els = list()
    print(etat)
    for c in etats[etat]:
        print(c)
        # print(constituants[c])
        e = copy.deepcopy(constituants[c][0])
        e.ref = constituants[c][1]
        els.append(e)
    etats[etat] = Ensemble(*els)

for etat in etats.keys()
    ensemble = etats[etat]
    print(ensemble.assemble())
    # print(ensemble.optimize('Ixx_max'))
    # raise
    # print('#######  bornes by optimization #########')
    # print(ensemble.bornesByOptimization())
    # raise

    # print('#######  MTC #########')
    # m,c,I = ensemble.domainAnalitic()MTC(ntirages = 50000)
    # print('m ',m.shape,m)
    # print('c ',c.shape, c)
    # print('I ',I.shape, I)

    print('########## analitic ###########')
    print(ensemble.domainAnalitic(approxMethod = True   ))
    # raise()
