import numpy as np


class BaseFragment():
    def __init__(self,structure):
        self.geometry = None


class Fragment():
    ''' Define fragments (Maybe replaced with https://github.com/FragIt/fragit-main)
    '''  
    def __init__(self, geometry,natom_per_fragment = None,atom_list = None, mm_charges=None):
        '''
        '''
        self._nfrag = 0
        self._qm_list = []
        self._geometry = geometry
        self._mm_list = []
        self._mm_charges = mm_charges
        self.natom_per_fragment = natom_per_fragment
        self.atom_list = atom_list
        self.get_qm_atom()
        self.get_mm_atom()
        self.connection = []
        self.get_connection()

    def get_qm_atom(self):
        '''
        '''
        if not (self.natom_per_fragment == None):
            natom_per_fragment = self.natom_per_fragment
            self._nfrag = len(self._geometry)//natom_per_fragment
            for i in range(self._nfrag):
                frag = np.arange(natom_per_fragment*i,natom_per_fragment*i+natom_per_fragment)
                #print(get_distance(geometry[frag[0]][1],geometry[frag[1]][1]))
                #print(get_distance(geometry[frag[0]][1],geometry[frag[2]][1]))
                #assert(get_distance(self._geometry[frag[0]][1],self._geometry[frag[1]][1])<1.0)
                #assert(get_distance(self._geometry[frag[0]][1],self._geometry[frag[2]][1])<1.0)
                self._qm_list.append(list(frag))
        elif not (self.atom_list==None):
            self._qm_list = self.atom_list
        else:
            #print(self._qm_list)
            assert(self._qm_list ==None)
            print('atom list or natom_per_fragment not given')
            exit()
    def get_mm_atom(self):
        '''
        '''
        list_all = list(range(self._nfrag))
        for i in range(self._nfrag):
            self._mm_list.append([x for x in list_all if x not in self._qm_list[i]])

    def get_connection(self):
        connection=[]
        natom = len(self._geometry)
        for i in range(natom):
            connection_tmp = []
            for j in range(natom):
                if j==i:
                    dist = 2
                else:
                    dist = get_distance(self._geometry[i][1],self._geometry[j][1])
                if dist < 1.75:
                    connection_tmp.append(j)
            connection.append(connection_tmp)
        self.connection = connection
        print(connection)