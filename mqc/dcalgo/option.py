from vqechem.set_options import set_options

class mbe_option(object):
    """
    attributes of mbe_option
    solver: method to get mbe energies
    isTI: whether the system is transition invariant
    link_atom: how to add link atoms to saturate the bonds
    ncas1 : number of orbitals in active space. 
    vqe options setting
    save_directory: directory to save Hamiltonians.
    """
    def __init__(self,**kwargs) -> None:
        ##  mbe setting
        self.solver : str = "pyscf_rhf"
        self.isTI: bool = False
        self.link_atom: str = "extend"
        self.ncas1 = None
        self.ncas2 = None
        self.ncas3 = None
        
        ## vqe
        self.algorithm = "adapt-vqe"
        
        ## scf
        self.ncas = None
        self.ncore = None
        self.mo_list = None
        self.qmmm_coords = None
        self.qmmm_charges = None
        self.shift = 0.5

        ## ops
        self.ops_class = 'fermionic'
        self.spin_sym = "sa"
        self.ops_pool = None

        ## ansatz
        self.method = "adapt"
        self.form = "unitary"
        self.Nu = 10
        
        ## opt
        self.maxiter = 300
        self.tol = 0.01

        ## orbital optimize
        self.basis = "minao"
        self.low_level = "mp2"
        self.type = "hf"
        self.diag = "vqe"

        ## file
        self.save_directory = None
        
        self.vqe_options = None

        for (key,value) in kwargs.items():
            if hasattr(self,key):
                setattr(self,key,value)

    def update(self,**kwargs):
        for (key,value) in kwargs.items():
            if hasattr(self,key):
                setattr(self,key,value)

    def save_option(self):
        pass

    def load_option(self):
        pass

    def make_vqe_options(self,**kwargs):    
        self.update(**kwargs)    
        self.vqe_options = {
                            'vqe' : {'algorithm':self.algorithm},
                            'scf' : {'ncas':self.ncas,'ncore':self.ncore,'mo_list':self.mo_list,
                                        'shift':self.shift,'qmmm_coords':self.qmmm_coords,
                                        'qmmm_charges':self.qmmm_charges},
                            'ops' : {'class':self.ops_class,'spin_sym':self.spin_sym,'ops_pool': self.ops_pool},
                            'ansatz' : {'method':self.method,'form':self.form,'Nu':self.Nu},
                            'opt' : {'maxiter':self.maxiter,'tol':self.tol},
                            'oo' : {'basis':self.basis,'low_level':self.low_level,'type':self.type,'diag':self.diag},
                            'file': {'save_directory':self.save_directory}
                            }
        return self.vqe_options



class dmet_option(object):
    def __init__(self) -> None:
        pass