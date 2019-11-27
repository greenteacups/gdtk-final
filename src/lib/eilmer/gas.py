# gas.py
# A Python wrapper for GasModel and GasState classes.
# PJ 2019-07-24: start of experiment with FFI.
#    2019-07-25: added Python wrapper
#
from cffi import FFI

ffi = FFI()
ffi.cdef("""
    int cwrap_gas_init();

    int gas_model_new(char* file_name);
    int gas_model_n_species(int gm_i);
    int gas_model_n_modes(int gm_i);
    int gas_model_species_name(int gm_i, int isp, char* name, int n);
    int gas_model_mol_masses(int gm_i, double* mm);

    int gas_state_new(int gm_i);
    int gas_state_set_scalar_field(int gs_i, char* field_name, double value);
    int gas_state_get_scalar_field(int gs_i, char* field_name, double* value);
    int gas_state_set_array_field(int gs_i, char* field_name, double* values, int n);
    int gas_state_get_array_field(int gs_i, char* field_name, double* values, int n);

    int gas_model_gas_state_update_thermo_from_pT(int gm_i, int gs_i);
    int gas_model_gas_state_update_thermo_from_rhou(int gm_i, int gs_i);
    int gas_model_gas_state_update_thermo_from_rhoT(int gm_i, int gs_i);
    int gas_model_gas_state_update_thermo_from_rhop(int gm_i, int gs_i);
    int gas_model_gas_state_update_thermo_from_ps(int gm_i, int gs_i, double s);
    int gas_model_gas_state_update_thermo_from_hs(int gm_i, int gs_i, double h, double s);
    int gas_model_gas_state_update_sound_speed(int gm_i, int gs_i);
    int gas_model_gas_state_update_trans_coeffs(int gm_i, int gs_i);

    int gas_model_gas_state_Cv(int gm_i, int gs_i, double* result);
    int gas_model_gas_state_Cp(int gm_i, int gs_i, double* result);
    int gas_model_gas_state_dpdrho_const_T(int gm_i, int gs_i, double* result);
    int gas_model_gas_state_R(int gm_i, int gs_i, double* result);
    int gas_model_gas_state_internal_energy(int gm_i, int gs_i, double* result);
    int gas_model_gas_state_enthalpy(int gm_i, int gs_i, double* result);
    int gas_model_gas_state_entropy(int gm_i, int gs_i, double* result);
    int gas_model_gas_state_molecular_mass(int gm_i, int gs_i, double* result);

    int gas_model_gas_state_enthalpy_isp(int gm_i, int gs_i, int isp, double* result);
    int gas_model_gas_state_entropy_isp(int gm_i, int gs_i, int isp, double* result);
    int gas_model_gas_state_gibbs_free_energy_isp(int gm_i, int gs_i, int isp, double* result);

    int gas_model_massf2molef(int gm_i, double* massf, double* molef);
    int gas_model_molef2massf(int gm_i, double* molef, double* massf);
    int gas_model_gas_state_get_molef(int gm_i, int gs_i, double* molef);
    int gas_model_gas_state_get_conc(int gm_i, int gs_i, double* conc);

    int chemical_reactor_new(char* file_name, int gm_i);
    int chemical_reactor_gas_state_update(int cr_i, int gs_i, double t_interval, double* dt_suggest);
""")
so = ffi.dlopen("libgas.so")
so.cwrap_gas_init()

# Service classes that wrap the C-API in a nice Pythonic API...

class GasModel(object):
    def __init__(self, file_name):
        self.file_name = file_name
        self.id = so.gas_model_new(bytes(self.file_name, 'utf-8'))
        self.species_names = []
        buf = ffi.new("char[]", b'\000'*32)
        for i in range(self.n_species):
            so.gas_model_species_name(self.id, i, buf, 32)
            self.species_names.append(ffi.string(buf).decode('utf-8'))
        return

    def __str__(self):
        text = 'GasModel(file="%s", id=%d, species=%s)' % \
            (self.file_name, self.id, self.species_names)
        return text
    
    @property
    def n_species(self):
        return so.gas_model_n_species(self.id)
    
    @property
    def n_modes(self):
        return so.gas_model_n_modes(self.id)

    @property
    def mol_masses(self):
        mm = ffi.new("double[]", [0.0]*self.n_species)
        so.gas_model_mol_masses(self.id, mm)
        return [mm[i] for i in range(self.n_species)]
    
    def update_thermo_from_pT(self, gstate):
        flag = so.gas_model_gas_state_update_thermo_from_pT(self.id, gstate.id)
        if flag < 0: raise Exception("could not update thermo from p,T.")
        return
    def update_thermo_from_rhou(self, gstate):
        flag = so.gas_model_gas_state_update_thermo_from_rhou(self.id, gstate.id)
        if flag < 0: raise Exception("could not update thermo from rho,u.")
        return
    def update_thermo_from_rhoT(self, gstate):
        flag = so.gas_model_gas_state_update_thermo_from_rhoT(self.id, gstate.id)
        if flag < 0: raise Exception("could not update thermo from rho,T.")
        return
    def update_thermo_from_rhop(self, gstate):
        flag = so.gas_model_gas_state_update_thermo_from_rhop(self.id, gstate.id)
        if flag < 0: raise Exception("could not update thermo from rho,p.")
        return
    def update_thermo_from_ps(self, gstate, s):
        flag = so.gas_model_gas_state_update_thermo_from_ps(self.id, gstate.id, s)
        if flag < 0: raise Exception("could not update thermo from p,s.")
        return
    def update_thermo_from_hs(self, gstate, h, s):
        flag = so.gas_model_gas_state_update_thermo_from_hs(self.id, gstate.id, h, s)
        if flag < 0: raise Exception("could not update thermo from h,s.")
        return
    def update_sound_speed(self, gstate):
        flag = so.gas_model_gas_state_update_sound_speed(self.id, gstate.id)
        if flag < 0: raise Exception("could not update sound speed.")
        return
    def update_trans_coeffs(self, gstate):
        flag = so.gas_model_gas_state_update_trans_coeffs(self.id, gstate.id)
        if flag < 0: raise Exception("could not update transport coefficients.")
        return

    def Cv(self, gstate):
        valuep = ffi.new("double *")
        flag = so.gas_model_gas_state_Cv(self.id, gstate.id, valuep)
        if flag < 0: raise Exception("could not compute Cv.")
        return valuep[0]
    def Cp(self, gstate):
        valuep = ffi.new("double *")
        flag = so.gas_model_gas_state_Cp(self.id, gstate.id, valuep)
        if flag < 0: raise Exception("could not compute Cp.")
        return valuep[0]
    def dpdrho_const_T(self, gstate):
        valuep = ffi.new("double *")
        flag = so.gas_model_gas_state_dpdrho_const_T(self.id, gstate.id, valuep)
        if flag < 0: raise Exception("could not compute dpdrho_const_T.")
        return valuep[0]
    def R(self, gstate):
        valuep = ffi.new("double *")
        flag = so.gas_model_gas_state_R(self.id, gstate.id, valuep)
        if flag < 0: raise Exception("could not compute R.")
        return valuep[0]
    def internal_energy(self, gstate):
        valuep = ffi.new("double *")
        flag = so.gas_model_gas_state_internal_energy(self.id, gstate.id, valuep)
        if flag < 0: raise Exception("could not compute internal energy.")
        return valuep[0]
    def enthalpy(self, gstate):
        valuep = ffi.new("double *")
        flag = so.gas_model_gas_state_enthalpy(self.id, gstate.id, valuep)
        if flag < 0: raise Exception("could not compute enthalpy.")
        return valuep[0]
    def entropy(self, gstate):
        valuep = ffi.new("double *")
        flag = so.gas_model_gas_state_entropy(self.id, gstate.id, valuep)
        if flag < 0: raise Exception("could not compute entropy.")
        return valuep[0]
    def molecular_mass(self, gstate):
        valuep = ffi.new("double *")
        flag = so.gas_model_gas_state_molecular_mass(self.id, gstate.id, valuep)
        if flag < 0: raise Exception("could not compute molecular mass.")
        return valuep[0]

    def enthalpy_isp(self, gstate, isp):
        valuep = ffi.new("double *")
        flag = so.gas_model_gas_state_enthalpy_isp(self.id, gstate.id, isp, valuep)
        if flag < 0: raise Exception("could not compute enthalpy for species.")
        return valuep[0]
    def entropy_isp(self, gstate, isp):
        valuep = ffi.new("double *")
        flag = so.gas_model_gas_state_entropy_isp(self.id, gstate.id, isp, valuep)
        if flag < 0: raise Exception("could not compute entropy for species.")
        return valuep[0]
    def gibbs_free_energy_isp(self, gstate, isp):
        valuep = ffi.new("double *")
        flag = so.gas_model_gas_state_gibbs_free_energy_isp(self.id, gstate.id, isp, valuep)
        if flag < 0: raise Exception("could not compute gibbs free energy for species.")
        return valuep[0]

    def massf2molef(self, massf_given):
        nsp = self.n_species
        if type(massf_given) == type([]):
            massf_list = massf_given.copy()
            assert len(massf_list) == self.n_species, "incorrect massf list length"
        elif type(massf_given) == type({}):
            massf_list = []
            for name in self.species_names:
                if name in massf_given.keys():
                    massf_list.append(massf_given[name])
                else:
                    massf_list.append(0.0)
        if abs(sum(massf_list) - 1.0) > 1.0e-6:
            raise Exception("mass fractions do not sum to 1.")
        my_massf = ffi.new("double[]", massf_list)
        my_molef = ffi.new("double[]", [0.0]*self.n_species)
        so.gas_model_massf2molef(self.id, my_massf, my_molef)
        return [my_molef[i] for i in range(self.n_species)]

    def molef2massf(self, molef_given):
        nsp = self.n_species
        if type(molef_given) == type([]):
            molef_list = molef_given.copy()
            assert len(molef_list) == self.n_species, "incorrect molef list length"
        elif type(molef_given) == type({}):
            molef_list = []
            for name in self.species_names:
                if name in molef_given.keys():
                    molef_list.append(molef_given[name])
                else:
                    molef_list.append(0.0)
        if abs(sum(molef_list) - 1.0) > 1.0e-6:
            raise Exception("mole fractions do not sum to 1.")
        my_molef = ffi.new("double[]", molef_list)
        my_massf = ffi.new("double[]", [0.0]*self.n_species)
        so.gas_model_molef2massf(self.id, my_molef, my_massf)
        return [my_massf[i] for i in range(self.n_species)]

    
class GasState(object):
    def __init__(self, gmodel):
        self.gmodel = gmodel
        self.id = so.gas_state_new(self.gmodel.id)

    def __str__(self):
        text = 'GasState(rho=%g' % self.rho
        text += ', p=%g' % self.p
        text += ', T=%g' % self.T
        text += ', u=%g' % self.u
        text += ', massf=%s' % str(self.massf)
        text += ', id=%d, gmodel.id=%d)' % (self.id, self.gmodel.id)
        return text
    
    @property
    def rho(self):
        valuep = ffi.new("double *")
        flag = so.gas_state_get_scalar_field(self.id, b"rho", valuep)
        if flag < 0: raise Exception("could not get density.")
        return valuep[0]
    @rho.setter
    def rho(self, value):
        flag = so.gas_state_set_scalar_field(self.id, b"rho", value)
        if flag < 0: raise Exception("could not set density.")
        return
    
    @property
    def p(self):
        valuep = ffi.new("double *")
        flag = so.gas_state_get_scalar_field(self.id, b"p", valuep)
        if flag < 0: raise Exception("could not get pressure.")
        return valuep[0]
    @p.setter
    def p(self, value):
        flag = so.gas_state_set_scalar_field(self.id, b"p", value)
        if flag < 0: raise Exception("could not set pressure.")
        return
    
    @property
    def T(self):
        valuep = ffi.new("double *")
        flag = so.gas_state_get_scalar_field(self.id, b"T", valuep)
        if flag < 0: raise Exception("could not get temperature.")
        return valuep[0]
    @T.setter
    def T(self, value):
        flag = so.gas_state_set_scalar_field(self.id, b"T", value)
        if flag < 0: raise Exception("could not set temperature.")
        return
    
    @property
    def u(self):
        valuep = ffi.new("double *")
        flag = so.gas_state_get_scalar_field(self.id, b"u", valuep)
        if flag < 0: raise Exception("could not get internal energy.")
        return valuep[0]
    @u.setter
    def u(self, value):
        flag = so.gas_state_set_scalar_field(self.id, b"u", value)
        if flag < 0: raise Exception("could not set internal energy.")
        return
    
    @property
    def a(self):
        valuep = ffi.new("double *")
        flag = so.gas_state_get_scalar_field(self.id, b"a", valuep)
        if flag < 0: raise Exception("could not get sound speed.")
        return valuep[0]
    
    @property
    def k(self):
        valuep = ffi.new("double *")
        flag = so.gas_state_get_scalar_field(self.id, b"k", valuep)
        if flag < 0: raise Exception("could not get conductivity.")
        return valuep[0]
    
    @property
    def mu(self):
        valuep = ffi.new("double *")
        flag = so.gas_state_get_scalar_field(self.id, b"mu", valuep)
        if flag < 0: raise Exception("could not get viscosity.")
        return valuep[0]

    @property
    def massf(self):
        nsp = self.gmodel.n_species
        mf = ffi.new("double[]", [0.0]*nsp)
        flag = so.gas_state_get_array_field(self.id, b"massf", mf, nsp)
        if flag < 0: raise Exception("could not get mass-fractions.")
        return [mf[i] for i in range(nsp)]
    @massf.setter
    def massf(self, mf_given):
        nsp = self.gmodel.n_species
        if type(mf_given) == type([]):
            mf_list = mf_given.copy()
        elif type(mf_given) == type({}):
            mf_list = []
            for name in self.gmodel.species_names:
                if name in mf_given.keys():
                    mf_list.append(mf_given[name])
                else:
                    mf_list.append(0.0)
        if abs(sum(mf_list) - 1.0) > 1.0e-6:
            raise Exception("mass fractions do not sum to 1.")
        mf = ffi.new("double[]", mf_list)
        flag = so.gas_state_set_array_field(self.id, b"massf", mf, nsp)
        if flag < 0: raise Exception("could not set mass-fractions.")
        return mf_list

    @property
    def molef(self):
        nsp = self.gmodel.n_species
        mf = ffi.new("double[]", [0.0]*nsp)
        flag = so.gas_model_gas_state_get_molef(self.gmodel.id, self.id, mf)
        if flag < 0: raise Exception("could not get mole-fractions.")
        return [mf[i] for i in range(nsp)]
    @molef.setter
    def molef(self, molef_given):
        nsp = self.gmodel.n_species
        mf_list = self.gmodel.molef2massf(molef_given)
        mf = ffi.new("double[]", mf_list)
        flag = so.gas_state_set_array_field(self.id, b"massf", mf, nsp)
        if flag < 0: raise Exception("could not set mass-fractions from mole-fractions.")
        # At this point, we may not have the mole-fractions as a list.
        return None

    @property
    def conc(self):
        nsp = self.gmodel.n_species
        myconc = ffi.new("double[]", [0.0]*nsp)
        flag = so.gas_model_gas_state_get_conc(self.gmodel.id, self.id, myconc)
        if flag < 0: raise Exception("could not get concentrations.")
        return [myconc[i] for i in range(nsp)]

    @property
    def u_modes(self):
        n = self.gmodel.n_modes
        if n == 0: return []
        um = ffi.new("double[]", [0.0]*n)
        flag = so.gas_state_get_array_field(self.id, b"u_modes", um, nsp)
        if flag < 0: raise Exception("could not get u_modes.")
        return [um[i] for i in range(n)]
    @u_modes.setter
    def u_modes(self, um_given):
        n = self.gmodel.n_modes
        if n == 0: return []
        if type(um_given) != type([]):
            raise Exception("u_modes needs to be supplied as a list.")
        um = ffi.new("double[]", um_given)
        flag = so.gas_state_set_array_field(self.id, b"u_modes", um, n)
        if flag < 0: raise Exception("could not set u_modes.")
        return um_given

    @property
    def T_modes(self):
        n = self.gmodel.n_modes
        if n == 0: return []
        Tm = ffi.new("double[]", [0.0]*n)
        flag = so.gas_state_get_array_field(self.id, b"T_modes", Tm, nsp)
        if flag < 0: raise Exception("could not get T_modes.")
        return [um[i] for i in range(n)]
    @T_modes.setter
    def T_modes(self, Tm_given):
        n = self.gmodel.n_modes
        if n == 0: return []
        if type(Tm_given) != type([]):
            raise Exception("T_modes needs to be supplied as a list.")
        Tm = ffi.new("double[]", Tm_given)
        flag = so.gas_state_set_array_field(self.id, b"T_modes", Tm, n)
        if flag < 0: raise Exception("could not set T_modes.")
        return Tm_given

    @property
    def k_modes(self):
        n = self.gmodel.k_modes
        if n == 0: return []
        km = ffi.new("double[]", [0.0]*n)
        flag = so.gas_state_get_array_field(self.id, b"k_modes", km, nsp)
        if flag < 0: raise Exception("could not get k_modes.")
        return [km[i] for i in range(n)]
            
    def update_thermo_from_pT(self):
        self.gmodel.update_thermo_from_pT(self)
        return
    def update_thermo_from_rhou(self):
        self.gmodel.update_thermo_from_rhou(self)
        return
    def update_thermo_from_rhoT(self):
        self.gmodel.update_thermo_from_rhoT(self)
        return
    def update_thermo_from_rhop(self):
        self.gmodel.update_thermo_from_rhop(self)
        return
    def update_thermo_from_ps(self, s):
        self.gmodel.update_thermo_from_ps(self, s)
        return
    def update_thermo_from_hs(self, h, s):
        self.gmodel.update_thermo_from_hs(self, h, s)
        return
    def update_sound_speed(self):
        self.gmodel.update_sound_speed(self)
        return
    def update_trans_coeffs(self):
        self.gmodel.update_trans_coeffs(self)
        return

    @property
    def Cv(self):
        return self.gmodel.Cv(self)
    @property
    def Cp(self):
        return self.gmodel.Cp(self)
    @property
    def dpdrho_const_T(self):
        return self.gmodel.dpdrho_const_T(self)
    @property
    def R(self):
        return self.gmodel.R(self)
    @property
    def internal_energy(self):
        return self.gmodel.internal_energy(self)
    @property
    def enthalpy(self):
        return self.gmodel.enthalpy(self)
    @property
    def entropy(self):
        return self.gmodel.entropy(self)
    @property
    def molecular_mass(self):
        return self.gmodel.molecular_mass(self)

    def enthalpy_isp(self, isp):
        return self.gmodel.enthalpy_isp(self, isp)
    def entropy_isp(self, isp):
        return self.gmodel.entropy_isp(self, isp)
    def gibbs_free_energy_isp(self, isp):
        return self.gmodel.gibbs_free_energy_isp(self, isp)


class ChemicalReactor(object):
    def __init__(self, file_name, gmodel):
        self.file_name = file_name
        self.gmodel = gmodel
        self.id = so.chemical_reactor_new(bytes(self.file_name, 'utf-8'), self.gmodel.id)

    def __str__(self):
        text = 'ChemicalReactor(file="%s", id=%d, gmodel.id=%d)' % \
            (self.file_name, self.id, self.gmodel.id)
        return text

    def update_state(self, gstate, t_interval, dt_suggest):
        dt_suggestp = ffi.new("double *")
        dt_suggestp[0] = dt_suggest
        flag = so.chemical_reactor_gas_state_update(self.id, gstate.id,
                                                    t_interval, dt_suggestp)
        if flag < 0: raise Exception("could not update state.")
        return dt_suggestp[0]