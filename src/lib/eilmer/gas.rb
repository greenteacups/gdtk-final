# gas.rb
# A Ruby wrapper for GasModel and GasState classes.
# PJ 2019-07-24: start of experiment.
#    2019-07-27: add Ruby wrapping code
#
require 'fiddle'
require 'fiddle/import'

module Gas
  extend Fiddle::Importer
  dlload 'libgas.so'
  extern 'int cwrap_gas_init()'

  extern 'int gas_model_new(char* file_name)'
  extern 'int gas_model_n_species(int gm_i)'
  extern 'int gas_model_n_modes(int gm_i)'
  extern 'int gas_model_species_name(int gm_i, int isp, char* name, int n)'
  extern 'int gas_model_mol_masses(int gm_i, double* mm)'

  extern 'int gas_state_new(int gm_i)'
  extern 'int gas_state_set_scalar_field(int gs_i, char* field_name, double value)'
  extern 'int gas_state_get_scalar_field(int gs_i, char* field_name, double* value)'
  extern 'int gas_state_set_array_field(int gs_i, char* field_name, double* values, int n)'
  extern 'int gas_state_get_array_field(int gs_i, char* field_name, double* values, int n)'

  extern 'int gas_model_gas_state_update_thermo_from_pT(int gm_i, int gs_i)'
  extern 'int gas_model_gas_state_update_thermo_from_rhou(int gm_i, int gs_i)'
  extern 'int gas_model_gas_state_update_thermo_from_rhoT(int gm_i, int gs_i)'
  extern 'int gas_model_gas_state_update_thermo_from_rhop(int gm_i, int gs_i)'
  extern 'int gas_model_gas_state_update_thermo_from_ps(int gm_i, int gs_i, double s)'
  extern 'int gas_model_gas_state_update_thermo_from_hs(int gm_i, int gs_i, double h, double s)'
  extern 'int gas_model_gas_state_update_sound_speed(int gm_i, int gs_i)'
  extern 'int gas_model_gas_state_update_trans_coeffs(int gm_i, int gs_i)'

  extern 'int gas_model_gas_state_Cv(int gm_i, int gs_i, double* result)'
  extern 'int gas_model_gas_state_Cp(int gm_i, int gs_i, double* result)'
  extern 'int gas_model_gas_state_dpdrho_const_T(int gm_i, int gs_i, double* result)'
  extern 'int gas_model_gas_state_R(int gm_i, int gs_i, double* result)'
  extern 'int gas_model_gas_state_internal_energy(int gm_i, int gs_i, double* result)'
  extern 'int gas_model_gas_state_enthalpy(int gm_i, int gs_i, double* result)'
  extern 'int gas_model_gas_state_entropy(int gm_i, int gs_i, double* result)'
  extern 'int gas_model_gas_state_molecular_mass(int gm_i, int gs_i, double* result)'

  extern 'int gas_model_gas_state_enthalpy_isp(int gm_i, int gs_i, int isp, double* result)'
  extern 'int gas_model_gas_state_entropy_isp(int gm_i, int gs_i, int isp, double* result)'
  extern 'int gas_model_gas_state_gibbs_free_energy_isp(int gm_i, int gs_i, int isp, double* result)'

  extern 'int gas_model_massf2molef(int gm_i, double* massf, double* molef)'
  extern 'int gas_model_molef2massf(int gm_i, double* molef, double* massf)'
  extern 'int gas_model_gas_state_get_molef(int gm_i, int gs_i, double* molef)'
  extern 'int gas_model_gas_state_get_conc(int gm_i, int gs_i, double* conc)'

  extern 'int chemical_reactor_new(char* file_name, int gm_i)'
  extern 'int chemical_reactor_gas_state_update(int cr_i, int gs_i, double t_interval, double* dt_suggest)'
end

Gas.cwrap_gas_init()

# Service classes that wrap the C-API in a nice Ruby API...

class GasModel
  include Gas
  attr_reader :id
  attr_reader :species_names
  
  def initialize(file_name)
    @file_name = file_name
    @id = Gas.gas_model_new(file_name)
    nsp = Gas.gas_model_n_species(@id)
    @species_names = []
    buf = Fiddle::Pointer.malloc(32)
    nsp.times do |i|
      Gas.gas_model_species_name(@id, i, buf, 32)
      @species_names << buf.to_s
    end
  end

  def to_s()
    "GasModel(file=\"#{@file_name}\", id=#{@id}, species=#{@species_names})"
  end
  
  def n_species()
    Gas.gas_model_n_species(@id)
  end
  def n_modes()
    Gas.gas_model_n_modes(@id)
  end
  def mol_masses()
    nsp = Gas.gas_model_n_species(@id)
    mm = Fiddle::Pointer.malloc(Fiddle::SIZEOF_DOUBLE*nsp)
    Gas.gas_model_mol_masses(@id, mm)
    return mm[0, mm.size].unpack("d*")
  end

  def update_thermo_from_pT(gstate)
    flag = Gas.gas_model_gas_state_update_thermo_from_pT(@id, gstate.id)
    if flag < 0 then raise "could not update thermo from p,T." end
  end
  def update_thermo_from_rhou(gstate)
    flag = Gas.gas_model_gas_state_update_thermo_from_rhou(@id, gstate.id)
    if flag < 0 then raise "could not update thermo from rho,u." end
  end
  def update_thermo_from_rhoT(gstate)
    flag = Gas.gas_model_gas_state_update_thermo_from_rhoT(@id, gstate.id)
    if flag < 0 then raise "could not update thermo from rho,T." end
  end
  def update_thermo_from_rhop(gstate)
    flag = Gas.gas_model_gas_state_update_thermo_from_rhop(@id, gstate.id)
    if flag < 0 then raise "could not update thermo from rho,p." end
  end
  def update_thermo_from_ps(gstate, s)
    flag = Gas.gas_model_gas_state_update_thermo_from_ps(@id, gstate.id, s)
    if flag < 0 then raise "could not update thermo from p,s." end
  end
  def update_thermo_from_hs(gstate, h, s)
    flag = Gas.gas_model_gas_state_update_thermo_from_hs(@id, gstate.id, h, s)
    if flag < 0 then raise "could not update thermo from h,s." end
  end
  def update_sound_speed(gstate)
    flag = Gas.gas_model_gas_state_update_sound_speed(@id, gstate.id)
    if flag < 0 then raise "could not update sound speed." end
  end
  def update_trans_coeffs(gstate)
    flag = Gas.gas_model_gas_state_update_trans_coeffs(@id, gstate.id)
    if flag < 0 then raise "could not update transport coefficients." end
  end

  def Cv(gstate)
    valuep = Fiddle::Pointer.malloc(Fiddle::SIZEOF_DOUBLE)
    flag = Gas.gas_model_gas_state_Cv(@id, gstate.id, valuep)
    if flag < 0 then raise "could not compute Cv." end
    return valuep[0, valuep.size].unpack("d")[0]
  end
  def Cp(gstate)
    valuep = Fiddle::Pointer.malloc(Fiddle::SIZEOF_DOUBLE)
    flag = Gas.gas_model_gas_state_Cp(@id, gstate.id, valuep)
    if flag < 0 then raise "could not compute Cp." end
    return valuep[0, valuep.size].unpack("d")[0]
  end
  def dpdrho_const_T(gstate)
    valuep = Fiddle::Pointer.malloc(Fiddle::SIZEOF_DOUBLE)
    flag = Gas.gas_model_gas_state_dpdrho_const_T(@id, gstate.id, valuep)
    if flag < 0 then raise "could not compute dpdrho_const_T." end
    return valuep[0, valuep.size].unpack("d")[0]
  end
  def R(gstate)
    valuep = Fiddle::Pointer.malloc(Fiddle::SIZEOF_DOUBLE)
    flag = Gas.gas_model_gas_state_R(@id, gstate.id, valuep)
    if flag < 0 then raise "could not compute R." end
    return valuep[0, valuep.size].unpack("d")[0]
  end
  def internal_energy(gstate)
    valuep = Fiddle::Pointer.malloc(Fiddle::SIZEOF_DOUBLE)
    flag = Gas.gas_model_gas_state_internal_energy(@id, gstate.id, valuep)
    if flag < 0 then raise "could not compute internal energy." end
    return valuep[0, valuep.size].unpack("d")[0]
  end
  def enthalpy(gstate)
    valuep = Fiddle::Pointer.malloc(Fiddle::SIZEOF_DOUBLE)
    flag = Gas.gas_model_gas_state_enthalpy(@id, gstate.id, valuep)
    if flag < 0 then raise "could not compute enthalpy." end
    return valuep[0, valuep.size].unpack("d")[0]
  end
  def entropy(gstate)
    valuep = Fiddle::Pointer.malloc(Fiddle::SIZEOF_DOUBLE)
    flag = Gas.gas_model_gas_state_entropy(@id, gstate.id, valuep)
    if flag < 0 then raise "could not compute entropy." end
    return valuep[0, valuep.size].unpack("d")[0]
  end
  def molecular_mass(gstate)
    valuep = Fiddle::Pointer.malloc(Fiddle::SIZEOF_DOUBLE)
    flag = Gas.gas_model_gas_state_molecular_mass(@id, gstate.id, valuep)
    if flag < 0 then raise "could not compute molecular mass." end
    return valuep[0, valuep.size].unpack("d")[0]
  end

  def enthalpy_isp(gstate, isp)
    valuep = Fiddle::Pointer.malloc(Fiddle::SIZEOF_DOUBLE)
    flag = Gas.gas_model_gas_state_enthalpy_isp(@id, gstate.id, isp, valuep)
    if flag < 0 then raise "could not compute enthalpy for species." end
    return valuep[0, valuep.size].unpack("d")[0]
  end
  def entropy_isp(gstate, isp)
    valuep = Fiddle::Pointer.malloc(Fiddle::SIZEOF_DOUBLE)
    flag = Gas.gas_model_gas_state_entropy_isp(@id, gstate.id, isp, valuep)
    if flag < 0 then raise "could not compute entropy for species." end
    return valuep[0, valuep.size].unpack("d")[0]
  end
  def gibbs_free_energy_isp(gstate, isp)
    valuep = Fiddle::Pointer.malloc(Fiddle::SIZEOF_DOUBLE)
    flag = Gas.gas_model_gas_state_gibbs_free_energy_isp(@id, gstate.id, isp, valuep)
    if flag < 0 then raise "could not compute gibbs free energy for species." end
    return valuep[0, valuep.size].unpack("d")[0]
  end

  def massf2molef(massf_given)
    nsp = Gas.gas_model_n_species(@id)
    if massf_given.class == [].class then
      massf_array = massf_given
    elsif massf_given.class == {}.class then
      massf_array = []
      @species_names.each do |name|
        if massf_given.has_key?(name) then
          massf_array << massf_given[name]
        else
          massf_array << 0.0
        end
      end
    end
    mf_sum = 0.0; massf_array.each do |mfi| mf_sum += mfi end
    if (mf_sum - 1.0).abs > 1.0e-6 then
      raise "mass fractions do not sum to 1."
    end
    my_massf = massf_array.pack("d*")
    my_molef = Fiddle::Pointer.malloc(Fiddle::SIZEOF_DOUBLE*nsp)
    Gas.gas_model_massf2molef(@id, my_massf, my_molef)
    return my_molef[0, my_molef.size].unpack("d*")
  end

  def molef2massf(molef_given)
    nsp = Gas.gas_model_n_species(@id)
    if molef_given.class == [].class then
      molef_array = molef_given
    elsif molef_given.class == {}.class then
      molef_array = []
      @species_names.each do |name|
        if molef_given.has_key?(name) then
          molef_array << molef_given[name]
        else
          molef_array << 0.0
        end
      end
    end
    mf_sum = 0.0; molef_array.each do |mfi| mf_sum += mfi end
    if (mf_sum - 1.0).abs > 1.0e-6 then
      raise "mole fractions do not sum to 1."
    end
    my_molef = molef_array.pack("d*")
    my_massf = Fiddle::Pointer.malloc(Fiddle::SIZEOF_DOUBLE*nsp)
    Gas.gas_model_molef2massf(@id, my_molef, my_massf)
    return my_massf[0, my_massf.size].unpack("d*")
  end
end


class GasState
  include Gas
  attr_reader :id
  
  def initialize(gmodel)
    @gmodel = gmodel
    @id = Gas.gas_state_new(gmodel.id)
  end

  def to_s()
    text = "GasState(rho=#{self.rho}"
    text << ", p=#{self.p}"
    text << ", T=#{self.T}"
    text << ", u=#{self.u}"
    text << ", massf=#{self.massf}"
    text << ", id=#{@id}, gmodel.id=#{@gmodel.id})"
  end
    
  def rho()
    valuep = Fiddle::Pointer.malloc(Fiddle::SIZEOF_DOUBLE)
    flag = Gas.gas_state_get_scalar_field(@id, "rho", valuep)
    if flag < 0 then raise "could not get density." end
    return valuep[0, valuep.size].unpack("d")[0]
  end
  def rho=(value)
    flag = Gas.gas_state_set_scalar_field(@id, "rho", value)
    if flag < 0 then raise "could not set density." end
  end
    
  def p()
    valuep = Fiddle::Pointer.malloc(Fiddle::SIZEOF_DOUBLE)
    flag = Gas.gas_state_get_scalar_field(@id, "p", valuep)
    if flag < 0 then raise "could not get pressure." end
    return valuep[0, valuep.size].unpack("d")[0]
  end
  def p=(value)
    flag = Gas.gas_state_set_scalar_field(@id, "p", value)
    if flag < 0 then raise "could not set pressure." end
  end
    
  def T()
    valuep = Fiddle::Pointer.malloc(Fiddle::SIZEOF_DOUBLE)
    flag = Gas.gas_state_get_scalar_field(@id, "T", valuep)
    if flag < 0 then raise "could not get temperature." end
    return valuep[0, valuep.size].unpack("d")[0]
  end
  def T=(value)
    flag = Gas.gas_state_set_scalar_field(@id, "T", value)
    if flag < 0 then raise "could not set temperature." end
  end
    
  def u()
    valuep = Fiddle::Pointer.malloc(Fiddle::SIZEOF_DOUBLE)
    flag = Gas.gas_state_get_scalar_field(@id, "u", valuep)
    if flag < 0 then raise "could not get internal energy." end
    return valuep[0, valuep.size].unpack("d")[0]
  end
  def u=(value)
    flag = Gas.gas_state_set_scalar_field(@id, "u", value)
    if flag < 0 then raise "could not set internal energy." end
  end
    
  def a()
    valuep = Fiddle::Pointer.malloc(Fiddle::SIZEOF_DOUBLE)
    flag = Gas.gas_state_get_scalar_field(@id, "a", valuep)
    if flag < 0 then raise "could not get sound speed." end
    return valuep[0, valuep.size].unpack("d")[0]
  end
    
  def k()
    valuep = Fiddle::Pointer.malloc(Fiddle::SIZEOF_DOUBLE)
    flag = Gas.gas_state_get_scalar_field(@id, "k", valuep)
    if flag < 0 then raise "could not get conductivity." end
    return valuep[0, valuep.size].unpack("d")[0]
  end
    
  def mu()
    valuep = Fiddle::Pointer.malloc(Fiddle::SIZEOF_DOUBLE)
    flag = Gas.gas_state_get_scalar_field(@id, "mu", valuep)
    if flag < 0 then raise "could not get viscosity." end
    return valuep[0, valuep.size].unpack("d")[0]
  end

  def massf()
    nsp = @gmodel.n_species
    mf = Fiddle::Pointer.malloc(Fiddle::SIZEOF_DOUBLE*nsp)
    flag = Gas.gas_state_get_array_field(@id, "massf", mf, nsp)
    if flag < 0 then raise "could not get mass-fractions." end
    return mf[0, mf.size].unpack("d*")
  end
  def massf=(mf_given)
    nsp = @gmodel.n_species
    if mf_given.class == [].class then
      mf_array = mf_given
    elsif mf_given.class == {}.class then
      mf_array = []
      @gmodel.species_names.each do |name|
        if mf_given.has_key?(name) then
          mf_array << mf_given[name]
        else
          mf_array << 0.0
        end
      end
    end
    mf_sum = 0.0; mf_array.each do |mfi| mf_sum += mfi end
    if (mf_sum - 1.0).abs > 1.0e-6 then
      raise "mass fractions do not sum to 1."
    end
    mf = mf_array.pack("d*")
    flag = Gas.gas_state_set_array_field(@id, "massf", mf, nsp)
    if flag < 0 then raise "could not set mass-fractions." end
    return mf_array
  end

  def molef()
    nsp = @gmodel.n_species
    mf = Fiddle::Pointer.malloc(Fiddle::SIZEOF_DOUBLE*nsp)
    flag = Gas.gas_model_gas_state_get_molef(@gmodel.id, @id, mf)
    if flag < 0 then raise "could not get mole-fractions." end
    return mf[0, mf.size].unpack("d*")
  end
  def molef=(molef_given)
    nsp = @gmodel.n_species
    mf_array = @gmodel.molef2massf(molef_given)
    mf = mf_array.pack("d*")
    flag = Gas.gas_state_set_array_field(@id, "massf", mf, nsp)
    if flag < 0 then raise "could not set mass-fractions from mole-fractions." end
    # At this point, we may not have the mole-fractions as an array.
    return nil
  end

  def conc()
    nsp = @gmodel.n_species
    mc = Fiddle::Pointer.malloc(Fiddle::SIZEOF_DOUBLE*nsp)
    flag = Gas.gas_model_gas_state_get_conc(@gmodel.id, @id, mc)
    if flag < 0 then raise "could not get concentrations." end
    return mc[0, mc.size].unpack("d*")
  end

  def u_modes()
    n = @gmodel.n_modes
    um = Fiddle::Pointer.malloc(Fiddle::SIZEOF_DOUBLE*n)
    flag = Gas.gas_state_get_array_field(@id, "u_modes", um, n)
    if flag < 0 then raise "could not get u_modes." end
    return um[0, um.size].unpack("d*")
  end
  def u_modes=(um_given)
    n = @gmodel.n_modes
    if um_given.class == [].class then
      raise "u_modes needs to be supplied as an array."
    end
    um = um_given.pack("d*")
    flag = Gas.gas_state_set_array_field(@id, "u_modes", um, n)
    if flag < 0 then raise "could not set u_modes." end
    return um_given
  end

  def T_modes()
    n = @gmodel.n_modes
    myTm = Fiddle::Pointer.malloc(Fiddle::SIZEOF_DOUBLE*n)
    flag = Gas.gas_state_get_array_field(@id, "T_modes", myTm, n)
    if flag < 0 then raise "could not get T_modes." end
    return myTm[0, myTm.size].unpack("d*")
  end
  def T_modes=(myTm_given)
    n = @gmodel.n_modes
    if myTm_given.class == [].class then
      raise "T_modes needs to be supplied as an array."
    end
    myTm = myTm_given.pack("d*")
    flag = Gas.gas_state_set_array_field(@id, "T_modes", myTm, n)
    if flag < 0 then raise "could not set T_modes." end
    return myTm_given
  end
 
  def k_modes()
    n = @gmodel.n_modes
    km = Fiddle::Pointer.malloc(Fiddle::SIZEOF_DOUBLE*n)
    flag = Gas.gas_state_get_array_field(@id, "k_modes", km, n)
    if flag < 0 then raise "could not get k_modes." end
    return km[0, km.size].unpack("d*")
  end
 
  def update_thermo_from_pT()
    @gmodel.update_thermo_from_pT(self)
  end
  def update_thermo_from_rhou()
    @gmodel.update_thermo_from_rhou(self)
  end
  def update_thermo_from_rhoT()
    @gmodel.update_thermo_from_rhoT(self)
  end
  def update_thermo_from_rhop()
    @gmodel.update_thermo_from_rhop(self)
  end
  def update_thermo_from_ps(s)
    @gmodel.update_thermo_from_ps(self, s)
  end
  def update_thermo_from_hs(h, s)
    @gmodel.update_thermo_from_hs(self, h, s)
  end
  def update_sound_speed()
    @gmodel.update_sound_speed(self)
  end
  def update_trans_coeffs()
    @gmodel.update_trans_coeffs(self)
  end

  def Cv()
    @gmodel.Cv(self)
  end
  def Cp()
    @gmodel.Cp(self)
  end
  def dpdrho_const_T()
    @gmodel.dpdrho_const_T(self)
  end
  def R()
    @gmodel.R(self)
  end
  def internal_energy()
    @gmodel.internal_energy(self)
  end
  def enthalpy()
    @gmodel.enthalpy(self)
  end
  def entropy()
    @gmodel.entropy(self)
  end
  def molecular_mass()
    @gmodel.molecular_mass(self)
  end

  def enthalpy_isp(isp)
    @gmodel.enthalpy(self, isp)
  end
  def entropy_isp(isp)
    @gmodel.entropy(self, isp)
  end
  def gibbs_free_energy_isp(isp)
    @gmodel.gibbs_free_energy(self, isp)
  end
end


class ChemicalReactor
  include Gas
  attr_reader :id
  
  def initialize(file_name, gmodel)
    @file_name = file_name
    @gmodel = gmodel
    @id = Gas.chemical_reactor_new(file_name, gmodel.id)
  end

  def to_s()
    text = "ChemicalReactor(file=#{@file_name}"
    text << ", id=#{@id}, gmodel.id=#{@gmodel.id})"
  end
    
  def update_state(gstate, t_interval, dt_suggest)
    dt_suggestp = [dt_suggest].pack("d*")
    flag = Gas.chemical_reactor_gas_state_update(@id, gstate.id,
                                                 t_interval, dt_suggestp)
    if flag < 0 then raise "could not update state." end
    return dt_suggestp[0, dt_suggestp.size].unpack("d")[0]
  end
end