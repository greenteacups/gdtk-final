-- Authors: RJG and KAD
-- Date: 2022-05-23
-- History: 2022-06-11
--          Update names and settings after talk with
--          Kyle D and Nick G.
--

NewtonKrylovGlobalConfigHidden = {
   
   -- global control based on step number
   set_reference_residuals_at_step = 1,
   freeze_limiter_at_step = -1,

   -- stopping criterion
   max_newton_steps = 1000,
   stop_on_relative_residual = 1.0e-99,
   stop_on_absolute_residual = 1.0e-99,
   stop_on_mass_balance = -1.0,

   -- CFL control
   cfl_max = 1.0e8,
   cfl_min = 0.001,
   cfl_schedule = { {0, 1.0}, {100, 1.0} },

   -- phase control
   number_of_phases = 1,
   phase_changes_at_steps = {},

   -- Newton stepping control and continuation
   use_local_timestep = true,
   inviscid_cfl_only = true,
   use_line_search = true,
   use_physicality_check = true,
   physicality_check_allowable_change = 0.2,

   -- linear solver and preconditioner
   max_linear_solver_iterations = 10,
   max_linear_solver_restarts = 0,
   use_scaling = true,
   frechet_derivative_perturbation = 1.0e-30,
   use_preconditioner = true,
   preconditioner_perturbation = 1.0e-30,
   preconditioner = "ilu",
   --
   -- ILU preconditioner settings
   --
   ilu_fill = 0,
   --
   -- SGS preconditioner settings
   --
   sgs_relaxation_iterations = 4,
   
   -- Output and diagnostics
   total_snapshots = 5,
   steps_between_snapshots = 10,
   steps_between_diagnostics = 10,
   steps_between_loads_update = 20,

   __index = function (t, k)
      return NewtonKrylovGlobalConfigHidden[k]
   end,
   __newindex = function (t, k, v)
      if NewtonKrylovGlobalConfigHidden[k] == nil then
	 print(string.format("The field '%s' cannot be set in 'NewtonKrylovGlobalConfig' table.", k))
      else
	 NewtonKrylovGlobalConfigHidden[k] = v
      end
   end,
   __call = function (_, t)
      for k, v in pairs(t) do
	 NewtonKrylovGlobalConfigHidden.__newindex(t, k, v)
      end
   end
}

NewtonKrylovGlobalConfig = {}
setmetatable(NewtonKrylovGlobalConfig, NewtonKrylovGlobalConfigHidden)

NewtonKrylovPhaseDefaults = {
   residual_interpolation_order = 2,
   jacobian_interpolation_order = 2, -- documentation explanation.
   frozen_preconditioner = true,
   steps_between_preconditioner_update = 10,
   use_adaptive_preconditioner = false,
   ignore_stopping_criteria = true,
   frozen_limiter_for_jacobian = true,

   -- Linear solver control
   linear_solve_tolerance = 0.01,
   
   -- Auto CFL control
   use_auto_cfl = false,
   threshold_residual_drop_for_cfl_growth = 0.99,
   start_cfl = 1.0,
   max_cfl = 1000.0,
   auto_cfl_exponent = 0.75,

}

local NewtonKrylovPhases = {}

local NewtonKrylovPhase = {
   myType = "NewtonKrylovPhase"
}

function NewtonKrylovPhase:new(o)
   local flag = type(self)=='table' and self.myType=='NewtonKrylovPhase'
   if not flag then
      error("Make sure that you are using NewtonKrylovPhase:new{} and not NewtonKrylovPhase.new{}", 2)
   end
   o = o or {}
   allowedNames = {}
   for k,_ in pairs(NewtonKrylovPhaseDefaults) do
      allowedNames[#allowedNames+1] = k
   end
   flag = checkAllowedNames(o, allowedNames)
   if not flag then
      error("Invalid name for item supplied to NewtonKrylovPhase constructor.", 2)
   end
   setmetatable(o, self)
   self.__index = self
   -- Make a record of this phase for later construction in config file.
   -- Note that we want the phase id to start at zero for the D code.
   o.id = #(NewtonKrylovPhases)
   NewtonKrylovPhases[#NewtonKrylovPhases+1] = o
   o.label = o.label or string.format("NewtonKrylovPhase-%d", o.id)
   -- Set up defaults to use for this phase
   -- For getting started...
   defaultsForThisPhase = NewtonKrylovPhaseDefaults
   if o.id > 0 then -- on subsequent phases, inherit setting from previous phase
      defaultsForThisPhase = NewtonKrylovPhases[o.id-1]
   end
   for k,v in pairs(defaultsForThisPhase) do
      -- Take values as set or use default value.
      o[k] = o[k] or v
   end
   return o
end -- end NewtonKrylovPhase:new(o)

function NewtonKrylovPhase:tojson()
   local str = string.format('"NewtonKrylovPhase_%d": {\n', self.id)
   str = str .. string.format('    "residual_interpolation_order": %d,\n', self.residual_interpolation_order)
   str = str .. string.format('    "jacobian_interpolation_order": %d,\n', self.jacobian_interpolation_order)
   str = str .. string.format('    "frozen_preconditioner": %s,\n', tostring(self.frozen_preconditioner))
   str = str .. string.format('    "steps_between_preconditioner_update": %d,\n', self.steps_between_preconditioner_update)
   str = str .. string.format('    "use_adaptive_preconditioner": %s,\n', tostring(self.use_adaptive_preconditioner))
   str = str .. string.format('    "ignore_stopping_criteria": %s,\n', tostring(self.ignore_stopping_criteria))
   str = str .. string.format('    "frozen_limiter_for_jacobian": %s,\n', tostring(self.frozen_limiter_for_jacobian))
   str = str .. string.format('    "linear_solve_tolerance": %.18e,\n', self.linear_solve_tolerance)
   str = str .. string.format('    "use_auto_cfl": %s,\n', tostring(self.use_auto_cfl))
   if self.use_auto_cfl then
      str = str .. string.format('    "threshold_residual_drop_for_cfl_growth": %.18e,\n', self.threshold_residual_drop_for_cfl_growth)
      str = str .. string.format('    "start_cfl": %.18e,\n', self.start_cfl)
      str = str .. string.format('    "max_cfl": %.18e,\n', self.max_cfl)
      str = str .. string.format('    "auto_cfl_exponent": %.18e,\n', self.auto_cfl_exponent)
   end
   str = str .. '    "dummy_entry_without_trailing_comma": 0\n' -- no comma on last entry
   str = str .. '}'
   return str
end

local function setIgnoreFlagInPhases(nkPhases)
   for i=1,#nkPhases-1 do
      nkPhases[i].ignore_stopping_criteria = true
   end
   nkPhases[#nkPhases].ignore_stopping_criteria = false
end

local function writeNKConfigToFile(nkConfig, nkPhases, fileName)
   local f = assert(io.open(fileName, "w"))
   f:write("{\n")
   -- global control based on step
   f:write(string.format('"set_reference_residuals_at_step": %d,\n', nkConfig.set_reference_residuals_at_step))
   f:write(string.format('"freeze_limiter_at_step": %d,\n', nkConfig.freeze_limiter_at_step))
   -- stopping criterion
   f:write(string.format('"max_newton_steps": %d,\n', nkConfig.max_newton_steps))
   f:write(string.format('"stop_on_relative_residual": %.18e,\n', nkConfig.stop_on_relative_residual))
   f:write(string.format('"stop_on_absolute_residual": %.18e,\n', nkConfig.stop_on_absolute_residual))
   f:write(string.format('"stop_on_mass_balance": %.18e,\n', nkConfig.stop_on_mass_balance))
   -- CFL control
   f:write(string.format('"cfl_max": %.18e,\n', nkConfig.cfl_max))
   f:write(string.format('"cfl_min": %.18e,\n', nkConfig.cfl_min))
   f:write('"cfl_schedule": [ ')
   for i,e in ipairs(nkConfig.cfl_schedule) do
      f:write(string.format(' [ %d, %.3e ]', e[1], e[2]))
      if i < #(nkConfig.cfl_schedule) then f:write(', ') end
   end
   f:write('],\n')
   -- phase control
   f:write(string.format('"number_of_phases": %d,\n', nkConfig.number_of_phases))
   f:write('"phase_changes_as_steps": [ ')
   for i,e in ipairs(nkConfig.phase_changes_at_steps) do
      f:write(string.format('%d', e))
      if i < #(nkConfig.phase_change_at_steps) then f:write(', ') end
   end
   f:write('],\n')
   -- Newton stepping control and continuation
   f:write(string.format('"use_local_timestep": %s,\n', tostring(nkConfig.use_local_timestep)))
   f:write(string.format('"inviscid_cfl_only": %s,\n', tostring(nkConfig.inviscid_cfl_only)))
   f:write(string.format('"use_line_search": %s,\n', tostring(nkConfig.use_line_search)))
   f:write(string.format('"use_physicality_check": %s,\n', tostring(nkConfig.use_physicality_check)))
   f:write(string.format('"physicality_check_allowable_change": %.18e,\n', nkConfig.physicality_check_allowable_change))
   -- linear solver and preconditioner
   f:write(string.format('"max_linear_solver_iterations": %d,\n', nkConfig.max_linear_solver_iterations))
   f:write(string.format('"max_linear_solver_restarts": %d,\n', nkConfig.max_linear_solver_restarts))
   f:write(string.format('"use_scaling": %s,\n', tostring(nkConfig.use_scaling)))
   f:write(string.format('"frechet_derivative_perturbation": %.18e,\n', nkConfig.frechet_derivative_perturbation))
   f:write(string.format('"use_preconditioner": %s,\n', tostring(nkConfig.use_preconditioner)))
   f:write(string.format('"preconditioner_perturbation": %.18e,\n', nkConfig.preconditioner_perturbation))
   f:write(string.format('"preconditioner": "%s",\n', nkConfig.preconditioner))
   f:write(string.format('"ilu_fill": %d,\n', nkConfig.ilu_fill))
   f:write(string.format('"sgs_relaxation_iterations": %d,\n', nkConfig.sgs_relaxation_iterations))
   -- output and diagnostics
   f:write(string.format('"total_snapshots": %d,\n', nkConfig.total_snapshots))
   f:write(string.format('"steps_between_snapshots": %d,\n', nkConfig.steps_between_snapshots))
   f:write(string.format('"steps_between_diagnostics": %d,\n', nkConfig.steps_between_diagnostics))
   f:write(string.format('"steps_between_loads_update": %d,\n', nkConfig.steps_between_loads_update))
   -- write out phases
   for i=1,#nkPhases do
      f:write(nkPhases[i]:tojson() .. ",\n")
   end
   f:write('"dummy_entry_without_trailing_comma": 0\n') -- no comma on last entry
   f:write('}\n')
   f:close()
end

return {
   NewtonKrylovGlobalConfig = NewtonKrylovGlobalConfig,
   NewtonKrylovPhases = NewtonKrylovPhases,
   NewtonKrylovPhase = NewtonKrylovPhase,
   writeNKConfigToFile = writeNKConfigToFile,
   setIgnoreFlagInPhases = setIgnoreFlagInPhases
   
}


