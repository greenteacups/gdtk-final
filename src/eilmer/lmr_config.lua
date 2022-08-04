-- lmr_config.lua
--
-- A module for helper functions to query and interact with
-- Eilmer program config. Here, we are referring to static
-- configuration of the software itself, not the configuration
-- related to individual simulations.
--
-- Authors: RJG, PJ, KAD, NNG
--

local lmr_config = {}

local json = require 'json'

function lmr_config.lmrConfigAsTable()
   local lmrCfgFile = os.getenv("DGD") .. "/etc/lmr.cfg"
   local f = assert(io.open(lmrCfgFile, "r"))
   local jsonStr = f:read("*a")
   f:close()
   local jsonData = json.parse(jsonStr)
   return jsonData
end

return lmr_config

