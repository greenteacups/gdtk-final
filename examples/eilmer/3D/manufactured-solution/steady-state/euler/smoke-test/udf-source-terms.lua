-- udf-source-template.lua
-- Lua template for the source terms of a Manufactured Solution.
--
-- PJ, 29-May-2011
-- RJG, 06-Jun-2014
--   Declared maths functions as local

local sin = math.sin
local cos = math.cos
local exp = math.exp
local pi = math.pi

function sourceTerms(t, cell)
   src = {}
   x = cell.x
   y = cell.y
   z = cell.z


fmass = 0.18*pi*(15.0*sin(1.0471975511965977*x) - 25.0*cos(4.7123889803846899*y) + 35.0*cos(3.1415926535897932*z) +800.0)*sin(4.7123889803846899*z) + 0.05*pi*(-75.0*sin(1.5707963267948966*x) + 40.0*cos(2.0943951023931954*y) -30.0*cos(3.9269908169872415*z) + 800.0)*sin(1.5707963267948966*y) - 26.6666666666667*pi*(0.15*sin(3.1415926535897932*x) - 0.1*cos(1.5707963267948966*y) -0.12*cos(4.7123889803846899*z) + 1.0)*sin(2.0943951023931954*y) - 35.0*pi*(0.15*sin(3.1415926535897932*x) - 0.1*cos(1.5707963267948966*y) -0.12*cos(4.7123889803846899*z) + 1.0)*sin(3.1415926535897932*z) + 75.0*pi*(0.15*sin(3.1415926535897932*x) - 0.1*cos(1.5707963267948966*y) -0.12*cos(4.7123889803846899*z) + 1.0)*cos(4.7123889803846899*x) + 0.15*pi*(50.0*sin(4.7123889803846899*x) - 30.0*cos(1.8849555921538759*y) -18.0*cos(1.5707963267948966*z) + 800.0)*cos(3.1415926535897932*x)



fxmom = 9.0*pi*(15.0*sin(1.0471975511965977*x) - 25.0*cos(4.7123889803846899*y) + 35.0*cos(3.1415926535897932*z) +800.0)*(0.15*sin(3.1415926535897932*x) - 0.1*cos(1.5707963267948966*y) - 0.12*cos(4.7123889803846899*z) +1.0)*sin(1.5707963267948966*z) + 0.18*pi*(15.0*sin(1.0471975511965977*x) - 25.0*cos(4.7123889803846899*y) +35.0*cos(3.1415926535897932*z) + 800.0)*(50.0*sin(4.7123889803846899*x) - 30.0*cos(1.8849555921538759*y) -18.0*cos(1.5707963267948966*z) + 800.0)*sin(4.7123889803846899*z) + 18.0*pi*(-75.0*sin(1.5707963267948966*x) + 40.0*cos(2.0943951023931954*y) -30.0*cos(3.9269908169872415*z) + 800.0)*(0.15*sin(3.1415926535897932*x) - 0.1*cos(1.5707963267948966*y) -0.12*cos(4.7123889803846899*z) + 1.0)*sin(1.8849555921538759*y) + 0.05*pi*(-75.0*sin(1.5707963267948966*x) + 40.0*cos(2.0943951023931954*y) -30.0*cos(3.9269908169872415*z) + 800.0)*(50.0*sin(4.7123889803846899*x) - 30.0*cos(1.8849555921538759*y) -18.0*cos(1.5707963267948966*z) + 800.0)*sin(1.5707963267948966*y) - 26.6666666666667*pi*(0.15*sin(3.1415926535897932*x) - 0.1*cos(1.5707963267948966*y) -0.12*cos(4.7123889803846899*z) + 1.0)*(50.0*sin(4.7123889803846899*x) - 30.0*cos(1.8849555921538759*y) -18.0*cos(1.5707963267948966*z) + 800.0)*sin(2.0943951023931954*y) - 35.0*pi*(0.15*sin(3.1415926535897932*x) - 0.1*cos(1.5707963267948966*y) -0.12*cos(4.7123889803846899*z) + 1.0)*(50.0*sin(4.7123889803846899*x) - 30.0*cos(1.8849555921538759*y) -18.0*cos(1.5707963267948966*z) + 800.0)*sin(3.1415926535897932*z) + 150.0*pi*(0.15*sin(3.1415926535897932*x) - 0.1*cos(1.5707963267948966*y) -0.12*cos(4.7123889803846899*z) + 1.0)*(50.0*sin(4.7123889803846899*x) - 30.0*cos(1.8849555921538759*y) -18.0*cos(1.5707963267948966*z) + 800.0)*cos(4.7123889803846899*x) + 0.15*pi*(50.0*sin(4.7123889803846899*x) - 30.0*cos(1.8849555921538759*y) -18.0*cos(1.5707963267948966*z) + 800.0)^2*cos(3.1415926535897932*x) - 40000.0*pi*sin(6.2831853071795865*x)



fymom = 0.18*pi*(15.0*sin(1.0471975511965977*x) - 25.0*cos(4.7123889803846899*y) + 35.0*cos(3.1415926535897932*z) +800.0)*(-75.0*sin(1.5707963267948966*x) + 40.0*cos(2.0943951023931954*y) - 30.0*cos(3.9269908169872415*z) +800.0)*sin(4.7123889803846899*z) + 37.5*pi*(15.0*sin(1.0471975511965977*x) - 25.0*cos(4.7123889803846899*y) +35.0*cos(3.1415926535897932*z) + 800.0)*(0.15*sin(3.1415926535897932*x) - 0.1*cos(1.5707963267948966*y) -0.12*cos(4.7123889803846899*z) + 1.0)*sin(3.9269908169872415*z) + 0.05*pi*(-75.0*sin(1.5707963267948966*x) + 40.0*cos(2.0943951023931954*y) -30.0*cos(3.9269908169872415*z) + 800.0)^2*sin(1.5707963267948966*y) - 53.3333333333333*pi*(-75.0*sin(1.5707963267948966*x) + 40.0*cos(2.0943951023931954*y) -30.0*cos(3.9269908169872415*z) + 800.0)*(0.15*sin(3.1415926535897932*x) - 0.1*cos(1.5707963267948966*y) -0.12*cos(4.7123889803846899*z) + 1.0)*sin(2.0943951023931954*y) - 35.0*pi*(-75.0*sin(1.5707963267948966*x) + 40.0*cos(2.0943951023931954*y) -30.0*cos(3.9269908169872415*z) + 800.0)*(0.15*sin(3.1415926535897932*x) - 0.1*cos(1.5707963267948966*y) -0.12*cos(4.7123889803846899*z) + 1.0)*sin(3.1415926535897932*z) + 75.0*pi*(-75.0*sin(1.5707963267948966*x) + 40.0*cos(2.0943951023931954*y) -30.0*cos(3.9269908169872415*z) + 800.0)*(0.15*sin(3.1415926535897932*x) - 0.1*cos(1.5707963267948966*y) -0.12*cos(4.7123889803846899*z) + 1.0)*cos(4.7123889803846899*x) + 0.15*pi*(-75.0*sin(1.5707963267948966*x) + 40.0*cos(2.0943951023931954*y) -30.0*cos(3.9269908169872415*z) + 800.0)*(50.0*sin(4.7123889803846899*x) - 30.0*cos(1.8849555921538759*y) -18.0*cos(1.5707963267948966*z) + 800.0)*cos(3.1415926535897932*x) - 37.5*pi*(0.15*sin(3.1415926535897932*x) - 0.1*cos(1.5707963267948966*y) -0.12*cos(4.7123889803846899*z) + 1.0)*(50.0*sin(4.7123889803846899*x) - 30.0*cos(1.8849555921538759*y) -18.0*cos(1.5707963267948966*z) + 800.0)*cos(1.5707963267948966*x) + 50000.0*pi*cos(3.1415926535897932*y)



fzmom = 0.18*pi*(15.0*sin(1.0471975511965977*x) - 25.0*cos(4.7123889803846899*y) + 35.0*cos(3.1415926535897932*z) +800.0)^2*sin(4.7123889803846899*z) + 0.05*pi*(15.0*sin(1.0471975511965977*x) - 25.0*cos(4.7123889803846899*y) +35.0*cos(3.1415926535897932*z) + 800.0)*(-75.0*sin(1.5707963267948966*x) + 40.0*cos(2.0943951023931954*y) -30.0*cos(3.9269908169872415*z) + 800.0)*sin(1.5707963267948966*y) - 26.6666666666667*pi*(15.0*sin(1.0471975511965977*x) - 25.0*cos(4.7123889803846899*y) +35.0*cos(3.1415926535897932*z) + 800.0)*(0.15*sin(3.1415926535897932*x) - 0.1*cos(1.5707963267948966*y) -0.12*cos(4.7123889803846899*z) + 1.0)*sin(2.0943951023931954*y) - 70.0*pi*(15.0*sin(1.0471975511965977*x) - 25.0*cos(4.7123889803846899*y) +35.0*cos(3.1415926535897932*z) + 800.0)*(0.15*sin(3.1415926535897932*x) - 0.1*cos(1.5707963267948966*y) -0.12*cos(4.7123889803846899*z) + 1.0)*sin(3.1415926535897932*z) + 75.0*pi*(15.0*sin(1.0471975511965977*x) - 25.0*cos(4.7123889803846899*y) +35.0*cos(3.1415926535897932*z) + 800.0)*(0.15*sin(3.1415926535897932*x) - 0.1*cos(1.5707963267948966*y) -0.12*cos(4.7123889803846899*z) + 1.0)*cos(4.7123889803846899*x) + 0.15*pi*(15.0*sin(1.0471975511965977*x) - 25.0*cos(4.7123889803846899*y) +35.0*cos(3.1415926535897932*z) + 800.0)*(50.0*sin(4.7123889803846899*x) - 30.0*cos(1.8849555921538759*y) -18.0*cos(1.5707963267948966*z) + 800.0)*cos(3.1415926535897932*x) + 37.5*pi*(-75.0*sin(1.5707963267948966*x) + 40.0*cos(2.0943951023931954*y) -30.0*cos(3.9269908169872415*z) + 800.0)*(0.15*sin(3.1415926535897932*x) - 0.1*cos(1.5707963267948966*y) -0.12*cos(4.7123889803846899*z) + 1.0)*sin(4.7123889803846899*y) + 5.0*pi*(0.15*sin(3.1415926535897932*x) - 0.1*cos(1.5707963267948966*y) -0.12*cos(4.7123889803846899*z) + 1.0)*(50.0*sin(4.7123889803846899*x) - 30.0*cos(1.8849555921538759*y) -18.0*cos(1.5707963267948966*z) + 800.0)*cos(1.0471975511965977*x) - 11666.6666666667*pi*cos(1.0471975511965977*z)



fe = 0.18*pi*((1.0/2.0)*(15.0*sin(1.0471975511965977*x) -25.0*cos(4.7123889803846899*y) + 35.0*cos(3.1415926535897932*z) + 800.0)^2 + (1.0/2.0)*(-75.0*sin(1.5707963267948966*x) + 40.0*cos(2.0943951023931954*y)- 30.0*cos(3.9269908169872415*z) + 800.0)^2 + (1.0/2.0)*(50.0*sin(4.7123889803846899*x) - 30.0*cos(1.8849555921538759*y) - 18.0*cos(1.5707963267948966*z) +800.0)^2 + 2.5*(50000.0*sin(3.1415926535897932*y) -35000.0*sin(1.0471975511965977*z) + 20000.0*cos(6.2831853071795865*x) + 100000.0)/(0.15*sin(3.1415926535897932*x) - 0.1*cos(1.5707963267948966*y) -0.12*cos(4.7123889803846899*z) + 1.0))*(15.0*sin(1.0471975511965977*x) - 25.0*cos(4.7123889803846899*y) +35.0*cos(3.1415926535897932*z) + 800.0)*sin(4.7123889803846899*z) + 0.05*pi*((1.0/2.0)*(15.0*sin(1.0471975511965977*x) - 25.0*cos(4.7123889803846899*y) +35.0*cos(3.1415926535897932*z) + 800.0)^2 + (1.0/2.0)*(-75.0*sin(1.5707963267948966*x) + 40.0*cos(2.0943951023931954*y) - 30.0*cos(3.9269908169872415*z) +800.0)^2 + (1.0/2.0)*(50.0*sin(4.7123889803846899*x) -30.0*cos(1.8849555921538759*y) - 18.0*cos(1.5707963267948966*z) + 800.0)^2 + 2.5*(50000.0*sin(3.1415926535897932*y) - 35000.0*sin(1.0471975511965977*z) +20000.0*cos(6.2831853071795865*x) + 100000.0)/(0.15*sin(3.1415926535897932*x) - 0.1*cos(1.5707963267948966*y) -0.12*cos(4.7123889803846899*z) + 1.0))*(-75.0*sin(1.5707963267948966*x) + 40.0*cos(2.0943951023931954*y) -30.0*cos(3.9269908169872415*z) + 800.0)*sin(1.5707963267948966*y) - 26.6666666666667*pi*((1.0/2.0)*(15.0*sin(1.0471975511965977*x) - 25.0*cos(4.7123889803846899*y) + 35.0*cos(3.1415926535897932*z) +800.0)^2 + (1.0/2.0)*(-75.0*sin(1.5707963267948966*x)+ 40.0*cos(2.0943951023931954*y) - 30.0*cos(3.9269908169872415*z) + 800.0)^2 + (1.0/2.0)*(50.0*sin(4.7123889803846899*x) - 30.0*cos(1.8849555921538759*y) -18.0*cos(1.5707963267948966*z) + 800.0)^2 + 2.5*(50000.0*sin(3.1415926535897932*y) - 35000.0*sin(1.0471975511965977*z) + 20000.0*cos(6.2831853071795865*x) +100000.0)/(0.15*sin(3.1415926535897932*x) - 0.1*cos(1.5707963267948966*y) - 0.12*cos(4.7123889803846899*z) +1.0))*(0.15*sin(3.1415926535897932*x) - 0.1*cos(1.5707963267948966*y) - 0.12*cos(4.7123889803846899*z) +1.0)*sin(2.0943951023931954*y) - 35.0*pi*((1.0/2.0)*(15.0*sin(1.0471975511965977*x) - 25.0*cos(4.7123889803846899*y) + 35.0*cos(3.1415926535897932*z) +800.0)^2 + (1.0/2.0)*(-75.0*sin(1.5707963267948966*x)+ 40.0*cos(2.0943951023931954*y) - 30.0*cos(3.9269908169872415*z) + 800.0)^2 + (1.0/2.0)*(50.0*sin(4.7123889803846899*x) - 30.0*cos(1.8849555921538759*y) -18.0*cos(1.5707963267948966*z) + 800.0)^2 + 2.5*(50000.0*sin(3.1415926535897932*y) - 35000.0*sin(1.0471975511965977*z) + 20000.0*cos(6.2831853071795865*x) +100000.0)/(0.15*sin(3.1415926535897932*x) - 0.1*cos(1.5707963267948966*y) - 0.12*cos(4.7123889803846899*z) +1.0))*(0.15*sin(3.1415926535897932*x) - 0.1*cos(1.5707963267948966*y) - 0.12*cos(4.7123889803846899*z) +1.0)*sin(3.1415926535897932*z) + 75.0*pi*((1.0/2.0)*(15.0*sin(1.0471975511965977*x) - 25.0*cos(4.7123889803846899*y) + 35.0*cos(3.1415926535897932*z) +800.0)^2 + (1.0/2.0)*(-75.0*sin(1.5707963267948966*x)+ 40.0*cos(2.0943951023931954*y) - 30.0*cos(3.9269908169872415*z) + 800.0)^2 + (1.0/2.0)*(50.0*sin(4.7123889803846899*x) - 30.0*cos(1.8849555921538759*y) -18.0*cos(1.5707963267948966*z) + 800.0)^2 + 2.5*(50000.0*sin(3.1415926535897932*y) - 35000.0*sin(1.0471975511965977*z) + 20000.0*cos(6.2831853071795865*x) +100000.0)/(0.15*sin(3.1415926535897932*x) - 0.1*cos(1.5707963267948966*y) - 0.12*cos(4.7123889803846899*z) +1.0))*(0.15*sin(3.1415926535897932*x) - 0.1*cos(1.5707963267948966*y) - 0.12*cos(4.7123889803846899*z) +1.0)*cos(4.7123889803846899*x) + 0.15*pi*((1.0/2.0)*(15.0*sin(1.0471975511965977*x) - 25.0*cos(4.7123889803846899*y) + 35.0*cos(3.1415926535897932*z) +800.0)^2 + (1.0/2.0)*(-75.0*sin(1.5707963267948966*x)+ 40.0*cos(2.0943951023931954*y) - 30.0*cos(3.9269908169872415*z) + 800.0)^2 + (1.0/2.0)*(50.0*sin(4.7123889803846899*x) - 30.0*cos(1.8849555921538759*y) -18.0*cos(1.5707963267948966*z) + 800.0)^2 + 2.5*(50000.0*sin(3.1415926535897932*y) - 35000.0*sin(1.0471975511965977*z) + 20000.0*cos(6.2831853071795865*x) +100000.0)/(0.15*sin(3.1415926535897932*x) - 0.1*cos(1.5707963267948966*y) - 0.12*cos(4.7123889803846899*z) +1.0))*(50.0*sin(4.7123889803846899*x) - 30.0*cos(1.8849555921538759*y) - 18.0*cos(1.5707963267948966*z) +800.0)*cos(3.1415926535897932*x) + (15.0*sin(1.0471975511965977*x) - 25.0*cos(4.7123889803846899*y) +35.0*cos(3.1415926535897932*z) + 800.0)*(0.15*sin(3.1415926535897932*x) - 0.1*cos(1.5707963267948966*y) -0.12*cos(4.7123889803846899*z) + 1.0)*(-35.0*pi*(15.0*sin(1.0471975511965977*x) - 25.0*cos(4.7123889803846899*y)+ 35.0*cos(3.1415926535897932*z) + 800.0)*sin(3.1415926535897932*z) + 37.5*pi*(-75.0*sin(1.5707963267948966*x) + 40.0*cos(2.0943951023931954*y) -30.0*cos(3.9269908169872415*z) + 800.0)*sin(3.9269908169872415*z) + 9.0*pi*(50.0*sin(4.7123889803846899*x) - 30.0*cos(1.8849555921538759*y) -18.0*cos(1.5707963267948966*z) + 800.0)*sin(1.5707963267948966*z) - 29166.6666666667*pi*cos(1.0471975511965977*z)/(0.15*sin(3.1415926535897932*x) -0.1*cos(1.5707963267948966*y) - 0.12*cos(4.7123889803846899*z) + 1.0) - 0.45*pi*(50000.0*sin(3.1415926535897932*y) - 35000.0*sin(1.0471975511965977*z) +20000.0*cos(6.2831853071795865*x) + 100000.0)*sin(4.7123889803846899*z)/(0.15*sin(3.1415926535897932*x) -0.1*cos(1.5707963267948966*y) - 0.12*cos(4.7123889803846899*z) + 1.0)^2) - 11666.6666666667*pi*(15.0*sin(1.0471975511965977*x) - 25.0*cos(4.7123889803846899*y) + 35.0*cos(3.1415926535897932*z) +800.0)*cos(1.0471975511965977*z) + (-75.0*sin(1.5707963267948966*x) + 40.0*cos(2.0943951023931954*y) -30.0*cos(3.9269908169872415*z) + 800.0)*(0.15*sin(3.1415926535897932*x) - 0.1*cos(1.5707963267948966*y) -0.12*cos(4.7123889803846899*z) + 1.0)*(37.5*pi*(15.0*sin(1.0471975511965977*x) - 25.0*cos(4.7123889803846899*y)+ 35.0*cos(3.1415926535897932*z) + 800.0)*sin(4.7123889803846899*y) - 26.6666666666667*pi*(-75.0*sin(1.5707963267948966*x) + 40.0*cos(2.0943951023931954*y) -30.0*cos(3.9269908169872415*z) + 800.0)*sin(2.0943951023931954*y) + 18.0*pi*(50.0*sin(4.7123889803846899*x) - 30.0*cos(1.8849555921538759*y) -18.0*cos(1.5707963267948966*z) + 800.0)*sin(1.8849555921538759*y) + 125000.0*pi*cos(3.1415926535897932*y)/(0.15*sin(3.1415926535897932*x) - 0.1*cos(1.5707963267948966*y) - 0.12*cos(4.7123889803846899*z) +1.0) - 0.125*pi*(50000.0*sin(3.1415926535897932*y) -35000.0*sin(1.0471975511965977*z) + 20000.0*cos(6.2831853071795865*x) + 100000.0)*sin(1.5707963267948966*y)/(0.15*sin(3.1415926535897932*x) - 0.1*cos(1.5707963267948966*y) - 0.12*cos(4.7123889803846899*z) +1.0)^2) + 50000.0*pi*(-75.0*sin(1.5707963267948966*x) +40.0*cos(2.0943951023931954*y) - 30.0*cos(3.9269908169872415*z) + 800.0)*cos(3.1415926535897932*y) +(0.15*sin(3.1415926535897932*x) - 0.1*cos(1.5707963267948966*y) - 0.12*cos(4.7123889803846899*z) +1.0)*(50.0*sin(4.7123889803846899*x) - 30.0*cos(1.8849555921538759*y) - 18.0*cos(1.5707963267948966*z) +800.0)*(5.0*pi*(15.0*sin(1.0471975511965977*x) - 25.0*cos(4.7123889803846899*y) + 35.0*cos(3.1415926535897932*z)+ 800.0)*cos(1.0471975511965977*x) - 37.5*pi*(-75.0*sin(1.5707963267948966*x) + 40.0*cos(2.0943951023931954*y) -30.0*cos(3.9269908169872415*z) + 800.0)*cos(1.5707963267948966*x) + 75.0*pi*(50.0*sin(4.7123889803846899*x) - 30.0*cos(1.8849555921538759*y) -18.0*cos(1.5707963267948966*z) + 800.0)*cos(4.7123889803846899*x) - 100000.0*pi*sin(6.2831853071795865*x)/(0.15*sin(3.1415926535897932*x) - 0.1*cos(1.5707963267948966*y) - 0.12*cos(4.7123889803846899*z) +1.0) - 0.375*pi*(50000.0*sin(3.1415926535897932*y) -35000.0*sin(1.0471975511965977*z) + 20000.0*cos(6.2831853071795865*x) + 100000.0)*cos(3.1415926535897932*x)/(0.15*sin(3.1415926535897932*x) - 0.1*cos(1.5707963267948966*y) - 0.12*cos(4.7123889803846899*z) +1.0)^2) - 40000.0*pi*(50.0*sin(4.7123889803846899*x) -30.0*cos(1.8849555921538759*y) - 18.0*cos(1.5707963267948966*z) + 800.0)*sin(6.2831853071795865*x) -26.6666666666667*pi*(50000.0*sin(3.1415926535897932*y) -35000.0*sin(1.0471975511965977*z) + 20000.0*cos(6.2831853071795865*x) + 100000.0)*sin(2.0943951023931954*y) - 35.0*pi*(50000.0*sin(3.1415926535897932*y) - 35000.0*sin(1.0471975511965977*z) + 20000.0*cos(6.2831853071795865*x) + 100000.0)*sin(3.1415926535897932*z) + 75.0*pi*(50000.0*sin(3.1415926535897932*y) - 35000.0*sin(1.0471975511965977*z) + 20000.0*cos(6.2831853071795865*x) +100000.0)*cos(4.7123889803846899*x)



   src.mass = fmass
   src.momentum_x = fxmom
   src.momentum_y = fymom
   src.momentum_z = fzmom
   src.total_energy = fe
   return src
end