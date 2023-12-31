/**
 * A Lua interface for the D grid module.
 *
 * Authors: Rowan G. and Peter J.
 * Date: 2016-11-30
 *
 * We need a base class to collect the common wrapper
 * implementations between structured and unstructured grids.
 * The following templates will generate specific functions at
 * compile time for Unstructured and Structured Grids.
 * These specific functions can be then registered into
 * the appropriate table of the Lua interpreter at run time.
 *
 * Note that the Grid class in the Lua domain is a pure Lua class
 * that holds a reference to one of these wrapped StructuredGrid
 * or UnstructuredGrid objects.
 *
 */

module geom.luawrap.luagrid;

import std.conv;
import std.string;
import nm.complex;
import nm.number;

import util.lua;
import util.lua_service;
import geom;
import geom.luawrap.luageom;

extern(C) int get_type(T, string MTname)(lua_State* L)
{
    int narg = lua_gettop(L); // assume narg == 1; This is a getter
    auto grid = checkObj!(T, MTname)(L, 1);
    lua_pushstring(L, gridTypeName(grid.grid_type).toStringz);
    return 1;
}

extern(C) int get_dimensions(T, string MTname)(lua_State* L)
{
    int narg = lua_gettop(L); // assume narg == 1; This is a getter
    auto grid = checkObj!(T, MTname)(L, 1);
    lua_pushnumber(L, grid.dimensions);
    return 1;
}

extern(C) int get_nvertices(T, string MTname)(lua_State* L)
{
    int narg = lua_gettop(L); // assume narg == 1; This is a getter
    auto grid = checkObj!(T, MTname)(L, 1);
    lua_pushnumber(L, grid.nvertices);
    return 1;
}

extern(C) int get_ncells(T, string MTname)(lua_State* L)
{
    int narg = lua_gettop(L); // assume narg == 1; This is a getter
    auto grid = checkObj!(T, MTname)(L, 1);
    lua_pushnumber(L, grid.ncells);
    return 1;
}

extern(C) int cellVolume(T, string MTname)(lua_State *L)
{
    auto grid = checkObj!(T, MTname)(L, 1);
    size_t indx = to!size_t(luaL_checkint(L, 2));
    number volume;
    Vector3 centroid;
    grid.compute_cell_properties(indx, true, centroid, volume);
    lua_pushnumber(L, volume);
    return 1;
}

extern(C) int cellCentroid(T, string MTname)(lua_State *L)
{
    auto grid = checkObj!(T, MTname)(L, 1);
    size_t indx = to!size_t(luaL_checkint(L, 2));
    number volume;
    Vector3 centroid;
    grid.compute_cell_properties(indx, true, centroid, volume);
    return pushVector3(L, centroid);
}

extern(C) int write_to_vtk_file(T, string MTname)(lua_State* L)
{
    int narg = lua_gettop(L); // assume narg == 2;
    auto grid = checkObj!(T, MTname)(L, 1);
    auto fileName = to!string(luaL_checkstring(L, 2));
    grid.write_to_vtk_file(fileName);
    return 0;
}

extern(C) int write_to_stl_file(T, string MTname)(lua_State* L)
{
    int narg = lua_gettop(L); // args: 1.grid-object, 2.fileName, 3.scale;
    auto grid = checkObj!(T, MTname)(L, 1);
    auto fileName = to!string(luaL_checkstring(L, 2));
    double scale = 1.0;
    if (narg >= 3) {
        if (lua_isnumber(L, 3)) {
            scale = to!double(lua_tonumber(L, 3));
        }
    }
    grid.write_to_stl_file(fileName, scale);
    return 0;
}

extern(C) int write_to_su2_file(T, string MTname)(lua_State* L)
{
    int narg = lua_gettop(L); // assume narg == 2;
    auto grid = checkObj!(T, MTname)(L, 1);
    auto fileName = to!string(luaL_checkstring(L, 2));
    grid.write_to_su2_file(fileName);
    return 0;
}

extern(C) int write_to_gzip_file(T, string MTname)(lua_State* L)
{
    int narg = lua_gettop(L); // assume narg == 2;
    auto grid = checkObj!(T, MTname)(L, 1);
    auto fileName = to!string(luaL_checkstring(L, 2));
    grid.write_to_gzip_file(fileName);
    return 0;
}

extern(C) int write_to_raw_binary_file(T, string MTname)(lua_State* L)
{
    int narg = lua_gettop(L); // assume narg == 2;
    auto grid = checkObj!(T, MTname)(L, 1);
    auto fileName = to!string(luaL_checkstring(L, 2));
    grid.write_to_raw_binary_file(fileName);
    return 0;
}
