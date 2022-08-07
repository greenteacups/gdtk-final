/** lmrconfig.d
 * Module for configuration of Eilmer program itself.
 *
 * Authors: RJG, PJ, KAD, NNG
 * Date: 2022-08-06
 */

module lmrconfig;

import std.format : format;
import std.process : environment;
import std.json;
import json_helper : readJSONfile;


JSONValue lmrConfig;

/**
 * Read Eilmer program configuration file.
 *
 * Authors: RJG
 * Date: 2022-08-06
 */
void readLmrConfig()
{
    auto lmrCfgFile = environment.get("DGD") ~ "/etc/lmr.cfg";
    lmrConfig = readJSONfile(lmrCfgFile);
}

/**
 * Return the simulation config filename as a string.
 *
 * Authors: RJG
 * Date: 2022-08-06
 */
string simulationConfigFilename()
{
    return lmrConfig["config-directory"].str ~ "/" ~ lmrConfig["config-filename"].str;
}

/**
 * Return the simulation control filename as a string.
 *
 * Authors: RJG
 * Date: 2022-08-06
 */
string simulationControlFilename()
{
    return lmrConfig["config-directory"].str ~ "/" ~ lmrConfig["control-filename"].str;
}

/**
 * Return NK config filename as a string.
 *
 * Authors: RJG
 * Date: 2022-08-07
 */
string nkConfigFilename() {
    return lmrConfig["config-directory"].str ~ "/" ~ lmrConfig["newton-krylov-config-filename"].str;
}
    

/**
 * Return the grid filename for a single block ('id') as a string.
 *
 * Authors: RJG
 * Date: 2022-08-06
 */
string gridFilenameWithoutExt(int id)
{
    string gName = lmrConfig["grid-directory"].str;
    gName ~= "/";
    gName ~= format(lmrConfig["block-filename-format"].str, id);
    return gName;
}

/**
 * Return the directory name for a snapshot of a flow field.
 *
 * Authors: RJG
 * Date: 2022-08-07
 */
string steadyFlowDirectory(int snapshot)
{
    string dName = lmrConfig["snapshot-directory"].str;
    dName ~= "/";
    dName ~= format(lmrConfig["snapshot-index-format"].str, snapshot);
    dName ~= "/";
    dName ~= lmrConfig["flow-directory"].str;
    return dName;
                             
}

/**
 * Return the flow solution filename for a single block ('id') as a string.
 *
 * Authors: RJG
 * Date: 2022-08-06
 */
string steadyFlowFilename(int snapshot, int blkId)
{
    string fName = steadyFlowDirectory(snapshot);
    fName ~= "/";
    fName ~= format(lmrConfig["block-filename-format"].str, blkId);
    fName ~= ".";
    fName ~= lmrConfig["zip-extension"].str;
    return fName;
}

static this()
{
    readLmrConfig();
}
