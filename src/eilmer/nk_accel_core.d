/** nk_accel_core.d
 * Core set of functions used in the Newton-Krylov updates for steady-state convergence.
 *
 * Authors: RJG and KAD
 * Date: 2022-02-28
 * History:
 *   2022-02-28 Fairly aggressive refactor of code that was in: steadystate_core.d
 */

module nk_accel_core;

import core.stdc.stdlib : exit;
import std.algorithm : min;
import std.algorithm.searching : countUntil;
import std.datetime : Clock;
import std.parallelism : parallel, defaultPoolThreads;
import std.stdio : File, writeln, writefln;
import std.array : appender;
import std.format : formattedWrite;
import std.json : JSONValue;
import std.math;
import std.string;
import std.conv : to;
import std.typecons : Tuple, tuple;

import nm.complex;
import nm.number : number;
import nm.smla;
import nm.bbla;
import json_helper;
import geom;

import conservedquantities : ConservedQuantities;

import globalconfig;
import globaldata;
import simcore_exchange;
import simcore_gasdynamic_step : detect_shocks;
import sfluidblock : SFluidBlock;
import user_defined_source_terms : getUDFSourceTermsForCell;

version(mpi_parallel) {
    import mpi;
}

/*---------------------------------------------------------------------
 * Exception class to signal N-K specific exceptions.
 *---------------------------------------------------------------------
 */
class NewtonKrylovException : Exception {
    @nogc
    this(string message, string file=__FILE__, size_t line=__LINE__,
         Throwable next=null)
    {
        super(message, file, line, next);
    }
}

/*---------------------------------------------------------------------
 * Enums for preconditioners
 *---------------------------------------------------------------------
 */
enum PreconditionerType { lusgs, diagonal, jacobi, sgs, ilu }

string preconditionerTypeName(PreconditionerType i)
{
    final switch (i) {
    case PreconditionerType.lusgs: return "lusgs";
    case PreconditionerType.diagonal: return "diagonal";
    case PreconditionerType.jacobi: return "jacobi";
    case PreconditionerType.sgs: return "sgs";
    case PreconditionerType.ilu: return "ilu";
    }
} 

PreconditionerType preconditionerTypeFromName(string name)
{
    switch (name) {
    case "lusgs", "lu_sgs": return PreconditionerType.lusgs;
    case "diagonal": return PreconditionerType.diagonal;
    case "jacobi": return PreconditionerType.jacobi;
    case "sgs": return PreconditionerType.sgs;
    case "ilu": return PreconditionerType.ilu;
    default:
        string errMsg = "The selected 'preconditioner' is unavailable.\n";
        errMsg ~= format("You selected: '%s'\n", name);
        errMsg ~= "The available strategies are: \n";
        errMsg ~= "   'lusgs'\n";
        errMsg ~= "   'diagonal'\n";
        errMsg ~= "   'jacobi'\n";
        errMsg ~= "   'sgs'\n";
        errMsg ~= "   'ilu'\n";
        errMsg ~= "Check your selection or its spelling in the input file.\n";
        throw new Error(errMsg);
    }
} 


/*---------------------------------------------------------------------
 * Structs for configuration: global and phase-specific
 *---------------------------------------------------------------------
 */

struct NKGlobalConfig {
    // global control based on step
    int setReferenceResidualsAtStep = 1;
    int freezeLimiterAtStep = -1;
    // stopping criterion
    int maxNewtonSteps = 1000;
    double stopOnRelativeResidual = 1.0e-99;
    double stopOnAbsoluteResidual = 1.0e-99;
    double stopOnMassBalance = -1.0;
    // CFL control
    double cflMax = 1.0e8;
    double cflMin = 0.001;
    Tuple!(int, "step", double, "cfl")[] cflSchedule;
    // phase control
    int numberOfPhases = 1;
    int[] phaseChangesAtSteps;
    // Newton stepping and continuation
    bool useLocalTimestep = true;
    bool inviscidCFLOnly = true;
    bool useLineSearch = true;
    bool usePhysicalityCheck = true;
    double physicalityCheckAllowableChange = 0.2;
    // Linear solver and preconditioning
    int maxLinearSolverIterations = 10;
    int maxLinearSolverRestarts = 0;
    bool useScaling = true;
    double frechetDerivativePerturbation = 1.0e-30;
    bool usePreconditioner = true;
    double preconditionerPerturbation = 1.0e-30;
    PreconditionerType preconditioner = PreconditionerType.ilu;
    // ILU setting
    int iluFill = 0;
    // SGS setting
    int sgsRelaxationIterations = 4;
    // output and diagnostics
    int totalSnapshots = 5;
    int stepsBetweenSnapshots = 10;
    int stepsBetweenDiagnostics = 10;
    int stepsBetweenLoadsUpdate = 20;

    void readValuesFromJSON(JSONValue jsonData)
    {
        setReferenceResidualsAtStep = getJSONint(jsonData, "set_reference_residuals_at_step", setReferenceResidualsAtStep);
        freezeLimiterAtStep = getJSONint(jsonData, "freeze_limiter_at_step", freezeLimiterAtStep);
        maxNewtonSteps = getJSONint(jsonData, "max_newton_steps", maxNewtonSteps);
        stopOnRelativeResidual = getJSONdouble(jsonData, "stop_on_relative_residual", stopOnRelativeResidual);
        stopOnAbsoluteResidual = getJSONdouble(jsonData, "stop_on_absolute_residual", stopOnAbsoluteResidual);
        stopOnMassBalance = getJSONdouble(jsonData, "stop_on_mass_balance", stopOnMassBalance);
        cflMax = getJSONdouble(jsonData, "cfl_max", cflMax);
        cflMin = getJSONdouble(jsonData, "cfl_min", cflMin);
        auto jsonArray = jsonData["cfl_schedule"].array;
        foreach (entry; jsonArray) {
            auto values = entry.array;
            cflSchedule ~= tuple!("step", "cfl")(values[0].get!int, values[1].get!double);
        }
        numberOfPhases = getJSONint(jsonData, "number_of_phases", numberOfPhases);
        phaseChangesAtSteps = getJSONintarray(jsonData, "phase_changes_at_steps", phaseChangesAtSteps);
        useLocalTimestep = getJSONbool(jsonData, "use_local_timestep", useLocalTimestep);
        inviscidCFLOnly = getJSONbool(jsonData, "inviscid_cfl_only", inviscidCFLOnly);
        useLineSearch = getJSONbool(jsonData, "use_line_search", useLineSearch);
        usePhysicalityCheck = getJSONbool(jsonData, "use_physicality_check", usePhysicalityCheck);
        physicalityCheckAllowableChange = getJSONdouble(jsonData, "physicality_check_allowable_change", physicalityCheckAllowableChange);
        maxLinearSolverIterations = getJSONint(jsonData, "max_linear_solver_iterations", maxLinearSolverIterations);
        maxLinearSolverRestarts = getJSONint(jsonData, "max_linear_solver_restarts", maxLinearSolverRestarts);
        useScaling = getJSONbool(jsonData, "use_scaling", useScaling);
        frechetDerivativePerturbation = getJSONdouble(jsonData, "frechet_derivative_perturbation", frechetDerivativePerturbation);
        usePreconditioner = getJSONbool(jsonData, "use_preconditioner", usePreconditioner);
        preconditionerPerturbation = getJSONdouble(jsonData, "preconditioner_perturbation", preconditionerPerturbation);
        auto pString = getJSONstring(jsonData, "preconditioner", "NO_SELECTION_SUPPLIED");
        preconditioner = preconditionerTypeFromName(pString);
        iluFill = getJSONint(jsonData, "ilu_fill", iluFill);
        sgsRelaxationIterations = getJSONint(jsonData, "sgs_relaxation_iterations", sgsRelaxationIterations);
        totalSnapshots = getJSONint(jsonData, "total_snapshots", totalSnapshots);
        stepsBetweenSnapshots = getJSONint(jsonData, "steps_between_snapshots", stepsBetweenSnapshots);
        stepsBetweenDiagnostics = getJSONint(jsonData, "steps_between_diagnostics", stepsBetweenDiagnostics);
        stepsBetweenLoadsUpdate = getJSONint(jsonData, "steps_between_loads_update", stepsBetweenLoadsUpdate);
    }
}
NKGlobalConfig nkCfg;

struct NKPhaseConfig {
    int residualInterpolationOrder = 2;
    int jacobianInterpolationOrder = 2;
    bool frozenPreconditioner = true;
    int stepsBetweenPreconditionerUpdate = 10;
    bool useAdaptivePreconditioner = false;
    bool ignoreStoppingCriteria = true;
    bool frozenLimiterForJacobian = true;
    double linearSolveTolerance = 0.01;
    // Auto CFL control
    bool useAutoCFL = false;
    double thresholdResidualDropForCFLGrowth = 0.99;
    double startCFL = 1.0;
    double maxCFL = 1000.0;
    double autoCFLExponent = 0.75;

    void readValuesFromJSON(JSONValue jsonData)
    {
        residualInterpolationOrder = getJSONint(jsonData, "residual_interpolation_order", residualInterpolationOrder);
        jacobianInterpolationOrder = getJSONint(jsonData, "jacobian_interpolation_order", jacobianInterpolationOrder);
        frozenPreconditioner = getJSONbool(jsonData, "frozen_preconditioner", frozenPreconditioner);
        stepsBetweenPreconditionerUpdate = getJSONint(jsonData, "steps_between_preconditioner_update", stepsBetweenPreconditionerUpdate);
        useAdaptivePreconditioner = getJSONbool(jsonData, "use_adaptive_preconditioner", useAdaptivePreconditioner);
        ignoreStoppingCriteria = getJSONbool(jsonData, "ignore_stopping_criteria", ignoreStoppingCriteria);
        frozenLimiterForJacobian = getJSONbool(jsonData, "frozen_limiter_for_jacobian", frozenLimiterForJacobian);
        linearSolveTolerance = getJSONdouble(jsonData, "linear_solver_tolerance", linearSolveTolerance);
        useAutoCFL = getJSONbool(jsonData, "use_auto_cfl", useAutoCFL);
        thresholdResidualDropForCFLGrowth = getJSONdouble(jsonData, "threshold_residual_drop_for_cfl_growth", thresholdResidualDropForCFLGrowth);
        startCFL = getJSONdouble(jsonData, "start_cfl", startCFL);
        maxCFL = getJSONdouble(jsonData, "max_cfl", maxCFL);
        autoCFLExponent = getJSONdouble(jsonData, "auto_cfl_exponent", autoCFLExponent);
    }
}

NKPhaseConfig[] nkPhases;
NKPhaseConfig activePhase;



/*---------------------------------------------------------------------
 * Class to handle CFL selection
 *---------------------------------------------------------------------
 */

/**
 * Interface to define the behaviour of a generic CFL selector.
 *
 * Authors: RJG and KAD
 * Date: 2022-03-08
 */
interface CFLSelector {
    @nogc double nextCFL(double cfl, int step, double currResidual, double prevResidual);
}

/**
 * A CFL selctor that simply returns a linear interpolation between
 * start and end points (in terms of steps).
 *
 * The CFL is constant off the ends the step range, as shown below.
 *
 *                   endStep    
 *                       |            cflEnd
 *                       +------------------
 *                      /
 *                     /
 *                    /
 *                   /
 *                  /
 *                 /
 *                /
 *               /
 *   cflStart   /
 * ------------+
 *             |
 *          startStep
 *
 * Authors: RJG and KAD
 * Date: 2022-03-08
 */

class LinearRampCFL : CFLSelector {
    this(int startStep, int endStep, double startCFL, double endCFL)
    {
        mStartStep = startStep;
        mEndStep = endStep;
        mStartCFL = startCFL;
        mEndCFL = endCFL;
    }

    @nogc
    override double nextCFL(double cfl, int step, double currResidual, double prevResidual)
    {
        if (step <= mStartStep) return mStartCFL;
        if (step >= mEndStep) return mEndCFL;
        double frac = (step - mStartStep)/(cast(double)(mEndStep - mStartStep));
        return (1.0-frac)*mStartCFL + frac*mEndCFL;
    }

private:
    int mStartStep;
    int mEndStep;
    double mStartCFL;
    double mEndCFL;
}

/**
 * CFL selector based on residual drop.
 *
 * Authors: RJG and KAD
 * Date: 2022-03-08
 */

class ResidualBasedAutoCFL : CFLSelector {
    this(double p, double cfl_max)
    {
        mP = p;
        mMaxCFL = cfl_max;
    }

    @nogc
    override double nextCFL(double cfl, int step, double currResidual, double prevResidual)
    {
        auto residRatio = prevResidual/currResidual;
        double cflTrial = cfl*pow(residRatio, mP);
        // Apply some safeguads on the value.
        cflTrial = fmin(cflTrial, mLimitOnCFLIncreaseRatio*cfl);
        cflTrial = fmax(cflTrial, mLimitOnCFLDecreaseRatio*cfl);
        cflTrial = fmin(cflTrial, mMaxCFL);
        return cflTrial;
    }

private:
    immutable double mLimitOnCFLIncreaseRatio = 2.0;
    immutable double mLimitOnCFLDecreaseRatio = 0.1;
    double mP;
    double mMaxCFL;
}



/*---------------------------------------------------------------------
 * Module-local globals
 *---------------------------------------------------------------------
 */

static int fnCount = 0;
immutable double dummySimTime = 0.0;
immutable double minScaleFactor = 1.0;
immutable string refResidFname = "config/reference-residuals.saved";
immutable string diagFname = "config/nk-diagnostics.dat";
    
ConservedQuantities referenceResiduals, currentResiduals, scale;


// Module-local, global memory arrays and matrices
// TODO: Think about these, maybe they shouldn't be globals
number[] g0;
number[] g1;
number[] h;
number[] hR;
Matrix!number H0;
Matrix!number H1;
Matrix!number Gamma;
Matrix!number Q0;
Matrix!number Q1;

/*---------------------------------------------------------------------
 * Locally used data structures
 *---------------------------------------------------------------------
 */

struct RestartInfo {
    double pseudoSimTime;
    double dt;
    double cfl;
    int step;
    double globalResidual;
    ConservedQuantities residuals;

    this(size_t n)
    {
        residuals = new ConservedQuantities(n);
    }
}
RestartInfo[] snapshots;

struct LinearSystemInput {
    int step;
    double dt;
    bool computePreconditioner;
};
LinearSystemInput lsi;

struct GMRESInfo {
    int nRestarts;
    double initResidual;
    double finalResidual;
    double iterationCount;
};
GMRESInfo gmresInfo;

/*---------------------------------------------------------------------
 * Main iteration algorithm
 *---------------------------------------------------------------------
 */


void performNewtonKrylovUpdates(int snapshotStart, int maxCPUs, int threadsPerMPITask)
{
    alias cfg = GlobalConfig;
    string jobName = cfg.base_file_name;

    if (cfg.verbosity_level > 1) writeln("Read N-K config file.");
    JSONValue jsonData = readJSONfile("config/"~cfg.base_file_name~".nkconfig");
    nkCfg.readValuesFromJSON(jsonData);
    
    double referenceGlobalResidual, globalResidual, prevGlobalResidual;
    bool residualsUpToDate = false;
    int nWrittenSnapshots;
    int startStep = 0;
    bool finalStep = false;
    double cfl;
    double dt;
    CFLSelector cflSelector;
    
    /*----------------------------------------------
     * Initialisation
     *----------------------------------------------
     */
    setAndReportThreads(maxCPUs, threadsPerMPITask);
    if (nkCfg.usePreconditioner) initPreconditioner();
    size_t nConserved = cfg.cqi.n;
    referenceResiduals = new ConservedQuantities(nConserved);
    currentResiduals = new ConservedQuantities(nConserved);
    scale = new ConservedQuantities(nConserved);
    initialiseDiagnosticsFile(diagFname);
    allocateGlobalGMRESWorkspace();
    foreach (blk; localFluidBlocks) {
        blk.allocate_GMRES_workspace();
    }
    /* solid blocks don't work just yet.
    allocate_global_solid_workspace();
    foreach (sblk; localSolidBlocks) {
        sblk.allocate_GMRES_workspace();
    }
    */

    // Look for global CFL schedule and use to set CFL
    if (nkCfg.cflSchedule.length > 0) {
        foreach (i, startRamp; nkCfg.cflSchedule[0 .. $-1]) {
            if (startStep >= startRamp.step) {
                auto endRamp = nkCfg.cflSchedule[i+1];
                cflSelector = new LinearRampCFL(startRamp.step, endRamp.step, startRamp.cfl, endRamp.cfl);
                break;
            }
        }
        // Or check we aren't at end of cfl schedule
        auto lastEntry = nkCfg.cflSchedule[$-1];
        if (startStep >= lastEntry.step) {
            // Set a flat CFL beyond limit of scheule.
            cflSelector = new LinearRampCFL(lastEntry.step, nkCfg.maxNewtonSteps, lastEntry.cfl, lastEntry.cfl);
        }
    }
    
    if (snapshotStart > 0) {
        /*----------------------------------------------
         * Special actions on restart
         *----------------------------------------------
         *  + extract restart information
         *  + read in reference residuals from file
         *  + determine how many snapshots have already been written
         *  + determine phase and set as active phase
         */
        extractRestartInfoFromTimesFile(jobName);
        referenceGlobalResidual = setReferenceResidualsFromFile();
        nWrittenSnapshots = determineNumberOfSnapshots();
        RestartInfo restart = snapshots[snapshotStart];
        startStep = restart.step + 1;
        globalResidual = restart.globalResidual;
        prevGlobalResidual = globalResidual;
        // Determine phase
        foreach (phase, phaseStep; nkCfg.phaseChangesAtSteps) {
            if (startStep < phaseStep) {
                setPhaseSettings(phase); 
                break;
            }
        }
        // end condition when step is past final phase
        if (startStep >= nkCfg.phaseChangesAtSteps[$-1]) {
            auto finalPhase = nkCfg.phaseChangesAtSteps.length;
            setPhaseSettings(finalPhase);
        }
        if (activePhase.useAutoCFL) {
            cflSelector = new ResidualBasedAutoCFL(activePhase.autoCFLExponent, activePhase.maxCFL);
            cfl = restart.cfl;
        }
        else { // Assume we have a global (phase-independent) schedule
            cfl = cflSelector.nextCFL(-1.0, startStep, -1.0, -1.0);
        }
    }
    else {
        // On fresh start, the phase setting must be at 0
        setPhaseSettings(0);
        if (activePhase.useAutoCFL) {
            cflSelector = new ResidualBasedAutoCFL(activePhase.autoCFLExponent, activePhase.maxCFL);
            cfl = activePhase.startCFL;
        }
        else { // Assume we have a global (phase-independent) schedule
            cfl = cflSelector.nextCFL(-1.0, startStep, -1.0, -1.0);
        }
    }

    // Start timer right at beginning of stepping.
    auto wallClockStart = Clock.currTime();
    double wallClockElapsed;
    foreach (step; startStep .. nkCfg.maxNewtonSteps) {
        /*---
         * 0. Check for any special actions based on step to perform at START of step
         *---
         *    a. change of phase
         *    b. step to set reference residuals
         *    c. limiter freezing
         *    d. set the timestep
         */
        residualsUpToDate = false;
        // 0a. change of phase 
        size_t currentPhase = countUntil(nkCfg.phaseChangesAtSteps, step);
        if (currentPhase != -1) { // start of new phase detected
            setPhaseSettings(currentPhase);
            if (activePhase.useAutoCFL) {
                // When we change phase, we reset CFL to user's selection if using auto CFL.
                cfl = activePhase.startCFL;
            }
        }

        // 0b. Check if setting reference residuals.
        if (step == nkCfg.setReferenceResidualsAtStep) {
            if (step == 0 && GlobalConfig.is_master_task) {
                string errMsg = "Reference residuals cannot be set at START of step 0\n"~
                    "since they have not been computed yet.\n"~
                    "Instead, select 'set_reference_residuals_at_step = 1' in your Lua configuration.";
                throw new Error(errMsg);
            }
            referenceGlobalResidual = globalResidual;
            if (!residualsUpToDate) {
                computeResiduals(currentResiduals);
            }
            referenceResiduals = currentResiduals;
        }

        // 0c. Check if we need to freeze limiter
        if (step == nkCfg.freezeLimiterAtStep) {
            // We need to compute the limiter a final time before freezing it.
            // This is achieved via evalRHS
            if (GlobalConfig.frozen_limiter == false) {
                evalResidual(0);
            }
            GlobalConfig.frozen_limiter = true;
        }

        if (step < nkCfg.freezeLimiterAtStep) {
            // Make sure that limiter is on in case it has been set
            // to frozen during a Jacobian evaluation.
            GlobalConfig.frozen_limiter = true;
        }

        // 0d. Set the timestep for this step
        if (nkCfg.useLocalTimestep) {
            setDtLocalInCells(cfl);
        }
        else {
            dt = determineMinDt(cfl);
        }
        
        /*---
         * 1. Perforn Newton update
         *---
         */
        globalResidual = solveNewtonStep(dt);
        // FIX ME -- 2022-07-02
        //        double omega = nkCfg.usePhysicalityCheck ? determineRelaxationFactor() : 1.0;
        double omega = 1.0;
        if (omega >= nkCfg.physicalityCheckAllowableChange) {
            // Things are good. Apply omega-scaled update and continue on.
            // We think??? If not, we bail at this point.
            try {
                applyNewtonUpdate(omega);
            }
            catch (NewtonKrylovException e) {
                // We need to bail out at this point.
                // User can probably restart with less aggressive CFL schedule.
                if (GlobalConfig.is_master_task) {
                    writeln("Update failure in Newton step.");
                    writefln("step= %d, CFL= %e, dt= %e, global-residual= %e ", step, cfl, dt, globalResidual);
                    writeln("Error message from failed update:");
                    writefln("%s", e.msg);
                    writeln("You might be able to try a smaller CFL.");
                    writeln("Bailing out!");
                    exit(1);
                }
            }
            cfl = cflSelector.nextCFL(cfl, step, globalResidual, prevGlobalResidual);
        }
        else {
            if (GlobalConfig.is_master_task) {
                writeln("WARNING: relaxation factor for Newton update is very small.");
                writefln("step= %d, relaxation factor= %f");
                writeln("Bailing out!");
                exit(1);
            }
            /*
            cfl = nkCfg.cflReductionFactorOnFail * cfl;
            // Return flow states to their original state for next attempt.
            foreach (blk; parallel(localFluidBlocks,1)) {
                foreach (cell; blk.cells) {
                    cell.decode_conserved(0, 0, 0.0);
                }
            }
            */
            // [THINK] What do we have to halt repeated reductions in CFL?
        }
        
        /*---
         * 2. Post-update actions
         *---
         * Here we need to do some house-keeping and see if we continue with iterations.
         */
        // We can now set previous residual in preparation for next step.
        prevGlobalResidual = globalResidual;
        
        /*---
         * 2a. Stopping checks.
         *---
         */
        if (step == nkCfg.maxNewtonSteps) {
            finalStep = true;
            if (GlobalConfig.is_master_task) {
                writeln("STOPPING: Reached maximum number of steps.");
            }
        }
        if (globalResidual <= nkCfg.stopOnAbsoluteResidual) {
            finalStep = true;
            if (GlobalConfig.is_master_task) {
                writeln("STOPPING: The absolute global residual is below target value.");
                writefln("         current global residual= %.6e  target value= %.6e", globalResidual, nkCfg.stopOnAbsoluteResidual);
            }
        }
        if ((globalResidual/referenceGlobalResidual) <= nkCfg.stopOnRelativeResidual) {
            finalStep = true;
            if (GlobalConfig.is_master_task) {
                writeln("STOPPING: The relative global residual is below target value.");
                writefln("         current residual= %.6e  target value= %.6e", (globalResidual/referenceGlobalResidual), nkCfg.stopOnRelativeResidual);
            }
        }
        // [TODO] Add in a halt_now condition.
        /*---
         * 2b. Reporting (to files and screen)
         *---
         */
        if (((step % nkCfg.stepsBetweenDiagnostics) == 0) || finalStep) {
            //writeDiagnostics(step, dt, cfl, residualsUpToDate);
        }

        // [TODO] Write loads. We only need one lot of loads.
        // Any intermediate loads before steady-state have no physical meaning.
        // They might have some diagnostic purpose?

        // Reporting to screen on progress.
        if ( ((step % GlobalConfig.print_count) == 0) || finalStep ) {
            wallClockElapsed = 1.0e-3*(Clock.currTime() - wallClockStart).total!"msecs"();
            //printStatusToScreen(step, cfl, dt, wallClockElapsed, residualsUpToDate);
        }
        
    }
}



/*---------------------------------------------------------------------
 * Auxiliary functions related to initialisation of stepping
 *---------------------------------------------------------------------
 */

void setAndReportThreads(int maxCPUs, int threadsPerMPITask)
{
    alias cfg = GlobalConfig;

    // 1. Check we aren't using more task threads than blocks
    int extraThreadsInPool;
    auto nBlocksInThreadParallel = localFluidBlocks.length;
    version(mpi_parallel) {
        extraThreadsInPool = min(threadsPerMPITask-1, nBlocksInThreadParallel-1);
    } else {
        extraThreadsInPool = min(maxCPUs-1, nBlocksInThreadParallel-1);
    }
    defaultPoolThreads(extraThreadsInPool);

    // 2. Report out the thread configuration for run-time
    version(mpi_parallel) {
        writefln("MPI-task %d : running with %d threads.", cfg.mpi_rank_for_local_task, extraThreadsInPool+1);
    }
    else {
        writefln("Single process running with %d threads.", extraThreadsInPool+1); // +1 for main thread.
    }
}

void initPreconditioner()
{
    if (nkCfg.usePreconditioner) {
        evalResidual(0);
        // initialize the flow Jacobians used as local precondition matrices for GMRES
        final switch (nkCfg.preconditioner) {
        case PreconditionerType.jacobi:
            foreach (blk; localFluidBlocks) { blk.initialize_jacobian(-1, nkCfg.preconditionerPerturbation); }
            break;
        case PreconditionerType.ilu:
            foreach (blk; localFluidBlocks) { blk.initialize_jacobian(0, nkCfg.preconditionerPerturbation); }
            break;
        case PreconditionerType.sgs:
            foreach (blk; localFluidBlocks) { blk.initialize_jacobian(0, nkCfg.preconditionerPerturbation); }
            break;
        case PreconditionerType.lusgs:
            // do nothing
            break;
        case PreconditionerType.diagonal:
            // do nothing
            break; 
        } // end switch
    }
}

void extractRestartInfoFromTimesFile(string jobName)
{
    size_t nConserved = GlobalConfig.cqi.n;
    RestartInfo restartInfo = RestartInfo(nConserved);
    // Start reading the times file, looking for the snapshot index
    auto timesFile = File("./config/" ~ jobName ~ ".times");
    auto line = timesFile.readln().strip();
    while (line.length > 0) {
        if (line[0] != '#') {
            // Process a non-comment line.
            auto tokens = line.split();
            auto idx = to!int(tokens[0]);
            restartInfo.pseudoSimTime = to!double(tokens[1]);
            restartInfo.dt = to!double(tokens[2]);
            restartInfo.cfl = to!double(tokens[3]);
            restartInfo.step = to!int(tokens[4]);
            restartInfo.globalResidual = to!double(tokens[5]);
            size_t startIdx = 6;
            foreach (ivar; 0 .. nConserved) {
                restartInfo.residuals.vec[ivar] = to!double(tokens[startIdx+ivar]);
            }
            snapshots ~= restartInfo;
        }
        line = timesFile.readln().strip();
    }
    timesFile.close();
    return;
}

double setReferenceResidualsFromFile()
{
    double refGlobalResidual;
    size_t nConserved = GlobalConfig.cqi.n;
    
    auto refResid = File(refResidFname, "r");
    auto line = refResid.readln().strip();
    auto tokens = line.split();
    refGlobalResidual = to!double(tokens[0]);
    size_t startIdx = 1;
    foreach (ivar; 0 .. nConserved) {
        referenceResiduals.vec[ivar] = to!double(tokens[startIdx+ivar]);
    }
    refResid.close();
    return refGlobalResidual;
}

int determineNumberOfSnapshots()
{
    string jobName = GlobalConfig.base_file_name;
    int nWrittenSnapshots = 0;
    auto timesFile = File("./config/" ~ jobName ~ ".times");
    auto line = timesFile.readln().strip();
    while (line.length > 0) {
        if (line[0] != '#') {
            nWrittenSnapshots++;
        }
        line = timesFile.readln().strip();
    }
    timesFile.close();
    nWrittenSnapshots--; // We don't count the initial solution as a written snapshot
    return nWrittenSnapshots; 
}

void initialiseDiagnosticsFile(string diagFname)
{
    alias cfg = GlobalConfig;
    size_t mass = cfg.cqi.mass;
    size_t xMom = cfg.cqi.xMom;
    size_t yMom = cfg.cqi.yMom;
    size_t zMom = cfg.cqi.zMom;
    size_t totEnergy = cfg.cqi.totEnergy;
    size_t rhoturb = cfg.cqi.rhoturb;
    size_t species = cfg.cqi.species;
    size_t modes = cfg.cqi.modes;

    if (cfg.is_master_task) {
        File fDiag;
        fDiag = File(diagFname, "w");
        fDiag.writeln("#  1: step");
        fDiag.writeln("#  2: dt");
        fDiag.writeln("#  3: CFL");
        fDiag.writeln("#  4: eta");
        fDiag.writeln("#  5: nRestarts");
        fDiag.writeln("#  6: nIters");
        fDiag.writeln("#  7: nFnCalls");
        fDiag.writeln("#  8: wall-clock, s");
        fDiag.writeln("#  9: global-residual-abs");
        fDiag.writeln("# 10: global-residual-rel");
        fDiag.writeln("# 11: mass-balance");
        fDiag.writeln("# 12: linear-solve-residual");
        fDiag.writeln("# 13: omega");
        fDiag.writeln("# 14: PC");
        int nVarStart = 15;
        fDiag.writefln("#  %02d: mass-abs", nVarStart + (2*mass));
        fDiag.writefln("# %02d: mass-rel", nVarStart + (2*mass+1));
        fDiag.writefln("# %02d: x-mom-abs", nVarStart + (2*xMom));
        fDiag.writefln("# %02d: x-mom-rel", nVarStart + (2*xMom+1));
        fDiag.writefln("# %02d: y-mom-abs", nVarStart + (2*yMom));
        fDiag.writefln("# %02d: y-mom-rel", nVarStart + (2*yMom+1));
        if ( cfg.dimensions == 3 ) {
            fDiag.writefln("# %02d: z-mom-abs", nVarStart + (2*zMom));
            fDiag.writefln("# %02d: z-mom-rel", nVarStart + (2*zMom+1));
        }
        fDiag.writefln("# %02d: energy-abs", nVarStart + (2*totEnergy));
        fDiag.writefln("# %02d: energy-rel", nVarStart + (2*totEnergy+1));
        auto nt = cfg.turb_model.nturb;
        foreach(it; 0 .. nt) {
            string tvname = cfg.turb_model.primitive_variable_name(it);
            fDiag.writefln("# %02d: %s-abs", nVarStart + (2*(rhoturb+it)), tvname);
            fDiag.writefln("# %02d: %s-rel", nVarStart + (2*(rhoturb+it)+1), tvname);
        }
        auto nsp = cfg.gmodel_master.n_species;
        if ( nsp > 1) {
            foreach(isp; 0 .. nsp) {
                string spname = cfg.gmodel_master.species_name(isp);
                fDiag.writefln("# %02d: %s-abs", nVarStart + (2*(species+isp)), spname);
                fDiag.writefln("# %02d: %s-rel", nVarStart + (2*(species+isp)+1), spname);
            }
        }
        auto nmodes = cfg.gmodel_master.n_modes;
        foreach(imode; 0 .. nmodes) {
            string modename = "T_MODES["~to!string(imode)~"]"; //GlobalConfig.gmodel_master.energy_mode_name(imode);
            fDiag.writefln("# %02d: %s-abs", nVarStart + (2*(modes+imode)), modename);
            fDiag.writefln("# %02d: %s-rel", nVarStart + (2*(modes+imode)+1), modename);
        }
        fDiag.close();
    }
}

void allocateGlobalGMRESWorkspace()
{
    size_t m = to!size_t(nkCfg.maxLinearSolverIterations);
    g0.length = m+1;
    g1.length = m+1;
    h.length = m+1;
    hR.length = m+1;
    H0 = new Matrix!number(m+1, m);
    H1 = new Matrix!number(m+1, m);
    Gamma = new Matrix!number(m+1, m+1);
    Q0 = new Matrix!number(m+1, m+1);
    Q1 = new Matrix!number(m+1, m+1);
}

/*---------------------------------------------------------------------
 * Auxiliary functions related to iteration algorithm
 *---------------------------------------------------------------------
 */

void setPhaseSettings(size_t phase)
{
    activePhase = nkPhases[phase];
    foreach (blk; parallel(localFluidBlocks,1)) blk.set_interpolation_order(activePhase.residualInterpolationOrder);
}

void computeResiduals(ref ConservedQuantities residuals)
{
    size_t nConserved = GlobalConfig.cqi.n;

    foreach (blk; parallel(localFluidBlocks,1)) {
        blk.residuals.copy_values_from(blk.cells[0].dUdt[0]);
        foreach (ivar; 0 .. nConserved) blk.residuals.vec[ivar] = fabs(blk.residuals.vec[ivar]);

        foreach (cell; blk.cells) {
            foreach (ivar; 0 .. nConserved) blk.residuals.vec[ivar] = fmax(blk.residuals.vec[ivar], fabs(cell.dUdt[0].vec[ivar]));
        }
    }
    // Do next bit in serial to reduce information to current thread
    residuals.copy_values_from(localFluidBlocks[0].residuals);
    foreach (blk; localFluidBlocks) {
        foreach (ivar; 0 .. nConserved) residuals.vec[ivar] = fmax(residuals.vec[ivar], blk.residuals.vec[ivar]);
    }
    // and for MPI a reduce onto master rank
    version(mpi_parallel) {
        foreach (ivar; 0 .. nConserved) {
            if (GlobalConfig.is_master_task) {
                MPI_Reduce(MPI_IN_PLACE, &(residuals.vec[ivar].re), 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            }
            else {
                MPI_Reduce(&(residuals.vec[ivar].re), &(residuals.vec[ivar].re), 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            }
        }
    }
}

/**
 * This function determines the minimum dt across all cells.
 *
 * Authors: RJG and KAD
 * Date: 2022-03-08
 */
double determineMinDt(double cfl)
{
    double signal;
    double dt;
    bool inviscidCFLOnly = nkCfg.inviscidCFLOnly;
    bool firstCell;
    foreach (blk; parallel(localFluidBlocks,1)) {
        firstCell = true;
        foreach (cell; blk.cells) {
            signal = cell.signal_frequency();
            if (inviscidCFLOnly) signal = cell.signal_hyp.re;
            cell.dt_local = cfl / signal;
            if (firstCell) {
                blk.dtMin = cell.dt_local;
                firstCell = false;
            }
            else {
                blk.dtMin = fmin(blk.dtMin, cell.dt_local);
            }
        }
    }
    // Find smallest dt across these local blocks in serial search
    dt = localFluidBlocks[0].dtMin;
    foreach (blk; localFluidBlocks) {
        dt = fmin(dt, blk.dtMin);
    }

    version(mpi_parallel) {
        // Find smallest dt globally if using distributed memory.
        MPI_Allreduce(MPI_IN_PLACE, &dt, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    }
    return dt;
}

/**
 * Set dt_local in cells based on CFL.
 *
 * Authors: RJG and KAD
 * Date: 2022-03-08
 */
void setDtLocalInCells(double cfl)
{
    double signal;
    bool inviscidCFLOnly = nkCfg.inviscidCFLOnly;
    foreach (blk; parallel(localFluidBlocks,1)) {
        foreach (cell; blk.cells) {
            signal = cell.signal_frequency();
            if (inviscidCFLOnly) signal = cell.signal_hyp.re;
            cell.dt_local = cfl / signal;
        }
    }
}


/*---------------------------------------------------------------------
 * Mixins to handle shared-memory dot product and norm
 *---------------------------------------------------------------------
 */

string dotOverBlocks(string dot, string A, string B)
{
    return `
foreach (blk; parallel(localFluidBlocks,1)) {
   blk.dotAcc = 0.0;
   foreach (k; 0 .. blk.nvars) {
      blk.dotAcc += blk.`~A~`[k].re*blk.`~B~`[k].re;
   }
}
`~dot~` = 0.0;
foreach (blk; localFluidBlocks) `~dot~` += blk.dotAcc;`;

}

string norm2OverBlocks(string norm2, string blkMember)
{
    return `
foreach (blk; parallel(localFluidBlocks,1)) {
   blk.normAcc = 0.0;
   foreach (k; 0 .. blk.nvars) {
      blk.normAcc += blk.`~blkMember~`[k].re*blk.`~blkMember~`[k].re;
   }
}
`~norm2~` = 0.0;
foreach (blk; localFluidBlocks) `~norm2~` += blk.normAcc;
`~norm2~` = sqrt(`~norm2~`);`;

}


/**
 * This function solves a linear system to provide a Newton step for the flow field.
 *
 * The particular linear solver used here is GMRES. It uses right-preconditioning,
 * scaling and possibly restarts.
 *
 * The linear solver is baked in-place here because the "sytem" is so closely
 * tied to the the flow field. In other words, we require several flow field
 * residual evaluations. This coupling between linear solver and fluid updates
 * makes it difficult to extract as a stand-alone routine.
 *
 * NOTE: GMRES algorithm does not need to compute the global residual.
 * However, it's convenient and efficient to do this computation here as a side effect.
 * The reason is because we've done all the work of packing the residual vector, R.
 * While the data is current, we can do the parallel computation of norm on that vector.
 *
 * Algorithm overview:
 *    0. Preparation for iterations.
 *       0a) Evaluate dUdt and store in R.
 *       0b) Determine scale factors.
 *       0c) Compute global residual. (Efficiency measure)
 *       0d) Compute scaled r0.
 *       0e) Compute preconditioner matrix
 *    1. Start loop on restarts
 *       1a) Perform inner iterations.
 *       1b) Check tolerance for convergence
 *    2. Post-iteration actions
 *       2a) Unscale values
 *     
 * Authors: RJG and KAD
 * Date: 2022-03-02
 * History:
 *    2022-03-02  Major clean-up as part of refactoring work.
 */
double solveNewtonStep(double dt)
{
    alias cfg = GlobalConfig;

    bool isConverged = false;
    /*---
     * 0. Preparation for iterations.
     *---
     */
    evalResidual(0);
    setResiduals();
    double globalResidual = computeGlobalResidual();
    determineScaleFactors(scale);
    // r0 = A*x0 - b
    compute_r0(scale);
    // beta = ||r0||
    number beta = computeLinearSystemResidual();
    number beta0 = beta; // Store a copy as initial residual
                         // because we will look for relative drop in residual
                         // compared to this.
    // v = r0/beta
    prepareKrylovSpace(beta);
    auto targetResidual = activePhase.linearSolveTolerance * beta;
    if (lsi.computePreconditioner) {
        computePreconditioner();
    }

    /*---
     * 1. Outer loop of restarted GMRES
     *---
     */
    size_t r_break;
    int maxIterations = nkCfg.maxLinearSolverIterations;
    // We add one here because input is to do with number of *restarts*.
    // We need at least one attempt (+1) plus the number of restarts chosen.
    int nAttempts = nkCfg.maxLinearSolverRestarts + 1;
    int iterationCount;
    foreach (r; 0 .. nAttempts) {
        // Initialise some working arrays and matrices for this step.
        g0[] = to!number(0.0);
        g1[] = to!number(0.0);
        H0.zeros();
        H1.zeros();
        // Set first residual entry.
        g0[0] = beta;

        // Delegate inner iterations
        isConverged = performIterations(maxIterations, dt, beta0, targetResidual,
                                        scale, iterationCount);
        int m = iterationCount;

        // At end H := R up to row m
        //        g := gm up to row m
        upperSolve!number(H1, to!int(m), g1);
        // In serial, distribute a copy of g1 to each block
        foreach (blk; localFluidBlocks) blk.g1[] = g1[];
        foreach (blk; parallel(localFluidBlocks,1)) {
            nm.bbla.dot!number(blk.V, blk.nvars, m, blk.g1, blk.zed);
        }

        // unscale 'z'
        unscaleVector("z");

        // Prepare dU values (for Newton update)
        /* FIX ME -- 2022-07-02
        if (nkCfg.usePreconditioner) {
            // Remove preconditioner effect from values.
            removePreconditioning();
        }
        else {
            foreach(blk; parallel(localFluidBlocks,1)) {
                blk.dU[] = blk.zed[];
            }
        }
        */

        foreach (blk; parallel(localFluidBlocks,1)) {
            foreach (k; 0 .. blk.nvars) blk.dU[k] += blk.x0[k];
        }

        if (isConverged || (r == nkCfg.maxLinearSolverRestarts)) {
            // We are either converged, or
            // we've run out of restart attempts.
            // In either case, we can leave now.
            r_break = r;
            break;
        }

        // If we get here, we need to prepare for next restart.
        // This requires setting x0[] and r0[].
        // We'll compute r0[] using the approach of Fraysee et al. (2005)
        foreach (blk; parallel(localFluidBlocks, 1)) {
            blk.x0[] = blk.dU[];
        }

        foreach (blk; localFluidBlocks) copy(Q1, blk.Q1);
        // Set all values in g0 to 0.0 except for final (m+1) value
        foreach (i; 0 .. m) g0[i] = 0.0;
        foreach (blk; localFluidBlocks) blk.g0[] = g0[];
        foreach (blk; parallel(localFluidBlocks,1)) {
            nm.bbla.dot(blk.Q1, m, m+1, blk.g0, blk.g1);
        }
        foreach (blk; parallel(localFluidBlocks,1)) {
            nm.bbla.dot(blk.V, blk.nvars, m+1, blk.g1, blk.r0);
        }

        mixin(dotOverBlocks("beta", "r0", "r0"));
        version(mpi_parallel) {
            MPI_Allreduce(MPI_IN_PLACE, &(beta.re), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            version(complex_numbers) { MPI_Allreduce(MPI_IN_PLACE, &(beta.im), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); }
        }
        beta = sqrt(beta);

        foreach (blk; parallel(localFluidBlocks,1)) {
            foreach (k; 0 .. blk.nvars) {
                blk.v[k] = blk.r0[k]/beta;
                blk.V[k,0] = blk.v[k];
            }
        }
    }

    // Set some information before leaving. This might be used in diagnostics file.
    gmresInfo.nRestarts = to!int(r_break);
    gmresInfo.initResidual = beta0.re;
    gmresInfo.finalResidual = beta.re;
    gmresInfo.iterationCount = iterationCount;
    
    // NOTE: global residual is value at START, before applying Newton update.
    // The caller applies the update, so we can't compute the new global
    // residual in here.
    return globalResidual;
}

/**
 * Copy values from dUdt into R.
 *
 * Authors: RJG and KAD
 * Date: 2022-03-02
 */
void setResiduals()
{
    size_t nConserved = GlobalConfig.cqi.n;
    foreach (blk; parallel(localFluidBlocks,1)) {
        size_t startIdx = 0;
        foreach (i, cell; blk.cells) {
            blk.R[startIdx .. startIdx+nConserved] = cell.dUdt[0].vec[0 .. nConserved];
            startIdx += nConserved;
        }
    }
}

/**
 * Determine scale factors per conserved quantity.
 *
 * The scale factors are typically the maximum rate of change found globally
 * (over all cells) per each conserved quantity. However, we place some gaurds
 * on determining those scales when rates of change are small.
 * 
 * Authors: RJG and KAD
 * Date: 2022-03-02
 */
void determineScaleFactors(ref ConservedQuantities scale)
{
    scale.vec[] = to!number(0.0);
    // First do this for each block.
    size_t nConserved = GlobalConfig.cqi.n;
    foreach (blk; parallel(localFluidBlocks,1)) {
        blk.maxR.vec[] = to!number(0.0);
        foreach (cell; blk.cells) {
            foreach (ivar; 0 .. nConserved) {
                blk.maxR.vec[ivar] = fmax(blk.maxR.vec[ivar], fabs(cell.dUdt[0].vec[ivar]));
            }
        }
    }
    // Next, reduce that maxR information across all blocks and processes
    foreach (blk; localFluidBlocks) {
        foreach (ivar; 0 .. nConserved) {
            scale.vec[ivar] = fmax(scale.vec[ivar], blk.maxR.vec[ivar]);
        }
    }
    // In distributed memory, reduce max values and make sure everyone has a copy.
    version(mpi_parallel) {
        foreach (ivar; 0 .. nConserved) {
            MPI_Allreduce(MPI_IN_PLACE, &(scale.vec[ivar].re), 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        }
    }
    // Use a guard on scale values if they get small
    foreach (ivar; 0 .. nConserved) {
        scale.vec[ivar] = fmax(scale.vec[ivar], minScaleFactor);
    }
    // Value is presently maxRate. Store as scale = 1/maxRate.
    foreach (ivar; 0 .. nConserved) {
        scale.vec[ivar] = 1./scale.vec[ivar];
    }
}


/**
 * Compute the global residual based on vector R.
 *
 * For certain turbulence models, we scale the contribution in the global residual
 * so that certain quantities do not dominate this norm. For example, the turbulent
 * kinetic energy in the k-omega model has its contribution scaled down.
 *
 * Authors: KAD and RJG
 * Date: 2022-03-02
 */
double computeGlobalResidual()
{
    double globalResidual;
    mixin(dotOverBlocks("globalResidual", "R", "R"));
    version(mpi_parallel) {
        MPI_Allreduce(MPI_IN_PLACE, &globalResidual, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }
    globalResidual = sqrt(globalResidual);

    return globalResidual;
}


/**
 * Compute the initial scaled residual vector for GMRES method.
 *
 * r0 = b - A*x0
 *
 * With x0 = [0] (as is common), r0 = b = R
 * However, we apply scaling also at this point.
 *
 * Authors: KAD and RJG
 * Date: 2022-03-02
 */
void compute_r0(ConservedQuantities scale)
{
    size_t nConserved = GlobalConfig.cqi.n;
    foreach (blk; parallel(localFluidBlocks,1)) {
        blk.x0[] = to!number(0.0);
        foreach (i; 0 .. blk.r0.length) {
            size_t ivar = i % nConserved;
            blk.r0[i] = scale.vec[ivar]*blk.R[i];
        }
    }
}

/**
 * Compute the residual of the linear system from r0.
 *
 * Authors: RJG and KAD
 * Date: 2022-03-02
 */
number computeLinearSystemResidual()
{
    number beta;
    mixin(dotOverBlocks("beta", "r0", "r0"));
    version(mpi_parallel) {
        MPI_Allreduce(MPI_IN_PLACE, &(beta.re), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        version(complex_numbers) { MPI_Allreduce(MPI_IN_PLACE, &(beta.im), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); }
    }
    beta = sqrt(beta);
    return beta;
}

/**
 * Prepare the Krylov space for iterating.
 *
 * Authors: RJG and KAD
 * Date: 2022-03-02
 */
void prepareKrylovSpace(number beta)
{
    g0[0] = beta;
    foreach (blk; parallel(localFluidBlocks,1)) {
        blk.v[] = (1./beta)*blk.r0[];
        foreach (k; 0 .. blk.nvars) {
            blk.V[k,0] = blk.v[k];
        }
    }
}

/**
 * Computes the preconditioner for use in GMRES solver.
 *
 * Authors: KAD and RJG
 * Date: 2022-03-02
 */
void computePreconditioner()
{
    size_t nConserved = GlobalConfig.cqi.n;
    double dt = lsi.dt;
    
    final switch (nkCfg.preconditioner) {
    case PreconditionerType.lusgs:
        // do nothing
        break;
    case PreconditionerType.diagonal:
        goto case PreconditionerType.sgs;
    case PreconditionerType.jacobi:
        goto case PreconditionerType.sgs;
    case PreconditionerType.sgs:
        foreach (blk; parallel(localFluidBlocks,1)) {
            blk.evaluate_jacobian();
            blk.flowJacobian.augment_with_dt(blk.cells, dt, blk.cells.length, nConserved);
            nm.smla.invert_block_diagonal(blk.flowJacobian.local, blk.flowJacobian.D, blk.flowJacobian.Dinv, blk.cells.length, nConserved);
        }
        break;
    case PreconditionerType.ilu:
        foreach (blk; parallel(localFluidBlocks,1)) {
            blk.evaluate_jacobian();
            blk.flowJacobian.augment_with_dt(blk.cells, dt, blk.cells.length, nConserved);
            nm.smla.decompILU0(blk.flowJacobian.local);
        }
        break;
    }
}

/*---------------------------------------------------------------------
 * Mixins to scale and unscale vectors
 *---------------------------------------------------------------------
 */

string scaleVector(string blkMember)
{
    return `
foreach (k; 0 .. blk.nvars) {
    size_t ivar = k % nConserved;
    blk.`~blkMember~`[k] *= scale.vec[ivar];
}`;
}

string unscaleVector(string blkMember)
{
    return `
foreach (k; 0 .. blk.nvars) {
    size_t ivar = k % nConserved;
    blk.`~blkMember~`[k] /= scale.vec[ivar];
}`;
}


/**
 * Perform GMRES iterations to fill Krylov subspace and get solution estimate.
 *
 * Returns boolean indicating converged or not.
 *
 * Authors: RJG and KAD
 * Date: 2022-03-02
 */
bool performIterations(int maxIterations, double dt, number beta0, number targetResidual,
                       ref ConservedQuantities scale, ref int iterationCount)
{
    alias cfg = GlobalConfig;

    bool isConverged = false;
    size_t nConserved = cfg.cqi.n;

    foreach (j; 0 .. maxIterations) {
        iterationCount = j+1;

        // 1. Unscale v
        // v is scaled earlier when r0 copied in.
        // However, to compute Jv via Frechet, we will need
        // unscaled values.
        foreach (blk; parallel(localFluidBlocks,1)) {
            unscaleVector("v");
        }

        // 2. Apply preconditioning (if requested)
        /* FIX ME -- 2022-07-02
        if (nkCfg.usePreconditioner) {
            applyPreconditioning();
        }
        else {
            foreach (blk; parallel(localFluidBlocks,1)) {
                blk.zed[] = blk.v[];
            }
        }
        */

        // 3. Jacobian-vector product
        // 3a. Prepare w vector with 1/dt term.
        //      (I/dt)(P^-1)v
        foreach (blk; parallel(localFluidBlocks,1)) {
            size_t startIdx = 0;
            foreach (cell; blk.cells) {
                number dtInv = nkCfg.useLocalTimestep ? 1.0/cell.dt_local : 1.0/dt;
                blk.w[startIdx .. startIdx + nConserved] = dtInv*blk.zed[startIdx .. startIdx + nConserved];
                startIdx += nConserved;
            }
        }
        // 3b. Determine perturbation size, sigma
        // Kyle's experiments show one needs to recompute on every step
        // if using the method to estimate a perturbation size based on
        // vector 'v'.
        number sigma;
        version (complex_numbers) {
            // For complex-valued Frechet derivative, a very small perturbation
            // works well (almost) all the time.
            sigma = nkCfg.frechetDerivativePerturbation;
        }
        else {
            // For real-valued Frechet derivative, we may need to attempt to compute
            // a perturbation size.
            sigma =  (nkCfg.frechetDerivativePerturbation < 0.0) ? computePerturbationSize() : nkCfg.frechetDerivativePerturbation;
        }

        // 3b. Evaluate Jz and place result in z
        evalJacobianVectorProduct(sigma);

        // 3c. Complete the calculation of w
        foreach (blk; parallel(localFluidBlocks,1)) {
            foreach (k; 0 .. blk.nvars)  blk.w[k] = blk.w[k] - blk.zed[k];
            scaleVector("w");
        }

        // 4. The remainder of the algorithm looks a lot like any standard
        // GMRES implementation (for example, see smla.d)
        foreach (i; 0 .. j+1) {
            foreach (blk; parallel(localFluidBlocks,1)) {
                // Extract column 'i'
                foreach (k; 0 .. blk.nvars ) blk.v[k] = blk.V[k,i];
            }
            number H0_ij;
            mixin(dotOverBlocks("H0_ij", "w", "v"));
            version(mpi_parallel) {
                MPI_Allreduce(MPI_IN_PLACE, &(H0_ij.re), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                version(complex_numbers) { MPI_Allreduce(MPI_IN_PLACE, &(H0_ij.im), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); }
            }
            H0[i,j] = H0_ij;
            foreach (blk; parallel(localFluidBlocks,1)) {
                foreach (k; 0 .. blk.nvars) blk.w[k] -= H0_ij*blk.v[k];
            }
        }
        number H0_jp1j;
        mixin(dotOverBlocks("H0_jp1j", "w", "w"));
        version(mpi_parallel) {
            MPI_Allreduce(MPI_IN_PLACE, &(H0_jp1j.re), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            version(complex_numbers) { MPI_Allreduce(MPI_IN_PLACE, &(H0_jp1j.im), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); }
        }
        H0_jp1j = sqrt(H0_jp1j);
        H0[j+1,j] = H0_jp1j;

        foreach (blk; parallel(localFluidBlocks,1)) {
            foreach (k; 0 .. blk.nvars) {
                blk.v[k] = blk.w[k]/H0_jp1j;
                blk.V[k,j+1] = blk.v[k];
            }
        }

        // Build rotated Hessenberg progressively
        if (j != 0) {
            // Extract final column in H
            foreach (i; 0 .. j+1) h[i] = H0[i,j];
            // Rotate column by previous rotations (stored in Q0)
            nm.bbla.dot(Q0, j+1, j+1, h, hR);
            // Place column back in H
            foreach (i; 0 .. j+1) H0[i,j] = hR[i];
        }
        // Now form new Gamma
        Gamma.eye();
        auto denom = sqrt(H0[j,j]*H0[j,j] + H0[j+1,j]*H0[j+1,j]);
        auto s_j = H0[j+1,j]/denom;
        auto c_j = H0[j,j]/denom;
        Gamma[j,j] = c_j; Gamma[j,j+1] = s_j;
        Gamma[j+1,j] = -s_j; Gamma[j+1,j+1] = c_j;
        // Apply rotations
        nm.bbla.dot(Gamma, j+2, j+2, H0, j+1, H1);
        nm.bbla.dot(Gamma, j+2, j+2, g0, g1);
        // Accumulate Gamma rotations in Q.
        if (j == 0) {
            copy(Gamma, Q1);
        }
        else {
            nm.bbla.dot!number(Gamma, j+2, j+2, Q0, j+2, Q1);
        }

        // Prepare for next step
        copy(H1, H0);
        g0[] = g1[];
        copy(Q1, Q0);

        // Get residual
        auto resid = fabs(g1[j+1]);
        auto linSolResid = (resid/beta0).re;
        if (resid <= targetResidual) {
            isConverged = true;
            break;
        }
    }
    return isConverged;
}


/**
 * Apply preconditioning to GMRES iterations.
 *
 * Authors: KAD and RJG
 * Date: 2022-03-02
 */
/***** TODO: fix this up. Disabled presently
void applyPreconditioning()
{
    final switch (nkCfg.preconditioner) {
    case PreconditionerType.jacobi:
        foreach (blk; parallel(localFluidBlocks,1)) {
            blk.flowJacobian.x[] = blk.v[];
            nm.smla.multiply(blk.flowJacobian.local, blk.flowJacobian.x, blk.zed);
        }
        break;
    case PreconditionerType.ilu:
        foreach (blk; parallel(localFluidBlocks,1)) {
            blk.zed[] = blk.v[];
            nm.smla.solve(blk.flowJacobian.local, blk.zed);
        }
        break;
    case PreconditionerType.sgs:
        foreach (blk; parallel(localFluidBlocks,1)) {
            blk.zed[] = blk.v[];
            nm.smla.sgs(blk.flowJacobian.local, blk.flowJacobian.diagonal, blk.zed, to!int(nConserved), blk.flowJacobian.D, blk.flowJacobian.Dinv);
        }
        break;
    case PreconditionerType.sgs_relax:
        foreach (blk; parallel(localFluidBlocks,1)) {
            int local_kmax = nkCfg.sgsRelaxationIterations;
            blk.zed[] = blk.v[];
            nm.smla.sgsr(blk.flowJacobian.local, blk.zed, blk.flowJacobian.x, to!int(nConserved), local_kmax, blk.flowJacobian.Dinv);
        }
        break;
    case PreconditionerType.lu_sgs:
        mixin(lusgs_solve("zed", "v"));
        break;
    } // end switch
}
******/

/**
 * Remove preconditioning on values and place in dU.
 *
 * Authors: KAD and RJG
 * Date: 2022-03-03
 */
/***** FIX ME -- 2022-07-02
void removePreconditioning()
{
    int nConserved = to!int(GlobalConfig.cqi.n);
    
    final switch (GlobalConfig.sssOptions.preconditionMatrixType) {
    case PreconditionMatrixType.jacobi:
        foreach(blk; parallel(localFluidBlocks,1)) {
            nm.smla.multiply(blk.flowJacobian.local, blk.zed, blk.dU);
        }
        break;
    case PreconditionMatrixType.ilu:
        foreach(blk; parallel(localFluidBlocks,1)) {
            blk.dU[] = blk.zed[];
            nm.smla.solve(blk.flowJacobian.local, blk.dU);
        }
        break;
    case PreconditionMatrixType.sgs:
        foreach(blk; parallel(localFluidBlocks,1)) {
            blk.dU[] = blk.zed[];
            nm.smla.sgs(blk.flowJacobian.local, blk.flowJacobian.diagonal, blk.dU, nConserved, blk.flowJacobian.Dinv, blk.flowJacobian.Dinv);
        }
        break;
    case PreconditionMatrixType.sgs_relax:
        int local_kmax = GlobalConfig.sssOptions.maxSubIterations;
        foreach(blk; parallel(localFluidBlocks,1)) {
            blk.dU[] = blk.zed[];
            nm.smla.sgsr(blk.flowJacobian.local, blk.dU, blk.flowJacobian.x, nConserved, local_kmax, blk.flowJacobian.Dinv);
        }
        break;
    case PreconditionMatrixType.lu_sgs:
        mixin(lusgs_solve("dU", "zed"));
        break;
    } // end switch
}
******/

/**
 * Compute perturbation size estimate for real-valued Frechet derivative.
 *
 * REFERENCE: [ASK KYLE]
 *
 * Authors: KAD and RJG
 * Date: 2022-03-02
 */
number computePerturbationSize()
{
    // calculate sigma without scaling
    number sumv = 0.0;
    mixin(dotOverBlocks("sumv", "zed", "zed"));
    version(mpi_parallel) {
        MPI_Allreduce(MPI_IN_PLACE, &(sumv.re), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }

    size_t nConserved = GlobalConfig.cqi.n;
    auto eps0 = 1.0e-6; // FIX ME: Check with Kyle about what to call this
                        //         and how to set the value.
    number N = 0.0;
    number sume = 0.0;
    foreach (blk; parallel(localFluidBlocks,1)) {
        int cellCount = 0;
        foreach (cell; blk.cells) {
            foreach (val; cell.U[0].vec) {
                sume += eps0*abs(val) + eps0;
                N += 1;
            }
            cellCount += nConserved;
        }
    }
    version(mpi_parallel) {
        MPI_Allreduce(MPI_IN_PLACE, &(sume.re), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }
    version(mpi_parallel) {
        MPI_Allreduce(MPI_IN_PLACE, &(N.re), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }

    number sigma = (sume/(N*sqrt(sumv))).re;
    return sigma;
}

/**
 * Evaluate J*v via a Frechet derivative.
 *
 * This is just a delegator function through to a complex-valued perturbation evaulation
 * or a real-valued perturbation evalulation.
 * Authors: KAD and RJG
 * Date: 2022-03-02
 */
void evalJacobianVectorProduct(number sigma)
{
    version (complex_number) {
        evalComplexMatVecProd(sigma);
    }
    else {
        evalRealMatVecProd(sigma.re);
    }
}



void evalResidual(int ftl)
{
    fnCount++;
    int gtl = 0;
    double dummySimTime = -1.0;

    foreach (blk; parallel(localFluidBlocks,1)) {
        blk.clear_fluxes_of_conserved_quantities();
        foreach (cell; blk.cells) cell.clear_source_vector();
    }
    exchange_ghost_cell_boundary_data(dummySimTime, 0, ftl);
    foreach (blk; localFluidBlocks) {
        blk.applyPreReconAction(dummySimTime, 0, ftl);
    }

    // We don't want to switch between flux calculator application while
    // doing the Frechet derivative, so we'll only search for shock points
    // at ftl = 0, which is when the F(U) evaluation is made.
    if (ftl == 0 && GlobalConfig.do_shock_detect) { detect_shocks(0, ftl); }

    // We need to apply the copy_cell_data BIE at this point to allow propagation of
    // "shocked" cell information (fs.S) to the boundary interface BEFORE the convective
    // fluxes are evaluated. This is important for a real-valued Frechet derivative
    // with adaptive fluxes, to ensure that each interface along the boundary uses a consistent flux
    // calculator for both the baseline residual R(U) and perturbed residual R(U+dU) evaluations.
    // [TODO] KD 2021-11-30 This is a temporary fix until a more formal solution has been decided upon.
    foreach (blk; parallel(localFluidBlocks,1)) {
        foreach(boundary; blk.bc) {
            if (boundary.preSpatialDerivActionAtBndryFaces[0].desc == "CopyCellData") {
                boundary.preSpatialDerivActionAtBndryFaces[0].apply(dummySimTime, gtl, ftl);
            }
        }
    }

    bool allow_high_order_interpolation = true;
    foreach (blk; parallel(localFluidBlocks,1)) {
        blk.convective_flux_phase0(allow_high_order_interpolation, gtl);
    }

    // for unstructured blocks we need to transfer the convective gradients before the flux calc
    if (allow_high_order_interpolation && (GlobalConfig.interpolation_order > 1)) {
        exchange_ghost_cell_boundary_convective_gradient_data(dummySimTime, gtl, ftl);
    }

    foreach (blk; parallel(localFluidBlocks,1)) {
        blk.convective_flux_phase1(allow_high_order_interpolation, gtl);
    }
    foreach (blk; localFluidBlocks) {
        blk.applyPostConvFluxAction(dummySimTime, gtl, ftl);
    }

    if (GlobalConfig.viscous) {
        foreach (blk; localFluidBlocks) {
            blk.applyPreSpatialDerivActionAtBndryFaces(dummySimTime, gtl, ftl);
            blk.applyPreSpatialDerivActionAtBndryCells(dummySimTime, gtl, ftl);
        }
        foreach (blk; parallel(localFluidBlocks,1)) {
            blk.flow_property_spatial_derivatives(0);
        }
        // for unstructured blocks employing the cell-centered spatial (/viscous) gradient method,
        // we need to transfer the viscous gradients before the flux calc
        exchange_ghost_cell_boundary_viscous_gradient_data(dummySimTime, gtl, ftl);
        foreach (blk; parallel(localFluidBlocks,1)) {
            // we need to average cell-centered spatial (/viscous) gradients to get approximations of the gradients
            // at the cell interfaces before the viscous flux calculation.
            if (blk.myConfig.spatial_deriv_locn == SpatialDerivLocn.cells) {
                foreach(f; blk.faces) {
                    f.average_cell_deriv_values(0);
                }
            }
            blk.estimate_turbulence_viscosity();
        }
        // we exchange boundary data at this point to ensure the
        // ghost cells along block-block boundaries have the most
        // recent mu_t and k_t values.
        exchange_ghost_cell_turbulent_viscosity();
        foreach (blk; parallel(localFluidBlocks,1)) {
            blk.viscous_flux();
        }
        foreach (blk; localFluidBlocks) {
            blk.applyPostDiffFluxAction(dummySimTime, 0, ftl);
        }
    }

    foreach (blk; parallel(localFluidBlocks,1)) {
        // the limit_factor is used to slowly increase the magnitude of the
        // thermochemical source terms from 0 to 1 for problematic reacting flows
        double limit_factor = 1.0;
        if (blk.myConfig.nsteps_of_chemistry_ramp > 0) {
            double S = SimState.step/to!double(blk.myConfig.nsteps_of_chemistry_ramp);
            limit_factor = min(1.0, S);
        }
        foreach (i, cell; blk.cells) {
            cell.add_inviscid_source_vector(0, 0.0);
            if (blk.myConfig.viscous) {
                cell.add_viscous_source_vector();
            }
            if (blk.myConfig.reacting) {
                cell.add_thermochemical_source_vector(blk.thermochem_source, limit_factor);
            }
            if (blk.myConfig.udf_source_terms) {
                size_t i_cell = cell.id;
                size_t j_cell = 0;
                size_t k_cell = 0;
                if (blk.grid_type == Grid_t.structured_grid) {
                    auto sblk = cast(SFluidBlock) blk;
                    assert(sblk !is null, "Oops, this should be an SFluidBlock object.");
                    auto ijk_indices = sblk.to_ijk_indices_for_cell(cell.id);
                    i_cell = ijk_indices[0];
                    j_cell = ijk_indices[1];
                    k_cell = ijk_indices[2];
                }
                getUDFSourceTermsForCell(blk.myL, cell, 0, dummySimTime, blk.myConfig, blk.id, i_cell, j_cell, k_cell);
                cell.add_udf_source_vector();
            }
            cell.time_derivatives(0, ftl);
        }
    }
}


/**
 * Evaluate J*v via a Frechet derivative with perturbation in imaginary plane.
 *
 * J*v = Im( R(U + sigma*j)/sigma )
 *
 * Authors: KAD and RJG
 * Date: 2022-03-03
 */
void evalComplexMatVecProd(double sigma)
{
    version(complex_numbers) {
        alias cfg = GlobalConfig;
        
        foreach (blk; parallel(localFluidBlocks,1)) { blk.set_interpolation_order(activePhase.jacobianInterpolationOrder); }
        if (activePhase.frozenLimiterForJacobian) {
            foreach (blk; parallel(localFluidBlocks,1)) { GlobalConfig.frozen_limiter = true; }
        }
        // Make a stack-local copy of conserved quantities info
        size_t nConserved = GlobalConfig.cqi.n;

        // We perform a Frechet derivative to evaluate J*D^(-1)v
        foreach (blk; parallel(localFluidBlocks,1)) {
            blk.clear_fluxes_of_conserved_quantities();
            foreach (cell; blk.cells) cell.clear_source_vector();
            size_t startIdx = 0;
            foreach (cell; blk.cells) {
                cell.U[1].copy_values_from(cell.U[0]);
                foreach (ivar; 0 .. nConserved) {
                    cell.U[1].vec[ivar] += complex(0.0, sigma * blk.zed[startIdx+ivar].re);
                }
                cell.decode_conserved(0, 1, 0.0);
                startIdx += nConserved;
            }
        }
        evalResidual(1);
        foreach (blk; parallel(localFluidBlocks,1)) {
            size_t startIdx = 0;
            foreach (cell; blk.cells) {
                foreach (ivar; 0 .. nConserved) {
                    blk.zed[startIdx+ivar] = cell.dUdt[1].vec[ivar].im/(sigma);
                }
                startIdx += nConserved;
            }
            // we must explicitly remove the imaginary components from the cell and interface flowstates
            foreach(cell; blk.cells) { cell.fs.clear_imaginary_components(); }
            foreach(bc; blk.bc) {
                foreach(ghostcell; bc.ghostcells) { ghostcell.fs.clear_imaginary_components(); }
            }
            foreach(face; blk.faces) { face.fs.clear_imaginary_components(); }
        }
        foreach (blk; parallel(localFluidBlocks,1)) { blk.set_interpolation_order(activePhase.residualInterpolationOrder); }
    } else {
        throw new Error("Oops. Steady-State Solver setting: useComplexMatVecEval is not compatible with real-number version of the code.");
    }
}

/**
 * Evaluate J*v via a Frechet derivative with perturbation in imaginary plane.
 *
 * J*v = (R(U + sigma) - R(U)) / sigma
 *
 * Authors: KAD and RJG
 * Date: 2022-03-03
 */
void evalRealMatVecProd(double sigma)
{
    foreach (blk; parallel(localFluidBlocks,1)) { blk.set_interpolation_order(activePhase.jacobianInterpolationOrder); }
    if (activePhase.frozenLimiterForJacobian) {
        foreach (blk; parallel(localFluidBlocks,1)) { GlobalConfig.frozen_limiter = true; }
    }
    // Make a stack-local copy of conserved quantities info
    size_t nConserved = GlobalConfig.cqi.n;

    // We perform a Frechet derivative to evaluate J*D^(-1)v
    foreach (blk; parallel(localFluidBlocks,1)) {
        blk.clear_fluxes_of_conserved_quantities();
        foreach (cell; blk.cells) cell.clear_source_vector();
        size_t startIdx;
        foreach (cell; blk.cells) {
            cell.U[1].copy_values_from(cell.U[0]);
            foreach (ivar; 0 .. nConserved) {
                cell.U[1].vec[ivar] += sigma*blk.zed[startIdx+ivar];
            }
            cell.decode_conserved(0, 1, 0.0);
            startIdx += nConserved;
        }
    }
    evalResidual(1);
    foreach (blk; parallel(localFluidBlocks,1)) {
        size_t startIdx = 0;
        foreach (cell; blk.cells) {
            foreach (ivar; 0 .. nConserved) {
                blk.zed[startIdx+ivar] = (cell.dUdt[1].vec[ivar] - blk.R[startIdx+ivar])/(sigma);
            }
            cell.decode_conserved(0, 0, 0.0);
            startIdx += nConserved;
        }
    }
    foreach (blk; parallel(localFluidBlocks,1)) { blk.set_interpolation_order(activePhase.residualInterpolationOrder); }
}

/**
 * Determine a relaxation factor.
 *
 * In this algorithm, the relaxation factor keeps falling as we search across cells in order.
 * This is efficient since we're searching for a worst case. If at any point, the relaxation
 * factor gets smaller than what we're prepared to accept then we just break the search
 * in that block of cells.
 *
 * Authors: KAD and RJG
 * Date: 2022-03-05
 */
/* FIX ME -- 2022-07-02
      Need to do this as Kyle has suggested. 
double determineRelaxationFactor()
{
    alias cfg = GlobalConfig;

    double deltaAllowable = nkCfg.allowableRelativeMassChange;
    double omegaMinAllowable = nkCfg.minimumAllowableRelaxationFactor;
    double omegaDecrement = nkCfg.decrementOnRelaxationFactor;
    
    size_t nConserved = cfg.cqi.n;
    size_t massIdx = cfg.cqi.mass;
    
    double omega = 1.0;
    foreach (blk; parallel(localFluidBlocks,1)) {
        int startIdx = 0;
        blk.omega_local = 1.0;
        foreach (cell; blk.cells) {
            //---
            // 1. check and set omega based on mass change.
            //---
            //
            /// relative mass change projected by update
            deltaProjected = fabs(blk.dU[startIdx+massIdx]/cell.U[0].vec[massIdx]);
            /// mass change ratio
            massChangeRatio = deltaProjected/deltaAllowable;
            /// inverse ratio (so we can compare to relaxation factor)
            invRatio = 1./massChangeRatio;
            /// relaxation factor is smaller of invRatio (based on mass change) or whatever our smallest so far is
            blk.omegaLocal = fmin(invRatio, blk.omegaLocal);
            if (blk.omegaLocal < omegaMinAllowable) {
                // No point continuing.
                break;
            }
            //---
            // 2. check that flow state primitives remain valid or adjust relaxation factor.
            //---
            //
            bool failedDecode;
            do {
                // Set to false until we find out otherwise.
                failedDecode = false;
                // Attempt to change cell and decode.
                cell.U[1].copy_values_from(cell.U[0]);
                foreach (ivar; 0 .. nConserved) {
                    cell.U[1].vec[ivar] = cell.U[0].vec[ivar] + blk.omegaLocal*blk.dU[startIdx+ivar];
                }
                try {
                    cell.decode_conserved(0, 1, 0.0);
                }
                catch (FlowSolverException e) {
                    failedDecode = true;
                }
                if (failedDecode) blk.omega_local -= omegaDecrement;
                if (blk.omegaLocal < omegaMinAllowable) {
                    // No point continuing
                    break;
                }
                // return cell to original state
                cell.decode_conserved(0, 0, 0.0);
            }
            while (failedDecode);

            startIdx += nConserved;
        }
    }
    // In serial, find minimum omega across all blocks.
    foreach (blk; localFluidBlocks) omega = fmin(omega, blk.omegaLocal);
    version (mpi_parallel) {
        // In parallel, find minimum and communicate to all processes
        MPI_Allreduce(MPI_IN_PLACE, &(omega.re), 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    }
    return omega;
}
****/

/**
 * Apply Newton update, scaled by relaxation factor, omega.
 *
 * Authors: KAD and RJG
 * Date: 2022-03-05
 */
void applyNewtonUpdate(double relaxationFactor)
{
    size_t nConserved = GlobalConfig.cqi.n;
    double omega = relaxationFactor;
    
    foreach (blk; parallel(localFluidBlocks, 1)) {
        size_t startIdx = 0;
        foreach (cell; blk.cells) {
            foreach (ivar; 0 .. nConserved) {
                cell.U[0].vec[ivar] += omega * blk.dU[startIdx + ivar];
            }
            try {
                cell.decode_conserved(0, 0, 0.0);
            }
            catch (FlowSolverException e) {
                // If the physicality check + relaxation factor is in play,
                // then we should never fail here since we've pre-tested
                // the update before applying.
                // However, we may not be using that check.
                // In which case, it's probably best to bail out here.
                string errMsg;
                errMsg ~= "Failed update for cell when applying Newton step.\n";
                errMsg ~= format("blk-id: %d", blk.id);
                errMsg ~= format("cell-id: %d", cell.id);
                errMsg ~= format("cell-pos: %s", cell.pos[0]);
                errMsg ~= "Error message from decode_conserved:\n";
                errMsg ~= e.msg;
                throw new NewtonKrylovException(errMsg);
            }
            startIdx += nConserved;
        }
    }
}

/****
void writeDiagnostics(int step, double cfl, double dt, ref bool residualsUpToDate)
{
    double massBalance = computeMassBalance();
    if (!residualsUpToDate) {
        computeResiduals(currentResiduals);
        residualsUpToDate = true;
    }
    // [TODO] finish.
}
****/

/****
void printStatusToScreen(int step, double cfl, double dt, double wallClockElapsed, ref bool residualsUpToDate)
{
    if (!residualsUpToDate) {
        computeResiduals(currentResiduals);
        residualsUpToDate = true;
    }
    if (GlobalConfig.is_master_task) {
        auto cqi = GlobalConfig.cqi;
        auto writer = appender!string();
        string hrule = "--------------------------------------------------------------\n";
        formattedWrite(writer, hrule);
        formattedWrite(writer, "step= %6d\tcfl=%10.3e\tdt=%10.3e\tWC=%.1f\n",
                       step, cfl, dt, wallClockElapsed);
        formattedWrite(writer, hrule);
        formattedWrite(writer, "residuals\t\trelative\t\tabsolute\n");
        formattedWrite(writer, "global\t\t%106e\t\t%10.6e", globalResidual/referenceGlobalResidual, globalResidual);
        // species density residuals
        foreach (isp; 0 .. GlobalConfig.gmodel_master.n_species) {
            
        }
        // [TODO] finish this once we decide on which equation set we solve.
    }        
    
}
****/
