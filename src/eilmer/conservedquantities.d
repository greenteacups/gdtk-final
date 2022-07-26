/**
 * conservedquantities.d
 * Class for the vector of conserved quantities, for use in the CFD codes.
 *
 * Author: Peter J., Rowan G. and Kyle Damm
 * Version:
 * 2014-07-17: initial cut, to explore options.
 * 2021-05-10: Change to array storage.
 */

module conservedquantities;

import std.string;
import std.format;
import std.conv;
import nm.complex;
import nm.number;
import geom;
import gas;


// Underlying definition of the conserved quantities collection,
// as seen by the transient solver.

class ConservedQuantities {
public:
    number[] vec;

    this(size_t n)
    {
        vec.length = n;
    }

    this(ref const(ConservedQuantities) other)
    {
        vec.length = other.vec.length;
        foreach (i, ref e; vec) { e = other.vec[i]; }
    }

    @nogc void copy_values_from(ref const(ConservedQuantities) src)
    {
        foreach (i, ref e; vec) { e = src.vec[i]; }
    }

    @nogc void clear()
    {
        foreach (ref e; vec) { e = 0.0; }
    }

    @nogc void add(ref const(ConservedQuantities) other, double factor=1.0)
    {
        foreach (i, ref e; vec) e += other.vec[i] * factor;
    }

    @nogc void scale(double factor)
    {
        foreach (ref e; vec) { e *= factor; }
    }

    override string toString() const
    {
        char[] repr;
        repr ~= "ConservedQuantities(vec=" ~ to!string(vec) ~ ")";
        return to!string(repr);
    }

    version(complex_numbers) {
        // When performing the complex-step Frechet derivative in the Newton-Krylov accelerator,
        // the conserved quantities accumulate imaginary components,
        // so we have to start with a clean slate, so to speak.
        @nogc void clear_imaginary_components()
        {
            foreach (ref e; vec) { e.im = 0.0; }
        }
    } // end version(complex_numbers)


    /+ Retain the old named items code, just for reference as we make the changes.
    number mass;           // density, kg/m**3
    Vector3 momentum;      // momentum/unit volume
    number total_energy;   // total energy
    version(multi_species_gas) {
        number[] massf;    // densities of each chemical species
    }
    version(multi_T_gas) {
        number[] energies; // modal energies
    }
    version(MHD) {
        Vector3 B;         // magnetic field, Tesla
        number psi;        // divergence cleaning parameter for MHD
        number divB;       // divergence of the magnetic field
    }
    version(turbulence) {
        number[2] rhoturb;    // turbulent conserved
    }
    +/
} // end class ConservedQuantities


// Now that the ConservedQuantities object is a simple vector of quantities,
// this collection of indices helps us select individual elements.

class ConservedQuantitiesIndices {
public:
    bool threeD;
    bool turb;
    bool MHD;
    size_t n;
    size_t n_species;
    size_t n_modes;
    size_t n_turb;
    size_t mass;
    size_t xMom;
    size_t yMom;
    size_t zMom;
    size_t totEnergy;
    size_t rhoturb;
    size_t xB;
    size_t yB;
    size_t zB;
    size_t psi;
    size_t divB;
    size_t species;
    size_t modes;

    this(int dimensions, size_t nturb, bool MHD, size_t nspecies, size_t nmodes) {

        bool put_mass_in_last_position = false;
        version (nk_accelerator) {
            // we will drop the mass continuity equation if we are running a multi-species calculation with
            // the steady-state solver, note that we still need an entry in the conserved quantities vector
            // for the mass (some parts of the code expect it), so we will place it in the last position
            if (nspecies > 1) { put_mass_in_last_position = true; }
        }

        if (put_mass_in_last_position) {
            xMom = 0; names ~= "x-mom";
            yMom = 1; names ~= "y-mom";
            if (dimensions == 3) {
                threeD = true;
                zMom = 2; names ~= "z-mom";
                totEnergy = 3; names ~= "tot-energy";
                n = 4;
            } else {
                // Do not carry z-momentum for 2D simulations.
                threeD = false;
                totEnergy = 2; names ~= "tot-energy";
                n = 3;
            }
            n_turb = nturb;
            if (nturb > 0) {
                turb = true;
                rhoturb = n; // Start of turbulence elements.
                n += nturb;
                foreach (i; 0 .. nturb) names ~= "turb-quant-" ~ to!string(i);
            } else {
                turb = false;
            }
            this.MHD = MHD;
            if (MHD) {
                xB = n; names ~= "x-B";
                yB = n+1; names ~= "y-B";
                zB = n+2; names ~= "z-B";
                psi = n+3; names ~= "psi";
                divB = n+4; names ~= "div-B";
                n += 5;
            }
            n_species = nspecies;
            species = n; // Start of species elements.
            n += nspecies;
            foreach (i; 0 .. nspecies) names ~= "species-" ~ to!string(i);
            n_modes = nmodes;
            if (nmodes > 0) {
                modes = n; // Start of modes elements.
                n += nmodes;
                foreach (i; 0 .. nmodes) names ~= "mode-" ~ to!string(i);
            }
            // we still need the mass in the conserved quantities vector in some places of the code
            mass = n; names ~= "mass";
            n += 1;
        } else { // fill out the array using our standard ordering
            mass = 0; names ~= "mass";
            xMom = 1; names ~= "x-mom";
            yMom = 2; names ~= "y-mom";
            if (dimensions == 3) {
                threeD = true;
                zMom = 3; names ~= "z-mom";
                totEnergy = 4; names ~= "tot-energy";
                n = 5;
            } else {
                // Do not carry z-momentum for 2D simulations.
                threeD = false;
                totEnergy = 3; names ~= "tot-energy";
                n = 4;
            }
            n_turb = nturb;
            if (nturb > 0) {
                turb = true;
                rhoturb = n; // Start of turbulence elements.
                n += nturb;
                foreach (i; 0 .. nturb) names ~= "turb-quant-" ~ to!string(i);
            } else {
                turb = false;
            }
            this.MHD = MHD;
            if (MHD) {
                xB = n; names ~= "x-B";
                yB = n+1; names ~= "y-B";
                zB = n+2; names ~= "z-B";
                psi = n+3; names ~= "psi";
                divB = n+4; names ~= "div-B";
                n += 5;
            }
            n_species = nspecies;
            if (nspecies > 1) {
                species = n; // Start of species elements.
                n += nspecies;
                foreach (i; 0 .. nspecies) names ~= "species-" ~ to!string(i);
                // Note that we only carry species in the conserved-quantities vector
                // if we have a multi-species gas model.
                // A single-species gas model assumes a species fraction on 1.0
                // throughout the flow solver code.
            }
            n_modes = nmodes;
            if (nmodes > 0) {
                modes = n; // Start of modes elements.
                n += nmodes;
                foreach (i; 0 .. nmodes) names ~= "mode-" ~ to!string(i);
            }
        }
    } // end constructor

    this(const(ConservedQuantitiesIndices) other)
    {
        threeD = other.threeD;
        turb = other.turb;
        MHD = other.MHD;
        n = other.n;
        n_turb = other.n_turb;
        n_species = other.n_species;
        n_modes = other.n_modes;
        mass = other.mass;
        xMom = other.xMom;
        yMom = other.yMom;
        zMom = other.zMom;
        totEnergy = other.totEnergy;
        rhoturb = other.rhoturb;
        xB = other.xB;
        yB = other.yB;
        zB = other.zB;
        psi = other.psi;
        divB = other.divB;
        species = other.species;
        modes = other.modes;
        names.length = other.names.length;
        names[] = other.names[];
    } // end copy constructor

    override string toString() const
    {
        char[] repr;
        repr ~= "ConservedQuantitiesIndices(";
        repr ~= format("threeD=%s, turb=%s, MHD=%s", threeD, turb, MHD);
        repr ~= format(", n=%d, n_turb=%d, n_species=%d, n_modes=%d", n, n_turb, n_species, n_modes);
        repr ~= format(", mass=%d, xMom=%d, yMom=%d, zMom=%d, totEnergy=%d", mass, xMom, yMom, zMom, totEnergy);
        repr ~= format(", rhoturb=%d, xB=%d, yB=%d, zB=%d, psi=%d, divB=%d", rhoturb, xB, yB, zB, psi, divB);
        repr ~= format(", species=%d, modes=%d", species, modes);
        repr ~= ")";
        return to!string(repr);
    }

    string nameFromIndex(size_t i)
    {
        return names[i];
    }

private:
    string[] names;
} // end ConvservedQuantitiesIndices
