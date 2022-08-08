/**
 * A module to provide a command line interface for Eilmer.
 *
 * Authors: RJG, PJ, KAD, NNG
 * Date: 2022-08-08
 */

import std.getopt;
import std.stdio;

import command;
import flow2vtk;
import prep_grids;

Command[string] commands;
Command helpCmd;

static this()
{
    // Initialise helpCmd in this module so that it has access to
    // all commands.
    helpCmd.main = &printHelp;
    helpCmd.description = "Display help information for a topic/command or a general overview.";
    helpCmd.shortDescription = "Display help about using Eilmer.";

    // Add commands here.
    commands["help"] = helpCmd;
    commands["flow2vtk"] = flow2vtkCmd;
    commands["prep-grids"] = prepGridCmd; commands["prep-grid"] = commands["prep-grids"]; // add alias
}

void main(string[] args)
{
    bool helpWanted = false;
    getopt(args,
           "help|h", &helpWanted,
           std.getopt.config.stopOnFirstNonOption,
    );

    auto cmd = args[1];

    if (cmd in commands) {
        return (*commands[cmd].main)(args);
    }
    // If we've made it here, then we've chosen a bad command.
    writefln("lmr: '%s' is not an lmr command. See 'lmr help'.", cmd);
    return;
}

void printHelp(string[] args)
{

    if (args.length >= 3) {
        auto cmd = args[2];
        if (cmd in commands) {
            writeln(commands[cmd].helpMsg);
            return;
        }
        // If we've made it here, then we've chosen a bad command.
        writefln("lmr: '%s' is not an lmr command. See 'lmr help'.", cmd);
    }
    // else just print general help.
    writeln("Some general help goes here.");
    return;
}


