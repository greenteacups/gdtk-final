module command;

struct Command
{
    /// pointer to function for command action
    void function(string[]) main;

    string description;
    string shortDescription;
    string helpMsg;
    
}

