#include "namedPipe.hpp"

// write
int main()
{
    NamePiped fifo(comm_path, User);
    if (fifo.OpenForWrite())
    {
        std::cout << "client open namd pipe done" << std::endl;
        while (true)
        {
            std::cout << "Please Enter> ";
            std::string message;
            std::getline(std::cin, message);
            fifo.WriteNamedPipe(message);
        }
    }

    return 0;
}