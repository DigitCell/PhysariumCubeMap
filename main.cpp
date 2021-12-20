#include <iostream>
#include "stdio.h"

#include "common/entry/entry_p.h"
#include "common/entry/entry.h"
#include "common/entry/entry_sdl.h"
#include "common/entry/entry_sdl.cpp"

#include "mainloop.hpp"


using namespace std;
//using namespace entry;

int main(int _argc, char** _argv)
{
    printf("Start Caterpiller project");
    MainLoop mainLoop(_argc, _argv, entry::s_ctx);
    mainLoop.run(entry::s_ctx);

    return 0;
}
