#pragma once

#include <iostream>
#include "mainloop.hpp"
#include "stdio.h"

using namespace std;

int _main_(int _argc, char** _argv)
{
    cout << "Hello Squids!" << endl;
    MainLoop mainLoop( _argc, _argv);
    mainLoop.run();

}
