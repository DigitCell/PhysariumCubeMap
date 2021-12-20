/*
 * Copyright 2011-2021 Branimir Karadzic. All rights reserved.
 * License: https://github.com/bkaradzic/bgfx#license-bsd-2-clause
 */

#include "entry_p.h"
#include "entry_sdl.h"

#if ENTRY_CONFIG_USE_SDL

#if BX_PLATFORM_LINUX || BX_PLATFORM_BSD
#	if ENTRY_CONFIG_USE_WAYLAND
#		include <wayland-egl.h>
#	endif
#elif BX_PLATFORM_WINDOWS
#	define SDL_MAIN_HANDLED
#endif

#include <bx/os.h>

#include <SDL2/SDL.h>

BX_PRAGMA_DIAGNOSTIC_PUSH()
BX_PRAGMA_DIAGNOSTIC_IGNORED_CLANG("-Wextern-c-compat")
#include <SDL2/SDL_syswm.h>
BX_PRAGMA_DIAGNOSTIC_POP()

#include <bgfx/platform.h>
#if defined(None) // X11 defines this...
#	undef None
#endif // defined(None)

#include <bx/mutex.h>
#include <bx/thread.h>
#include <bx/handlealloc.h>
#include <bx/readerwriter.h>
#include <tinystl/allocator.h>
#include <tinystl/string.h>


namespace entry
{

const Event* poll()
{
    return s_ctx->m_eventQueue.poll();
}

const Event* poll(WindowHandle _handle)
{
    return s_ctx->m_eventQueue.poll(_handle);
}

void release(const Event* _event)
{
    s_ctx->m_eventQueue.release(_event);
}

WindowHandle createWindow(int32_t _x, int32_t _y, uint32_t _width, uint32_t _height, uint32_t _flags, const char* _title)
{
    bx::MutexScope scope(s_ctx->m_lock);
    WindowHandle handle = { s_ctx->m_windowAlloc.alloc() };

    if (UINT16_MAX != handle.idx)
    {
        Msg* msg = new Msg;
        msg->m_x      = _x;
        msg->m_y      = _y;
        msg->m_width  = _width;
        msg->m_height = _height;
        msg->m_title  = _title;
        msg->m_flags  = _flags;

        sdlPostEvent(SDL_USER_WINDOW_CREATE, handle, msg);
    }

    return handle;
}

void destroyWindow(WindowHandle _handle)
{
    if (UINT16_MAX != _handle.idx)
    {
        sdlPostEvent(SDL_USER_WINDOW_DESTROY, _handle);

        bx::MutexScope scope(s_ctx->m_lock);
        s_ctx->m_windowAlloc.free(_handle.idx);
    }
}

void setWindowPos(WindowHandle _handle, int32_t _x, int32_t _y)
{
    Msg* msg = new Msg;
    msg->m_x = _x;
    msg->m_y = _y;

    sdlPostEvent(SDL_USER_WINDOW_SET_POS, _handle, msg);
}

void setWindowSize(WindowHandle _handle, uint32_t _width, uint32_t _height)
{
    Msg* msg = new Msg;
    msg->m_width  = _width;
    msg->m_height = _height;

    sdlPostEvent(SDL_USER_WINDOW_SET_SIZE, _handle, msg);
}

void setWindowTitle(WindowHandle _handle, const char* _title)
{
    Msg* msg = new Msg;
    msg->m_title = _title;

    sdlPostEvent(SDL_USER_WINDOW_SET_TITLE, _handle, msg);
}

void setWindowFlags(WindowHandle _handle, uint32_t _flags, bool _enabled)
{
    Msg* msg = new Msg;
    msg->m_flags = _flags;
    msg->m_flagsEnabled = _enabled;
    sdlPostEvent(SDL_USER_WINDOW_SET_FLAGS, _handle, msg);
}

void toggleFullscreen(WindowHandle _handle)
{
    sdlPostEvent(SDL_USER_WINDOW_TOGGLE_FULL_SCREEN, _handle);
}

void setMouseLock(WindowHandle _handle, bool _lock)
{
    sdlPostEvent(SDL_USER_WINDOW_MOUSE_LOCK, _handle, NULL, _lock);
}

int32_t MainThreadEntry::threadFunc(bx::Thread* _thread, void* _userData)
{
    BX_UNUSED(_thread);

    MainThreadEntry* self = (MainThreadEntry*)_userData;
    int32_t result = mainTH(self->m_argc, self->m_argv);

    SDL_Event event;
    SDL_QuitEvent& qev = event.quit;
    qev.type = SDL_QUIT;
    SDL_PushEvent(&event);
    return result;
}

}
#endif // ENTRY_CONFIG_USE_SDL
