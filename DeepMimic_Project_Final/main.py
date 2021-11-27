import glfw
from OpenGL.GL import *
import numpy as np
import ctypes
import imgui
from imgui.integrations.glfw import GlfwRenderer

from program import Program
from camera import Camera
from mesh import Mesh
from texture import Texture
from context import Context

import dartpy as dart

# print(world.getNumSkeletons())
# for i in range(world.getNumSkeletons()):
#     skeleton = world.getSkeleton(i)
#     print(skeleton.getName())
#     ncount = skeleton.getNumBodyNodes()
#     for j in range(ncount):
#         body = skeleton.getBodyNode(j)
#         shapes = body.getShapeNodes()
#         print(shapes)
#         for shape in shapes:
#             v = shape.getVisualAspect()
#             if not v:
#                 continue
#             s = shape.getShape()
#             print("color:", v.getRGB(), "size:", s.getSize())
#         transform = body.getTransform()
#         pose = transform.translation()
#         print(pose)
#         rotation = transform.rotation()
#         print(rotation)

def cursor_pos(window, x, y):
    context = glfw.get_window_user_pointer(window)
    context.cursor_pos(x, y)

def mouse_button(window, button, action, mods):
    # print("button: {}, action: {}, mods: {}".format(button, action, mods))
    pos = glfw.get_cursor_pos(window)
    context = glfw.get_window_user_pointer(window)
    context.mouse_button(button, action, mods, pos[0], pos[1])

def framebuffer_size(window, width, height):
    context = glfw.get_window_user_pointer(window)
    context.framebuffer_size(width, height)

def drop_file(window, filenames):
    print("dropped file:", filenames)
    context = glfw.get_window_user_pointer(window)
    context.env.load_world(filenames[0])
    context.env.set_body_skeleton(1)

def main():
    if not glfw.init():
        print("failed to initialize glfw")
        return

    try:
        width = 800
        height = 600

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)

        window = glfw.create_window(width, height, "DeepMimic Project", None, None)
        glfw.make_context_current(window)

        glfw.set_mouse_button_callback(window, mouse_button)
        glfw.set_cursor_pos_callback(window, cursor_pos)
        glfw.set_framebuffer_size_callback(window, framebuffer_size)
        glfw.set_drop_callback(window, drop_file)

        context = Context.create(width, height)
        glfw.set_window_user_pointer(window, context)

        version = glGetString(GL_VERSION)
        print("OpenGL context version:", version)

        imgui.create_context()
        imgui_glfw = GlfwRenderer(window)

        prev_mouse_pos = (0, 0)
        prev_time = glfw.get_time()
        while not glfw.window_should_close(window):
            glfw.poll_events()
            mouse_pos = glfw.get_cursor_pos(window)
            if mouse_pos[0] != prev_mouse_pos[0] or mouse_pos[1] != prev_mouse_pos[1]:
                cursor_pos(window, mouse_pos[0], mouse_pos[1])
                prev_mouse_pos = mouse_pos

            imgui_glfw.process_inputs()
            imgui.new_frame()

            curr_time = glfw.get_time()
            delta_time = curr_time - prev_time
            context.render_ui()
            context.render_scene(delta_time)
            prev_time = curr_time

            imgui.render()
            imgui_glfw.render(imgui.get_draw_data())

            glfw.swap_buffers(window)
    except Exception as e:
        print(e)

    glfw.terminate()

if __name__ == "__main__":
    main()