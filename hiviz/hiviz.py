import torch
import numpy as np
import glfw
from OpenGL.GL import *
import cuda.cudart as cudart
import cuda.cuda as cuda

class TorchGLInterop:
    def __init__(self, width, height, channels=4):
        self.width = width
        self.height = height
        self.channels = channels

        # initialize CUDA context
        cudart.cudaFree(0)

        # create OpenGL texture
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        # register texture with CUDA
        self.pbo = glGenBuffers(1)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self.pbo)
        glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * channels, None, GL_DYNAMIC_DRAW)

        # get the GL buffer device pointer
        ret, self.cuda_graphics_resource = cuda.cuGraphicsGLRegisterBuffer(
            int(self.pbo),
            int(cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard)
        )

    def tensor_to_texture(self, tensor):
        assert tensor.is_cuda, 'Input tensor must be on CUDA device'
        assert tensor.shape == (self.height, self.width, self.channels), 'Tensor shape mismatch'

        # map graphics resource
        cuda.cuGraphicsMapResources(1, self.cuda_graphics_resource, 0)

        # copy tensor data to mapped memory
        tensor_ptr = tensor.data_ptr()
        _, dev_ptr, size = cuda.cuGraphicsResourceGetMappedPointer(self.cuda_graphics_resource)
        cuda.cuMemcpy(dev_ptr, tensor_ptr, size)

        # unmap resource
        cuda.cuGraphicsUnmapResources(1, self.cuda_graphics_resource, 0)

        # update texture from PBO
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self.pbo)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.width, self.height, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)

    def cleanup(self):
        cuda.cuGraphicsUnregisterResource(self.cuda_graphics_resource)
        glDeleteBuffers(1, [self.pbo])
        glDeleteTextures([self.texture])

class TextureRenderer:
    def __init__(self):
        # vertex shader for rendering a textured quad
        vertex_shader = """
        #version 330 core
        layout (location = 0) in vec3 position;
        layout (location = 1) in vec2 texCoord;

        out vec2 TexCoord;

        void main() {
            gl_Position = vec4(position, 1.0);
            TexCoord = texCoord;
        }
        """

        # fragment shader for texture sampling
        fragment_shader = """
        #version 330 core
        in vec2 TexCoord;

        uniform sampler2D textureSampler;

        out vec4 FragColor;

        void main() {
            FragColor = texture(textureSampler, TexCoord);
        }
        """

        # compile shaders
        vertex = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(vertex, vertex_shader)
        glCompileShader(vertex)

        fragment = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(fragment, fragment_shader)
        glCompileShader(fragment)

        # create shader program
        self.shader_program = glCreateProgram()
        glAttachShader(self.shader_program, vertex)
        glAttachShader(self.shader_program, fragment)
        glLinkProgram(self.shader_program)

        # clean up shaders
        glDeleteShader(vertex)
        glDeleteShader(fragment)

        # Vertex data for a quad that fills the screen
        vertices = np.array([
             1.0,  1.0, 0.0,  1.0, 1.0,  # top right
            -1.0,  1.0, 0.0,  0.0, 1.0,  # top left
             1.0, -1.0, 0.0,  1.0, 0.0,  # bottom right
            -1.0, -1.0, 0.0,  0.0, 0.0,  # bottom left
            -1.0,  1.0, 0.0,  0.0, 1.0,  # top left
             1.0, -1.0, 0.0,  1.0, 0.0,  # bottom right
        ], dtype=np.float32)

        # create VAO and VBO
        self.vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)

        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        # position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * 4, None)
        glEnableVertexAttribArray(0)

        # texture coord attribute
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(1)

    def render(self, texture_id):
        glUseProgram(self.shader_program)

        # bind texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glUniform1i(glGetUniformLocation(self.shader_program, 'textureSampler'), 0)

        # draw quad
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, 6)

    def cleanup(self):
        glDeleteVertexArrays(1, [self.vao])
        glDeleteProgram(self.shader_program)

class HiViz:
    def __init__(self, width, height):
        # initialize glfw
        if not glfw.init():
            raise Exception('Failed to initialize GLFW')

        # create window
        self.window = glfw.create_window(width, height, 'HiViz', None, None)
        if not self.window:
            glfw.terminate()
            raise Exception('Failed to create window')

        # set opengl context to this window's context
        glfw.make_context_current(self.window)

        # initialize interop
        self.interop = TorchGLInterop(width, height, 4)
        self.renderer = TextureRenderer()

    # cleanup
    def __del__(self):
        self.renderer.cleanup()
        self.interop.cleanup()
        glfw.terminate()

    # main render loop
    def animate(self, generate):
        # set the viewport to match the window size
        fb_width, fb_height = glfw.get_framebuffer_size(self.window)
        glViewport(0, 0, fb_width, fb_height)

        for tensor in generate():
            # handle window close
            if glfw.window_should_close(self.window):
                break

            # update viewport on window resize
            new_fb_width, new_fb_height = glfw.get_framebuffer_size(self.window)
            if (new_fb_width, new_fb_height) != (fb_width, fb_height):
                fb_width, fb_height = new_fb_width, new_fb_height
                glViewport(0, 0, fb_width, fb_height)

            # copy tensor to texture
            self.interop.tensor_to_texture(tensor)

            # render texture
            glClear(GL_COLOR_BUFFER_BIT)
            self.renderer.render(self.interop.texture)

            # swap buffers
            glfw.swap_buffers(self.window)
            glfw.poll_events()

if __name__ == '__main__':
    # animation parameters
    nx, ny = 512, 512
    delta = 0.01

    # generate animation frames
    def generate():
        t = 0
        while True:
            xt = torch.linspace(0, 1, nx, device='cuda')
            yt = torch.linspace(0, 1, ny, device='cuda')
            x, y = torch.meshgrid(xt, yt, indexing='ij')
            v = 127.5 * (1 + torch.sin((x * y + t) * 2 * np.pi))
            ones = torch.ones_like(v, device='cuda')
            test_tensor = torch.stack([v, v, v, ones], dim=-1)
            yield test_tensor.byte()
            t = (t + delta) % 1

    # animate frames
    viz = HiViz(512, 512)
    viz.animate(generate)
