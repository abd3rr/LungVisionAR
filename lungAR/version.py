from OpenGL.GL import glGetString, GL_VERSION
from OpenGL.GLUT import glutInit

def get_opengl_version():
    # Initialize GLUT - this is necessary for OpenGL context
    glutInit()
    
    # Get the version string and decode to a regular Python string (from bytes)
    version = glGetString(GL_VERSION)
    if version:
        return version.decode('utf-8')
    else:
        return "OpenGL version not found"

if __name__ == "__main__":
    version = get_opengl_version()
    print(f"OpenGL Version: {version}")
