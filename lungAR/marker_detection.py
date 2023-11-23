import numpy as np
import cv2
import cv2.aruco as aruco
from OpenGL.GL import *
from OpenGL.GLUT import *
import pywavefront

# Constants for camera calibration
MATRIX_COEFFICIENTS = np.array([[800.0, 0, 320.0], [0, 800.0, 240.0], [0, 0, 1.0]], dtype=np.float64)
DISTORTION_COEFFICIENTS = np.array([0, 0, 0, 0, 0], dtype=np.float64)
lung_model = pywavefront.Wavefront('data/3dModel/lung.obj', collect_faces=True)


def initialize_camera():
    """Initialize and return the webcam object."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    return cap

def detect_markers(frame, dictionary, parameters):
    """Detect ArUco markers in the given frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return aruco.detectMarkers(gray, dictionary, parameters=parameters)

def estimate_pose(corners, marker_size, camera_matrix, dist_coeffs):
    """Estimate the pose of the ArUco marker."""
    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)
    return rvecs, tvecs

""" Creating a FrameBuffer object / OPENGL SET UP """
def render_model(fbo, rvec, tvec, width, height):
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    glViewport(0, 0, width, height)
    # ... [Set up and clear color and depth buffers] ...
    # ... [Apply camera and model transformations based on rvec, tvec] ...
    # Render the lung model
    glBegin(GL_TRIANGLES)
    for mesh in lung_model.mesh_list:
        for face in mesh.faces:
            for vertex_i in face:
                glVertex3f(*lung_model.vertices[vertex_i])
    glEnd()
    glBindFramebuffer(GL_FRAMEBUFFER, 0)

def init_gl(width, height):
    """Initialize OpenGL for offscreen rendering."""
    # Create a framebuffer (FBO)
    fbo = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)

    # Create a texture and attach it to the FBO
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0)

    # Create a renderbuffer for depth and attach to the FBO
    rbo = glGenRenderbuffers(1)
    glBindRenderbuffer(GL_RENDERBUFFER, rbo)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height)
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo)

    # Check if FBO is complete
    if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
        print("Error: Framebuffer is not complete.")
        return None

    glBindFramebuffer(GL_FRAMEBUFFER, 0)

    # Return the FBO, texture, and RBO identifiers
    return fbo, texture, rbo

def get_image_from_texture(texture, width, height):
    """Read the texture into a NumPy array."""
    glBindTexture(GL_TEXTURE_2D, texture)
    image = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE)
    image = np.frombuffer(image, dtype=np.uint8).reshape(height, width, 3)
    image = cv2.flip(image, 0)  # Flip vertically
    glBindTexture(GL_TEXTURE_2D, 0)
    return image


def main():
    cap = initialize_camera()
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_5X5_250)
    parameters = aruco.DetectorParameters()
    marker_size = 0.05  # Marker size in meters

    # Initialize OpenGL
    glutInit()
    fbo, texture, rbo = init_gl(640, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        corners, ids, _ = detect_markers(frame, dictionary, parameters)
        if ids is not None:
            rvecs, tvecs = estimate_pose(corners, marker_size, MATRIX_COEFFICIENTS, DISTORTION_COEFFICIENTS)
            for rvec, tvec in zip(rvecs, tvecs):
                # Render the 3D model to the off-screen buffer
                render_model(fbo, rvec, tvec, 640, 480)

                # Capture the rendered image
                rendered_image = get_image_from_texture(texture, 640, 480)

                # Overlay this rendered image onto the OpenCV frame
                # ... [Overlay logic: blending, replacing, etc.] ...

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()