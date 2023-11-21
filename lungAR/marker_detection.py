import numpy as np
import cv2
import cv2.aruco as aruco
import pywavefront
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *


# Load the 3D model
lung_model = pywavefront.Wavefront('data/3dModel/lung.obj', collect_faces=True)

# Constants for camera calibration
MATRIX_COEFFICIENTS = np.array([[800.0, 0, 320.0], [0, 800.0, 240.0], [0, 0, 1.0]], dtype=np.float64)
DISTORTION_COEFFICIENTS = np.array([0, 0, 0, 0, 0], dtype=np.float64)

def initialize_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    return cap

def detect_markers(frame, dictionary, parameters):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return aruco.detectMarkers(gray, dictionary, parameters=parameters)

def estimate_pose(corners, marker_size, camera_matrix, dist_coeffs):
    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)
    return rvecs, tvecs

def render_scene(rvec, tvec):
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    
    # Apply the camera transformation
    tvec = tvec.flatten()
    gluLookAt(0, 0, 0, tvec[0], tvec[1], tvec[2], 0, -1, 0)

    # Render the model
    glBegin(GL_TRIANGLES)
    for mesh in lung_model.mesh_list:
        for face in mesh.faces:
            for vertex_i in face:
                glVertex3f(*lung_model.vertices[vertex_i])
    glEnd()

def main():
    cap = initialize_camera()
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_5X5_250)
    parameters = aruco.DetectorParameters()
    marker_size = 0.05  # Size of the marker in meters

    # Initialize GLUT
    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(640, 480)
    glutCreateWindow("3D Lung Visualization")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        corners, ids, _ = detect_markers(frame, dictionary, parameters)
        if ids is not None:
            for i in range(len(ids)):
                rvec, tvec = estimate_pose(corners[i], marker_size, MATRIX_COEFFICIENTS, DISTORTION_COEFFICIENTS)
                render_scene(rvec, tvec)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
