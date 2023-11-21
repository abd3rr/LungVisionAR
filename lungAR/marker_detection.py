import numpy as np
import cv2
import cv2.aruco as aruco

# Constants for camera calibration
MATRIX_COEFFICIENTS = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
DISTORTION_COEFFICIENTS = np.array([0, 0, 0, 0, 0])

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

def draw_axes(frame, corners, rvecs, tvecs, camera_matrix, dist_coeffs):
    """Draw axes on a frame for a detected marker."""
    for i in range(len(rvecs)):
        corner = corners[i][0]
        point1 = tuple(corner[0].astype(int))
        point2 = tuple(corner[1].astype(int))
        point3 = tuple(corner[2].astype(int))
        point4 = tuple(corner[3].astype(int))
        cv2.line(frame, point1, point2, (0, 255, 0), 2)
        cv2.line(frame, point2, point3, (0, 0, 255), 2)
        cv2.line(frame, point3, point4, (255, 0, 0), 2)

def main():
    """Main function to handle the marker tracking and pose estimation."""
    try:
        cap = initialize_camera()
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_5X5_250)
        parameters = aruco.DetectorParameters_create() if hasattr(aruco, 'DetectorParameters_create') else aruco.DetectorParameters()
        marker_size = 0.05  # size of the marker in meters

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            corners, ids, _ = detect_markers(frame, dictionary, parameters)
            if ids is not None:
                aruco.drawDetectedMarkers(frame, corners)

                for i in range(len(ids)):
                    rvecs, tvecs = estimate_pose(corners[i], marker_size, MATRIX_COEFFICIENTS, DISTORTION_COEFFICIENTS)
                    draw_axes(frame, corners[i], rvecs, tvecs, MATRIX_COEFFICIENTS, DISTORTION_COEFFICIENTS)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
