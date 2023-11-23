import numpy as np
import cv2
import cv2.aruco as aruco

# Constants for camera calibration
MATRIX_COEFFICIENTS = np.array([[800.0, 0, 320.0], [0, 800.0, 240.0], [0, 0, 1.0]], dtype=np.float64)
DISTORTION_COEFFICIENTS = np.array([0, 0, 0, 0, 0], dtype=np.float64)

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

def main():
    """Main function to handle the marker tracking and pose estimation."""
    try:
        cap = initialize_camera()
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_5X5_250)
        parameters = aruco.DetectorParameters()  # Updated to the correct method
        marker_size = 0.05  # size of the marker in meters

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            corners, ids, _ = detect_markers(frame, dictionary, parameters)
            if ids is not None:
                aruco.drawDetectedMarkers(frame, corners)
                rvecs, tvecs = estimate_pose(corners, marker_size, MATRIX_COEFFICIENTS, DISTORTION_COEFFICIENTS)
                
                # Use cv2.drawFrameAxes to draw the axes
                for rvec, tvec in zip(rvecs, tvecs):
                    frame = cv2.drawFrameAxes(frame, MATRIX_COEFFICIENTS, DISTORTION_COEFFICIENTS, rvec, tvec, marker_size)

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