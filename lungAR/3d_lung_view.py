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

def draw_axes(frame, rvecs, tvecs, camera_matrix, dist_coeffs):
    """Draw axes on the frame for each detected marker."""
    axis_length = 0.1  # Length of the axes to be drawn
    axis_points = np.float32([[axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]]).reshape(-1, 3)

    for rvec, tvec in zip(rvecs, tvecs):
        # Ensure rvecs and tvecs are correct shape and type
        rvec = rvec.reshape(-1, 1, 3).astype(np.float64)
        tvec = tvec.reshape(-1, 1, 3).astype(np.float64)

        img_points, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)
        
        # Draw the axes lines
        point_origin = tuple(img_points[0].ravel())
        point_x_axis = tuple(img_points[1].ravel())
        point_y_axis = tuple(img_points[2].ravel())
        point_z_axis = tuple(img_points[2].ravel())

        frame = cv2.line(frame, point_origin, point_x_axis, (255, 0, 0), 3)
        frame = cv2.line(frame, point_origin, point_y_axis, (0, 255, 0), 3)
        frame = cv2.line(frame, point_origin, point_z_axis, (0, 0, 255), 3)

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

            corners, ids, rejectedImgPoints = detect_markers(frame, dictionary, parameters)
            if ids is not None:
                aruco.drawDetectedMarkers(frame, corners)
                rvecs, tvecs = estimate_pose(corners, marker_size, MATRIX_COEFFICIENTS, DISTORTION_COEFFICIENTS)
                draw_axes(frame, rvecs, tvecs, MATRIX_COEFFICIENTS, DISTORTION_COEFFICIENTS)

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
