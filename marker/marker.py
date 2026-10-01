import cv2
import numpy as np

def generate_marker(marker_id=25,
                    dictionary_name=cv2.aruco.DICT_4X4_50,
                    marker_size=400,
                    border_size=100,
                    output_file="marker_25.png"):

    # Load ArUco dictionary
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_name)

    # Generate the marker
    marker = np.zeros((marker_size, marker_size), dtype=np.uint8)
    cv2.aruco.drawMarker(dictionary, marker_id, marker_size, marker, 1)

    # Create white background (bigger image)
    total_size = marker_size + 2 * border_size
    canvas = np.ones((total_size, total_size), dtype=np.uint8) * 255

    # Place marker in the center
    canvas[border_size:border_size+marker_size,
           border_size:border_size+marker_size] = marker

    # Save the marker
    cv2.imwrite(output_file, canvas)

    print(f"Marker saved as {output_file}")

    # Display marker
    cv2.imshow("Generated Marker", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Run once
generate_marker()