import cv2
from util import get_parking_spots_bboxes, empty_or_not
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate the absolute difference in mean pixel values between two images (or regions)
def calc_diff(im1, im2):
    return np.abs(np.mean(im1) - np.mean(im2))

# Path to the mask image used for detecting parking spots
mask = './data/mask_1920_1080.png'
# Path to the video file that contains footage of the parking lot
video_path = './data/parking_1920_1080_loop.mp4'

# Load the mask image in grayscale mode (0 indicates a single channel)
mask = cv2.imread(mask, 0)

# Open the video file using OpenCV's VideoCapture
cap = cv2.VideoCapture(video_path)

# Identify connected components in the mask image (for finding individual regions)
connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)

# Get bounding boxes for detected parking spots using the utility function
spots = get_parking_spots_bboxes(connected_components)

# Initialize the status (empty or occupied) and pixel differences for each parking spot
spots_status = [None for j in spots]
diffs = [None for j in spots]

# Variable to store the previous frame for calculating pixel differences
prev_frame = None
# Frame counter to track the frame number
frame_nb = 0
# Variable to indicate if frames are being read successfully
ret = True
# Process every 30th frame for performance optimization
step = 30

# Main loop to process video frames
while ret:
    # Read a frame from the video
    ret, frame = cap.read()

    # Process every 30th frame if we have a previous frame for comparison
    if frame_nb % step == 0 and prev_frame is not None:
        for spot_idx, spot in enumerate(spots):
            x1, y1, w, h = spot  # Get coordinates of the parking spot bounding box

            # Crop the region corresponding to the current parking spot
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]

            # Calculate the difference between the current spot and the previous frame's spot region
            diffs[spot_idx] = calc_diff(spot_crop, prev_frame[y1:y1 + h, x1:x1 + w, :])
        '''
        # Plot a histogram of the differences to analyze the distribution and determine a cutoff
        print([diffs[j] for j in np.argsort(diffs)][::-1])
        plt.figure()
        plt.hist([diffs[j] / np.amax(diffs) for j in np.argsort(diffs)][::-1], bins=20)
        if frame_nb == 300:
            plt.show()
        '''
    # Check the status of parking spots every 30th frame
    if frame_nb % step == 0:
        # If there's no previous frame, process all spots
        if prev_frame is None:
            arr_ = range(len(spots))
        else:
            # Otherwise, process spots with a pixel difference above the threshold (0.4)
            arr_ = [j for j in np.argsort(diffs) if diffs[j] / np.amax(diffs) > 0.4]
        for spot_idx in arr_:
            spot = spots[spot_idx]
            x1, y1, w, h = spot

            # Crop the region corresponding to the parking spot
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]

            # Check if the parking spot is empty or not using the utility function
            spot_status = empty_or_not(spot_crop)
            spots_status[spot_idx] = spot_status

    # Update the previous frame to the current frame (for pixel difference calculation)
    if frame_nb % step == 0:
        prev_frame = frame.copy()

    # Draw rectangles around parking spots based on their status
    for spot_idx, spot in enumerate(spots):
        spot_status = spots_status[spot_idx]
        x1, y1, w, h = spots[spot_idx]

        if spot_status:  # If the spot is empty, draw a green rectangle
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
        else:  # If the spot is occupied, draw a red rectangle
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)

    # Apply a blur to a specific region of the frame
    frame[20:80, 80:710] = cv2.blur(frame[20:80, 80:710], (50, 50))
    # Display the number of available spots
    cv2.putText(frame, 'Available spots: {} out of {} total'.format(str(sum(spots_status)), str(len(spots_status))),
                (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the frame in a window
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame', frame)
    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    # Increment the frame counter
    frame_nb += 1

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
