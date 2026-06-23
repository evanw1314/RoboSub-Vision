import cv2
import numpy as np

# CONFIGURATION CONSTANTS
DEFAULT_WEBCAM_IDX = 0
MIN_CONTOUR_AREA = 500
MORPHOLOGY_KERNEL_SIZE = (5, 5)
REFRESH_TIME = 1

# APPEARANCE CONSTANTS
COLOR_WHITE = (255, 255, 255)
CIRCLE_THICKNESS = 3
RED_LABEL_FONT_SCALE = 1
RED_LABEL_THICKNESS = 2

# HSV BOUNDARIES FOR RED
LOWER_RED_1 = np.array([0, 120, 50])
UPPER_RED_1 = np.array([10, 255, 255])
LOWER_RED_2 = np.array([170, 120, 50])
UPPER_RED_2 = np.array([180, 255, 255])


def main():
    cap = cv2.VideoCapture(DEFAULT_WEBCAM_IDX)
    if not cap.isOpened():
        print("Camera Error")
        exit()

    print("Press 'Q' to quit.")

    kernel = np.ones(MORPHOLOGY_KERNEL_SIZE, dtype=np.uint8)

    while True:
        captured, bgr_image = cap.read()
        if not captured:
            print("Error: Failed to grab frame.")
            break

        hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(hsv_image, LOWER_RED_1, UPPER_RED_1)
        mask2 = cv2.inRange(hsv_image, LOWER_RED_2, UPPER_RED_2)
        combined = cv2.bitwise_or(mask1, mask2)

        red_mask = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        output = bgr_image.copy()

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > MIN_CONTOUR_AREA:
                (center_x, center_y), radius = cv2.minEnclosingCircle(contour)

                center = (int(center_x), int(center_y))
                radius = int(radius)
                cv2.circle(output, center, radius, COLOR_WHITE, CIRCLE_THICKNESS)

                label = f"Red Area: {int(area)} squared pixels"
                cv2.putText(
                    output,
                    label,
                    center,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    RED_LABEL_FONT_SCALE,
                    COLOR_WHITE,
                    RED_LABEL_THICKNESS,
                )

        cv2.imshow("Masked Image", red_mask)
        cv2.imshow("Color Detection", output)

        if cv2.waitKey(REFRESH_TIME) & 0xFF == ord("q"):
            print("'Q' pressed, exiting.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
