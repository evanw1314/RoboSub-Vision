import cv2
import numpy as np

# ABSORPTION & GLOW
MAX_RED_ABSORPTION = 0.85
MAX_GREEN_ABSORPTION = 0.4
MAX_BLUE_ABSORPTION = 0.1
BLUE_GLOW_MULTIPLIER = 45
GREEN_GLOW_MULTIPLIER = 18

# HAZE LAYER
HAZE_BLEND = 0.35
MURKY_TEAL_BLUE = 110
MURKY_TEAL_GREEN = 85
MURKY_TEAL_RED = 60

# VIGNETTE & BLUR
GAUSSIAN_KERNEL_SIGMAX = 0
VIGNETTE_MULTIPLIER = 0.65

# MARINE SNOW PARTICLES
SNOW_MIN_DEPTH = 0.15
SNOW_CHANCE_MULTIPLIER = 0.00096
SNOW_DARKEST = 180
SNOW_BRIGHTEST = 255

# UI & APPLICATION SETTINGS
DEFAULT_CAMERA_IDX = 0
DEPTH_PERCENT_STARTPOS = 30
DEPTH_PERCENT_MAXVAL = 100
REFRESH_TIME = 1

# LABEL STYLING
LABEL_THICKNESS = 2
LABEL_FONT_SCALE = 1
LABEL_RGB = (200, 220, 255)
LABEL_COORDS_TOPLEFT = (12, 30)


def apply_underwater_effect(frame, depth):
    """Applies an underwater effect to a BGR image."""
    # Convert to float32 to avoid overflow/underflow amidst calculations
    degraded = frame.astype(np.float32)
    height, width = degraded.shape[:2]

    # Effect 1: Color Absorption (Long wavelengths absorbed faster than short
    # wavelengths)
    degraded[:, :, 2] *= max(0.0, 1.0 - depth * MAX_RED_ABSORPTION)
    degraded[:, :, 1] *= max(0.0, 1.0 - depth * MAX_GREEN_ABSORPTION)
    degraded[:, :, 0] *= max(0.0, 1.0 - depth * MAX_BLUE_ABSORPTION)

    # Effect 2: Blue-Green Glow (Simulates light scattering)
    degraded[:, :, 0] = np.clip(
        degraded[:, :, 0] + depth * BLUE_GLOW_MULTIPLIER, 0, 255
    )
    degraded[:, :, 1] = np.clip(
        degraded[:, :, 1] + depth * GREEN_GLOW_MULTIPLIER, 0, 255
    )

    # Effect 3: Haze/Fog Layer (Simulates underwater murkiness caused by dirt & algae)
    haze_strength = depth * HAZE_BLEND
    fog_layer = np.zeros_like(degraded)
    fog_layer[:, :, 0] = MURKY_TEAL_BLUE
    fog_layer[:, :, 1] = MURKY_TEAL_GREEN
    fog_layer[:, :, 2] = MURKY_TEAL_RED
    degraded = cv2.addWeighted(
        degraded, 1.0 - haze_strength, fog_layer, haze_strength, 0.0
    )

    # Effect 4: Blur (Simulates light scattering caused by microparticles)
    blue_kernel = int(depth * 2) * 2 + 1  # Ensures odd kernel size
    degraded = cv2.GaussianBlur(
        degraded, (blue_kernel, blue_kernel), GAUSSIAN_KERNEL_SIGMAX
    )

    # Effect 5: Vignette (Simulates the gradual loss of light from the lens center)
    y, x = np.ogrid[:height, :width]
    cx, cy = width / 2.0, height / 2.0
    dist_from_center = np.sqrt(((x - cx) / cx) ** 2 + ((y - cy) / cy) ** 2)
    vignette_mask = np.clip(
        1.0 - dist_from_center * depth * VIGNETTE_MULTIPLIER, 0.0, 1.0
    )
    degraded *= vignette_mask[:, :, np.newaxis]

    # Effect 6: Marine Snow (Simulates organic detritus drifting in the water)
    if depth > SNOW_MIN_DEPTH:
        particle_mask = np.random.random((height, width)) < (
            depth * SNOW_CHANCE_MULTIPLIER
        )
        brightness = np.random.randint(
            SNOW_DARKEST, SNOW_BRIGHTEST, size=(height, width)
        ).astype(np.float32)
        degraded[particle_mask] = brightness[particle_mask, np.newaxis]

    return np.clip(degraded, 0, 255).astype(np.uint8)


def main():
    cap = cv2.VideoCapture(DEFAULT_CAMERA_IDX)
    if not cap.isOpened():
        print("Camera Error. Exiting")
        exit()

    cv2.namedWindow("Underwater Simulation")
    cv2.createTrackbar(
        "Depth %",
        "Underwater Simulation",
        DEPTH_PERCENT_STARTPOS,
        DEPTH_PERCENT_MAXVAL,
        lambda x: None,
    )

    print("Drag the slider to change the effect")
    print("Press 'Q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        depth_percent = cv2.getTrackbarPos("Depth %", "Underwater Simulation")
        depth = depth_percent / 100.0

        result = apply_underwater_effect(frame, depth)
        label = f"Simulated depth: {depth_percent}%"

        cv2.putText(
            result,
            label,
            LABEL_COORDS_TOPLEFT,
            cv2.FONT_HERSHEY_SIMPLEX,
            LABEL_FONT_SCALE,
            LABEL_RGB,
            LABEL_THICKNESS,
        )

        cv2.imshow("Original", frame)
        cv2.imshow("Underwater Simulation", result)

        if cv2.waitKey(REFRESH_TIME) & 0xFF == ord("q"):
            print("'Q' pressed, exiting.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
