import cv2
import numpy as np

# Applies an underwater effect to an image using 6 effects.
# frame is the image (3d array) to apply the effects on. depth is a float from 0.0 to 1.0 that describes how deep underwater we are simulating (surface to very deep).
def apply_underwater_effect(frame, depth):
    
    # Generally, camera frames (like this one) are read as uint8, where each pixel is given a value from 0-255.
    # uint8 risks (over/under)flowing. For example, assigning a uint8 to 256 --> 0, 257 --> 1, 300 --> 44, -1 --> 255.
    # Numpy's astype function allows us to convert an image's data type. 
    # We convert uint8 to float32 because float32 can handle floating point values up to 3.4 * 10^38.
    degraded = frame.astype(np.float32) 
    
    # Numpy's shape function returns the number of elements along each axis of our array.
    # Generally, color images in openCV have three dimensions: 1) Height, 2) Width, 3) Color Channels.
    # 1) Height: The vertical height of our image (in pixels)
    # 2) Width: The horizontal width of our image (in pixels)
    # 3) Color Channels: The color of each pixel (in BGR format)
    height, width = degraded.shape[:2]
    
    # Effect 1: Color Absorption
    # As we descend underwater, water molecules increasingly absorb light. Long wavelengths (like Red) are absorbed the most, and short wavelengths (like Blue) are absorbed the least.
    # The format [:, :, #] means select all rows and all columns, but only select color channel index #
    # BGR Format: 0 = Blue, 1 = Green, 2 = Red
    degraded[:, :, 2] *= max(0.0, 1.0-depth*0.85) # Absorbs Red (Up to 85% of OG Brightness)
    degraded[:, :, 1] *= max(0.0, 1.0-depth*0.4) # Absborbs Green (Up to 40% of OG Brightness)
    degraded[:, :, 0] *= max(0.0, 1.0-depth*0.1) # Absorbs Blue (Up to 10% of OG Brightness)
    
    # Effect 2: Blue-Green Glow
    # While underwater, light often bounces off water molecules and into the camera, creating a slight glow of blue and green.
    # Numpy's clip function takes in an array and returns it with all values within the min and max values as passed in as arguments (0 to 255).
    # Blue is brightened 2.5 times more than that of green
    degraded[:, :, 0] = np.clip(degraded[:, :, 0] + depth * 45, 0, 255) # Brightens Blue
    degraded[:, :, 1] = np.clip(degraded[:, :, 1] + depth * 18, 0, 255) # Brightens Green
    
    # Effect 3: Haze/Fog Layer
    # Simulates the murkiness underwater caused by dirt and algae. At maximum depth, the haze can account for 35% of the image.
    # Numpy's zeros_like function creates a completely black canvas with dimensions and datatype identical to its argument.
    # cv2's addWeighted function blends 2 images together by applying the formula: result = (α * img1) + (β * img2) + γ to every pixel in the image
    # cv2's addWeighted arguments: (image1, α, image2, β, γ), where α = importance of image1, β = importance of image2, γ = final brightness adjustment
    haze_strength = depth * 0.35
    fog_layer = np.zeros_like(degraded)
    fog_layer[:, :, 0] = 110 # Sets the blue value for murky teal
    fog_layer[:, :, 1] = 85 # Sets the green value for murky teal
    fog_layer[:, :, 2] = 60 # Sets the red value for murky teal
    degraded = cv2.addWeighted(degraded, 1.0 - haze_strength, fog_layer, haze_strength, 0) # Blends the current image with murky teal. Increased blending of murky teal at deeper depths
    
    # Effect 4: Blur
    # Simulates how microparticles (microplastics, minerals, microorganisms, etc.) underwater scatter light, preventing specific details from entering the camera lens.
    # cv2's GaussianBlur function blurs an image by averaging a pixel with its neighbors using a kernel.
    # GaussianBlur multiplies each pixel in the kernel with a weight using a Gaussian formula, then sums up all those multiplied values to become the new value for the center pixel.
    # GaussianBlur arguments: (image, kernel size, standard deviation for the Gaussian formula)
    # Keeping the kernel small ensures subtle softening while keeping computations low.
    blue_kernel = int(depth * 2) * 2 + 1 # Guarantees odd value
    degraded = cv2.GaussianBlur(degraded, (blue_kernel, blue_kernel), 0)
    
    # Effect 5: Vignette 
    # Simulates underwater vignette, where the center of the frame stays bright, while the corners and edges slowly fade into black. The edge pixels' brightness can be reduced by 91.93% at max depth.
    # Numpy's ogrid creates 2 separate coordinate vectors, which can later be combined to create a 2D grid. Y = vertical column vector, X = horizontal row vector.
    # dist_from_center uses the Pythagorean theorem and combines X and Y to create a 2D grid, where each value on the grid represents the distance from the center of the grid (Normalized: 0.0 to 1.414)
    # Because each pixel's value should be a representation of its final brightness, vignette_mask inverts dist_from_center's values so that those at the center become the largest while those at the edges become the smallest.
    # Multiplying 2 numpy arrays together with different dimensions will throw an error, so we add a fake 3rd dimension (of length 1) to vignette_mask to bypass that error.
    Y, X = np.ogrid[:height, :width]
    cx, cy = width / 2.0, height / 2.0
    dist_from_center = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
    vignette_mask = np.clip(1.0 - dist_from_center * depth * 0.65, 0.0, 1.0)
    degraded *= vignette_mask[:, :, np.newaxis]
    
    # Effect 6: Marine Snow
    # Simulates organic detritus drifting through the ocean, which in a camera would look like tiny, bright white specks scattered across the frame.
    # Numpy's random.random function generates an array of random floating-point values from 0.0 to 1.0. We take each value and compare them against depth * 0.006 to create particle_mask, a 2D boolean array that describes which pixels on the frame should be marine snow.
    # Numpy's random.randint function generates an array of random integers (from 180 to 255 in our case) which describes the brightness of each speck of marine snow.
    # Effectively, we are first determining which pixels on the frame should be marine snow, and then determining how bright each speck of marine snow should be.
    # Lastly, we replace the appropriate pixels' original BGR values with the corresponding BGR value from brightness.
    if depth > 0.15:
        particle_mask = np.random.random((height, width)) < (depth * 0.00096)
        brightness = np.random.randint(180, 255, size=(height, width)).astype(np.float32)
        degraded[particle_mask] = brightness[particle_mask, np.newaxis]

    # Convert the frame's datatype back into uint8 so openCV doesn't crash
    return np.clip(degraded, 0, 255).astype(np.uint8)

# Corrects a degraded underwater image by 1) increasing the red channel, 2) achieving the gray world assumption, and 3) using histogram equalization.
def correct_underwater_image(frame):
    
    degraded = frame.astype(np.float32)
    
    # Step 1: Increase Red Channel
    # Red light is absorbed the deeper one goes underwater. We add this red back into the image using the amount of blue as a reference.
    
    # np.mean calculates the mean value of a grid. Here, we find the mean red, blue, and green in the image.
    mean_red = np.mean(degraded[:, :, 2])
    mean_green = np.mean(degraded[:, :, 1])
    mean_blue = np.mean(degraded[:, :, 0])
    
    # We add 50% of the average difference between blue and red back into the image's red channels.
    # We don't add 100% of the difference because that would assume the image naturally contains equal amounts of red and blue, which is almost never true in ocean scenes.
    difference = mean_blue - mean_red
    degraded[:, :, 2] = np.clip(degraded[:, :, 2] + difference * 0.5, 0, 255)
    
    # Step 2: Achieve Gray World Assumption, which states that in a natural scene, the average brightness of each color channel should be equal to the average brightness of the entire image.
    # mean_bgr is the average brightness of the entire image.
    mean_red = np.mean(degraded[:, :, 2])
    mean_bgr = (mean_red + mean_green + mean_blue) / 3.0
    
    # If the average brightness of a color channel is not equal average brightness of the image, we correct it by scaling each pixel of that color channel by: (avg image brightness) / (avg brightness of color channel)
    # We use 1e-6 as a safety net in case the average brightness of a color channel is 0 (which would result in a divide by 0 error).
    degraded[:, :, 2] = np.clip(degraded[:, :, 2] * mean_bgr / max(mean_red, 1e-6), 0, 255)
    degraded[:, :, 1] = np.clip(degraded[:, :, 1] * mean_bgr / max(mean_green, 1e-6), 0, 255)
    degraded[:, :, 0] = np.clip(degraded[:, :, 0] * mean_bgr / max(mean_blue, 1e-6), 0, 255)
    
    degraded = degraded.astype(np.uint8)
    
    # lab color space: l = lightness (from pure black to pure white), a = green (-128) --> red (127), b = blue (-128) --> yellow (127)
    lab = cv2.cvtColor(degraded, cv2.COLOR_BGR2LAB)
    
    # cv2.split splits a color channel into its components (each a single 2d array).
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # use CLAHE instead of standard HE because it ensures the entire image's histogram isn't improperly spread out by implementing a clipLimit and utilizing billinear interpolation (using the closest 4 tiles' center pixels' mapping functions to determine the final value of each pixel).
    # clipLimit is a multiplier to determine the actual pixel ceiling. Calculation: (clipLimit) * (# of pixels per tile) / (256).
    # createCLAHE's tileGridSize parameter specificies what kind of grid we want to split our image into. Choosing to split the image into an 8x8 grid (or 64 tiles) is the most common choice.
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    
    l_enhanced = clahe.apply(l_channel)
    
    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    
    degraded = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    return degraded
    
cap = cv2.VideoCapture(0)

# isOpened tells us whether the camera or file has been successfully connected or not.
if not cap.isOpened():
    print("Camera Error. Exiting")
    exit()

# cv2's namedWindow function allows us to add UI elements to a window before displaying it. In contrast, imshow creates and display the window automatically, so that's why we don't use it here.
cv2.namedWindow("Color Correction")

# cv2's createTrackbar function adds an interactive slider to the window. cv2's trackbars always range from 0 to max_value (max_value passed in as an argument).
# arguments: (Slider name, Window to add Slider to, intial starting position of slider, max_value, function when adjusted)
# Because the createTrackbar function requires a 5th argument, which is the function to call when the trackbar is used (but we have none), we pass in "lambda x: None", which effectively is a function that does nothing.
cv2.createTrackbar("Depth %", "Color Correction", 50, 100, lambda x: None)

print("Left: Live Feed || Center: Degraded || Right: Corrected")
print("Press 'Q' to quit")

while True:

    # the read function grabs the next frame from the camera.
    # ret: whether the next frame was successfully read or not
    # frame: the actual next frame (3d array) from the camera.
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to grab frame.")
        break
    
    depth_percent = cv2.getTrackbarPos("Depth %", "Color Correction")
    depth = depth_percent / 100.0
    
    result = apply_underwater_effect(frame, depth)
    corrected = correct_underwater_image(result)
    
    height, width = frame.shape[:2]
    
    def add_label(img, text):
        out = img.copy()
        
        # cv2.rectangle args: (img to overwrite, coord of top-left corner, coord of bottom-right corner, color, thickness --> -1 = fill)
        cv2.rectangle(out, (0, 0), (width, 28), (0, 0, 0), -1)
        cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return out
    
    panel_original = add_label(frame, "Original")
    panel_degraded = add_label(result, f"Degraded (Depth: {int(depth*100)}%)")
    panel_corrected = add_label(corrected, "Corrected")
    
    # np.hstack takes a list of arrays and attaches them together side-by-side along their horizontal axis (It's required that the arrays' heights be equal)
    comparison = np.hstack([panel_original, panel_degraded, panel_corrected])
    cv2.imshow("Color Correction", comparison)

    # waitKey forces the program to wait 1 ms before switching frames (vid), and also returns the ASCII value of key pressed during that time frame.
    # We use & 0xFF (bitwise AND) in order to retrieve the actual ASCII value
    # ord gives the ASCII value of the letter passed in.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("'Q' pressed, exiting.")
        break

cap.release()
cv2.destroyAllWindows()