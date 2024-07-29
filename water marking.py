import cv2

# Read images
img_path = r"C:\Users\SARAN\OneDrive\Pictures\HD-wallpaper-owl-eyes-owl-eyes-dark-staring-bird-black.jpg"
wm_path = r"C:\Users\SARAN\OneDrive\Pictures\Prints and desktop backgrounds by 3D artist George Grie Ghost ship (1).jpg"
img = cv2.imread(img_path)
wm = cv2.imread(wm_path)

if img is None:
    print(f"Error: Could not load the image from {img_path}")
if wm is None:
    print(f"Error: Could not load the watermark from {wm_path}")

# Get dimensions
h_wm, w_wm = wm.shape[:2]
h_img, w_img = img.shape[:2]

print(f"Image size: {w_img}x{h_img}")
print(f"Watermark size: {w_wm}x{h_wm}")

# Ensure watermark is smaller than the image
if h_wm > h_img or w_wm > w_img:
    print("Error: The watermark is larger than the image.")
else:
    # Calculate the position
    center_x = int(w_img / 2)
    center_y = int(h_img / 2)
    top_y = center_y - int(h_wm / 2)
    left_x = center_x - int(w_wm / 2)
    bottom_y = top_y + h_wm
    right_x = left_x + w_wm

    # Debugging prints for position
    print(f"Top-left corner: ({left_x}, {top_y})")
    print(f"Bottom-right corner: ({right_x}, {bottom_y})")

    # Create the region of interest
    roi = img[top_y:bottom_y, left_x:right_x]

    # Blend the images
    result = cv2.addWeighted(roi, 1, wm, 0.3, 0)

    # Place the blended result back into the original image
    img[top_y:bottom_y, left_x:right_x] = result

    # Display the result
    cv2.imshow("Watermarked Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
