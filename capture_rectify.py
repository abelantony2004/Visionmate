import cv2
import numpy as np
import os
import glob

# === Paths ===
calib_file = "stereo_calib.npz"
image_dir = "calib"
output_dir = "rectified"
os.makedirs(output_dir, exist_ok=True)

# === Load calibration data ===
data = np.load(calib_file)
K1 = data['cameraMatrix1']
D1 = data['distCoeffs1']
K2 = data['cameraMatrix2']
D2 = data['distCoeffs2']
R1 = data['R1']
R2 = data['R2']
P1 = data['P1']
P2 = data['P2']
Q = data['Q']

# === Image pairs ===
left_images = sorted(glob.glob(os.path.join(image_dir, "left*.png")))
right_images = sorted(glob.glob(os.path.join(image_dir, "right*.png")))

# === Get image size from first image
img_shape = cv2.imread(left_images[0]).shape[:2][::-1]  # (width, height)

# === Init rectification maps
map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, img_shape, cv2.CV_16SC2)
map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, img_shape, cv2.CV_16SC2)

# === Stereo matcher setup (SGBM)
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=128,   # Must be divisible by 16
    blockSize=5,
    P1=8 * 3 * 5**2,
    P2=32 * 3 * 5**2,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

for i, (l_path, r_path) in enumerate(zip(left_images, right_images)):
    imgL = cv2.imread(l_path)
    imgR = cv2.imread(r_path)

    rectL = cv2.remap(imgL, map1x, map1y, cv2.INTER_LINEAR)
    rectR = cv2.remap(imgR, map2x, map2y, cv2.INTER_LINEAR)

    # Save rectified images
    cv2.imwrite(os.path.join(output_dir, f"rect_left_{i:02d}.png"), rectL)
    cv2.imwrite(os.path.join(output_dir, f"rect_right_{i:02d}.png"), rectR)

    # Convert to grayscale for disparity
    grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)

    disp = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
    disp_vis = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX)
    disp_vis = np.uint8(disp_vis)

    # Save and display disparity
    cv2.imwrite(os.path.join(output_dir, f"disp_{i:02d}.png"), disp_vis)
    cv2.imshow("Disparity", disp_vis)
    cv2.waitKey(0)

cv2.destroyAllWindows()
print("Rectification and disparity complete. Saved in:", output_dir)