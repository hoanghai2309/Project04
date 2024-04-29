import cv2 as cv2
import numpy as np
# This function will get click pixel coordinate that source image will be pasted to destination image
def inversewarping(src, T, coordinate):
    T_inv = np.linalg.inv(T)
    x_max, y_max = src.shape[0] - 1, src.shape[1] - 1
    x, y, z = T_inv @ np.array(coordinate)
    if x > x_max or x < 0:
        return 0
    if y > y_max or y < 0:
        return 0
    x_top = int(x // 1)
    x_down = int(x_top + 1)
    if x_down>x_max:
      x_down=x_max
    x_red = (x * 100) % 100 / 100
    y_left = int(y // 1)
    y_right = int(y_left + 1)
    if y_right>y_max:
      y_right=y_max
    y_red = (y * 100) % 100 / 100
    top_left = src[x_top, y_left, :]
    top_right = src[x_top, y_right, :]
    down_left = src[x_down, y_left, :]
    down_right = src[x_down, y_right, :]
    return x_red * y_red * top_left + x_red * (1 - y_red) * top_right + (1 - x_red) * y_red * down_left + (1 - x_red) * (1 - y_red) * top_left

def ProjectiveWarping(image, matrix, img_transformed):
  for i, row in enumerate(img_transformed):
    for j, col in enumerate(row):
        coordinate = np.array([i, j, 1])
        img_transformed[i, j] = inversewarping(image,matrix,coordinate)
def get_des_position(event, x, y, flags, paste_coordinate_list):
    cv2.imshow('collect coordinate', img_dest_copy)
    if event == cv2.EVENT_LBUTTONUP:
        # Draw circle right in click position
        cv2.circle(img_dest_copy, (x, y), 2, (0, 0, 255), -1)
        # Append new clicked coordinate to paste_coordinate_list
        paste_coordinate_list.append([x, y])
def get_src_position(event, x, y, flags, paste_coordinate_list):
    cv2.imshow('collect coordinate', img_src_copy)
    if event == cv2.EVENT_LBUTTONUP:
        # Draw circle right in click position
        cv2.circle(img_src_copy, (x, y), 2, (0, 0, 255), -1)
        # Append new clicked coordinate to paste_coordinate_list
        paste_coordinate_list.append([x, y])
if __name__ == '__main__':
    # Read source image
    img_src = cv2.imread('src.png', cv2.IMREAD_COLOR)
    # Read destination image
    img_dest = cv2.imread('des.png', cv2.IMREAD_COLOR)
    # copy destination image for get_paste_position (Just avoid destination image will be draw)
    img_dest_copy = img_dest.copy()#np.tile(img_dest, 1)
    img_src_copy = img_src.copy()
    # paste_coordinate in destination image
    paste_coordinate = []
    # Get source image parameter: [[left,top], [left,bottom], [right, top], [right, bottom]]
    img_src_coordinate = []
    cv2.namedWindow('collect coordinate')
    cv2.setMouseCallback('collect coordinate', get_des_position, paste_coordinate)
    while True:
        cv2.waitKey(1)
        if len(paste_coordinate) == 4:
            break
    paste_coordinate = np.array(paste_coordinate)
    print(paste_coordinate)
    cv2.namedWindow('collect coordinate')
    cv2.setMouseCallback('collect coordinate', get_src_position, img_src_coordinate)
    while True:
        cv2.waitKey(1)
        if len(img_src_coordinate) == 4:
            break
    img_src_coordinate = np.array(img_src_coordinate)
    print(img_src_coordinate)
    # Get perspective matrix
    matrix= cv2.getPerspectiveTransform(np.float32(img_src_coordinate), np.float32(paste_coordinate))
    print(f'matrix: {matrix}')

    # perspective_img = np.empty((1000, 1000, img_src.shape[2]))
    # perspective_img =ProjectiveWarping(img_src, matrix, perspective_img)


    perspective_img = cv2.warpPerspective(img_src, matrix, (img_dest.shape[0], img_dest.shape[1]))


    cv2.imshow('img', perspective_img)
    # Create an empty mask with the same size as the destination image
    mask = np.zeros_like(img_dest)

    # Fill in the region within img_src_coordinate in the mask
    cv2.fillPoly(mask, [np.int32(paste_coordinate)], (255, 255, 255))

    # Convert the mask to grayscale (cv2.copyTo requires a single channel mask)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Now use this mask in the cv2.copyTo function
    cv2.copyTo(src=perspective_img, mask=mask, dst=img_dest)

    cv2.imshow('result', img_dest)
    cv2.waitKey()
    cv2.destroyAllWindows()