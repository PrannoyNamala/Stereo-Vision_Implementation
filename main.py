# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


from functions import *
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dset_number = input("Enter the dataset number to test\n1 for Dataset 1\n2 for Dataset 2\n3 for Dataset 3 ")
    img_list = image_loader(dset_number)

    pointslist_img1, pointslist_img2 = feature_matching(img_list)


    # f = matrix_estimate(pointslist_img1, pointslist_img2)
    # print("My f", f)
    # f = cv2.findFundamentalMat(np.array(pointslist_img1)[:, :2], np.array(pointslist_img2)[:, :2], cv2.FM_LMEDS)
    # f = f[0]
    # print("Inbuilt F", f)
    f = matrix_bulider(pointslist_img1[0:8], pointslist_img2[0:8])
    u, l, v = np.linalg.svd(f)
    l = np.array([[l[0], 0, 0], [0, l[1], 0], [0, 0, 0]])
    F = np.dot(u, l)
    f_best = np.dot(F, v)
    f = f_best / f_best[2, 2]
    print("8 rows F", f)
    # raise SystemExit()

    properties_dict = getText(dset_number)
    # print(properties_dict)

    e = (np.transpose(properties_dict['cam0']).dot(f)).dot(properties_dict['cam1'])
    # raise SystemExit()

    pairs = get_solutions(e)
    positions = []
    for pair in pairs:
        X = lin_triangulation(properties_dict['cam0'], np.zeros((3, 1)), np.eye(3), pair[1], pair[0],
                              np.array(pointslist_img1), np.array(pointslist_img2))
        positions.append(X)

    X, R, C = disambiguate_pose(pairs, positions)

    X = nonlin_tirangulation(properties_dict['cam0'], np.array(pointslist_img1), np.array(pointslist_img2), X,
                             np.eye(3), np.zeros((3, 1)),
                             R, C)

    print("Rotation Matrix:", R)
    print("Translation Matrix:", C)

    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(np.array(pointslist_img2[:40]), 2, f)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = drawlines(img_list[0], img_list[1], lines1, pointslist_img1[:40], pointslist_img2[:40])
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(np.array(pointslist_img1[:40]), 1, f)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines(img_list[1], img_list[0], lines2, pointslist_img2[:40], pointslist_img1[:40])

    cv2.imwrite("unrectified_1.png", img5)
    cv2.imwrite("unrectified_2.png", img3)
    plt.subplot(121), plt.imshow(img5)
    plt.subplot(122), plt.imshow(img3)
    plt.show()

    h1, w1 = img_list[0].shape
    h2, w2 = img_list[1].shape
    # print(w1, h1)
    _, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(pointslist_img1[:40]), np.float32(pointslist_img2[:40]), f,
                                              imgSize=(w1, h1))  # imgSize=(w1, h1)
    img1_rectified = cv2.warpPerspective(img5, H1, (w1, h1))
    img1_rectified_without_lines = cv2.warpPerspective(img_list[0], H1, (w1, h1))
    # feature_locations_img1_transformed = cv2.warpPerspective(feature_binary(img5, pointslist_img1[:40]), H1, (w1, h1))

    img2_rectified = cv2.warpPerspective(img3, H2, (w2, h2))
    img2_rectified_without_lines = cv2.warpPerspective(img_list[1], H2, (w2, h2))
    # feature_locations_img2_transformed = cv2.warpPerspective(feature_binary(img3, pointslist_img2[:40]), H2, (w2, h2))

    cv2.imwrite("rectified_1.png", img1_rectified)
    cv2.imwrite("rectified_2.png", img2_rectified)
    cv2.imwrite("rectified_stacked.png", np.hstack((img1_rectified, img2_rectified)))
    cv2.imshow("...", np.hstack((img1_rectified, img2_rectified)))
    cv2.waitKey(0)

    img1_rectified_without_lines = cv2.resize(img1_rectified_without_lines, (640, 480))
    img2_rectified_without_lines = cv2.resize(img2_rectified_without_lines, (640, 480))

    template_window_length = 21
    half_window = int(template_window_length / 2)
    templates_dict = get_windows(template_window_length, img1_rectified_without_lines)
    depth_map = np.zeros_like(img1_rectified_without_lines)
    disparity_map = np.zeros_like(img1_rectified_without_lines)
    print("Number of templates:", len(templates_dict.keys()))
    count = 1
    for point in templates_dict.keys():
        match_point = template_match(templates_dict[point], img2_rectified_without_lines)
        depth, disparity = compute_disparity(point, match_point, properties_dict['cam0'][0, 0],
                                      float(properties_dict['baseline']))
        print(point)
        disparity_map[int(point[1]) - half_window:int(point[1]) + half_window,
        int(point[0]) - half_window:int(point[0]) + half_window] = disparity
        depth_map[int(point[1]) - half_window:int(point[1]) + half_window,
        int(point[0]) - half_window:int(point[0]) + half_window] = depth
        print("Template Done", count)
        count += 1

    plt.figure()
    plt.imshow(disparity_map, cmap="gray")
    plt.colorbar()
    plt.show()
    plt.close()

    plt.figure()
    plt.imshow(depth_map, cmap="plasma")
    plt.colorbar()
    plt.show()
    plt.close()
