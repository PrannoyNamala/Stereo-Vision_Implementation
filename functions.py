import cv2
import glob
import numpy as np
import docx
import scipy.optimize as opt


def array_from_doc(str_to_convert):
    split_string = str_to_convert.split('[')
    split_string = split_string[-1].split(']')
    split_string = split_string[0].split(';')
    _toconvert = []
    for i in range(0, len(split_string)):
        split_row = split_string[i].split()
        split_row = list(map(float, split_row))
        _toconvert.append(split_row)
    numbers_array = np.array(_toconvert)
    return numbers_array


def getText(number):
    doc = docx.Document('Dataset ' + number + '/Groundtruth.docx')
    fulltext = []
    for para in doc.paragraphs:
        fulltext.append(para.text)
    properties_dict = {}
    for entry in fulltext:
        split_pair = entry.split('=')
        properties_dict[split_pair[0]] = split_pair[1]

    properties_dict['cam0'] = array_from_doc(properties_dict['cam0'])
    properties_dict['cam1'] = array_from_doc(properties_dict['cam1'])

    return properties_dict


def image_loader(number):
    image_list = []
    for filename in glob.glob('Dataset ' + number + '/*.png'):
        im = cv2.imread(filename)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = cv2.resize(im, (1920, 1080), interpolation=cv2.INTER_AREA)
        image_list.append(im)
    return image_list


def feature_matching(img_list):
    # Initiate ORB detector
    orb = cv2.ORB_create()
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img_list[0], None)
    kp2, des2 = orb.detectAndCompute(img_list[1], None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1, des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)
    # Draw first 10 matches.
    img3 = cv2.drawMatches(img_list[0], kp1, img_list[1], kp2, matches, None,
                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    pointslist_img1 = []
    pointslist_img2 = []
    for match in matches:
        pointslist_img1.append(np.array(((kp1[match.queryIdx].pt[0]), (kp1[match.queryIdx].pt[1]), (1))))
        pointslist_img2.append(np.array(((kp2[match.trainIdx].pt[0]), (kp2[match.trainIdx].pt[1]), (1))))

    return pointslist_img1, pointslist_img2


def matrix_bulider(img_1points, img_2points):
    A = []
    for point_1, point_2 in zip(img_1points, img_2points):
        row = [point_1[0] * point_2[0], point_1[1] * point_2[0], point_2[0], point_1[0] * point_2[1],
               point_1[1] * point_2[1], point_2[1], point_1[0], point_1[1], 1]
        A.append(row)

    A = np.array(A)

    u, l, vt = np.linalg.svd(A)
    f = np.reshape(vt[-1, :], (3, 3)).transpose()
    return f


def matrix_estimate(pointslist_img1, pointslist_img2):
    n = 0
    N = len(pointslist_img1)
    iterations = 0
    k = np.log(1 - 0.5) / np.log(1 - (0.5 ** 8))
    f_best = matrix_bulider(pointslist_img1[0:8], pointslist_img2[0:8])
    while iterations < k:
        int_list = np.random.randint(0, N, size=8)
        img1_selection = []
        img2_selection = []
        for ii in int_list:
            img1_selection.append(pointslist_img1[ii])
            img2_selection.append(pointslist_img2[ii])
        f = matrix_bulider(img1_selection, img2_selection)
        s = 0
        for j in range(0, N):
            prod = np.abs(np.transpose(pointslist_img2[j]).dot(f.dot(pointslist_img1[j])))
            if prod < 0.0005:
                s += 1

        if n < s:
            n = s
            f_best = f
        iterations += 1

    u, l, v = np.linalg.svd(f_best)
    l = np.array([[l[0], 0, 0], [0, l[1], 0], [0, 0, 0]])
    F = np.dot(u, l)
    f_best = np.dot(F, v)

    return f_best / f_best[2, 2]


def get_solutions(e):
    u, d, vt = np.linalg.svd(e)
    w = np.array(((0, -1, 0), (1, 0, 0), (0, 0, 1)))
    c = u[:, 2].reshape(3, 1)
    r1 = u.dot(w.dot(vt))
    r2 = u.dot(np.transpose(w).dot(vt))

    return [(r1, c), (r1, -c), (r2, c), (r2, -c)]


def skew(x):
    return np.array([[0, -x[2], x[1]], [x[2], 0, x[0]], [x[1], x[0], 0]])


def lin_triangulation(K, C1, R1, C2, R2, X1, X2):
    i3c = np.hstack((np.eye(3), -C1))
    P1 = K.dot(R1).dot(i3c)
    i3c = np.hstack((np.eye(3), -C2))
    P2 = K.dot(R2).dot(i3c)
    sz = X1.shape[0]

    X = np.zeros((sz, 3))

    for i in range(sz):
        skew1 = skew(X1[i, :])
        skew2 = skew(X2[i, :])
        A = np.vstack((np.dot(skew1, P1), np.dot(skew2, P2)))
        _, _, v = np.linalg.svd(A)
        x = v[-1] / v[-1, -1]
        x = np.reshape(x, (len(x), -1))
        X[i, :] = x[0:3].T

    return X


def disambiguate_pose(pairs, positions):
    best = 0
    for i in range(len(pairs)):
        N = positions[i].shape[0]
        n = 0
        for j in range(N):
            if (np.dot(pairs[i][0][2, :], (positions[i][j].reshape(3, 1) - pairs[i][1])) > 0) and \
                    positions[0][0][-1] >= 0:
                n = n + 1

        if n > best:
            C = pairs[i][1]
            R = pairs[i][0]
            X = positions[i]
            best = n

    return X, R, C


def minimizeFunction(init, K, x1, x2, R1, C1, R2, C2):
    sz = len(x1)
    X = np.reshape(init, (sz, 3))

    X = np.hstack((X, np.ones((sz, 1))))

    i3c = np.hstack((np.eye(3), -C1))
    P1 = K.dot(R1).dot(i3c)
    i3c = np.hstack((np.eye(3), -C2))
    P2 = K.dot(R2).dot(i3c)

    u1 = np.divide((np.dot(P1[0, :], X.T).T), (np.dot(P1[2, :], X.T).T))
    v1 = np.divide((np.dot(P1[1, :], X.T).T), (np.dot(P1[2, :], X.T).T))
    u2 = np.divide((np.dot(P2[0, :], X.T).T), (np.dot(P2[2, :], X.T).T))
    v2 = np.divide((np.dot(P2[1, :], X.T).T), (np.dot(P2[2, :], X.T).T))

    #     print(u1.shape,x1.shape)
    # assert u1.shape[0] == x1.shape[0], "shape not matched"

    error1 = ((x1[:, 0] - u1) + (x1[:, 1] - v1))
    error2 = ((x2[:, 0] - u2) + (x2[:, 1] - v2))
    #     print(error1.shape)
    error = sum(error1, error2)

    return sum(error)


def nonlin_tirangulation(K, x1, x2, X_init, R1, C1, R2, C2):
    sz = len(x1)

    init = X_init.flatten()
    #     Tracer()()
    optimized_params = opt.least_squares(
        fun=minimizeFunction,
        x0=init,
        method="dogbox",
        args=[K, x1, x2, R1, C1, R2, C2])

    X = np.reshape(optimized_params.x, (sz, 3))

    return X


def drawlines(img1, img2, lines, pts1, pts2):

    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple((int(pt1[0]), int(pt1[1]))), 5, color, -1)
        img2 = cv2.circle(img2, tuple((int(pt2[0]), int(pt2[1]))), 5, color, -1)
    return img1, img2


def feature_binary(img, points_list):
    img_dup = np.zeros_like(img)
    for point in points_list:
        img_dup[point[1], point[0]] = 255

    return img_dup


def template_match(template, img):
    res = cv2.matchTemplate(img, template, cv2.TM_SQDIFF)
    _, _, min_loc, _ = cv2.minMaxLoc(res)
    return min_loc


def get_windows(template_length, img):
    h, w = img.shape
    dict_return = {}
    center_location = (template_length + 1) / 2
    for i in range(0, h - template_length):
        for j in range(0, w - template_length):
            template = img[i:i + template_length, j:j + template_length]
            location = (j + center_location, i + center_location)
            dict_return[location] = template

    return dict_return


def compute_disparity(point_img1, point_img2, f, b):
    d = point_img1[0] - point_img2[0]
    if d == 0:
        d = 0.001
    return f * b / d, d
