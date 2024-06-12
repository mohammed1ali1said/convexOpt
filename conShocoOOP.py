import sqlite3
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import cvxpy as cp
import sqlite3

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# Using SIFT feature detection
sift = cv.SIFT_create()
# Using Flann-based matcher
flann = cv.FlannBasedMatcher()


products = []
productsPnP =[]
database_dir_coffee = '../ConvexClassify/3d models/pro coffee/dbdbdbdbd.db'
database_dir_nuts = '../ConvexClassify/3d models/pro nuts/nutseeez.db'

class Keypoint:
    def __init__(self,x,y,z,r,g,b):
        self.x=x
        self.y=y
        self.z=z
        self.r=r
        self.g=g
        self.b=b


class Product:
    def __init__(self,name):
        self.keypoints = []
        self.descriptors = []
        self.name=name

    def load_key_points(self,path):
     self.keypoints = []
     with open(path, 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue  # Skip comment lines
            data = line.strip().split()
            x, y, z =  data[1:4]
            self.keypoints.append([x, y, z])


    def load_descriptors(self, path):
        connection = sqlite3.connect(path)
        cursor = connection.cursor()

        # Fetch descriptors from the database
        query = 'SELECT data FROM descriptors'
        cursor.execute(query)
        descriptors_data = cursor.fetchall()

        # Close the database connection
        connection.close()

        # Process the descriptors
        self.descriptors = []
        for row in descriptors_data:
            descriptor_data = row[0]
            descriptor_np = np.frombuffer(descriptor_data, dtype=np.uint8)
            descriptor_np = descriptor_np.reshape(-1, 128)
            self.descriptors.append(descriptor_np)

        # Convert the list of descriptors to a single numpy array
        self.descriptors = np.concatenate(self.descriptors, axis=0, dtype=np.float32)

def filter(matches,kp1,kp2):

    matched_indices = set()
    for match_list in matches:
        for match in match_list:
            matched_indices.add(match.queryIdx)

    matched_keypoints = []
    matched_kp2 = []
    for idx,match in enumerate(matches):
        if(idx>=len(kp2) or idx>= len(kp1)):
            return matched_keypoints,matched_kp2
        idx1 = match[0].queryIdx  # Index in kp1
        idx2 = match[0].queryIdx  # Index in kp3


        matched_keypoints.append(kp1[idx1])
        matched_kp2.append(kp2[idx2])
    return matched_keypoints,matched_kp2

def load_products():
    #Create and append products to the products array
    product1 = Product("Nuts")
    product1.load_descriptors(database_dir_nuts)
    # product1.load_key_points('3d models/pro nuts/points3D.txt')
    products.append(product1)

    product2 =Product("Coffee")
    product2.load_descriptors(database_dir_coffee)
    products.append(product2)



def solve_pnp(keypoints_frame, keypoints_model):

    with np.load('CameraParams.npz') as data:
        camera_matrix = data['cameraMatrix']
        dist = data['dist']

    keypoints_model = np.array([[kp[0], kp[1],kp[2]] for kp in keypoints_model],dtype=np.float32)
    keypoints_frame = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints_frame],dtype=np.float32)


    success, rvec, tvec= cv.solvePnP(keypoints_model,keypoints_frame, camera_matrix, dist,flags=0)

    if success:
        return tvec, rvec
    else:
        return None, None


def pixel_to_3d(pixel_points, camera_matrix):
    # Initialize an empty list to store the 3D points
    points_3d_cartesian = []

    # Convert each pixel point to 3D
    for pixel_point in pixel_points:
        # Step 1: Normalize pixel coordinates
        pixel_point = pixel_point.ravel()
        x_norm = pixel_point[0]
        y_norm = pixel_point[1]

        # Step 2: Convert to homogeneous coordinates
        pixel_homogeneous = np.array([x_norm, y_norm, 1])

        # Step 3: Apply inverse projection
        inverse_projection_matrix = np.linalg.inv(camera_matrix)

        point_3d_homogeneous = np.dot(inverse_projection_matrix, pixel_homogeneous)

        # Convert homogeneous to Cartesian coordinates
        point_3d_cartesian = point_3d_homogeneous[:3]
        points_3d_cartesian.append(point_3d_cartesian)
    return np.array(points_3d_cartesian)



def find_match(frame):
    # Convert frame to grayscale
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    keypoints_frame, descriptors_frame = sift.detectAndCompute(frame_gray, None)

    with np.load('CameraParams.npz') as data:
        camera_matrix = data['cameraMatrix']

    f_x = camera_matrix[0, 0]
    f_y = camera_matrix[1, 1]

    best_match_idx = -1  # Index of the best match
    best_match_distance = float('inf')  # Best match distance

    for idx, product in enumerate(products):
        product_matches = flann.knnMatch(descriptors_frame, product.descriptors, k=10)
        kp1, kp2 = filter(product_matches, keypoints_frame, product.keypoints)

        # Extract matched descriptors
        descriptors_matched_frame = []
        descriptors_matched_product = []
        for match in product_matches:
            query_idx = match[0].queryIdx
            train_idx = match[0].trainIdx
            descriptors_matched_frame.append(descriptors_frame[query_idx])
            descriptors_matched_product.append(product.descriptors[train_idx])

        descriptors_matched_frame = np.array(descriptors_matched_frame)
        descriptors_matched_product = np.array(descriptors_matched_product)


        distance = cp.sum_squares(descriptors_matched_product - descriptors_matched_frame)

        objective = cp.Minimize(distance)
        problem = cp.Problem(objective)
        problem.solve()

        # Calculate total distance
        total_distance = distance.value
        print(total_distance)

        # Update best match if smaller total distance found
        if total_distance is not None and total_distance < best_match_distance:
            best_match_distance = total_distance
            best_match_idx = idx

    # Print the best match
    if best_match_idx != -1:
        print(f"Best match: {products[best_match_idx].name} with total distance {best_match_distance}")
    else:
        print("No matches found")

    return best_match_idx



def visualize(rvecs, tvecs,keypoints1,keypoints2,f_x,f_y):

    # Initialize the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Camera position (translation vector)
    camera_position = tvecs.ravel()

    # Camera orientation (rotation vector to rotation matrix)
    R, _ = cv.Rodrigues(rvecs)

    # Plot camera position
    ax.scatter(camera_position[0], camera_position[1], camera_position[2], c='r', marker='o', label='Camera Position')


    keypoints2 = np.array(keypoints2, dtype=np.float32)
    keypoints1[:,0] = keypoints1[:,0]
    keypoints1[:,1] = keypoints1[:,1]

    print(keypoints1[:,2])
    print(keypoints2[:,2])


    keypoints2[:,0] = keypoints2[:,0]
    keypoints2[:,1] = keypoints2[:,1]
    #todo apply rotation to object points


    # Plot rescaled keypoints
    ax.scatter(keypoints2[:, 0], keypoints2[:, 1], keypoints2[:, 2], c='b', marker='o', label='Keypoints')
    ax.scatter(keypoints1[:, 0], keypoints1[:, 1], keypoints1[:, 2], c='black', marker='o', label='Keypoints2')

    # Add lines from camera position to keypoints
    for kp in keypoints2:
        ax.plot([camera_position[0], kp[0]], [camera_position[1], kp[1]], [camera_position[2], kp[2]], c='g')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Position and KeyPoints')
    plt.show()



load_products()
frame=cv.imread('coffee.jpg')
name=find_match(frame)

