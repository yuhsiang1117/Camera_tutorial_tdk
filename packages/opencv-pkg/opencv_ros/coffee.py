import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import open3d as o3d
from sklearn.cluster import KMeans

'''
Depth Camera Intrinsics:
Width: 848
Height: 480
PPX (principal point x): 429.2327575683594
PPY (principal point y): 236.95892333984375
Fx (focal length x): 427.5506286621094
Fy (focal length y): 427.5506286621094
Distortion model: distortion.brown_conrady
Distortion coefficients: [0.0, 0.0, 0.0, 0.0, 0.0]
Color Camera Intrinsics:
PPX: 329.0754699707031, PPY: 244.79696655273438
Fx: 606.8665771484375, Fy: 606.635986328125
'''

class Coffee(Node):
    def __init__(self):
        super().__init__('coffee_node')
        self.bridge = CvBridge()
        self.img_sub = self.create_subscription(
            Image,
            "/camera/camera/color/image_raw",
            self.image_callback,
            10
        )
        self.depth_sub = self.create_subscription(
            Image,
            "/camera/camera/depth/image_rect_raw",
            self.depth_callback,
            10
        )
        self.img_publisher = self.create_publisher(Image, 'coffee_detection', 10)
        self.debug_publisher = self.create_publisher(Image, 'coffee_debug', 10)

    def image_callback(self, msg):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.publish_processed_image()
        except Exception as e:
            self.get_logger().error(f'Error converting image: {e}')

    def depth_callback(self, msg):
        try:
            self.cv_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'Error converting depth image: {e}')

    def publish_processed_image(self):
        try:
            depth_m = self.cv_depth.astype(np.float32) / 1000.0
            mask = (depth_m > 0.2) & (depth_m < 1.0)
            binary_img = np.zeros_like(depth_m, dtype=np.uint8)
            binary_img[mask] = 255

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
            binary_img = cv2.erode(binary_img, kernel)
            binary_img = cv2.dilate(binary_img, kernel)

            contours, _ = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                max_region = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(max_region)
                cx = x + w // 2
                cy = y + h // 2

                depth_center = depth_m[cy, cx]
                self.get_logger().info(f"Depth at center ({cx}, {cy}): {depth_center:.3f} m")

                lower_depth = depth_center - 0.5
                upper_depth = depth_center + 0.5

                indices = np.where((depth_m >= lower_depth) & (depth_m <= upper_depth))

                margin = 50
                x_min = max(x - margin, 0)
                x_max = min(x + w + margin, depth_m.shape[1] - 1)
                y_min = max(y - margin, 0)
                y_max = min(y + h + margin, depth_m.shape[0] - 1)

                filtered_points = []
                for py, px in zip(indices[0], indices[1]):
                    if x_min <= px <= x_max and y_min <= py <= y_max:
                        filtered_points.append((px, py))

                if filtered_points:
                    pts = np.array(filtered_points)
                    new_x, new_y, new_w, new_h = cv2.boundingRect(pts)

                    fx_color = 606.8665771484375
                    physical_offset = 0.0148
                    pixel_offset = int(physical_offset * fx_color / depth_center)

                    new_x_corrected = new_x + pixel_offset
                    img_width = self.cv_image.shape[1]
                    if new_x_corrected + new_w > img_width:
                        new_x_corrected = img_width - new_w - 1
                    if new_x_corrected < 0:
                        new_x_corrected = 0

                    # 畫校正後框
                    cv2.rectangle(self.cv_image, (new_x_corrected, new_y), 
                                (new_x_corrected + new_w, new_y + new_h), (0, 255, 255), 3)

                    cx_corrected = cx + pixel_offset
                    if cx_corrected >= img_width:
                        cx_corrected = img_width - 1

                    # cv2.circle(self.cv_image, (cx_corrected, cy), 5, (0, 255, 255), -1)
                    # cv2.putText(self.cv_image, f"({cx_corrected},{cy})", (cx_corrected + 10, cy - 10),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    msg_debug = self.bridge.cv2_to_imgmsg(self.cv_image, encoding='bgr8')
                    self.debug_publisher.publish(msg_debug)

                    # 透視變換
                    src_pts = np.float32([
                        [new_x_corrected, new_y],
                        [new_x_corrected + new_w, new_y],
                        [new_x_corrected + new_w, new_y + new_h],
                        [new_x_corrected, new_y + new_h]
                    ])

                    dst_width, dst_height = new_w, new_h
                    dst_pts = np.float32([
                        [0, 0],
                        [dst_width - 1, 0],
                        [dst_width - 1, dst_height - 1],
                        [0, dst_height - 1]
                    ])

                    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                    warped = cv2.warpPerspective(self.cv_image, M, (dst_width, dst_height))

                    # 轉灰階+二值化
                    n_gray_img = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                    _, n_binary_img = cv2.threshold(n_gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 
                    n_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    n_binary_img = cv2.erode(n_binary_img, n_kernel)
                    n_binary_img = cv2.dilate(n_binary_img, n_kernel)
                    # n_blurred_img = cv2.GaussianBlur(n_binary_img, (3, 3), 0)
                    # 找輪廓
                    n_contours, _ = cv2.findContours(n_binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                    img_height, img_width = warped.shape[:2]
                    img_area = img_height * img_width

                    rectangles = []

                    for cnt in n_contours:
                        area = cv2.contourArea(cnt)
                        if area < 100 or area > img_area / 2:
                            continue
                        epsilon = 0.02 * cv2.arcLength(cnt, True)
                        approx = cv2.approxPolyDP(cnt, epsilon, True)
                        if len(approx) == 4 and cv2.isContourConvex(approx):
                            x, y, w, h = cv2.boundingRect(approx)
                            aspect_ratio = w / h if h != 0 else 0
                            if 0.5 < aspect_ratio < 2:
                                rectangles.append(cnt)
                                cv2.drawContours(warped, [approx], -1, (0, 255, 0), 2)

                    msg_warped = self.bridge.cv2_to_imgmsg(warped, encoding='bgr8')
                    self.img_publisher.publish(msg_warped)

            '''
            if not hasattr(self, 'cv_image') or not hasattr(self, 'cv_depth'):
                return

            # Depth Camera Intrinsics
            fx = 427.5506286621094
            fy = 427.5506286621094
            cx = 429.2327575683594
            cy = 236.95892333984375

            height, width = self.cv_depth.shape
            depth = self.cv_depth.astype(np.float32) / 1000.0  # mm → m

            points = []
            colors = []

            # 遍歷深度圖
            for v in range(0, height, 2):
                for u in range(0, width, 2):
                    z = depth[v, u]
                    if z == 0 or np.isnan(z):
                        continue
                    if z < 0.2 or z > 0.9:  # 只在 0.2~1.0m 之間保留
                        continue
                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy
                    points.append([x, y, z])
                    colors.append(self.cv_image[v, u] / 255.0)

            if len(points) < 100:
                self.get_logger().warn("Not enough valid depth points")
                return

            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(points)
            pc.colors = o3d.utility.Vector3dVector(colors)

            # 平面偵測
            plane_model, inliers = pc.segment_plane(distance_threshold=0.01,
                                                    ransac_n=3,
                                                    num_iterations=1000)
            [a, b, c, d] = plane_model
            self.get_logger().info(f"Plane equation: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")

            # 建立 mask
            inlier_pc = pc.select_by_index(inliers)
            inlier_points = np.asarray(inlier_pc.points)

            mask = np.zeros((height, width), dtype=np.uint8)
            for pt in inlier_points:
                u = int((pt[0] * fx) / pt[2] + cx)
                v = int((pt[1] * fy) / pt[2] + cy)
                if 0 <= u < width and 0 <= v < height:
                    mask[v, u] = 255

            mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)

            result = cv2.bitwise_and(self.cv_image, self.cv_image, mask=mask)

            msg = self.bridge.cv2_to_imgmsg(result, encoding='bgr8')
            self.img_publisher.publish(msg)
        '''
        except Exception as e:
            self.get_logger().error(f'Error in publish_processed_image: {e}')
        




def main(args=None):
    rclpy.init(args=args)
    node = Coffee()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()