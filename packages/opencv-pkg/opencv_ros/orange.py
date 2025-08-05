import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

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

class Orange(Node):
    def __init__(self):
        super().__init__('orange_node')
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
        self.img_publisher = self.create_publisher(Image, 'orange_detection', 10)

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
            # 創一個篩選出橘色的mask
            lower_bound = np.array([0, 50, 100]) # 三個參數分別對應：色相, 飽和度, 亮度
            upper_bound = np.array([40, 255, 255])
            mask = cv2.inRange(self.cv_image, lower_bound, upper_bound)
            orange_img = cv2.bitwise_and(self.cv_image, self.cv_image, mask=mask)
            gray_img = cv2.cvtColor(orange_img, cv2.COLOR_BGR2GRAY)
            _, binary_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            # 侵蝕後膨脹降低雜訊
            binary_img = cv2.erode(binary_img, kernel)
            binary_img = cv2.dilate(binary_img, kernel)
            blurred_img = cv2.GaussianBlur(binary_img, (15, 15), 0)
            contours, _ = cv2.findContours(blurred_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # 找到最大的輪廓並畫出邊界框
                max_region = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(max_region)
                cv2.rectangle(self.cv_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
                # 計算方框中心點
                cx = x + w // 2
                cy = y + h // 2

                depth_value = self.cv_depth[cy, cx] * 0.1
                self.get_logger().info(f"Depth at ({x}, {y}): {depth_value:.3f} cm")

                # 在圖上標記中心點
                cv2.circle(self.cv_image, (cx, cy), 3, (0, 0, 255), -1)
                cv2.putText(self.cv_image, f"({cx},{cy})", (cx + 10, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            msg = self.bridge.cv2_to_imgmsg(self.cv_image, encoding='bgr8')
            self.img_publisher.publish(msg)
        except Exception as e:
            self.get_logger().error(f'Error publishing processed image: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = Orange()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()