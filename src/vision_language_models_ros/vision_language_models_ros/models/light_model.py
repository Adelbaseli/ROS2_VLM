import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
from PIL import Image as PILImage
import time
import torch

print("CUDA available:", torch.cuda.is_available())  # Should return True

# ---------------------------
# LOCAL LLM SETUP
# ---------------------------
from transformers import pipeline

# Use a local image-to-text model (faster, offline)
# Make sure you downloaded the model beforehand with `from_pretrained(local_path)`
pipe = pipeline(
    "image-to-text",
    model="Salesforce/blip-image-captioning-base",
    device=0  # 0 = first CUDA GPU
)

# ---------------------------
# HELPER FUNCTION
# ---------------------------
def answer_with_local_model(image_cv):
    """Generate ultra-short description using local model."""
    pil_img = PILImage.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
    description = pipe(
        pil_img,
        max_new_tokens=100  # short output
    )
    return description[0]['generated_text']

# ---------------------------
# ROS 2 NODE
# ---------------------------
class LocalVLMNode(Node):
    def __init__(self):
        super().__init__('local_vlm_node')
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            Image,
            'image_raw',  # Camera topic
            self.image_callback,
            10
        )
        self.publisher_ = self.create_publisher(String, '/vlm/image_description', 10)
        self.last_time = time.time()

    def image_callback(self, msg):
        # Limit processing frequency to once per second
        if time.time() - self.last_time < 1.0:
            return
        self.last_time = time.time()

        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        description = answer_with_local_model(cv_image)
        self.publisher_.publish(String(data=description))
        self.get_logger().info(f"Published: {description}")

# ---------------------------
# MAIN
# ---------------------------
def main(args=None):
    rclpy.init(args=args)
    node = LocalVLMNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

