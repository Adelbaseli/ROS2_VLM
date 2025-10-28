import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
from PIL import Image as PILImage
import io
import google.generativeai as genai
import time

# ---------------------------
# CONFIGURE GEMINI
# ---------------------------
genai.configure(api_key="AIzaSyACo5D9BvnNxK17cDDh_dy38d3K93myiW0")
model = genai.GenerativeModel("gemini-2.5-flash")

# ---------------------------
# HELPER FUNCTION
# ---------------------------
def answer_with_gemini(image_cv):
    """Ask Gemini a question with an image (ultra-short)."""
    pil_img = PILImage.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    prompt = "Answer in a few words: What can you see in the image?"
    inputs = [prompt, {"mime_type": "image/jpeg", "data": img_bytes}]
    response = model.generate_content(inputs)
    return response.text.strip()

# ---------------------------
# ROS 2 NODE
# ---------------------------
class GeminiVisionNode(Node):
    def __init__(self):
        super().__init__('gemini_vision_node')
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            Image,
            'image_raw',  # Change to your camera topic
            self.image_callback,
            10
        )
        self.publisher_ = self.create_publisher(String, '/gemini/image_description', 10)
        self.last_time = time.time()

    def image_callback(self, msg):
        if time.time() - self.last_time < 5.0:
            return
        self.last_time = time.time()

        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        description = answer_with_gemini(cv_image)
        self.publisher_.publish(String(data=description))
        self.get_logger().info(f"Published: {description}")

def main(args=None):
    rclpy.init(args=args)
    node = GeminiVisionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

