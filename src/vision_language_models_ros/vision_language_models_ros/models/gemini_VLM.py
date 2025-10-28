import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
from PIL import Image as PILImage
import io
import google.generativeai as genai

# ---------------------------
# CONFIGURE GEMINI
# ---------------------------
genai.configure(api_key="AIzaSyACo5D9BvnNxK17cDDh_dy38d3K93myiW0")
model = genai.GenerativeModel("gemini-2.5-flash")

# ---------------------------
# HELPER FUNCTION
# ---------------------------
def answer_with_gemini(image_cv, run_number):
    """Ask Gemini a question optimized for navigation."""
    pil_img = PILImage.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    prompt = (
        "You are a robot vision assistant. Analyze the image for safe navigation. "
        "Answer very concisely with key info: "
        "- humans and their locations "
        "- obstacles (desks, chairs, bags, cables, walls, doors) "
        "- is the path ahead clear or blocked? "
        "- only relevant info for navigation, max 2-3 short sentences."
    )

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
        self.latest_image = None
        self.run_counter = 1

        # Subscribe to camera images
        self.image_sub = self.create_subscription(
            Image,
            'image_raw',  # Change to your camera topic
            self.image_callback,
            10
        )

        # Publisher for navigation info
        self.publisher_ = self.create_publisher(String, '/gemini/image_description', 10)

        # Timer to run every 5 seconds
        self.timer = self.create_timer(5.0, self.timer_callback)

    def image_callback(self, msg):
        """Save the latest image for the timer to process."""
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def timer_callback(self):
        """Process the latest image every 5 seconds."""
        if self.latest_image is None:
            return

        run_number = self.run_counter
        self.run_counter += 1

        description = answer_with_gemini(self.latest_image, run_number)
        self.publisher_.publish(String(data=description))
        self.get_logger().info(f"Published: [Run {run_number}] {description}")


def main(args=None):
    rclpy.init(args=args)
    node = GeminiVisionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

