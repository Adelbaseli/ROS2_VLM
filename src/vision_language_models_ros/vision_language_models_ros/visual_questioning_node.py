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
def answer_with_gemini(image_cv, run_number, prev_summary=None):
    """Ask Gemini a question optimized for navigation, with optional previous-frame context."""
    pil_img = PILImage.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    # Enhanced prompt
    prompt = (
        "You are a robot visual perception and navigation assistant. Analyze this image carefully "
        "and provide a structured summary for safe robot motion.\n"
        "Divide the image into 9 regions numbered 1 to 9 "
        "(top-left=1, top-center=2, top-right=3, middle-left=4, middle-center=5, middle-right=6, "
        "bottom-left=7, bottom-center=8, bottom-right=9).\n"
        "For each detected object and human, provide:\n"
        "- Name/type\n"
        "- Region number\n"
        "- Distance (very close, close, far, very far)\n"
        "- Motion direction if visible\n\n"
        "Also include:\n"
        "1. Path: clear, partially blocked, or blocked.\n"
        "2. Risk assessment (low, medium, high).\n"
        "3. Weighted control recommendations: suggest path planning vs obstacle avoidance weights "
        "based on obstacles/humans.\n"
        "4. Suggested safe movement or waypoint adjustments.\n"
        "5. Comparison with previous frame: newly appeared objects/humans, objects/humans no longer visible.\n"
        "Answer concisely in structured numbered format."
    )

    if prev_summary:
        prompt += (
            f"\nPrevious frame summary:\n{prev_summary}\n"
            "Compare this frame with the previous one and highlight changes in objects/humans and path status."
        )

    inputs = [prompt, {"mime_type": "image/jpeg", "data": img_bytes}]
    response = model.generate_content(inputs)
    return response.text.strip()


class GeminiVisionNode(Node):
    def __init__(self):
        super().__init__('gemini_vision_node')
        self.bridge = CvBridge()
        self.latest_image = None
        self.run_counter = 1
        self.prev_description = None  # buffer for previous Gemini summary

        self.image_sub = self.create_subscription(
            Image, 'image_raw', self.image_callback, 10
        )
        self.publisher_ = self.create_publisher(String, '/gemini/image_description', 10)

        # Run every 20 seconds
        self.timer = self.create_timer(20.0, self.timer_callback)

    def image_callback(self, msg):
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def timer_callback(self):
        if self.latest_image is None:
            return

        run_number = self.run_counter
        self.run_counter += 1

        # Pass previous description for temporal comparison
        description = answer_with_gemini(self.latest_image, run_number, self.prev_description)
        self.prev_description = description  # store for next iteration

        self.publisher_.publish(String(data=description))
        self.get_logger().info(f"\n========================\nRun #{run_number}\n{description}\n========================")


def main(args=None):
    rclpy.init(args=args)
    node = GeminiVisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

