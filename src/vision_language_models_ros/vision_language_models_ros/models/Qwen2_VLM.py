#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from PIL import Image as PILImage
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import time

class VisualQuestioningNode(Node):
    def __init__(self):
        super().__init__('visual_questioning_node')
        self.get_logger().info("Initializing Qwen2-VL-2B-Instruct model...")

        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"The model will run on: {self.device}")

        model_id = "Qwen/Qwen2-VL-2B-Instruct"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            device_map="auto"
        )

        self.get_logger().info(f"Selected model device: {self.model.device}")

        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            10
        )
        self.publisher_ = self.create_publisher(String, '/robot/vision_answer', 10)

        # Timer variables
        self.last_run_time = 0.0
        self.interval = 5.0  # seconds between model runs
        self.run_counter = 0
        self.latest_image = None

        self.get_logger().info("Node ready. Listening on /image_raw")

    def image_callback(self, msg):
        """Save the most recent image; only process every self.interval seconds."""
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        current_time = time.time()

        if current_time - self.last_run_time >= self.interval:
            self.last_run_time = current_time
            if self.latest_image is not None:
                self.run_inference(self.latest_image)

    def run_inference(self, cv_image):
        self.run_counter += 1
        image = PILImage.fromarray(cv_image[:, :, ::-1])

        # Simple plain-text navigation prompt
        prompt = (
            "You are a robot vision assistant observing the scene. "
            "Describe the environment in one concise sentence. "
            "Specifically, assess if the robot can move safely in each direction: front, left, right. "
            "Mention visible obstacles (doors, chairs, walls, cables, etc.) and humans. "
            "Based on safety, give a short movement suggestion: move forward, turn left, turn right, or stop. "
            "Always prioritize safety over movement. "
            "Do not output JSON, only one clear sentence for the robot to follow."
        )

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        text_input = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=image, text=text_input, return_tensors="pt").to(self.model.device)

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, max_new_tokens=64)

        output_text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]

        # Directly publish plain text
        message = f"Run #{self.run_counter}: {output_text}"
        self.get_logger().info(message)
        self.publisher_.publish(String(data=output_text))


def main(args=None):
    rclpy.init(args=args)
    node = VisualQuestioningNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

