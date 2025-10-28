#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from PIL import Image as PILImage
import torch
from transformers import AutoModel, AutoTokenizer
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import time

# ImageNet normalization
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def dynamic_preprocess(image, image_size=448, use_thumbnail=True):
    main_image = image.resize((image_size, image_size))
    return [main_image]

def load_image(image: PILImage.Image, input_size=448):
    """Preprocess a PIL image for CogAlign."""
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True)
    pixel_values = [transform(img) for img in images]
    return torch.stack(pixel_values)

class VisualQuestioningNode(Node):
    def __init__(self):
        super().__init__('visual_questioning_node')

        model_id = "Salesforce/cogalign-internvl2_5-mpo-1b"
        self.get_logger().info(f"Initializing CogAlign model ({model_id})...")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {self.device}")

        self.model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            use_flash_attn=False
        ).eval().to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_fast=False
        )

        self.get_logger().info(f"Model loaded on {self.device}")

        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            10
        )
        self.publisher_ = self.create_publisher(String, '/robot/vision_answer', 10)

        # Timing and state
        self.last_run_time = 0.0
        self.interval = 20  # seconds
        self.run_counter = 0
        self.latest_image = None

        # Scene memory buffer
        self.previous_description = ""  # Store previous frame description

        self.get_logger().info("Visual Questioning Node is ready.")

    def image_callback(self, msg):
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        current_time = time.time()

        if current_time - self.last_run_time >= self.interval:
            self.last_run_time = current_time
            if self.latest_image is not None:
                self.run_inference(self.latest_image)

    def run_inference(self, cv_image):
        self.run_counter += 1

        pil_image = PILImage.fromarray(cv_image[:, :, ::-1]).convert("RGB")
        pixel_values = load_image(pil_image).to(
            torch.bfloat16 if self.device.type=="cuda" else torch.float32
        ).to(self.device)

        prompt_text = (
            "You are a robot visual perception and navigation assistant. Analyze the current image "
            "and previous frame (if available). Divide the image into 9 regions numbered 1 to 9 "
            "(top-left=1, top-center=2, top-right=3, middle-left=4, middle-center=5, middle-right=6, "
            "bottom-left=7, bottom-center=8, bottom-right=9). For each detected object and human, provide:\n"
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
            "5. Comparison with previous frame: newly appeared objects/humans, objects/humans no longer visible.\n\n"
            "Answer concisely in structured numbered format."
        )


        generation_config = dict(
            max_new_tokens=2048,
            do_sample=True,
            top_p=0.9,
            temperature=0.7
        )

        response = self.model.chat(
            self.tokenizer,
            pixel_values,
            f"<image>\n{prompt_text}",
            generation_config
        )

        # Update previous description
        self.previous_description = response

        # Add separators for readability
        message = f"\n========================\nRun #{self.run_counter}\n========================\n{response}\n"
        self.get_logger().info(message)
        self.publisher_.publish(String(data=response))

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

