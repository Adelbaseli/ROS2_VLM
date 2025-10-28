from setuptools import setup

package_name = 'vision_language_models_ros'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    install_requires=[
        'setuptools',
        'google-generativeai',
        'Pillow',
        'opencv-python',
        'rclpy',
        'cv-bridge',
    ],
    zip_safe=True,
    author='Your Name',
    entry_points={
        'console_scripts': [
            'visual_questioning_node = vision_language_models_ros.visual_questioning_node:main',
        ],
    },
)

