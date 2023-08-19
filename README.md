# Runway Detection and Localization in Aerial Images using Hybrid Learning


## Project Description

In an effort to address landing accidents caused by poor visibility conditions, this project presents an innovative method for runway detection and localization using a hybrid learning approach. Foggy weather conditions often hinder the clarity of aerial images, reducing visibility and potentially leading to accidents during landing.

The project begins by employing digital image processing techniques to enhance the visibility of degraded images affected by atmospheric conditions such as fog. Subsequently, a deep learning technique is employed to detect the presence of a runway in the image. Once the runway is detected, a machine learning algorithm is utilized for localizing the exact path of the runway. The Hough transform algorithm is employed to highlight the runway lines, aiding pilots in ensuring a safe aircraft landing. By following the localized path, the project aims to reduce landing accidents by up to 90%.

## Key Features

- Removal of hazy content from aerial images using digital image processing.
- Deep learning-based runway detection.
- Machine learning-based runway localization.
- Use of the Hough transform algorithm for highlighting runway lines.
- Enhanced safety during aircraft landings.

## Technologies Used

- TensorFlow framework
- OpenCV
- NumPy
- Matplotlib
- Pandas
- Keras
- Python

### Hardware Requirements

- Processor: Pentium II class, 450 MHz or above
- Random Access Memory: 1GB or above
- Hard Disk Memory: 240 GB
- Monitor: Color Monitor
- Video: 800X600 resolution, 256 colors

### Software Requirements

- Operating System: Windows 10
- Front-end: Tkinter
- Integrated Development Environment: PyCharm

## Installation and Setup

1. Clone this repository to your local machine.
2. Install the required libraries and frameworks using the following command:
   
   pip install tensorflow opencv-python numpy matplotlib pandas keras
   
3. Open the project in PyCharm and run the main script.

## Usage

1. Launch the application.
2. Upload the aerial image for runway detection and localization.
3. View the enhanced image after hazy content removal.
4. Detect the presence of the runway using deep learning.
5. Localize the exact path of the runway using machine learning.
6. Observe highlighted runway lines using the Hough transform algorithm.
