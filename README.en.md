# Forward Collision Warning System

This project proposes a forward collision warning (FCW) system that integrates UFLDv2 lane detection, YOLO object detection, and monocular distance estimation, combined with a visualization interface. The system supports inference acceleration with PyTorch and TensorRT, and adopts a multi-threaded design to achieve both real-time performance and high accuracy.

------

## I. Software Architecture

#### 📌 The system adopts a modular design, mainly consisting of the following core modules:

- **Deep Learning Framework:** PyTorch + TensorRT (for inference acceleration)
- **Computer Vision:** OpenCV + PIL
- **Lane Detection:** Ultra-Fast-Lane-Detection-v2
- **Object Detection:** YOLOv5 / YOLOv8 / YOLOv11-seg
- **UI Development:** PyQt + custom futuristic components
- **Configuration Management:** JSON + Python class encapsulation

------

## II. Installation Guide

#### 📌 Clone the repository

```
git clone https://github.com/HHX300/forward-collision-warning-system.git
cd forward-collision-warning-system
```

#### 📌 Create a virtual environment

```
conda create -n fcw python=3.10
```

#### 📌 Activate the environment

```
conda activate fcw
```

#### 📌 Install dependencies

- **Install PyTorch dependencies**

  ```
  # Choose the appropriate version based on your CUDA/CPU setup,
  # or download from the official PyTorch website.
  # CUDA 11.8
  pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118
  # CUDA 12.1
  pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121
  # CUDA 12.4
  pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124
  # CPU only
  pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cpu
  ```

- **Install other dependencies**

  ```
  pip install -r requirements.txt
  ```

------

## III. Usage Instructions

#### 📌 Run the program

```
python main.py
```

#### 📌 Demo

![show](demo/demo1.gif)