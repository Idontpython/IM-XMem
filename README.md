# IM-XMem: Interactive Larvae Ventricle Segmentation Network

This repository contains the official PyTorch implementation of **IM-XMem**. 

Our framework introduces Mamba-based modules into the XMem architecture to achieve highly accurate, robust, and interactive segmentation of **larvae ventricles** in microscopy videos.This demonstrates strong generalization performance on challenging biological objects.

## 🎥 Interactive Demonstration

Watch IM-XMem in action, performing robust segmentation on a moving larvae ventricle:

https://github.com/user-attachments/assets/d4040cdb-9809-4bb6-84b2-c6b7004baaa3


## ⚙️ Environment Setup

**1. Clone the repository:**
```bash
git clone https://github.com/Idontpython/IM-XMem.git
cd IM-XMem
```

**2. Create a virtual environment and install dependencies:**
```bash
# Note: Python 3.11.11 is used in our development environment
conda create -n imxmem python=3.11.11
conda activate imxmem
pip install -r code/requirements.txt
```

**3. Install Mamba (Crucial for our state space models):**
Note on Wheel Selection: Since mamba installation heavily depends on the CUDA and Python version, please manually download the specific wheel (.whl) files for causal_conv1d and mamba_ssm that match your system (e.g., CUDA 11.8 and Python 3.11).
```bash
# Replace the .whl filenames below with your specifically downloaded versions
pip install causal_conv1d-*.whl
pip install mamba_ssm-*.whl
```


## 📁 Data Preparation & Pre-trained Weights 

To run the inference or training, you need to download the required datasets and weights:

**1. Weights:** Download the pre-trained weights from [https://drive.google.com/file/d/1G-ufRBjs-M8if4AAhx7IDypusfqHA7-c/view?usp=drive_link] and place them in the `code/saves/` directory.

**2. Data:** Download the larvae ventricle data from [https://drive.google.com/file/d/1akvvoqX2r7v46qszK6nhtdVGPAsehuj1/view?usp=drive_link] and place it in the `Data/` directory.


The folder structure should look exactly like this before running the code:

```text
IM-XMem/
├── code/
│   ├── saves/                 <-- Put weights here
│   ├── interactive_demo.py
│   └── ...
├── Data/                <-- Put dataset here
└── wheels/                    <-- Put mamba .whl files here
```

## 🚀 Quick Start (Inference)


Run the following command to evaluate the model on the test set. (For detailed command-line arguments, please refer to the comments inside `interactive_demo.py`):

```bash
cd code
python interactive_demo.py \
  --images <path_to_input_image_sequence> \
  --workspace <path_to_output_directory> \
  --model <path_to_main_model_weight.pth> \
  --s2m_model <path_to_s2m_model_weight.pth> \
  --size -1
```
Command Line Arguments Explanation:

**--images**: Path to the directory containing your target video frames (e.g., JPEG or PNG images).

**--workspace**: Path to the directory where the segmentation results and GUI outputs will be saved.

**--model**: Path to the main IM-XMem pre-trained weight file.

**--s2m_model**: Path to the secondary structural model (s2m) weight file.

**--size**: Target resolution for inference. (Use -1 to keep the original image size).

## 📊 Evaluation
After the inference is complete, convert the output masks into a binary format, and then evaluate the accuracy (e.g., J & F scores) using our provided scripts:

```bash
python mask2binary.py
python eval_jf.py
```

## 🖥️ Headless Server Setup (Optional)

If you are running the interactive demo on a remote server without a Graphical User Interface (GUI), you can establish a virtual display and a VNC tunnel using the following steps:

**Step 1: Set up the virtual display on the remote server**

```bash
# Start a virtual framebuffer
nohup Xvfb :5 -screen 0 1920x1080x24 &
export DISPLAY=:5

# Start a lightweight window manager
nohup fluxbox &

# Start the VNC server
nohup x11vnc -nopw -forever -noxdamage -rfbport 5900 &
```

**Step 2: Create an SSH tunnel from your local machine
Open a terminal on your local computer and run the following command (Please replace username@server_ip with your actual server credentials):**

```bash
ssh -L 5900:localhost:5900 username@server_ip
```

**Step 3: Connect via VNC Client
Open any VNC Client (e.g., VNC Viewer) on your local computer and connect to the following address:**

```bash
localhost:5900
```

## Contact
If you have any issues or questions about this paper or need assistance with reproducing the results, please contact me.

Minghao Wang

Shanghai Ocean University

Email:  m240751938@st.shou.edu.cn





