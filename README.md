# Dr. Leaf: Intelligent Plant Disease Detection System

Dr. Leaf (also known as Dr. Feuille) is a specialized AI ecosystem that combines conversational natural language processing with advanced computer vision to diagnose plant diseases. The system integrates a local Large Language Model (LLM) with a dedicated deep learning classifier through a custom communication protocol.

## Project Overview

The core objective of Dr. Leaf is to bridge the gap between complex neural network analysis and end-user accessibility. By utilizing a ChatBot interface, users can interact naturally with the system, upload leaf images, and receive expert-level diagnostic feedback with associated confidence scores.

## System Architecture

The project is built on a three-layered architecture designed for modularity and real-time performance.


### 1. Conversation Layer (Llama 3)
The user interface is powered by Llama3:instruct, a 7B parameter model optimized for following complex instructions.
* **Role**: Acts as the primary interface, handling general plant-related inquiries and triggering the image analysis workflow when a filename is detected.
* **Prompt Engineering**: The LLM is constrained to three specific tasks: detecting image requests, reformulating technical results into helpful advice, and maintaining a professional conversational tone.

### 2. Protocol Layer (Model Context Protocol - MCP)
A custom-implemented bridge that enables the conversational AI to communicate with the specialized vision model.
* **JSON Exchange**: The ChatBot intercepts filenames and sends a structured JSON request (`{"action": "analyse_image", "chemin": "filename.jpg"}`) to the analysis server.
* **Orchestration**: It manages the two-stage interaction pattern, ensuring the technical output from the vision model is successfully passed back to the LLM for final natural language formulation.


### 3. Analysis Layer (MobileNetV2 Specialist)
The "brain" of the diagnostic system, consisting of a high-performance Convolutional Neural Network (CNN).
* **Base Model**: MobileNetV2, chosen for its balance of high accuracy and computational efficiency on edge devices.
* **Custom Classifier Head**: Features a mapping from 1280 features to 512, followed by a 28-class output layer with dropout (0.4) to prevent overfitting.
* **Dual-Stage Training**: The model was first trained on the laboratory-controlled PlantVillage dataset and subsequently fine-tuned on the real-world PlantDoc dataset to ensure robustness against noise and lighting variations.


## Technical Performance

The system was rigorously tested on both benchmark datasets and "noisy" real-world images sourced from the internet.

| Metric | Value |
| :--- | :--- |
| **Validation Accuracy (PlantDoc)** | 85.23% |
| **Real-World Test Accuracy** | 100% (on unseen noisy images) |
| **Typical Confidence Range** | 50% - 80% |
| **Supported Classes** | 28 plant disease categories |


## Methodology and Robustness

To achieve reliable performance in natural environments, several optimization strategies were employed:
* **Data Augmentation**: Images were transformed with random rotations (+30°), flips, brightness/contrast adjustments, and Gaussian blur.
* **Regularization**: Implementation of weight decay ($5 \times 10^{-4}$) and dropout layers to mitigate the risk of overfitting common in small-scale agricultural datasets.
* **Layer Freezing**: The first 10 layers of MobileNetV2 were frozen to preserve general feature extraction capabilities learned from ImageNet.

## Installation and Setup

### Prerequisites
* Python 3.8+
* PyTorch & Torchvision
* Ollama (for the local LLM runtime)
* Flask (for the MCP server)

### Running the System
1. **Pull the LLM**: `ollama pull llama3:instruct`
2. **Start the MCP Server**: `python mcp_server.py` (Launches on port 5000)
3. **Start the ChatBot**: `python Chatbot.py`
