# TorchieVision

[![GitHub issues](https://img.shields.io/github/issues/aintburak/TorchieVision?style=for-the-badge&labelColor=blue)](https://github.com/aintburak/TorchieVision/issues) [![GitHub pull requests](https://img.shields.io/github/issues-pr/aintburak/TorchieVision?style=for-the-badge&labelColor=green)](https://github.com/aintburak/TorchieVision/pulls)  [![GitHub stars](https://img.shields.io/github/stars/aintburak/TorchieVision?style=for-the-badge&labelColor=yellow)](https://github.com/aintburak/TorchieVision/stargazers)  [![GitHub forks](https://img.shields.io/github/forks/aintburak/TorchieVision?style=for-the-badge&labelColor=orange)](https://github.com/aintburak/TorchieVision/forks)  [![GitHub watchers](https://img.shields.io/github/watchers/aintburak/TorchieVision?style=for-the-badge&labelColor=purple)](https://github.com/aintburak/TorchieVision/watchers)  ![GitHub](https://img.shields.io/github/license/aintburak/TorchieVision?style=for-the-badge)

###### tags: `Pytorch` `CNN` `ImageClassification`

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <h3 align="center">TorchieVision</h3>
  <div align="center">
    Convolutional Neural Network Project using Pytorch for Image Classification.
    <br />
    <a href="https://github.com/aintburak/TorchieVision">Repository Link</a>
    ·
    <a href="https://github.com/aintburak/TorchieVision/issues">Report Bug</a>
    ·
    <a href="https://github.com/aintburak/TorchieVision/pulls">Contribute</a>
  </div>
</div>

---


## :tada: Welcome to TorchieVision!
<!-- Add an introductory paragraph about the repository and its purpose. -->

### :book: Contents
<!-- Add links to each section. -->

1. [Project Overview](#grey_question-project-overview)
2. [Data Overview](#Data-Overview)
3. [Requirements](#computer-requirements)
4. [Usage](#key-usage)
5. [File Structure](#rocket-file-structure)
6. [Contribution](#Contribution)
7. [License](#memo-license)

---

## :grey_question: Project Overview

TorchieVision is a project aimed at providing a comprehensive solution for computer vision tasks using PyTorch. With a focus on simplicity and effectiveness, this project offers a range of functionalities to facilitate the development and deployment of computer vision models.

Key features of TorchieVision include:
- Simplified data loading and preprocessing for image datasets.
- Implementation of common computer vision models using PyTorch.
- Training, evaluation, and inference pipelines for seamless model development.
- Integration with popular datasets such as CIFAR10, ImageNet, and MNIST.
- Extensive documentation and example usage to support users at every stage of their computer vision projects.

Whether you're a beginner exploring computer vision concepts or an experienced practitioner developing advanced models, TorchieVision aims to be your go-to resource for all things related to computer vision with PyTorch.


## 📊 What about data?

When dealing with image, text, audio, or video data, standard Python packages that load data into a numpy array are often used. These arrays can then be converted into `torch.*Tensor`.

- For images, packages such as Pillow and OpenCV are useful.
- For audio, packages such as scipy and librosa are commonly used.
- For text, options include raw Python or Cython-based loading, as well as NLTK and SpaCy.

For vision tasks specifically, the `torchvision` package provides data loaders for common datasets such as ImageNet, CIFAR10, and MNIST, along with data transformers for images (`torchvision.datasets` and `torch.utils.data.DataLoader`). This package offers significant convenience and helps avoid writing boilerplate code.

For this project, you can use the CIFAR10 dataset, which consists of the following classes: 'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'. The images in CIFAR-10 are of size 3x32x32, representing 3-channel color images of 32x32 pixels.


The project consists of the following components:

1. **Data Loading**: The CIFAR-10 dataset is loaded and preprocessed using PyTorch's torchvision library.
2. **Model Definition**: A CNN model is defined using PyTorch's nn.Module API.
3. **Training**: The model is trained on the training dataset using stochastic gradient descent (SGD) optimization and cross-entropy loss.
4. **Testing**: The trained model is evaluated on the test dataset to measure its performance.
5. **Model Saving and Loading**: The trained model is saved to disk and can be loaded for future use.

![cifar10](https://imgur.com/ThgB5FZ.png)


## 👩🏼‍💻 Requirements

To run the code in this project, you'll need the following dependencies:

- Python 3.x
- PyTorch
- TorchVision
- Matplotlib
- NumPy

You can install the required packages using pip:

```bash
pip install torch torchvision matplotlib numpy
````

Additionally, make sure you have access to a CUDA-capable GPU if you wish to leverage GPU acceleration for training your neural network models.




## 🔑 Usage

1. **Clone the repository:**

    ```bash
    git clone https://github.com/your_username/TorchieVision.git
    ```

2. **Navigate to the project directory:**

    ```bash
    cd TorchieVision
    ```

3. **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Run the main.py file:**

    ```bash
    python main.py
    ```

5. **Follow the prompts or instructions displayed in the terminal to interact with the program.**


## 🚀 File Structure


The project directory structure is organized as follows:

```
TorchieVision/
│
├── pyTorchProject/
│ ├── data/ # Directory to store dataset files (automatically generated)
│ ├── main.py # Main Python script containing the PyTorch CNN project code
│ ├── cifar_net.pth # Trained model file (automatically generated after training)
│ ├── README.md # Project documentation file
│ └── requirements.txt # File containing project dependencies

```


In this structure:

- `data/`: This directory is automatically generated to store dataset files used in the project.
- `main.py`: This file contains the main Python script that implements the PyTorch CNN project.
- `cifar_net.pth`: This file is automatically generated after training and contains the trained model.
- `README.md`: This file serves as the project documentation.
- `requirements.txt`: This file contains a list of project dependencies required to run the code.


## 🦸🏻‍♀️ Contribution
Contributions to Flower4all are welcome! If you find any bugs, have feature requests, or want to contribute enhancements, please feel free to open an issue or submit a pull request.

### 🚦 How to Contribute
1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -am 'Add some feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.


## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.