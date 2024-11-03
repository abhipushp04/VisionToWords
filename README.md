# Exploring Lightweight and Complex Architectures in Image Captioning: A Performance Analysis on Flickr8k
---

This project explores and evaluates different deep learning architectures for generating captions for images, using the Flickr8k dataset. Three architectures are implemented and compared in terms of accuracy, contextual relevance, and computational efficiency:

1. **MobileNetV3Large with Bidirectional LSTM**
2. **DenseNet201 with LSTM**
3. **BLIP (Bootstrapping Language-Image Pre-training)**

The goal is to understand the trade-offs between these architectures in image captioning tasks, providing insights into model selection based on specific use cases. And make a image caption generation model

## Project Overview

Image captioning, the process of generating descriptive captions for images, is valuable in various applications, such as accessibility technology, social media, and autonomous systems. This repository includes code to preprocess the Flickr8k dataset, train the models, generate captions, and evaluate them using BLEU scores.

### Key Features

- **Dataset**: Flickr8k dataset, containing 8,000 images and multiple captions per image.
- **Models**: Comparison of CNN-RNN architectures (MobileNetV3Large and DenseNet201) with an advanced, pre-trained BLIP model.
- **Evaluation Metrics**: BLEU score for quantitative evaluation and qualitative analysis of caption relevance.
- **Tools**: Includes code for image processing, model training, caption generation, and performance visualization.
- **Training Environment**: Models trained on Kaggle T4 GPUs with specific parameters for optimization and stability.

## Architecture Details

| Model             | Speed               | Accuracy     | Context-Aware Captions | Computational Requirements |
|-------------------|---------------------|--------------|-------------------------|----------------------------|
| MobileNetV3Large  | High                | 87%         | Moderate                | Low                         |
| DenseNet201       | Moderate            | 30%         | Low                     | High                        |
| BLIP              | Moderate            | Best (approx. 90%) | High                    | Extensive pre-training      |

## Challenges Faced

The main challenges in developing robust image captioning models include balancing computational efficiency and accuracy, fine-tuning parameters for optimal performance, and ensuring contextually accurate captions. Details on these challenges and solutions are available in the project code and documentation.

### Training Notebook

For detailed code and training procedures, visit the Kaggle notebook: [Image Captioning Notebook](https://www.kaggle.com/code/abhipushpmaurya/caption-01).

## Future Work

Future research could explore larger datasets, advanced attention mechanisms, and optimization for real-time applications on edge devices. Further fine-tuning and leveraging transformer-based architectures could enhance context-awareness in captions.
