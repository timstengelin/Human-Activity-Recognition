# Team 15
- Florian Schatz  
  st191622@stud.uni-stuttgart.de
- Tim Stengelin  
  st190770@stud.uni-stuttgart.de

# Project I: Diabetic Retinopahty Detection
Diabetic retinopathy (DR) is a serious ocular condition that can significantly compromise the vision of individuals with diabetes. In this study, we developed multiple detection models based on deep convolutional neural networks and compared their performance. All models were trained using the IDRID dataset. Additionally, we employed data augmentation, transfer learning and ensemble learning techniques, enabling our best model to achieve an accuracy of 84%. Ultimately, our findings suggest that the size of the dataset is the most critical factor influencing detection performance.

## Content
* **Dataset:** Utilizes the Indian Diabetic Retinopathy Image Dataset (IDRiD) comprising 516 images.
* **Input Pipeline:** Employs TFRecord for efficient data loading and processing.
* **Image Preprocessing:** Involves cutting black edges, resizing, and normalization. Implements various data augmentation techniques to enhance model robustness.
* **Architecture Families:** Leverages well-known CNN architectures such as MobileNet, EfficientNet, and DenseNet.
* **Metrics:** Evaluates performance using binary accuracy and confusion matrix analyses.
* **Transfer Learning:** Applies transfer learning in model variants EfficientNetB3 and DenseNet201.
* **Ensemble Learning:** Constructs composed model from MobileNetV2, EfficientNetB0 with Augmentation and pre-trained EfficientNetB3, DenseNet201.
* **Hyperparameter Tuning:** Enhance the performance of the individual model variants.
* **Deep Visualization Techniques:** Use methods Grad-CAM, Guided Backpropagation, Guided Grad-CAM, and Integrated Gradients to interpret model decisions.


##  Results
Test accuracies for different model variants:

| **Architecture Familiy** | **Model Variants** | **Augmentation** | **Transfer Learning** | **Test Accuracy** |
|--------------------------|--------------------|------------------|-----------------------|-------------------|
| MobileNet                | MobileNetV2        | False            | False                 | 0.75              |
| "                        | "                  | True             | False                 | 0.80              |
| EfficientNet             | EfficientNetB0     | False            | False                 | 0.73              |
| "                        | "                  | True             | False                 | 0.80              |
| "                        | EfficientNetB3     | True             | True                  | 0.84              |
| DenseNet                 | DenseNet201        | True             | True                  | 0.79              |

Test accuracy of composed model from MobileNetV2, EfficientNetB0 with Augmentation and pre-trained EfficientNetB3, DenseNet201 results in: 0.80

## How to run the Code?
TODO

# Project II: Human Activity Recognition
TODO

## Content
TODO

##  Results
TODO

## How to run the Code?
TODO

