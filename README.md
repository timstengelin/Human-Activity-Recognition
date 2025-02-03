# Team 15
- Florian Schatz  
  st191622@stud.uni-stuttgart.de
- Tim Stengelin  
  st190770@stud.uni-stuttgart.de

# Project I: Diabetic Retinopahty Detection
Diabetic retinopathy (DR) is a serious ocular condition that can significantly compromise the vision of individuals with diabetes. In this study, we developed multiple detection models based on deep convolutional neural networks and compared their performance. All models were trained using the IDRID dataset. Additionally, we employed data augmentation, transfer learning and ensemble learning techniques, enabling our best model to achieve an accuracy of 84%. Ultimately, our findings suggest that the size of the dataset is the most critical factor influencing detection performance.

## Content
TODO

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

Test accuracy of composed model from MobileNetV2 and EfficientNetB0 with Augmentation, pre-trained EfficientNetB3 and DenseNet201 results in: 0.80

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

