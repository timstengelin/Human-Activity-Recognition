# Team 15
* Florian Schatz (E-Mail: st191622@stud.uni-stuttgart.de)
* Tim Stengelin (E-Mail: st190770@stud.uni-stuttgart.de)

# Project I: Diabetic Retinopathty Detection
Diabetic retinopathy (DR) is a serious ocular condition that can significantly compromise the vision of individuals with diabetes. In this study, we utilized multiple detection models based on deep convolutional neural networks and compared their performance. All models were trained using the IDRiD dataset. Additionally, we employed data augmentation, transfer learning and ensemble learning techniques, enabling our best model to achieve an accuracy of 84%. Ultimately, our findings suggest that the size of the dataset is the most critical factor influencing detection performance.

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

The test accuracy of the composed model from MobileNetV2, EfficientNetB0 with Augmentation and pre-trained EfficientNetB3, DenseNet201 results in: 0.80

## How to run the Code?
* **Step 0:** Download the Indian Diabetic Retinopathy Image Dataset (IDRiD) from [here](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid).
* **Step 1:** Adjust the paths of `load.img_dir` and `load.csv_dir` in `config.gin` to the images/labels folder of the data directory, the IDRiD dataset.
* **Step 2a:** Configure the program manually to be able to use the full range of functions.
  * **Step 2a.a:** Train a model
    * **Step 2a.a.1:** Set the parameter `main_logic.model_names` in `config.gin` to `['MobileNetV2']`/`['EfficientNetB0']`/`['EfficientNetB3_pretrained']`/`['DenseNet201_pretrained']` to train the model variant MobileNetV2/EfficientNetB0/EfficientNetB3/DenseNet201. The latter two model variants use a transfer learning apprach.
    * **Step 2a.a.2:** Set the parameter `main_logic.mode` in `config.gin` to `'train'`
    * **Step 2a.a.3:** Set the remaining parameters in `config.gin`.
  * **Step 2a.b:** Evaluate a model
    * **Step 2a.b.1:** Set the parameter `main_logic.model_names` in `config.gin` to `['MobileNetV2']`/`['EfficientNetB0']`/`['EfficientNetB3_pretrained']`/`['DenseNet201_pretrained']`/`['ComposedModel']` to evaluate the model variant MobileNetV2/EfficientNetB0/EfficientNetB3/DenseNet201 or a composed model. The model you want to evaluate has to be trained already!
    * **Step 2a.b.2:** Set the parameter `main_logic.mode` in `config.gin` to `'evaluate'`
    * **Step 2a.b.3:** Set the remaining parameters in `config.gin`.
  * **Step 2a.c:** Tune a model
    * **Step 2a.c.1:** Set the parameter `main_logic.model_names` in `config.gin` to `['MobileNetV2']`/`['EfficientNetB0']`/`['EfficientNetB3_pretrained']`/`['DenseNet201_pretrained']` to tune the model variant MobileNetV2/EfficientNetB0/EfficientNetB3/DenseNet201.  The model you want to tune has to be trained already!
    * **Step 2a.c.2:** Set the parameter `main_logic.mode` in `config.gin` to `'tune'`
    * **Step 2a.c.3:** Set the remaining parameters in `config.gin`.
  * **Step 2a.d:** Create a composed model
    * **Step 2a.d.1:** E.g., set the parameter `main_logic.model_names` in `config.gin` to `['MobileNetV2', 'EfficientNetB0', 'EfficientNetB3_pretrained', 'DenseNet201_pretrained']` to create a composed model from the model variants MobileNetV2, EfficientNetB0, EfficientNetB3, DenseNet201.  The models you want to compose have to be trained already.
    * **Step 2a.d.2:** Set the parameter `main_logic.mode` in `config.gin` to `'create_ensemble_model'`
    * **Step 2a.d.3:** Set the remaining parameters in `config.gin`.
* **Step 2b:** Use a Quickstart configuration to train or evaluate the EfficientNetB3 model and get the best results we achieved in this project.
  * **Step 2b.a:** Run `run_quickstart_train_efficientnetb3.sh` to train the EfficientNetB3 model variant.
  * **Step 2b.b:** Run `run_quickstart_evaluate_efficientnetb3.sh` to evaluate the EfficientNetB3 model variant you trained in Step2b.a. 

Note: Sequences of letters, e.g. 2a, 2b, represent alternatives. Number sequences, e.g. 0., 1., represent sequences!

# Project II: Human Activity Recognition
Human Activity Recognition (HAR) is a area of research in the field of ubiquitous computing and machine learning. It involves the automatic identification of activities performed by individuals through the analysis of data collected from various sensors. This work has the goal to implement, train and evaluate a classifier for those activities. In this study, we utilized multiple detection models based on deep recurrent neural networks and compared their performance. All models were trained using the HAPT dataset. Additionally, we employed hyperparameter optimization, enabling our best model to achieve an accuracy of 92%. Ultimately, our findings suggest the variance of the dataset is the most critical factor influencing detection performance and thus the reworking of the raw data is the most critical part.

## Content
* **Dataset:** Utilizes the Human Activities and Postural Transitions Dataset (HAPT) containing activity data from 30 volunteers from 19 to 48 years.
* **Input Pipeline:** Employs TFRecord for efficient data loading and processing. Calculate class imbalance within trainig dataset.
* **Weighted loss:** Calculated class imbalance is used to weight the loss within training, to prevent overfitting on the higher represented classes.
* **Architecture Families:** Different mix of well known recurrent-NN structures, used for time sequence processing.
* **Metrics:** Evaluates performance using accuracy, confusion matrix and visualization of sequence classification.
* **Hyperparameter Tuning:** Enhance the performance of the individual model variants by tuning different aspects like window-size, window-shift and size of the model.

##  Results
Test accuracies for different model variants (All models include a two layer dense classifier at the end using SoftMax):

| **Architecture Familiy** | **Test Accuracy** |
|--------------------------|-------------------|
| LSTM                     | 91.5%             |
| GRU                      | 91%               |
| Bidirectional LSTM       | 91.8%             |
| Conv + LSTM              | 90.6%             |

For the best model, following confusion matrix could be achieved:

BILD Confusion Matrix

With the model, as visible in the picture, a valid clasification of activities could be achieved.

## How to run the Code?
TODO

