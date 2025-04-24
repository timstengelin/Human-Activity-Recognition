# Human Activity Recognition
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

![d7ccfb92-7b1e-4a89-a9a0-d9da23114cf0](https://github.com/user-attachments/assets/329e4497-0df2-4b67-bb3a-a8b081e10a11)


With the model, as visible in the picture, a valid clasification of activities could be achieved.

![sequence_visualization](https://github.tik.uni-stuttgart.de/iss/dl-lab-24w-team15/assets/8809/7b8dcd0c-9291-49f4-a59c-6e04064e345b)

## How to run the Code?
* **Step 0:** Download the Human Activities and Postural Transitions Dataset (HAPT) from [here]([https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid](https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones)).
* **Step 1:** Adjust the paths of `load.data_dir` in `config.gin` to the datasets folder.
* **Step 2:** Adjust the parameters within `config.gin` so that they fit your request.
   * **Step 2a:** Within 'INPUT PIPELINE' configure the window size, shift and batch size.
   * **Step 2b:** Within 'LOCAL TRAINING PARAMETERS' configure the typicall local training paramerters.
   * **Step 2c:** Within 'ARCHITECTURE PARAMETERS' you can adjust the model parameters.
   * **Step 2d:** Within 'VISUALIZATION PARAMETERS' you can adjust the size of the sequence visualization.
* **Step 3:** If you want to tune the model, adjust the tuning configuration and parameters within 'PARAMETERS FOR MODE TUNE' according to Wandb documentation (Important: Add you own Wandb key)
* **Step 4:** For the actual run, there are following possibilities:
   * **Step 4a:** For training the model, you can run it directly from command line ('python main.py --train [Model_Name]) or start it via calling train.sh file (Adjust the model name within the file)
   * **Step 4b:** For evaluating the model, you can run it directly from command line ('python main.py --evaluate [Model_Name]) or start it via calling evaluate.sh file (Adjust the model name within the file)
   * **Step 4c:** For tune the model/problem, you can run it directly from command line ('python main.py --tune [Model_Name]) or start it via calling tune.sh file (No model needed here, because it is defined via config)
