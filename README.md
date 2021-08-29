# Deploying a Bank Note Authentication model using Flask
## This problem is part of Kaggle: https://www.kaggle.com/ritesaluja/bank-note-authentication-uci-data
### The objective of the problem is to distinguish between forged bank notes and authentic notes. The data comprises of 4 features and 1 target columns, and then create a Flask App which can be invoked via POSTMAN or localhost.

#### The features are: 'variance', 'skewness', 'kurtosis', and 'entropy'. The target column is labelled as 'class'.

The features were extracted from images that were taken from genuine and forged banknote-like specimens. For digitization, an industrial camera, usually used for print inspection was used. The final images have 400x 400 pixels. Due to the object lens and distance to the investigated object, gray-scale pictures with a resolution of about 660 dpi were gained. Wavelet Transform tool were used to extract features from images.

This is a simple classification problem without much requirement for feature engineering. With a default RandomForest Classifier model, we were able to achieve >99% validation accuracy. The trained model is then saved as a .pkl file (classifier.pkl) for later use.

In the first instance, we create a flask_api.py file, which can be run from localhost or using Postman service. The app requires 4 inputs, as the model has been trained on 4 features. It internally invokes a decorator to load the landing page and then invokes a predict_note_authentication with get() to load the pre-trained model (saved as classifier.pkl) to provide the prediction. While invoking predict_note_authentication, we call the /predict method and the pass the features in the invoking URL itself (as we have used get()): http://127.0.0.1:5000/predict?variance=2&skewness=3&curtosis=2&entropy=1. This returns the prediction as: "Hello The answer is: [0]. This is also possible via Postman-GET

We can also test the model on a test dataset: TestFile.csv, which we pass to the model using another method predict_note_file() which is invoked using post() this time, as we have a file and we can't pass all the features via a single URL. The host this, we use Postman exclusively, with POST.
