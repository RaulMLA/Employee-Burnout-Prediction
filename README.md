# Employee Burnout Prediction

The objective of this project is to build models with various preprocessing techniques. The topic is employee burnout: a company is concerned about the level of employee exhaustion and would like to create a model that predicts the likelihood of employees leaving the company, using a dataset collected by the human resources department.

# Content
- [Methods](#methods)
- [Feature Selection using Filter Methods](#feature-selection-using-filter-methods)
- [Authors](#authors)

# Methods

In this project, we will focus on building models to predict solar energy production. We will consider a ``Logistic Regression`` model as the base method without adjusting hyperparameters and choose a boosting method as the advanced approach with hyperparameter tuning. It is essential to preprocess the data appropriately, preferably using pipelines to ensure consistency and efficiency. 
The boosting methods selected for evaluation in this project are:

- ``AdaBoost``:
AdaBoost, short for Adaptive Boosting, is a popular boosting algorithm that combines weak learners (usually decision trees) to create a strong predictive model. AdaBoost assigns weights to each instance in the dataset, focusing more on the misclassified samples in subsequent iterations. This iterative process improves the model's performance by giving more attention to difficult-to-predict instances. AdaBoost is known for its ability to handle complex datasets and is resistant to overfitting.

- ``Gradient Boosting``:
Gradient Boosting is another powerful boosting technique that builds an ensemble of decision trees in a sequential manner. Unlike AdaBoost, Gradient Boosting uses a gradient descent optimization algorithm to minimize the loss function during each iteration. This algorithm allows subsequent trees to correct the errors made by previous trees. Gradient Boosting is highly flexible and capable of capturing complex interactions in the data, making it a popular choice for predictive modeling tasks.

- ``XGBoost``:
XGBoost, short for Extreme Gradient Boosting, is an optimized implementation of gradient boosting that provides enhanced performance and scalability. XGBoost incorporates additional regularization techniques to prevent overfitting and offers a variety of hyperparameters for fine-tuning the model. It is designed to handle large-scale datasets efficiently and has gained popularity in machine learning competitions due to its exceptional predictive power.

- ``LightGBM``:
LightGBM is another high-performance gradient boosting framework that aims to achieve faster training speed and lower memory usage compared to other boosting algorithms. It utilizes a technique called Gradient-based One-Side Sampling (GOSS) to select the most informative instances for building trees, resulting in reduced computational resources. LightGBM also supports categorical features directly and offers several advanced features, making it an attractive option for boosting models.

By comparing these models and analyzing their performance, we aim to identify the most effective approach for predicting solar energy production. We will also assess the impact of hyperparameter tuning on model performance and evaluate if the additional external libraries provide further improvements.

All the explanations and conclusions are included in the notebook [Solar Energy Production Prediction](AA_P2_Cuaderno.ipynb).

# Feature Selection using Filter Methods

In addition to model building, we will explore the use of filter-based feature selection methods available in scikit-learn, such as f_classif, mutual_info_classif, or chi2. Specifically, we will employ the SelectKBest method to select the most informative attributes. The objective is to determine if these feature selection techniques can enhance the results obtained in the previous section and gain insights into the importance of different attributes according to these methods.

By evaluating the performance and interpretability of the models after applying feature selection, we can draw conclusions about the impact of attribute relevance on solar energy production prediction. This analysis will provide valuable insights into the significant features and help streamline the modeling process.

# Authors
- [Raúl Manzanero López-Aguado](https://github.com/RaulMLA)
- [Adrián Sánchez del Cerro](https://github.com/adrisdc)
