# 2023-Machine-Learning
Class Project, Recommender Systems, Neural Networks, Interpretability

# Content
Developed a Restaurant Recommendation System in Evanston:
- **Implemented Popularity Matching**: Created raw and shrinkage versions, tailoring them for small-sized data.
- **Applied Content-Based Filtering**: Constructed distance matrices between items using numeric, categorical, and natural language features (TF-IDF, BERT).
- **Utilized Collaborative Filtering**: Formed distance matrices between customers, driven by features or score matrices.
- **Innovated Evaluation Metric**: Introduced a new metric named "hit rate" to evaluate the recommendation system.

Predicted Drunk Driving Percentage Across 50 US States:
- **Engineered a Neural Network**: Designed layers, optimizer, and loss criterion for regression and classification tasks.
- **Optimized Hyper-Parameters**: Improved model performance by 10% through grid-search methodology.
- **Enhanced with Transfer Learning**: Boosted model performance by 44% through fine-tuning, compared to the baseline model.

Classified Animal Images Using Deep Convolutional Neural Networks:
- **Identified Images**: Achieved 53% validation accuracy on the Animal-10N Image dataset (26% with linear regression).
- **Tuned Neural Networks**: Elevated performance by 12.8% by adjusting layers and optimizers.
- **Explored Model Explainability**: Used LIME and saliency maps with various noise levels to highlight influential image regions.



# Code
*Each Project file contains code and report, while some of them also include data*

1. Project1: data preprocessing, i.e., data wrangling and EDA
2. Project2: recommender systems
   - popularity matching
   - content filtering
   - collaborative filtering
3. Project3: predictive models
   - linear model
   - random forest
   - neural networks
   - transfer learning, i.e., generalization
   - visualization in map
4. Project4: interpretability
   - tabular data: weights, feature importance for rf, LIME(kind of "local linear interpretation")
   - image data(Animal_10N): SmoothGrad(i.e., gradient sensitivity map), LIME

