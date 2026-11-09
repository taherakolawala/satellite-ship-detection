# Evaluating Dimensionality Reduction Techniques on Satellite Imagery Classification

[Paper (PDF)](paper/satellite-ship-detection.pdf) · [LaTeX source](paper/satellite-ship-detection.tex)

**Taher Akolawala**

---

## Abstract

Using the 4,000 labeled 80×80 RGB crops in the Ships in Satellite Imagery dataset (San Francisco and San Pedro Bays), I compared logistic regression and linear SVM classifiers on raw pixels against pipelines that first reduce the 19,200 pixel features with PCA, Isomap, or LLE, tuning every pipeline for F1 with three-fold cross-validation on 3,200 training images and then scoring it once on the 800 held-out images. PCA with 150 components followed by a linear SVM did best on the holdout, at an F1 of 0.850 against 0.768 for the same SVM on raw pixels (which fit its training split perfectly and generalized worse), and it cut the mean candidate fit time during the grid search from 17.95 s to 8.95 s. Isomap landed between the two at 0.832 to 0.840 F1, while LLE collapsed to predicting no-ship for every image and scored an F1 of 0, so the choice of reduction method matters as much as the decision to reduce at all. Because all of this comes from one random, unstratified split of crops that may share scenes, I read it as a comparison of classical pipelines on this dataset and not as evidence about operational ship detection.

---

## Introduction

Automated ship detection in satellite imagery has become increasingly critical for maritime surveillance, port activity monitoring, and supply chain analysis. As commercial satellite providers like Planet capture imagery of the entire Earth daily, the volume of data has outgrown manual analysis capabilities, necessitating robust machine learning solutions. However, satellite imagery presents a unique challenge: each 80×80 RGB image contains 19,200 features, creating a high-dimensional feature space that significantly impacts computational efficiency and model performance. Thus, it becomes critical to understand how dimensionality reduction techniques can maintain classification accuracy while improving training and inference times, enabling real-time deployment in operational maritime monitoring systems. The main aim of the project can be summarized in one question:

*How can dimensionality reduction optimize ship detection performance while minimizing computational costs?*

To address this question, the analysis ran in four stages:

1. **Baseline classifiers:** logistic regression and a linear SVM trained on the raw 19,200 pixel features.
2. **PCA analysis:** how many principal components it takes to keep 90% of the variance, and how much reconstruction error that costs.
3. **PCA pipelines:** PCA inserted before each classifier, with the number of components tuned alongside the classifier's own hyperparameters.
4. **Manifold learning:** Isomap and LLE in place of PCA, to see whether a nonlinear reduction separates the classes better.

Every pipeline was tuned with three-fold cross-validation on the training split and then evaluated once on the held-out test split, and the test-split F1 scores are the numbers to compare.

---

## Data Set and Methodology

The dataset consists of 4,000 80×80 RGB images labeled with either a "ship" or "no-ship" classification. Images were derived from PlanetScope, which are orthorectified to a 3-meter pixel size. In order for any sort of machine learning analysis, I needed to ensure that the image data was properly structured. The raw images were flattened from their original three-dimensional shape into one-dimensional feature vectors, transforming each 80×80×3 image into a single array of 19,200 pixel values. This flattening process was essential for compatibility with traditional machine learning algorithms, but it also highlighted the severe curse of dimensionality inherent in image classification tasks. With nearly 20,000 features per sample, the computational cost of training models becomes prohibitive, making dimensionality reduction not just beneficial but necessary for practical implementation.

The dataset exhibits a significant class imbalance, with 1000 ship images and 3000 no-ship images. This 1:3 ratio reflects the real-world scenario where ships occupy relatively small portions of maritime surveillance imagery. To ensure fair comparison across all experiments, StandardScaler was applied to normalize the pixel values in all pipelines, preventing features with larger magnitudes from dominating the learning process. After all preprocessing steps were completed, the data was separated into 3,200 training and 800 test images with a random 80/20 split (seed 91). The split was not stratified, so the test set's class balance only approximately matches the full dataset's.

---

## Experimental Setup

The experimental framework was designed to systematically evaluate the impact of dimensionality reduction on classification performance and computational efficiency. I began with two baseline classifiers: Support Vector Machine (SVM) and Logistic Regression. SVM was chosen for its effectiveness in high-dimensional spaces and ability to find optimal decision boundaries through kernel methods, while Logistic Regression provided a simpler, interpretable baseline that is computationally efficient and well-suited for binary classification tasks. For both classifiers I ran GridSearchCV with 3-fold cross-validation, scored on F1. Training on 19,200 raw features is slow, so the grids were small: the SVM used a linear kernel with C in {0.01, 0.05, 0.1}, and logistic regression used an L2 penalty with C in {0.01, 0.05}.

<p align="center">
  <img src="images/image1.png" alt="SVM Grid Search" width = "25%"/>
  <img src="images/image2.png" alt="Logistic Regression Grid Search" width="22%"/>
</p>

With baseline performance established, I moved onto Principal Component Analysis to understand the intrinsic dimensionality of the image data. PCA was fit on the training set to decompose the 19,200-dimensional feature space into orthogonal components ranked by explained variance. I systematically examined the cumulative variance explained to identify the minimum number of components needed to preserve 90% of the dataset variance. Reconstruction quality was evaluated by calculating root mean squared error between original and reconstructed images. This analysis informed the hyperparameter search space for subsequent PCA-enhanced pipelines. I then also reconstructed the training images with the top components found. More to be discussed in the results section

<p align="center">
  <img src="images/image3.png" alt="PCA Analysis" width = "25%" />
</p>

Building on these insights, I integrated PCA as a preprocessing step in the classification pipelines, treating the number of components as a hyperparameter alongside classifier-specific parameters. This ensured optimal dimensionality was selected in conjunction with the best classifier configuration.

<p align="center">
  <img src="images/image4.png" alt="PCA Pipeline 1"  width = "22%"/>
  <img src="images/image5.png" alt="PCA Pipeline 2"  width = "25%"/>
</p>

Finally, I explored nonlinear dimensionality reduction through Isomap and Locally Linear Embedding (LLE). Isomap preserves global geometric properties through geodesic distances, while LLE preserves local neighborhood structures. Both methods were optimized through grid search, with manifold-reduced features fed into the same classifiers.

<p align="center">
  <img src="images/image6.png" alt="Isomap Grid Search"  width = "25%"/>
  <img src="images/image7.png" alt="LLE Grid Search"  width = "23%"/>
</p>

---

## Results

### Baseline Performance

All numbers in this section through the manifold learning results are three-fold cross-validation scores on the 3,200-image training split; the held-out test results follow in their own section. The baseline classifiers established strong benchmarks without any dimensionality reduction. Both models achieved cross-validated accuracy above 90%, with SVM slightly outperforming Logistic Regression at 90.84% compared to 90.66%. The F1 scores were similarly close, at 0.8308 and 0.8329 respectively, indicating balanced performance across both classes despite the 1:3 class imbalance. However, the computational costs were substantial: the mean fit time per grid-search candidate was 63.15 seconds for Logistic Regression and 17.95 seconds for SVM. These training times highlight the computational burden of working directly with the full 19,200-dimensional feature space.


| Model Type          | CV Accuracy    | CV F1    | Mean Fit Time |
|---------------------|----------------|----------|---------------|
| Logistic Regression | 0.9066         | 0.8329   | 63.15s        |
| SVM                 | 0.9084         | 0.8308   | 17.95s        |

### PCA Analysis

The PCA variance analysis revealed that 109 principal components were sufficient to explain 90% of the cumulative variance in the dataset. This represents a significant reduction from the original 19,200 features, compressing the data by over 99% while retaining most of the information. The figure below shows the cumulative explained variance curve, which rises steeply in the first 50 components before gradually plateauing.

<p align="center">
  <img src="images/image9.png" alt="PCA Cumulative Variance" width="600" />
</p>

To evaluate the quality of this compression, I reconstructed images from the reduced representation and calculated the root mean squared error between original and reconstructed images. The average RMSE was 13.24, indicating moderate reconstruction loss but acceptable fidelity for classification purposes. PCA Reconstruction shown below.

<p align="center">
  <img src="images/image10.png" alt="PCA Reconstruction" width="600"/>
</p>

### PCA Enhanced Classification

Integrating PCA into the classification pipelines produced mixed results. Logistic Regression with PCA maintained nearly identical performance to the baseline, with accuracy of 90.69% and F1 score of 0.8334. More importantly, training time dropped dramatically to just 5.71 seconds, a 91% reduction from the baseline. SVM with PCA actually improved performance, achieving 92.22% accuracy and 0.8447 F1 score, the highest scores across all experiments. Training time for SVM with PCA was 8.95 seconds, cutting the baseline time in half while simultaneously improving accuracy. These results demonstrate that PCA successfully reduced computational costs without sacrificing performance, and in the case of SVM, actually enhanced it by removing noise and irrelevant features.

| Model Type                  | CV Accuracy    | CV F1    | Mean Fit Time per Model |
|-----------------------------|----------------|----------|-------------------------|
| Logistic Regression w/ PCA  | 0.9069         | 0.8334   | 5.71s                   |
| SVM w/ PCA                  | 0.9222         | 0.8447   | 8.95s                   |

### Manifold Learning Results

The manifold learning techniques produced more varied results. Isomap performed reasonably well with both classifiers. Logistic Regression with Isomap achieved 0.8191 F1 score with an inference time of 0.229 seconds, while SVM with Isomap scored 0.8004 F1 with 0.235 second inference time. Both were slightly below PCA performance but still competitive with the baselines. The 2D Isomap projection in the figure below shows some clustering structure, with ship and no-ship samples occupying different regions of the manifold space, though with considerable overlap.

<p align="center">
  <img src="images/image12.png" alt="Isomap Results" width="50%"/>
</p>

LLE, however, completely failed on this dataset. Despite extensive hyperparameter tuning across different numbers of neighbors, regularization parameters, and component dimensions, both Logistic Regression and SVM with LLE achieved F1 scores of exactly 0.0. The confusion matrices reveal that both models simply predicted the majority class (no-ship) for every single sample, resulting in perfect recall for no-ships but zero recall for ships. The LLE projection in the figure below shows why this occurred: the manifold structure completely collapsed, with minimal separation between classes. This failure likely stems from LLE's assumption that the data lies on a locally linear manifold, which does not hold for this high-dimensional image dataset with complex nonlinear relationships.

<p align="center">
  <img src="images/image13.png" alt="LLE Results" width="50%"/>
</p>

I attempted to explore other manifold learning techniques such as t-SNE and UMAP, but these methods lack a transform function, meaning they cannot be applied to new test data. Since the goal was to build practical classification pipelines that generalize to unseen images, these techniques were not viable options despite their superior visualization capabilities.

| Model Type                     | Inference Time | CV F1    | Mean Fit Time per Model |
|--------------------------------|----------------|----------|-------------------------|
| Logistic Regression w/ Isomap  | 0.229s         | 0.8191   | 8.99s                   |
| SVM w/ Isomap                  | 0.235s         | 0.8004   | 15.78s                  |
| Logistic Regression w/ LLE     | 0.198s         | 0.0      | 6.29s                   |
| SVM w/ LLE                     | 0.195s         | 0.0      | 6.17s                   |

### Held-Out Test Results

Each tuned pipeline was then scored once on the 800 test images. The test results widen the gap that cross-validation showed: the raw-pixel SVM fit its training split perfectly (zero errors on all 3,200 training images) and fell to 0.768 F1 on unseen data, while PCA with SVM held up at 0.850.

| Pipeline                       | Test F1    | Prediction Time (800 images) |
|--------------------------------|------------|------------------------------|
| SVM (raw pixels)               | 0.7677     | 3.67s                        |
| Logistic Regression (raw)      | 0.7990     | 0.057s                       |
| **SVM w/ PCA**                 | **0.8495** | 0.161s                       |
| Logistic Regression w/ PCA     | 0.8333     | 0.149s                       |
| SVM w/ Isomap                  | 0.8324     | 1.72s                        |
| Logistic Regression w/ Isomap  | 0.8404     | 1.66s                        |
| SVM w/ LLE                     | 0.0        | 2.04s                        |
| Logistic Regression w/ LLE     | 0.0        | 1.86s                        |

Timings come from a single run on a shared Jupyter server, so they show rough ordering and should not be read as a latency benchmark.

---

## Discussion

### Optimal Pipeline Selection

Based on the comprehensive evaluation across all experiments, SVM with PCA emerges as the optimal pipeline for ship detection in satellite imagery. This pipeline achieved the highest held-out F1 of 0.8495 (and the best cross-validated accuracy, 92.22%), while reducing mean fit time to 8.95 seconds compared to 17.95 seconds for the baseline SVM. Its confusion matrix on the 3,200 training images has 66 false positives and 103 false negatives; I did not save a confusion matrix for the test split, which is the first thing I would add. This represents a favorable precision-recall balance, which is critical for operational deployment where both missing actual ships and triggering false alarms carry operational costs.

The success of SVM with PCA can be attributed to several factors. First, PCA effectively filtered out noise and redundant features from the high-dimensional pixel space, allowing the SVM to focus on the most informative variance directions. Second, the reduction acts as a regularizer: the raw SVM classified every training image correctly and still lost 0.08 F1 to the PCA pipeline on the test split, which points to overfitting in the full pixel space. Finally, the reduced feature space allowed the SVM's kernel computations to operate more efficiently, cutting training time in half.

### Misclassification Analysis

Examining misclassified samples reveals clear patterns. The figure below shows ships incorrectly predicted as "no-ship." Most mislabeled images share similar traits: they are low contrast images where ships blend into surrounding water, providing few distinct features for the models to detect. The images lack clear structural cues like strong edges or visible waves that models learn to associate with ships. Poor atmospheric conditions, smaller vessels, and certain sun angles that minimize shadows proved particularly problematic. 

<p align="center">
  <img src="images/image15.png" alt="Misclassification Analysis" />
</p>

To address these issues, I recommend data augmentation to generate low-contrast training examples, incorporating texture descriptors like edge detection filters, and implementing confidence thresholds to flag uncertain predictions for manual review in operational deployment.

---

## Conclusion

This paper investigated dimensionality reduction for ship detection in satellite imagery. Through systematic evaluation, I demonstrated that dimensionality reduction can simultaneously improve efficiency and accuracy. SVM with PCA reached 0.850 F1 on held-out images against 0.768 for the raw-pixel SVM, while halving mean fit time and compressing 19,200 features to 150 components. In contrast, LLE completely failed, highlighting that not all dimensionality reduction approaches suit image classification. Training-split errors pointed to low-contrast images as the primary challenge, suggesting future work should focus on augmentation and feature engineering. For operational deployment, SVM with PCA is the pipeline worth carrying forward, though it should be checked on a scene-held-out split and compared against a small CNN before anyone relies on it.
