Tiny ImageNet Challenge is the first course project of Introduction to Deep Learning. It runs similar to the ImageNet challenge and Stanford CS231n.

#### Dataset
The Tiny ImageNet dataset has 100 classes. Each class has 1,000 training images, 100 validation images. The test set contains 10,000 images in total. We have released the training and validation sets with images and labels. The test set is released without labels. You are asked to predict the class label of each test image.

In detail, all training and validation images are from ImageNet training set. For test set, 5,000 images (id 0~4999) are selected from ImageNet validation set, while another 5,000 images (id 5000-9999) from ImageNet-Adataset, which called Natural Adversarial Examples. Modern deep CNNs achieve very low accuracy on these natural adversarial examples. ImageNet-A examples cause mistakes due to occlusion, weather, and other complications encountered in the long tail of scene configurations. You are recommended to read the original paper for more information.
For this course project, you need to consider how to achieve high classification accuracy on both general ImageNet images and natural adversarial examples.

#### Submission
We use the test set accuracy to measure the performance. To submit your predictions on the test set, you need to upload a CSV file, which should be a two-column file with 10,001 lines (the first row is a header). Each line contains a pair of test image filename and its predicted class id (0-99). An example might look like:

Id,Category
0.jpg,0
1.jpg,1
2.jpg,2
...
9999.jpg,99
After the deadline, you are required to submit all codes and a short report (no less than one page). Plagiarism and labeling test images manually will be penalized.
