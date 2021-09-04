# Sensor_fusion_driving_situation
left and right camera with voice commands sensor fusion to detect driving situations

# Senfor fusion
Two VGG16 parallel for image feature extraction and VGGish for audio feature extraction.

Then merge features to detect safe and unsafe stiuation. See driving scene recognition conflicts with voice commands.

# How to use
You can use train.py to train the sensor fusion network. It ouputs accuracy and loss
You can use train2.py with k-fold cross validation. It gives you 10 times of accuracy, loss, val_accuracy and val_loss.


# Reading list
SIEVE: Secure In-Vehicle Automatic Speech Recognition Systems

Deep Multimodal Fusion by Channel Exchanging

A survey on machine learning for data fusion

A survey on data fusion in internet of things: Towards secure and privacy-preserving fusion

Book multi-sensor data fusion 2007 

CNN-based Sensor Fusion Techniques for Multimodal Human Activity Recognition

Handling Data Uncertainty and Inconsistency Using Multisensor Data Fusion

Planning and Decision-Making for Autonomous Vehicles

Toward Safe and Personalized Autonomous Driving: Decision-Making and Motion Control With DPF and CDT Techniques

Ethical Decision Making During Automated Vehicle Crashes 

Deep Learning for Safe Autonomous Driving: Current Challenges and Future Directions

Enabling Safe Autonomous Driving in Real-World City Traffic Using Multiple Criteria Decision Making

# TODO
1 The accuracy is still around 0.5 when using 200 image-audio pairs. So we need to review the architecture.

2 Generate more high quality data.

3 Failed to use keras callbacks for precision and recall.

