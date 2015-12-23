# perforated-cnn-matconvnet

PerforatedCNNs accelerate convolutional neural networks (CNNs) by skipping evaluation of the convolutional layers in some of the spatial positions. See the paper for more details:

Michael Figurnov, Dmitry Vetrov, Pushmeet Kohli. PerforatedCNNs: Acceleration through Elimination of Redundant Convolutions. _Under review as a conference paper at ICLR 2016_ [[arXiv](http://arxiv.org/abs/1504.08362)].

The code is based on MatConvNet from December 2014, with some backports (such as more aggressive memory savings for ReLU).

**Code is coming soon.**

**MatConvNet** is a MATLAB toolbox implementing *Convolutional Neural
Networks* (CNNs) for computer vision applications. It is simple,
efficient, and can run and learn state-of-the-art CNNs. Several
example CNNs are included to classify and encode images. Please visit
the [homepage](http://www.vlfeat.org/matconvnet) to know more.
