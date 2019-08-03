# deBot
Rates a debate speech based on speech quality and argument verity.

Before you begin, you must have the 50-dimensional glove embeddings file saved. You can download them from the link below and extract it into the correct directory.
- [glove.6B.50d.txt.w2v.zip](https://www.dropbox.com/s/c6m006wzrzb2p6t/glove.6B.50d.txt.w2v.zip?dl=0) (67 MB)

You will also need the train.csv and test.csv datasets from IBM, which can be obtained from the link below. (Extract the files into the correct directory, similarly to above).
- [IBM debater Argument Quality Datasets](https://www.research.ibm.com/haifa/dept/vst/files/IBM_Debater_(R)_EviConv-ACL-2019.v1.zip)


In order to make the ArgumentQuality test work, you must first run ArgumentQualityTrain.py, which trains the argument quality model on data from IBM. This only needs to be done once and it will create an ArgumentQualityModel.npy file, which contains the parameters of the trained model.
