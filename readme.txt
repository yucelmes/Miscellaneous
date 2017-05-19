FACE DETECTION AND BLUR
-----------------------
Original Algorithm: Haar Cascade Classifier as Implemented in OpenCV Library

Performances on the iMCD Image Set 
----------------------------------
Original algorithm: 59% of Recall 
Modified&Optimised algorithm: 89% of Recall 

Notebook File Name: FaceBlur.ipynb
----------------------------------
This notebook detects the regions of interest, blur these rois and save the blurred versions of the images in
a created folder on Desktop named TargetFolder. Images are read from a source file (as an example FaceSourceFile 
name is used here.).

External Dependency:
--------------------
cv2 (OpenCV)

This implementation is for Anaconda3 with Python 3.5.
All the relevant libraries/dependencies have been installed in .\Users\Mesut\Anaconda3\Final_FaceandNP.
Note that there is another Anaconda3 installed on the server side of this computer (C:\Program...).
However, this notebook should be run from .\Users\Mesut\Anaconda3 as the dependent scripts stay there.

Parallelized Face Detection
---------------------------
A Data Parallelism Problem: Process has been parallelised by making use of Python's ipyparallel library.
All images are distributed to the 'n' parallel engines.

Notebook File Name: FaceDetection_Parallel.ipynb
------------------------------------------------
Before running this script, the parallel engines should already be started to run first.
For this, in Power Shell: ipcluster start -n number_of_engines
This computer has 32 logical processors. Therefore, I recommend number_of_engines=32.
However, trying different numbers may also increase the performance.

Here, as an example, the images in a file named FileTest on Desktop are distributed to the engines.

Dependencies:
-------------
ipyparallel (installed in .\Users\Mesut\Anaconda3)
FaceBlur - this is a slightly different version of FaceBlur.ipynb. 
FaceBlur.py is a module that needs to be imported.

This parallel processing reads an image, process it, and saves its blurred version to the same location. 
The blurred version has the same name as the original one except that it has a suffix of '_FP'.
Example: 
Original Name: ABCDEF.jpg
Blurred Version's Name: ABCDEF_FP.jpg
 


NUMBER PLATE DETECTION AND BLUR
-------------------------------
Original Algorithm: Matthew Earl's ANPR - https://github.com/matthewearl/deep-anpr

Performances on the iMCD Image Set 
----------------------------------
Original algorithm: 17% of Recall 
Modified&Optimised algorithm: 76% of Recall 

Notebook File Name: NumberPlateBlur.ipynb
----------------------------------
This notebook detects the regions of interest, blur these rois and save the blurred versions of the images in
a created folder on Desktop named TargetFolder. Images are read from a source file (as an example NPSample 
name is used here.).

External Dependency:
--------------------
cv2 (OpenCV), tensorflow
common and model modules are parts of ANPR.

This script loads weights100.npz.
weights100.npz contain the weights of the neural network parameters.
We have obtained these weights from training the network for 100,000 steps.

This implementation is for Anaconda3 with Python 3.5.
All the relevant libraries/dependencies have been installed in .\Users\Mesut\Anaconda3\Final_FaceandNP.
Note that there is another Anaconda3 installed on the server side of this computer (C:\Program...).
However, this notebook should be run from .\Users\Mesut\Anaconda3 as the dependent scripts stay there.

Parallelized Number Plate Detection
---------------------------
Same Parallelism Problem: Process has been parallelised by making use of Python's ipyparallel library.
All images are distributed to the 'n' parallel engines.

Notebook File Name: NumberPlateDetection_Parallel.ipynb
------------------------------------------------
Before running this script, the parallel engines should already be started to run first.
For this, in Power Shell: ipcluster start -n number_of_engines
This computer has 32 logical processors. Therefore, I recommend number_of_engines=32.
However, trying different numbers may also increase the performance.

Here, as an example, the images in a file named FileTest on Desktop are distributed to the engines.

Dependencies:
-------------
ipyparallel (installed in .\Users\Mesut\Anaconda3)
NumberPlateBlur - this is a slightly different version of NumberPlateBlur.ipynb. 
NumberPlateBlur.py is a module that needs to be imported.

This parallel processing reads an image, process it, and saves its blurred version to the same location. 
The blurred version has the same name as the original one except that it has a suffix of '_NP'.
Example: 
Original Name: ABCDEF.jpg
Blurred Version's Name: ABCDEF_NP.jpg










