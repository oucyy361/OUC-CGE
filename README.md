# OUC-CGE
OUC-CGE (OUC Classroom Group Engagement Dataset), designed for recognizing student group engagement in classroom environments using visual cues, is the first benchmark for group engagement analysis in real - world classroom settings with pure visual signals. The dataset is divided into three categories of high, medium, and low student group engagement levels following an 8:1:1 ratio, which is applied when splitting it into training, validation, and test subsets to ensure balanced data representation for model training and evaluation. Comprising 7,705 meticulously annotated 10 - second classroom clips across STEM and Humanities domains, and differing from existing datasets by targeting the whole classroom, it has been tested with several classical models. Through a technical - pedagogical dual validation strategy, OUC - CGE exhibits good consistency and discriminability compared to existing datasets, thus qualifying as a benchmark for future research on recognizing student group engagement levels. 
# Experiments
We tried six methods to achieve student group engagement classification.The accuracy rates of different models are as follows: SlowFast 93.2%, C2D 94.3%, I3D 93.3%, X3D 96.8%, SLOW - NLN 97.4%, and SLOW 97.8%. These results highlight OUC-CGE1's reliability and effectiveness, providing crucial references for related research. The official source code of the above methods can be obtained by visiting https://github.com/facebookresearch/SlowFast.
# Demo


https://github.com/user-attachments/assets/b3dbb15f-b5d0-4e66-a18b-9679811f0f94

