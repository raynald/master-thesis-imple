Note:
======
Based on Pegasos' implementation
------
For technical details see:
"Pegasos: Primal Estimated sub-GrAdient SOlver for SVM".
Shai Shalev-Shwartz, Yoram Singer, and Nathan Srebro.
ICML, 2007.
The original code is distributed under GNU General Public License (see license.txt
for details).
Copyright (c) 2007 Shai Shalev-Shwartz. All Rights Reserved.

Author
-----
Raynald Chung @ETHZ

Usage
-----
 - make
 - ./optimize [options] <data-file>

Options:
------
 - -epoch number of epochs (default = 30)
 - -round number of round (default = 10)
 - -lambda regularization parameter (default = 0.0001)
 - -testFile name of test data file (default = noTestFile)
