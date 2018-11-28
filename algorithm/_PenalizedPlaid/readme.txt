

*************** Bayesian Penalized Plaid model for biclustering ************** 
*****************************************************************************    
		  Program tested with java version "1.7.0"


****** This program is distributed under the GNU General Public license. 
****** Please read the file COPYING.

****** If you use this software, please make a reference to the work given below in REFERENCES.

REFERENCES:

Thierry Chekouo and Alejandro Murua (2014), 
"The Penalized Biclustering Model And Related Algorithms," 
Journal of Applied Statistics, In press.



                ************************
                 LIST OF JAVA CLASSES
                ************************ 

1. mainProgram.class: the main java class  
2. mcmcMethods.class: It implements all the different steps of the MCMC algorithm including useful functions.
3. InitialValuesPenalized.class: It contains all the methods used for the
   initialization of the parameters.




                 ************************
                     REQUIREMENT
                 ************************

The weka package: Java Programs for Machine Learning, a GNU general public license version 2, June 1991. 
We used the K-means method implemented in that  package to initialize our bicluster memberships rho and kappa. 



                 ********************************
                  USAGE (examples are given below)
                 ********************************

1. Unzip the zip file penalizedplaid.zip; this will create a new directory called penalizedplaid.

2. Copy your data matrix (rows are genes and columns are conditions) to the same directory or to another directory. 

3. Create a directory to save the results: mkdir /home/home_name/resultsplaid/;
   It  can be the same directory with the matrix data.

4. Note that Java needs to be installed on your system for this to work; update your version to 1.7.
   The source files are in the PenalizedPlaid.jar file.  

5. On the command line, run this command:

java -jar PenalizedPlaid.jar "/home/home_name/resultsplaid/" 2ndarg 3rdarg 4tharg 5tharg 6tharg 7tharg 8tharg 

where 	
        2ndarg is the file name of the data matrix accompanied with its directory if necessary;

        3rdarg is the number of genes (rows). For instance, you can specify as nbrgenes=400;

        4rdarg is the number of conditions (columns). Any string can precede this number. 
        For 50 conditions for example, we can enter: nbrconditions=50 or nbrconditions:50 or nbrconditions50;  

        5tharg is the number of biclusters, K, to estimate. For 2 biclusters for example, we can enter: K=2;


        6tharg is the model used for the data. We have basically 4 methods: 
            
          a. GPE=the penalized plaid model with the sampling of the penalty parameter,
             lambda. The model is fitted with a Gibbs sampling.
          b. GPF=the penalized model with a fix value of the penalty parameter lambda
             and fitted with the  Gibbs sampling procedure
             b.1 when  lambda=0, we have the plaid model which does not assume
                 any constraint on the amount of overlapping genes and conditions between biclusters
             b.2 when lambda tends to infinity (lambda>=10^3 is recommended to speed up the algorithm), 
                 we assume that biclusters do not overlap as with the Cheng and Church model ("Biclustering of Expression Data," Int. Conf. Intelligent Systems for Molecular Biolog., 12:61-86, 2000).
           
          c. MPE=the penalized model fitted with a Metropolis-Hastings procedure
          d. MPF=the penalized model with a fix value of lambda and fitted with a Metropolis-Hastings procedure  
          
             
        7tharg is the number of MCMC samples after the burn in. We recommend more than 1000 samples; sample=1000

        8tharg is the number of burn-in samples. We recommend a number more than 1000. For example, burninsample=1000;

        If the models are GPF or MPF, a 9th argument is necessary to specify a fix value of lambda. 
        For example lambda=0, lambda=1000, lambda=0.1;


Example with the the GPE model:

java -jar PenalizedPlaid.jar "/home/home_name/resultsplaid/" "data.txt" nbrgenes=400 nbrconditions=50 nbrofbiclusters=2 GPE sample=1000 burnin=1000 
 
Example with the the MPE model:

java -jar PenalizedPlaid.jar "/home/home_name/resultsplaid/" "data.txt" nbrgenes=400 nbrconditions=50 nbrofbiclusters=2 MPE sample=1000 burnin=1000

Example with the the GPF model:

java -jar PenalizedPlaid.jar "/home/home_name/resultsplaid/" "data.txt" nbrgenes=400 nbrconditions=50 nbrofbiclusters=2 GPF sample=1000 burnin=1000 lambda=0;

Example with the the MPF model:

java -jar PenalizedPlaid.jar "/home/home_name/resultsplaid/" "data.txt" nbrgenes=400 nbrconditions=50 nbrofbiclusters=2 MPF sample=1000 burnin=1000 lambda=0;


When the number of true biclusters are known as in the case in simulated data, three extra arguments can be added
to compute the F1-measure defined in the paper; 
--the first is the number of true biclusters: For example K=2;
--the second is the true row membership. It is a binary matrix of order p (number of genes) times K (number of true biclusters) i.e each column k=1,…,K is a binary vector ( 1 if the row belong to bicluster k and 0 otherwise)
--the third  is the true column membership. It is a binary matrix of order q (number of conditions) times K (number of true biclusters);

For example we can run:

java -jar PenalizedPlaid.jar "/home/home_name/resultsplaid/" "data.txt" nbrgenes=400 nbrconditions=50 nbrofbiclusters=2 GPF sample=1000 burnin=1000 lambda=0 K=2 rhoknown.txt kappaknown.txt



			*************** 
			   OUTPUT 
			***************

All the output files are saved in the result directory.

The program saves the following results:

1. The estimated AIC and DIC
2. The estimated membership labels Rho and Kappa
3. The penalty parameter estimate if the model is either GPE or MPE: LambdaEstimate.
4. The overall mean estimate of each bicluster: MuEstimate (K times 1)
5. The row effects estimate matrix : AlphaEstimate (p times K)
6. the column effect estimate matrix: BetaEstimate (q times K)
7. the F1-measure if we know the true biclustering

Each saved file name contains the model name and the number of estimated biclusters.
For example, the F1 measure for 2 biclusters estimated with the GPE model is
saved as: F1.GPE.K2.txt.

Please contact tchekouo@mdanderson.org OR murua@dms.umontreal.ca to report any bug.

Enjoy!!!   
