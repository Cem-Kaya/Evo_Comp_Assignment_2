This solution contains the software implementation for the second assignment. 
A summary of relevant files and folders is given below. For more details, please see the documentation in the files.

-	graph.py: Contains the Graph class. Encapsulates the functionality to load the graph from text representation file and graph operations like applying partitioning as solution, moving nodes and calculating gains.
-	node.py: Node class file. Contains properties of a node, such as id, neighbors, lock status, etc.
-	node_linked.py: Implements linked list functionality, with previous, next, insert, remove, etc. Extends Node class.
-	fm_impl.py: Contains the FM class, which implements the Fiduccia-Mattheyses (FM) heuristic. Provides functionalities like initializing and managing buckets, running a FM pass and generating execution statistics.
-	mls.py: Implements MLS by running FM instances. Exports the results to pickle files and contains functionality to generate metrics and statistics.
-	ils.py: Implementation of ILS algorithm.
-	ils_adaptive.py: Extension of ILS with adaptive pursuit algorithm.
-	gls.py: Implementation of GLS algorithm.
-	experimentation.py: Contains methods that executes various experiments.
-	test_methods_1.py: Contains various test cases for implemented functions.
-	utils.py: Contains helper methods, such as loading datasets from pickle files, statistical significance test and formatting datasets for display.
-	pckl: This folder contains the archived experiment results. Archive format is pickle.
-   a_runner_*.ipynb: Python notebooks for running experiments.
-   a_*.ipynb: Python notebooks for result analysis, table and chart generations.