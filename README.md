# HutchNER
The Fred Hutch Named Entity Recognition general training and execution pipeline
================================================================================
To install from the src directory:
	-pip install -r requirements.txt

To train:
	-call the train_local.py pipeline
		-required args: -t path_to_text_files -a path_to_annotations -m name_of_model
		-optional args: -mt algorithm_type -at annotation_type
	
to evaluate:
	-call the  evaluate.py pipeline
		-required args: -t path_to_text_files -m name_of_model
		-optional args: -at annotation_type
