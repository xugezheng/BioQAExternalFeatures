export BIOCODES_DIR=/path/to/biobert/biocodes #BioBERT biocodes dir -> use their transformation script
export JAVA_DIR=/path/to/Evaluation-Measures #Official Evaluation dir
export OUTPUT_DIR=/path/to/prediction/output #Prediction result
export QA_DIR=/path/to/golden/answer #data should be downloaded from bioasq official website (need registration)



pre="BioASQ-test-factoid-6b-snippet-"
post="-2sent.json"

pregold="6B"
postgold="_golden.json"

for i in 1 2 3 4 5
do
  cd $BIOCODES_DIR
  python transform_n2b_factoid.py --nbest_path=$OUTPUT_DIR/$pre$i$post/nbest_predictions_.json --output_path=$OUTPUT_DIR/$pre$i$post
  echo "Tansfered!"
  cd $JAVA_DIR
  java -Xmx10G -cp ./flat/BioASQEvaluation/dist/BioASQEvaluation.jar evaluation.EvaluatorTask1b -phaseB -e 5 $QA_DIR/$pregold$i$postgold $OUTPUT_DIR/$pre$i$post/BioASQform_BioASQ-answer.json
  echo "Evaluation Done"
done