import weka.classifiers.Classifier;
import weka.classifiers.lazy.IBk;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Random;

/**
 * @author LBW
 */
public class ImprovedBaggingKNN {
    private static String[] dataSets = {"breast-w.arff","colic.arff","credit-a.arff","credit-g.arff","diabetes.arff", "hepatitis.arff","mozilla4.arff","pc1.arff","pc5.arff","waveform-5000.arff"};

    public static void main(String[] args) throws Exception{
        for (int i = 0; i < 10; i++) {
            DataSource dataSource = new DataSource("data/" + dataSets[i]);
            Instances data = dataSource.getDataSet();
            if (data.classIndex() == -1)
                data.setClassIndex(data.numAttributes()-1);


            //cross validation
            double sumOfAccuracy = 0;
            for (int n = 0; n < 10; n++) {
                Instances trainSet = data.trainCV(10, n);
                Instances testSet = data.testCV(10, n);

                //compute ratio of the number of samples per class
                HashMap<Double, Double> ratioClasses = getRatioMap(trainSet);



                ArrayList<Instances> trainSets = new ArrayList<>(100);
                //sample randomly to build 100 trainSets
                for (int j = 0; j < 100; j++) {
                    int numInstances = trainSet.numInstances();
                    trainSets.add(new Instances(trainSet));
                    trainSets.get(j).clear();
                    Random random = new Random();
                    for (int k = 0; k < numInstances; k++) {
                        trainSets.get(j).add(trainSet.get(random.nextInt(numInstances)));
                    }
                }

                //check the ratio of the number of samples per class in the trainSets.
                Iterator<Instances> it = trainSets.iterator();
                while(it.hasNext()) {
                    Instances instances = it.next();
                    HashMap<Double, Double> smallMap = getRatioMap(instances);
                    for (Double key: ratioClasses.keySet()) {
                        //if difference of ratio is larger than 5%, remove the trainSet.
                        if (Math.abs(ratioClasses.get(key) - smallMap.get(key)) >= 0.05) {
                            it.remove();
                            break;
                        }
                    }
                }

                int numTrainSets = trainSets.size();
                //add classifiers
                Classifier[] classifiers =  new Classifier[numTrainSets];
                for (int j = 0; j < numTrainSets; j++) {
                    IBk iBk = new IBk();
                    iBk.setOptions(new String[]{"-K", "5"});
                    //train the classifier
                    iBk.buildClassifier(trainSets.get(j));
                    classifiers[j] = iBk;
                }

                double predicateCorrectNum = 0;
                for (int j = 0; j < testSet.numInstances(); j++) {
                        if (vote(classifiers, testSet.get(j)) == testSet.get(j).classValue()) {
                            predicateCorrectNum += 1;
                        }
                }
                double accuracy = predicateCorrectNum / testSet.numInstances();
                //System.out.println(accuracy);
                sumOfAccuracy += accuracy;
            }
            double averageAccuracy = sumOfAccuracy / 10;
            System.out.println("Accuracy of improved bagging knn on " + dataSets[i] + ": " + averageAccuracy);
        }
    }

    private static double vote(Classifier[] classifiers, Instance instance) throws Exception{
        HashMap<Double, Double> hashMap = new HashMap<>();
        for (Classifier classifier: classifiers) {
            double classValue = classifier.classifyInstance(instance);
            if (!hashMap.containsKey(classValue)) {
                hashMap.put(classValue, 1.0);
            }
            else hashMap.put(classValue, hashMap.get(classValue)+1);
        }
        double resultKey = -1;
        double maxValue = Double.NEGATIVE_INFINITY;
        for (Double key: hashMap.keySet()) {
            if (hashMap.get(key) > maxValue) {
                maxValue = hashMap.get(key);
                resultKey = key;
            }
        }
        return resultKey;
    }

    private static HashMap<Double, Double> getRatioMap(Instances trainSet) {
        HashMap<Double, Double> ratioClasses = new HashMap<>();
        for (int j = 0; j < trainSet.numInstances(); j++) {
            Double classValue = trainSet.get(j).classValue();
            if (!ratioClasses.containsKey(classValue)) {
                ratioClasses.put(classValue, 1.0);
            }
            else ratioClasses.put(classValue, ratioClasses.get(classValue)+1);
        }
        for (Double key: ratioClasses.keySet()) {
            ratioClasses.put(key, ratioClasses.get(key)/trainSet.numInstances());
            //System.out.println(key + ": " + ratioClasses.get(key));
        }
        return ratioClasses;
    }
}
