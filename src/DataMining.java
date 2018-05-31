import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.Bagging;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.util.Random;

/**
 * @author LBW
 */
public class DataMining {

    private static String[] dataSets = {"breast-w.arff","colic.arff","credit-a.arff","credit-g.arff","diabetes.arff", "hepatitis.arff","mozilla4.arff","pc1.arff","pc5.arff","waveform-5000.arff"};
    private static String[] classifierNames = {"J48","NaiveBayes","SMO","MultilayerPerceptron","IBk(k=5)", "Bagging of J48","Bagging of NaiveBayes","Bagging of SMO","Bagging of MultilayerPerceptron","Bagging of IBk(k=5)"};
    public static void main(String[] args) throws Exception {
        Classifier[] classifiers = getClassifiers();

        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) {
                DataSource source = new DataSource("data/" + dataSets[j]);
                Instances data = source.getDataSet();
                if (data.classIndex() == -1) {
                    data.setClassIndex(data.numAttributes() - 1);
                }
                System.out.println("/********" + classifierNames[i] + " on " + dataSets[j] + "********/");

                Evaluation evaluation = new Evaluation(data);
                evaluation.crossValidateModel(classifiers[i], data, 10, new Random(1));
                double accuracy = (1 - evaluation.errorRate()) * 100;
                System.out.println("accuracy: " + accuracy);

                double AUC = 0;
                for (int k = 0; k < data.numClasses(); k++) {
                    AUC += evaluation.areaUnderROC(k);
                }
                double averAUC = AUC / data.numClasses();
                System.out.println("average AUC: " + averAUC);
            }
        }

    }
    private static Classifier[] getClassifiers() throws Exception{
        Classifier[] classifiers = new Classifier[10];
        classifiers[0] = new J48();
        classifiers[1] = new NaiveBayes();
        classifiers[2] = new SMO();
        classifiers[3] = new MultilayerPerceptron();
        IBk iBk = new IBk();
        iBk.setOptions(new String[]{"-K","5"});
        classifiers[4] = iBk;
        Bagging bagging = new Bagging();
        bagging.setOptions(new String[]{"-W","weka.classifiers.trees.J48"});
        classifiers[5] = bagging;

        bagging = new Bagging();
        bagging.setOptions(new String[]{"-W","weka.classifiers.bayes.NaiveBayes"});
        classifiers[6] = bagging;

        bagging = new Bagging();
        bagging.setOptions(new String[]{"-W","weka.classifiers.functions.SMO"});
        classifiers[7] = bagging;

        bagging = new Bagging();
        bagging.setOptions(new String[]{"-W","weka.classifiers.functions.MultilayerPerceptron"});
        classifiers[8] = bagging;

        bagging = new Bagging();
        bagging.setOptions(new String[]{"-W","weka.classifiers.lazy.IBk","--","-K","5"});
        classifiers[9] = bagging;

        return classifiers;
    }
}
