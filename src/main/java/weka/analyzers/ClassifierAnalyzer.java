package weka.analyzers;

import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Capabilities;
import weka.core.Option;
import weka.core.Utils;

import java.util.Enumeration;
import java.util.Vector;

/**
 * Abstract class for Analyzers that operate using a Classifier. Includes
 * functionality to set and get the classifier.
 */
public abstract class ClassifierAnalyzer extends Analyzer {

    /** The base classifier to use */
    protected Classifier classifier = new J48();

    /**
     * @return Name of the default Classifier
     */
    public String defaultClassifierString() {
        return "J48";
    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities capabilities  = super.getCapabilities();
        capabilities.and(classifier.getCapabilities());
        return capabilities;
    }

    @Override
    public Enumeration<Option> listOptions() {
        Vector<Option> newVector = new Vector<Option>(3);

        Enumeration<Option> enu = super.listOptions();
        while (enu.hasMoreElements()) {
            newVector.addElement(enu.nextElement());
        }

        newVector.addElement(new Option(
                "\tFull name of base classifier.\n"
                        + "\t(default: " + defaultClassifierString() +")",
                        "W", 1, "-W"));

        newVector.addElement(new Option(
                "",
                "", 0, "\nOptions specific to classifier "
                        + classifier.getClass().getName() + ":"));
        enu = classifier.listOptions();
        while (enu.hasMoreElements()) {
            newVector.addElement(enu.nextElement());
        }

        return newVector.elements();
    }

    @Override
    public void setOptions(String[] options) throws Exception {
        super.setOptions(options);

        String classifierName = Utils.getOption('W', options);
        if (classifierName.length() == 0) {
            classifierName = defaultClassifierString();
        }
        setClassifier(Classifier.forName(classifierName, null));
        setClassifier(Classifier.forName(classifierName,
                Utils.partitionOptions(options)));
    }

    @Override
    public String [] getOptions() {
        String [] classifierOptions = classifier.getOptions();
        int extraOptionsLength = classifierOptions.length;
        if (extraOptionsLength > 0) {
            extraOptionsLength++; // for the double hyphen
        }

        String [] superOptions = super.getOptions();
        String [] options = new String [superOptions.length +
                                        extraOptionsLength + 2];

        int current = 0;
        options[current++] = "-W";
        options[current++] = getClassifier().getClass().getName();

        System.arraycopy(superOptions, 0, options, current,
                superOptions.length);
        current += superOptions.length;

        if (classifierOptions.length > 0) {
            options[current++] = "--";
            System.arraycopy(classifierOptions, 0, options, current,
                    classifierOptions.length);
        }

        return options;
    }

    public void setClassifier(Classifier newClassifier) {
        classifier = newClassifier;
    }

    public Classifier getClassifier() {
        return classifier;
    }

    public String classifierTipText() {
        return "The classifier to be used";
    }
}
