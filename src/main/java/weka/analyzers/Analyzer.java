package weka.analyzers;


import weka.core.Capabilities;
import weka.core.CapabilitiesHandler;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Utils;
import weka.gui.Logger;

import java.io.Serializable;
import java.util.Enumeration;
import java.util.Vector;

/**
 * Super class for class that can be used to run computations
 * and generate visual and textual output for datasets.
 */
// Modelled after weka.classifiers.Classifier
public abstract class Analyzer implements OptionHandler,
        CapabilitiesHandler, Serializable {

    /** For serializations */
    private static final long serialVersionUID = 1499833768169062075L;

    /** Whether to run in debug mode */
    protected boolean m_Debug = false;

    /**
     * Run some kind of analysis on some Instances and return the result.
     *
     * @param data Instances to analyze
     * @param idAtt index of an attribute to be treated as an ID attribute,
     *              -1 if there is no such attribute
     * @param logger Logger to log status updates to
     * @return AnalyzerOutput containing text output and GenerateVisualizations
     * to express to the user the results of the analysis
     * @throws Exception if there was an error analyzing the data
     */
    public abstract AnalyzerOutput analyzeData(Instances data, int idAtt,
                                               Logger logger) throws Exception;

    /**
     * Returns the Capabilities of this Analyzer.
     *
     * @return the capabilities of this object
     */
    public Capabilities getCapabilities() {
        Capabilities result = new Capabilities(this);
        result.enableAll();
        return result;
    }

    /**
     * @return Option Enumeration of the ways this Analyzer can be configured
     * and the flags to use to configure it
     */
    public Enumeration<Option> listOptions() {
        Vector<Option> newVector = new Vector<Option>(1);
        newVector.addElement(new Option(
                "\tIf set, analyzer is run in debug mode and\n"
                        + "\tmay output additional info to the console",
                "D", 0, "-D"));
        return newVector.elements();
    }

    /**
     * Configures this Analyzer to an array of arguements.
     *
     * @param options String[] containing the arguements used
     *                to configure this
     * @throws Exception If there was an error configuring this
     */
    public void setOptions(String[] options) throws Exception {
        setDebug(Utils.getFlag('D', options));
    }

    /**
     * @return String[] of the arguments that would be used to
     * configure this to its current settings.
     */
    public String[] getOptions() {
        String[] options;
        if (getDebug()) {
            options = new String[1];
            options[0] = "-D";
        } else {
            options = new String[0];
        }
        return options;
    }

    public void setDebug(boolean debug) {
        m_Debug = debug;
    }

    public boolean getDebug() {
        return m_Debug;
    }

    public String debugTipText() {
        return "If set to true, analyzer may output additional info to " +
                "the console.";
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(this.getClass().getCanonicalName() + " ");
        String[] opts = getOptions();
        int i = 0;
        while(i < opts.length && opts[i].length() != 0) {
            sb.append(opts[i++] + " ");
        }
        return sb.toString();
    }
}
