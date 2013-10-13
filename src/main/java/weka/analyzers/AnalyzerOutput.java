package weka.analyzers;

/**
 * Output of an analyzer. Includes some text output and any number of
 * GenerateVisualizations.
 */
public class AnalyzerOutput {

    /** Textual output of the Analyzer */
    public final String textOutput;

    /** Null or list of Visualizations the Analyzer produced */
    public final GenerateVisualization[] visualizers;

    /**
     * Creates an AnalyzerOutput with no visualizations.
     *
     * @param out Text output the Analyzer returned
     */
    public AnalyzerOutput(String out) {
        this(out, null);
    }

    /**
     * Creates an AnalyzerOutput.
     *
     * @param out Text output the Analyzer returned
     * @param visualizers Visualizations the Analyzer returned
     */
    public AnalyzerOutput(String out, GenerateVisualization[] visualizers) {
        this.textOutput = out;
        this.visualizers = visualizers;
    }
}
