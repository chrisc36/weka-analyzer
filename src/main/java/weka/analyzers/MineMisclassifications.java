package weka.analyzers;

import weka.analyzers.AnalyzerUtils.GenerateTextWindow;
import weka.analyzers.mineData.BasicCachedRule;
import weka.analyzers.mineData.BitInstancesView;
import weka.analyzers.mineData.CachedRule;
import weka.analyzers.mineData.CachedRuleConjunction;
import weka.analyzers.mineData.CachedRuleDisjunction;
import weka.analyzers.mineData.CachedRuleSet;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.Utils;
import weka.gui.Logger;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Collection;
import java.util.Collections;
import java.util.Enumeration;
import java.util.Formatter;
import java.util.HashSet;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.Vector;

/**
 * Analyzer tha aims to find areas in the feature space that a particular
 * classifier is struggling to classify. This Analyzer builds and tests the
 * given classifier using cross validation, possibly for multiple iterations.
 * the average accuracy of the classifier is used to pick out a subset of
 * points that the classifier is struggling to classify correctly. Next a
 * number of rules are generated that partition the dataset, similar to the
 * rules a decision tree would use to split a dataset. A beam search is finally
 * used to identify subsets of the feature space as defined by conjunctions of
 * rules that are dense in hard to classify points.
 */
// TODO allow a prediction/target column so targets can be cached
// TODO if pruning, shorten rules and views to only the training data
// TODO preferable to use a single panel with a drop down menu to show rules
// TODO support regression tasks
public class MineMisclassifications extends ClassifierAnalyzer {

    public String globalInfo() {
        return " Analyzer tha aims to find areas in the feature space that a particular" +
                " classifier is struggling to classify. This Analyzer builds and tests the" +
                " given classifier using cross validation, possibly for multiple iterations." +
                " the average accuracy of the classifier is used to pick out a subset of" +
                " points that the classifier is struggling to classify correctly. Next a" +
                " number of rules are generated that partition the dataset, similar to the" +
                " rules a decision tree would use to split a dataset. A beam search is finally" +
                " used to identify subsets of the feature space as defined by conjunctions of" +
                " rules that are dense in hard to classify points.";
    }

    /**
     * Percent of the data set to hold out if we are pruning rules using a
     * validation set.
     */
    private final double VALIDATION_SIZE = .30;

    /**
     * How many unique values a feature must have before we consider using
     * quantiles to generate rules
     */
    private final int USE_QUANTILES = 8;

    /** When printing Instance ids, how long the lines should be */
    private final int ID_LINE_LENGTH = 110;

    /** Max number of examples to print */
    private final int MAX_EXAMPLES_TO_PRINT = 10;

    /** Max number attributes we can print examples for */
    private final int MAX_ATTRIBUTES_TO_PRINT = 40;

    /** Max number of ids to print */
    private final int MAX_IDS_TO_PRINT = 200;

   /** Number of folds to use when generating predictions */
    private int cvFolds = 4;

    /** Max number of rules to generate */
    private int maxRules = 10;

    /** Value to regularize laplace accuracy with */
    private int k = 20;

    /** Regularization term for each rule */
    private double rulePenalty = 0.01;

    /** Number of beams to use when searching for rules */
    private int beams = 4;

    /** Number of times to build the classifier and average the results */
    private int classificationIterations = 1;

    /**
     * Percent of classification iterations that need to classify the point
     * correctly for the point to not be considered a target
     */
    private double cutoff = .80;

    /** Whether to use the class attribute when rule building */
    private boolean useClass = true;

    /**
     * Max number of quantiles to use when building rules for
     * numeric features, -1 to use all useful split points
     */
    private int quantiles = 20;

    /** Whether to prune rules with a validation set */
    private boolean prune = true;

    /** Seed used for randomization */
    private long seed = 0L;

    /** Random object to use for randomization */
    // Seed is stored in addition to rand so we can return the set seed with getSeed()
    private Random rand = new Random(seed);

    /** Object used to evaluate rules */
    private LaplaceAccuracy ruleEvaluator = new LaplaceAccuracy();

    /**
     * Class used to evaluate a rule using LaplaceAccuracy.
     */
    private class LaplaceAccuracy implements Serializable {

        /** For Serialization */
        private static final long serialVersionUID = -7779955511680675241L;

        /**
         * Calculates the baseline score of a BitInstancesView, which can be
         * cached and used when evaluating rules against that View later.
         *
         * @param view, BitInstancesView to calculate the baseline for
         * @return the baseline
         */
        public double calcBaseline(BitInstancesView view) {
            return ((double)view.targets()) / view.size();
        }

        /**
         * Calculates the score of a rule given a particular BitInstancesView.
         * Also requires knowing the size of the rule.
         *
         * @param rule CachedRule to evaluate
         * @param size Size of the rule, usually number of basic rules
         * @param view BitInstancesView to evaluate against
         * @return the score
         */
        public double evalRule(CachedRule rule, int size, BitInstancesView view) {
            return evalRule(rule, size, view, calcBaseline(view));
        }

        /**
         * Calculates the score of a rule given a particular BitInstancesView.
         * Also requires knowing the rule size and baseline score of the view.
         *
         * @param rule CachedRule to evaluate
         * @param size Size of the rule, usually number of basic rules
         * @param view BitInstancesView to evaluate against
         * @param baseline baseline score of the view
         * @return the score
         */
        public double evalRule(CachedRule rule, int size,
                               BitInstancesView view, double baseline) {
            BitInstancesView.RuleEvaluation eval = view.evaluateRule(rule);
            return ((eval.targetsCovered + k*baseline) / (eval.covered + k)) -
                    size * rulePenalty;
        }

        /**
         * Calculates the score of a ruleSey given a particular BitInstancesView.
         * Also requires knowing the baseline of the view.
         *
         * @param ruleSet CachedRuleSet to evaluate
         * @param view BitInstancesView to evaluate against
         * @param baseline baseline score of the view
         * @return the score
         */
        public double evalRule(CachedRuleSet<? extends CachedRule> ruleSet,
                               BitInstancesView view, double baseline) {
            return evalRule(ruleSet, ruleSet.size(), view, baseline);
        }

        /**
         * Calculates the score of a ruleSet given a particular BitInstancesView.
         *
         * @param ruleSet CachedRuleSet to evaluate
         * @param view BitInstancesView to evaluate against
         * @return the score
         */
        public double evalRule(CachedRuleSet<? extends CachedRule> ruleSet,
                               BitInstancesView view) {
            return evalRule(ruleSet, ruleSet.size(), view);
        }
    }

    // Simple method to reduce verbosity
    private void printIfDebug(String msg) {
        if(getDebug())
            System.out.println(msg);
    }

    /**
     * Creates a String summary of how well a rule fits a data set. Includes
     * the number of Instances the rules covers, number of targets it covers,
     * accuracy or targets / covered, and the internal score.
     *
     * @param rule CachedRule to evaluate
     * @param size Size of that rule
     * @param data BitInstancesView to evaluate against
     * @return summary String
     */
    public String ruleReport(CachedRule rule, int size,
                             BitInstancesView data) {
        BitInstancesView.RuleEvaluation eval = data.evaluateRule(rule);
        return String.format("Covered: %d\tTargets: %d\tAccuracy: %.3f\tScored: %.3f",
                eval.covered, eval.targetsCovered, (double) eval.targetsCovered/eval.covered,
                ruleEvaluator.evalRule(rule, size, data));
    }

    /**
     * Class for evaluating CachedRules that are chained together as
     * conjunctions in linked lists.
     */
    private class RuleEval implements Comparable<RuleEval> {

        /**
         * Instances the conjunction of the rules in the list covers.
         * Can be null to indicate this has not been calculated
         */
        public BitInstancesView coveredInstances;

        /** Rule at the end of the list */
        public final CachedRule rule;

        /** Previous rule in the list, null if this is the first rule */
        public final RuleEval prev;

        /** Score of the conjunction of the list */
        public final double score;

        /** Number of rules in the linked list */
        public final int size;

        public RuleEval(CachedRule rule, RuleEval prev,
                        double score, BitInstancesView coveredInstances) {
            this.rule = rule;
            this.score = score;
            this.prev = prev;
            this.coveredInstances = coveredInstances;
            if(prev != null) {
                size = prev.size + 1;
            } else {
                size = 0; // Always started with empty rule
            }
        }

        public RuleEval(CachedRule rule, RuleEval prev, double score) {
            this(rule, prev, score, null);
        }

        @Override
        public int compareTo(RuleEval other) {
            return Double.compare(score, other.score);
        }

        /**
         * @return CachedRuleConjunction of the conjunction this represents
         */
        public CachedRuleConjunction<CachedRule> reconstructConjunction() {
            CachedRuleConjunction<CachedRule> rc = new CachedRuleConjunction<CachedRule>(rule.covered().size());
            RuleEval next = this;
            while(next.prev != null) {
                rc.add(next.rule);
                next = next.prev;
            }
            rc.reverse();
            return rc;
        }
    }

    /**
     * Finds the best rule conjunction we can using beam search.
     *
     * @param iv BitInstancesView of Instances to evaluate rules with
     * @param rules CachedRules to search through
     * @param baseline of the BitInstancesView
     * @return best rule conjunction found
     */
    public CachedRuleConjunction<CachedRule> greedyLearnBeams(BitInstancesView iv,
            Collection<CachedRule> rules, double baseline) {
        CachedRule emptyRule = BasicCachedRule.emptyRule(iv.size());

        // Marks beams which do not need to explore
        boolean[] doneBeams = new boolean[beams];
        Arrays.fill(doneBeams, 1, doneBeams.length, true); // Start with just one beam at the empty rule

        // Best rules as of the most recent completed iteration
        List<RuleEval> curBestRules = new ArrayList<RuleEval>(beams);

        // Best rules overall, even mid iteration, ordered from worst to best
        PriorityQueue<RuleEval> bestRuleQueue = new PriorityQueue<RuleEval>(beams);

        double baseScore = ruleEvaluator.evalRule(emptyRule, 0, iv, baseline);
        RuleEval empty = new RuleEval(emptyRule, null, baseScore, iv.copy());
        for(int i = 0; i < beams; i++) {
            bestRuleQueue.add(empty);
            curBestRules.add(empty);
        }

        boolean allBeamsDone = false;
        while(!allBeamsDone) {
            // Figure out what the best extension of the current best rules are
            for(int k = 0; k < beams; k++) {
                if(!doneBeams[k]) {
                    RuleEval rc = curBestRules.get(k);
                    for(CachedRule newRule : rules) {
                        double newScore = ruleEvaluator.evalRule(newRule, rc.size + 1,
                                rc.coveredInstances, baseline);
                        if(newScore > bestRuleQueue.peek().score) {
                            bestRuleQueue.poll();
                            bestRuleQueue.offer(new RuleEval(newRule, rc, newScore));
                        }
                    }
                }
            }
            curBestRules.clear();
            curBestRules.addAll(bestRuleQueue);

            // Update our best rules, precompute coverage and mark dead ends
            allBeamsDone = true;
            for(int j = 0; j < beams; j++) {
                RuleEval n = curBestRules.get(j);
                if(n.coveredInstances == null) {
                    // Precompute coverage of current best rules
                    n.coveredInstances = n.prev.coveredInstances.copy();
                    n.coveredInstances.filterByRule(n.rule);
                    allBeamsDone = false;
                    doneBeams[j] = false;
                } else {
                    // If we already have precomputed the coverage this rule was a best
                    // rule before, so no need to explore it further as it must be better than its children
                    doneBeams[j] = true;
                }
            }
        }
        Collections.sort(curBestRules);
        return curBestRules.get(curBestRules.size() - 1).reconstructConjunction();
    }

    /**
     * Returns the unique values found in attIndex in Instances, up to
     * some maximum.
     *
     * @param instances Instances to get unique values from
     * @param attIndex attribute index to get the values from
     * @param max max unique values to get before stopping
     * @return HashSet up to size max of the unique values found
     */
    private HashSet<Double> getUniqueValues(Instances instances, int attIndex, int max) {
        HashSet<Double> s = new HashSet<Double>();
        for(int i = 0; i < instances.numInstances(); i++) {
            s.add(instances.instance(i).value(attIndex));
            if(s.size() == max) {
                return s;
            }
        }
        return s;
    }

    /**
     * Builds a set of CachedRules to use when searching the rule space.
     *
     * @param instances Instances to build the rules from
     * @param targets boolean array marking which Instances are targets
     * @param idIndex index of the idAttribute
     * @return List of rules to use
     */
    public List<CachedRule> generateRules(Instances instances, boolean[] targets, int idIndex) {
        List<CachedRule> ruleList = new ArrayList<CachedRule>();
        Instances instancesCopy = new Instances(instances);
        for(int attIndex = 0; attIndex < instances.numAttributes(); attIndex++) {
            if(!useClass && attIndex == instances.classIndex() || attIndex == idIndex) {
                // skip id and possibly class attributes
                continue;
            }
            Attribute att = instances.attribute(attIndex);
            if(att.isNominal()) {
                // For nominal values build equals and not equals rules
                if(att.numValues() > 1) {
                    for(int j = 0; j < att.numValues(); j++) {
                        ruleList.add(BasicCachedRule.EqualsRule(instances, att, j));
                        if(att.numValues() > 2) {
                            // Redundant to use != rules if only 2 values
                            ruleList.add(BasicCachedRule.NotEqualsRule(instances, att, j));
                        }
                    }
                }
            } else {
                // Numeric values
                HashSet<Double> s = getUniqueValues(instances, attIndex, USE_QUANTILES);
                if(s.size() <= 3) {
                    // Treat as categorical if they are very limited
                    if(s.size() > 1) {
                        for(double val : s) {
                            ruleList.add(BasicCachedRule.EqualsRule(instances, att, val));
                            if(s.size() > 2) {
                                ruleList.add(BasicCachedRule.NotEqualsRule(instances, att, val));
                            }
                        }
                    }
                } else if(quantiles > 0 && s.size() >= USE_QUANTILES) {
                    // Use quantiles, look for all points where the value and class change
                    // as we scan the sorted Instances.
                    instancesCopy.sort(attIndex);
                    double prevQuantile = instancesCopy.firstInstance().value(attIndex);
                    for(int q = 1; q < quantiles + 2; q++) {
                        // Find quantiles by scanning down block of Instances
                        double quantile = instancesCopy.instance((instancesCopy.numInstances()*q)
                                / (quantiles + 2)).value(attIndex);
                        if(quantile != prevQuantile) {
                            ruleList.add(BasicCachedRule.GreaterOrEqualRule(instances, att, quantile));
                            ruleList.add(BasicCachedRule.SmallerThanRule(instances, att, quantile));
                        }
                        prevQuantile = quantile;
                    }
                } else {
                    // Do not use quartiles
                    int[] indices = Utils.sort(instances.attributeToDoubleArray(attIndex));

                    // True, False, null if instances with current or previous value are all
                    // targets, all non-targets, or mixed
                    Boolean prevClass = targets[indices[0]];
                    Boolean curClass = targets[indices[0]];

                    double prevVal = instances.instance(indices[0]).value(attIndex);
                    double firstVal = prevVal;
                    double lastVal = instances.instance(indices[indices.length-1]).value(attIndex);
                    for(int j = 0; j < indices.length; j++) {
                        int curIndex = indices[j];
                        double val = instances.instance(curIndex).value(attIndex);
                        if(prevVal == val) {
                            // If value becomes mixed, we might need to split between the previous level if we
                            // failed to do so already
                            if(prevClass != null && // If the old level was not mixed
                                    curClass != null && curClass != targets[curIndex] && // And cur level became mixed
                                    curClass == prevClass && // And cur/prev level were the same class
                                    val != firstVal // And this is not the first value
                                    ) {
                                ruleList.add(BasicCachedRule.SmallerThanRule(instances, att, val));
                                if(lastVal != val)
                                    ruleList.add(BasicCachedRule.GreaterOrEqualRule(instances, att, val));
                            }
                            // Set this level to mixed if we need to
                            if(curClass != null && curClass != targets[curIndex]) {
                                curClass = null;
                            }
                        } else {
                            prevClass = curClass;
                            curClass = targets[curIndex];;
                            if(curClass == null || prevClass == null ||
                                    prevClass != curClass) {
                                ruleList.add(BasicCachedRule.SmallerThanRule(instances, att, val));
                                if(lastVal != val)
                                    ruleList.add(BasicCachedRule.GreaterOrEqualRule(instances, att, val));
                            }
                        }
                        prevVal = val;
                    }
                }
            }
        }
        return ruleList;
    }

    /**
     * Find a set of rules conjunctions that define areas in the feature space that have a
     * relatively high number of targets using the current configuration.
     *
     * @param instances Instances to mine
     * @param targets BitSet where a bit is set iff the corresponding
     *                Instance in instances is considered a target.
     * @param rules Rules to consider when building the conjunctions
     * @param log Logger to output status updates to
     * @return The set of conjunctions found, sorted by our internal scoring mechanism
     */
    public CachedRuleDisjunction<CachedRuleConjunction<CachedRule>> mineData(Instances instances, BitSet targets,
            Collection<CachedRule> rules, Logger log) {
        BitInstancesView validationView;
        BitInstancesView trainView;
        if(prune) {
            int validationSize = (int) (instances.numInstances()*VALIDATION_SIZE);
            validationView = new BitInstancesView(targets, 0, validationSize);
            trainView = new BitInstancesView(targets, validationSize);
        } else {
            trainView = new BitInstancesView(targets, instances.numInstances());
            validationView = null;
        }
        CachedRuleDisjunction<CachedRuleConjunction<CachedRule>> ruleSet =
                new CachedRuleDisjunction<CachedRuleConjunction<CachedRule>>(instances.numInstances());
        double validationBaseLine = 0.0;
        if(validationView != null) {
            validationBaseLine  = ruleEvaluator.calcBaseline(validationView);
        }

        do {
            double trainBaseLine  = ruleEvaluator.calcBaseline(trainView);
            CachedRuleConjunction<CachedRule> newRule = greedyLearnBeams(trainView, rules, trainBaseLine);
            printIfDebug("New greedily learned " + newRule.toString());
            if(prune) {
                pruneRule(validationView, newRule, validationBaseLine);
                printIfDebug("Pruned rule " + newRule.toString());
            }
            if(newRule.size() == 0) {
                sortRules(ruleSet, new BitInstancesView(targets, instances.numInstances()));
                return ruleSet;
            }
            ruleSet.add(newRule);
            log.statusMessage("Found rule number " + ruleSet.size());
            trainView.removeCoveredTargets(newRule);
        } while (trainView.targets() != 0 && ruleSet.size() < maxRules);
        sortRules(ruleSet, new BitInstancesView(targets, instances.numInstances()));
        return ruleSet;
    }

    /**
     * Sorts rules in a CachedRuleSet by how well the rules score on some
     * data.
     *
     * @param rules CachedRuleSet to sort
     * @param view BitInstancesView to use when evaluating the rules
     */
    public void sortRules(
             CachedRuleSet<CachedRuleConjunction<CachedRule>> rules,
             BitInstancesView view) {
        double[] scores = new double[rules.size()];
        double baseLine = ruleEvaluator.calcBaseline(view);
        for(int i = 0; i < scores.length; i++) {
            scores[i] = ruleEvaluator.evalRule(rules.get(i), view, baseLine);
        }
        int[] indices = Utils.stableSort(scores);
        for(int i = indices.length - 1; i >= 0; i--) {
            // Take next best rule and swap it to the front  
            rules.swap(indices[i], indices.length - i - 1);
        }
    }

    /**
     * Prunes a CachedRuleConjunction by greedily removing rules whose 
     * presence reduces its score on a validation set.
     * 
     * @param iv BitInstancesView to evaluate the rules on
     * @param rules CachedRuleConjunction to prune
     * @param baseline baseline score of the BitInstancesView
     */
    public void pruneRule(BitInstancesView iv, 
                          CachedRuleConjunction<CachedRule> rules, double baseline) {
        printIfDebug("Starting pruning rule " + rules.toString());
        double bestScore = ruleEvaluator.evalRule(rules, iv, baseline);
        Integer bestRemoveIndex;
        do {
            bestRemoveIndex = null;
            // Try to remove every clause and see what gets the best score
            for(int i = 0; i < rules.size(); i++) {
                CachedRule removedRule = rules.remove(i);
                double newScore = ruleEvaluator.evalRule(rules, iv, baseline);
                if(bestScore <= newScore) {
                    bestScore = newScore;
                    bestRemoveIndex = i;
                }
                printIfDebug("\tConsidering: " + rules.toString() + " " + bestScore);
                rules.add(i, removedRule);
            }
            if(bestRemoveIndex != null) {
                printIfDebug("Pruned to: " + rules.toString() + " " + bestScore);
                rules.remove(bestRemoveIndex);
            }
        // Stop if nothing helped or at the Empty rule
        } while (bestRemoveIndex != null && rules.size() != 0);
    }

    // Translates a boolean to the representation of True or False for an attribute
    private double toValue(boolean input, Attribute att) {
        return input ? att.indexOfValue("True") : att.indexOfValue("False");
    }

    /**
     * Marks Instances with attributes to indicate whether each Instances was
     * a target and what rules applied to it. Creates categorical attributes
     * with values True and False for each rule (named after the rules) and
     * a "Was Target" attribute. Returns a modified copy of the data.
     *
     * @param data Instances to mark
     * @param rules CachedRuleDisjunction to mark the data with
     * @param targets targets to mark the data with
     * @return marked copy of the data
     */
    public Instances getMarkedDataset(Instances data,
                                      CachedRuleDisjunction<CachedRuleConjunction<CachedRule>> rules,
            boolean[] targets) {
        Instances newData = new Instances(data, targets.length);
        FastVector values = new FastVector();
        values.addElement("True");
        values.addElement("False");
        for(int i = 0; i < rules.size(); i++) {
            CachedRuleConjunction<CachedRule> rule = rules.get(i);
            newData.insertAttributeAt(
                    new Attribute(rule.toString(), values),
                    newData.numAttributes());
        }
        newData.insertAttributeAt(new Attribute("Was Target", values), newData.numAttributes());
        for(int i = 0; i < targets.length; i++) {
            double[] instanceValues = new double[newData.numAttributes()];
            Instance oldInst = data.instance(i);
            int curIndex = 0;
            for(int j = 0; j < data.numAttributes(); j++) {
                instanceValues[curIndex++] = oldInst.value(j);
            }
            for(int j = 0; j < rules.size(); j++) {
                instanceValues[curIndex++] =
                        toValue(rules.get(j).covered().get(i), newData.attribute(curIndex));
            }
            instanceValues[curIndex] = toValue(targets[i], newData.attribute(curIndex));
            newData.add(new Instance(1, instanceValues));
        }
        return newData;
    }

    /**
     * Marks Instances with attributes to indicate whether each Instances was
     * a target and what rules applied to it. Creates categorical attributes
     * with values True and False for each rule (named after the rules) and
     * a Was Target attribute. Returns a modified copy of the data. Calculates
     * the Rules based on the current configuration.
     *
     * @param data Instances to calculates the rules from
     * @param dataToMark Instances to mark with the results. Should have the same
     *                   ordering as daa
     * @param idIndex attribute index of data to treat as an ID
     * @param targets boolean[] of what Insances are targets
     * @param logger Logger to log incremental status updates to
     * @return Copy of dataToMark with rules and target attributes
     * @throws Exception If there was a problem marking the data
     */
    public Instances markData(Instances data, Instances dataToMark, int idIndex,
            boolean[] targets, Logger logger) throws Exception {
        BitSet targetBits = new BitSet(targets.length);
        for(int i = 0; i < targets.length; i++) {
            if(targets[i])
                targetBits.set(i);
        }
        List<CachedRule> ruleSet = generateRules(data, targets, idIndex);
        CachedRuleDisjunction<CachedRuleConjunction<CachedRule>> minedRules =
                mineData(data, targetBits, ruleSet, logger);
        return getMarkedDataset(dataToMark, minedRules, targets);
    }

    @Override
    public AnalyzerOutput analyzeData(Instances data, int idIndex, Logger logger) throws Exception {
        data = new Instances(data); // Ensure order will be preserved for our copy
        data.randomize(rand);

        // Pick out the targets
        double[][] predictions = RunClassifier.runClassifier(
                idIndex == -1 ? data : AnalyzerUtils.removeColumn(data, idIndex),
                classifier, cvFolds, classificationIterations,
                logger);
        boolean[] misclassifications = new boolean[predictions.length];
        BitSet misclassificationsBits = new BitSet(predictions.length);
        for(int i = 0; i < predictions.length; i++) {
            if(predictions[i][(int)data.instance(i).classValue()] /
                    (double) Utils.sum(predictions[i]) < cutoff) {
                misclassifications[i] = true;
                misclassificationsBits.set(i);
            }
        }

        // Build the rules
        logger.statusMessage("Generating Rules");
        List<CachedRule> ruleSet = generateRules(data, misclassifications, idIndex);

        // Build the rule conjunctions
        logger.statusMessage("Mining misclassifications");
        CachedRuleDisjunction<CachedRuleConjunction<CachedRule>> minedRules =
                mineData(data, misclassificationsBits, ruleSet, logger);


        // Build the text output
        BitInstancesView all = new BitInstancesView(misclassificationsBits, data.numInstances());
        StringBuilder msg = new StringBuilder();
        int errors = misclassificationsBits.cardinality();
        double accuracy = 1 - ((double) errors) / misclassifications.length;
        msg.append(String.format("Classifier made: %d mistakes\naccuracy: %.3f%n", errors, accuracy));
        msg.append("=== Confusion Matrix ===\n");
        String[] confusionMatrixHeader = new String[data.classAttribute().numValues()];
        for(int j = 0; j < data.classAttribute().numValues(); j++) {
            confusionMatrixHeader[j] = data.classAttribute().value(j);
        }
        AnalyzerUtils.writeSquareMatrix(msg, confusionMatrixHeader,
                AnalyzerUtils.confusionMatrix(data, predictions));

        msg.append("\nDetails written to ConfusionMatrix on the results list");
        msg.append("\nGenerated: " + ruleSet.size() + " potential rules.\n");
        msg.append("Final Rules:\n\n");
        for(int i = 0; i < minedRules.size(); i++) {
            CachedRuleConjunction<CachedRule> r = minedRules.get(i);
            msg.append("Rule: " + i + "\n");
            msg.append(r.toString() + "\n");
            msg.append(ruleReport(r, r.size(), all) + "\n\n");
        }

        // Build the visualizations
        GenerateVisualization[] gvs = new GenerateVisualization[minedRules.size() + 1];
        ConfusionMatrixVisualizer cmView =
                new ConfusionMatrixVisualizer("Confusion Matrix", 500, 500,
                        data, predictions, idIndex);
        gvs[0] = cmView;

        String formatStr = "";
        String[] attNames = new String[data.numAttributes()];
        for(int a = 0; a < data.numAttributes(); a++) {
            Attribute att = data.attribute(a);
            if(att.isNumeric()) {
                formatStr += "%-" + Integer.toString((int)Math.max(att.name().length(), 6)) + "s  ";
            } else {
                int max = att.name().length();
                Enumeration<String> e = att.enumerateValues();
                while(e.hasMoreElements()) {
                    max = (int) Math.max(max, e.nextElement().length());
                }
                formatStr += "%-" + Integer.toString(max) + "s  ";
            }
            attNames[a] = data.attribute(a).name();
        }
        int numInstances = data.numInstances();
        for(int i = 0; i < minedRules.size(); i++) {
            CachedRuleConjunction<CachedRule> r = minedRules.get(i);
            final StringBuilder sb = new StringBuilder();
            sb.append("Rule " + (i + 1) + ":\n" + r.toString() + "\n\n");
            sb.append("Stats:\n" + ruleReport(r, r.size(), all) + "\n\n");
            sb.append("Confusion Matrix:\n");
            sb.append(getRuleConfusionMatrix(data, predictions, r));
            sb.append("\nBreak down:\n\n");
            CachedRuleConjunction<CachedRule> rs = new CachedRuleConjunction<CachedRule>(numInstances);
            CachedRule emptyRule = BasicCachedRule.emptyRule(numInstances);
            sb.append("Baseline (empty rule):\nStats:" +  ruleReport(emptyRule, 0, all) + "\n\n");
            for(CachedRule clause : r) {
                rs.add(clause);
                sb.append("Added: " + clause.toString() + "\n");
                sb.append("New stats: " + ruleReport(rs, rs.size(), all) + "\n\n");
            }
            BitSet bs = r.covered();

            // Print out some ids if we can
            if(idIndex != -1) {
                if(MAX_IDS_TO_PRINT < bs.cardinality()) {
                    // TODO should print a few examples Instances anyway
                    sb.append("Too many IDs to print");
                } else {
                    Attribute idAtt = data.attribute(idIndex);
                    sb.append("Instances ids:\n");
                    int j = bs.nextSetBit(0);
                    String idStr = AnalyzerUtils.attValStr(data.attribute(idIndex),
                            data.instance(j).value(idAtt));
                    sb.append(idStr);
                    int lineLength = idStr.length();
                    j =  bs.nextSetBit(j+1);
                    for(; j>=0; j=bs.nextSetBit(j+1)) {
                        idStr = AnalyzerUtils.attValStr(data.attribute(idIndex),
                                data.instance(j).value(idAtt));
                        if(lineLength + 2 + idStr.length() > ID_LINE_LENGTH) {
                            sb.append("\n" + idStr);
                            lineLength = idStr.length();
                        } else {
                            sb.append(", " + idStr);
                            lineLength += 2 + idStr.length();
                        }
                    }
                    sb.append("\n\n");
                }
            }

            // Print some example instances
            if(data.numAttributes() <= MAX_ATTRIBUTES_TO_PRINT) {
                sb.append("Examples:\n");
                Formatter formatter = new Formatter(sb);
                formatter.format(formatStr,(Object[])attNames);
                sb.append("\n");
                int examplesShown = 0;
                for(int j = bs.nextSetBit(0); j>=0 && examplesShown < MAX_EXAMPLES_TO_PRINT;
                    j=bs.nextSetBit(j+1)) {
                    examplesShown++;
                    formatter.format(formatStr,(Object[])data.instance(j).toString().split(","));
                    sb.append("\n");
                }
                formatter.close();
            }
            String ruleNum = "Rule " + Integer.toString(i + 1);
            gvs[i + 1] =  new GenerateTextWindow(ruleNum, sb.toString(), 500, 550);
        }
        return new AnalyzerOutput(msg.toString(), gvs);
    }

    /**
     * Gets a text confusion matrix for some Instances covered by a .
     *
     * @param data Instances the rule applies to
     * @param predictions per Instance per class prediction counts
     * @param rule CachedRule to build the confusion matrix for
     * @return String confusion matrix, suitable for showing to a user.
     */
    public String getRuleConfusionMatrix(Instances data, double[][] predictions, CachedRule rule) {
        StringBuilder report = new StringBuilder();
        BitSet covered = rule.covered();
        int size = covered.cardinality();
        double[][] subsetPred = new double[size][predictions[0].length];
        int cur = 0;
        Instances subset = new Instances(data, covered.cardinality());
        for (int i = covered.nextSetBit(0); i >= 0; i = covered.nextSetBit(i+1)) {
            subset.add(data.instance(i));
            subsetPred[cur] = predictions[i];
            cur++;
        }
        double[][] confusionMatrix = AnalyzerUtils.confusionMatrix(subset, subsetPred);
        String[] labels = new String[data.classAttribute().numValues()];
        for(int i = 0; i < data.classAttribute().numValues(); i++) {
            labels[i] = data.classAttribute().value(i);
        }
        AnalyzerUtils.writeSquareMatrix(report, labels, confusionMatrix);
        return report.toString();
    }

    // Option setting methods

    @Override
    public Enumeration<Option> listOptions() {
        Vector<Option> newVector = new Vector<Option>();
        newVector.addElement(new Option(
                "\tCross validation folds\n",
                "V", 1, "-V"));
        newVector.addElement(new Option(
                "Classification Iterations\n",
                "I", 1, "-I"));
        newVector.addElement(new Option(
                "\tCutoff\n",
                "C", 1, "-C"));
        newVector.addElement(new Option(
                "\tCross validation folds\n",
                "V", 1, "-V"));
        newVector.addElement(new Option(
                "\tLalplace Accuracy\n",
                "K", 1, "-K"));
        newVector.addElement(new Option(
                "\tRule Penality\n",
                "T", 1, "-T"));
        newVector.addElement(new Option(
                "\tQuartiles\n",
                "Q", 1, "-Q"));
        newVector.addElement(new Option(
                "\tPrune Rules\n",
                "P", 0, "-P"));
        newVector.addElement(new Option(
                "\tUse Class\n",
                "C", 0, "-C"));
        newVector.addElement(new Option(
                "\tUse Prediction\n",
                "R", 0, "-R"));
        newVector.addElement(new Option(
                "\tRandom Seed\n",
                "S", 1, "-S"));
        newVector.addElement(new Option(
                "Max Rules\n",
                "M", 0, "-M"));

        Enumeration<Option> enu = super.listOptions();
        while (enu.hasMoreElements()) {
            newVector.addElement(enu.nextElement());
        }
        return newVector.elements();
    }

    @Override
    public void setOptions(String[] options) throws Exception {
        String cvString = Utils.getOption('V', options);
        cvFolds = cvString.length() != 0 ? Integer.parseInt(cvString) : 10;

        String kString = Utils.getOption('K', options);
        k = kString.length() != 0 ? Integer.parseInt(kString) : 20;

        String cIterationsString = Utils.getOption('I', options);
        classificationIterations = cIterationsString.length() != 0 ?
                Integer.parseInt(cIterationsString) : 1;

        String cutoffString = Utils.getOption('I', options);
        cutoff = cutoffString.length() != 0 ? Double.parseDouble(cutoffString) : 0.80;

        String rpString = Utils.getOption('T', options);
        rulePenalty = rpString.length() != 0 ? Double.parseDouble(rpString) : 0.0;

        String quartileString = Utils.getOption('Q', options);
        quantiles = quartileString.length() != 0 ? Integer.parseInt(quartileString) : 20;

        String maxRulesString = Utils.getOption('M', options);
        quantiles = maxRulesString.length() != 0 ? Integer.parseInt(maxRulesString) : 10;

        String seedString = Utils.getOption('S', options);
        seed = seedString.length() != 0 ? Long.parseLong(seedString) : 0;

        prune = Utils.getFlag('P', options);
        useClass = Utils.getFlag('C', options);

        super.setOptions(options);
    }

    @Override
    public String[] getOptions() {
          String[] superOptions = super.getOptions();
          String[] options = new String [19 + superOptions.length];

          int current = 0;
          options[current++] = "-V";
          options[current++] =  Integer.toString(cvFolds);
          options[current++] = "-I";
          options[current++] =  Integer.toString(classificationIterations);
          options[current++] = "-C";
          options[current++] =  Double.toString(cutoff);
          options[current++] = "-K";
          options[current++] =  Integer.toString(k);
          options[current++] = "-T";
          options[current++] =  Double.toString(rulePenalty);
          options[current++] = "-Q";
          options[current++] =  Integer.toString(quantiles);
          options[current++] = "-M";
          options[current++] =  Integer.toString(maxRules);
          options[current++] = "-S";
          options[current++] =  Long.toString(seed);

          if (prune) {
              options[current++] = "-P";
          }
          if (useClass) {
              options[current++] = "-C";
          }

          System.arraycopy(superOptions, 0, options, current,
                  superOptions.length);
          current += superOptions.length;
          while(current < options.length) {
              options[current++] = "";
          }
          return options;
      }

    // Getters, Setters, and Tooltips

    public int getK() {
        return k;
    }
    public void setK(int k) {
        this.k = k;
    }

    public boolean getUseClass() {
        return useClass;
    }
    public void setUseClass(boolean useClass) {
        this.useClass = useClass;
    }

    public int getMaxRules() {
        return maxRules;
    }
    public void setMaxRules(int maxRules) {
        this.maxRules = maxRules;
    }

    public int getClassificationIterations() {
        return classificationIterations;
    }
    public void setClassificationIterations(int classificationIterations) {
        this.classificationIterations = classificationIterations;
    }

    public double getCutoff() {
        return cutoff;
    }

    public void setCutoff(double cutoff) {
        this.cutoff = cutoff;
    }

    public int getBeams() {
        return beams;
    }
    public void setBeams(int beams) {
        this.beams = beams;
    }

    public int getQuantiles() {
        return quantiles;
    }
    public void setQuantiles(int quantiles) {
        this.quantiles = quantiles;
    }

    public int getCVFolds() {
        return cvFolds;
    }
    public void setCVFolds(int cv) {
        cvFolds = cv;
    }
    public long getSeed() {
        return seed;
    }
    public void setSeed(long seed) {
        this.seed = seed;
        this.rand = new Random(seed);
    }

    public boolean getPruneRule() {
        return prune;
    }
    public void setPruneRule(boolean prune) {
        this.prune = prune;
    }

    public double getRulePenalty() {
        return rulePenalty;
    }
    public void setRulePanlity(double rulePenality) {
        this.rulePenalty = rulePenality;
    }

    public String cutoffTipText() {
        return "Cutoff for deciding when an Instance is a target when using multiple " +
                "rounds of classification or regression.";
    }
    public String useClassTipText() {
        return "Use the class attribute to generate the rules. This in general allow" +
                " us to find better rules.";
    }
    public String MaxRulesTipText() {
        return "Max rules to generate, if negative we generate the maximum number of " +
                "rules possible (but will still eventually terminate).";
    }
    public String classificationIterationsTipText() {
        return "Number of times to classify the data, increasing this can make";
    }
    public String KTipText() {
        return "K value to use when evaluating rule, higher K favours " +
                "broader but less accurate rules";
    }
    public String CVFoldsTipText() {
        return "Number of folds to use when generating the cross validation data. More folds " +
                "will take longer but increase the accuracy of the labels we use when splitting" +
                " the data into misclassified examples and correctly classified examples.";
    }
    public String pruneRuleTipText() {
        return "Should the rules be pruned? Pruning will help stop the rules " +
                "overfitting, but might reduce our accuracy since some of that data" +
                " will need to be set aside for a validation set. It is normally" +
                " more effective to adjust rulePenality or K to combat overfitting.";
    }
    public String beamsRuleTipText() {
        return "Number of beams to search for rule conjunction with, more beams will " +
                "yield better rules and the cost of more computation.";
    }
    public String quartiles0RuleTipText() {
        return "For numeric attributes, the number of different quantiles rule's should" +
                " split the values of that attribute into. If less then 0 the attribute" +
                " rules will be built that can split that attribute at every useful point." +
                " More rules can be increase the accuracy of MisclassificationMiner at the" +
                "cost of computation.";
    }
    public String seedTipText() {
        return "Random seed to use.";
    }

    public String rulePenalityTipText() {
        return "Value to penalize a conjunction of rule's score " +
                "in proportion to the number of rules in the conjunction." +
                " A higher value will make MisclassificationMiner favour" +
                "shorter rules.";
    }
}
