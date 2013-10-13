package weka.analyzers.mineData;

import java.util.BitSet;

/**
 * Represents a subset of some indexed dataset and what Instances of that subset
 * are considered to be targets. Efficiently supports a number of calculations
 * and filtering operations given BitSet representations of additional subsets.
 */
public class BitInstancesView {

    /**
     * BitSet where a bit is set iff the corresponding Instance is a target.
     * Targets will not be modified and so this can be a reference.
     */
    protected final BitSet targets;

    /**
     * BitsSet where a bit is set iff the Instance at the
     * same index is considered to be in this BitInstancesView.
     */
    protected final BitSet covered;

    /**
     * Construct a new BitInstancesView that covers all instances
     * from start to stop.
     *
     * @param targets BitSet Marking the targets in the data, will
     *                not be modified
     * @param start the starting index of this data included in this
     * @param stop the ending index of this data included in this
     */
    public BitInstancesView(BitSet targets, int start, int stop) {
        this.targets = targets;
        covered = new BitSet(targets.size());
        covered.set(start, stop);
    }

    /**
     * Construct a new BitInstancesView that covers all instances
     * from the first instance to stop.
     *
     * @param targets BitSet Marking the targets in the data, will
     *                not be modified
     * @param stop the ending index of this data included in this
     */
    public BitInstancesView(BitSet targets, int stop) {
        this(targets, 0, stop);
    }

    /**
     * Construct a new BitInstancesView that covers all instances
     * covered by a BitSet.
     *
     * @param targets BitSet Marking the targets in the data, will
     *                not be modified
     * @param covered the Instances from the indexed dataset this covers
     */

    public BitInstancesView(BitSet targets, BitSet covered) {
        this.covered = covered;
        this.targets = targets;
    }

    /**
     * Remove all instances from this that are not covered by a Rule.
     *
     * @param rule Rule to filter this by
     */
    public void filterByRule(CachedRule rule) {
        covered.and(rule.covered());
    }

    /**
     * Remove instances from this dataset that are both covered by the given rule
     * and are targets.
     *
     * @param rule Rule to filter by
     */
    public void removeCoveredTargets(CachedRule rule) {
        BitSet coveredTargets = (BitSet) targets.clone();
        coveredTargets.and(rule.covered());
        covered.andNot(coveredTargets);
    }

    /**
     * Returns the of Instances covered and number of targets covered
     * by the intersection of the instances covered by this and the given rule.
     *
     * @param rule CachedRule to evaluate
     * @return int[] containing the number of instances covered and number of
     * targets covered
     */
    public int[] evaluateRule(CachedRule rule) {
        BitSet coveredCopy = (BitSet) covered.clone();
        coveredCopy.and(rule.covered());
        int totalCovered = coveredCopy.cardinality();
        coveredCopy.and(targets);
        int totalTargets = coveredCopy.cardinality();
        return new int[] {totalCovered, totalTargets};
    }

    /**
     * @return number of targets this includes
     */
    public int targets() {
        BitSet targetsCopy = (BitSet) targets.clone();
        targetsCopy.and(covered);
        return targetsCopy.cardinality();
    }

    /**
     * @return number of instances this includes
     */
    public int size() {
        return covered.cardinality();
    }

    /**
     * @return Deep copy of the this
     */
    public BitInstancesView copy() {
        return new BitInstancesView(targets, (BitSet) covered.clone());
    }
}