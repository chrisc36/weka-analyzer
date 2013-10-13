package weka.analyzers.mineData;


import java.util.BitSet;

/**
 * A Conjunction of other CachedRules.
 *
 * @param <T> the type of CachedRule this CachedRuleConjunction contains
 */
public class CachedRuleConjunction<T extends CachedRule> extends CachedRuleSet<T> {

    /**
     * Constructs a CachedRule designed to cover a given number of instances.
     *
     * @param numInstances
     */
    public CachedRuleConjunction(int numInstances) {
        super(numInstances, "AND");
    }

    @Override
    protected void addBitSet(BitSet other) {
        covered.and(other);
    }
}
